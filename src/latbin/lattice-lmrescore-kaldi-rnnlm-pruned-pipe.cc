// latbin/lattice-lmrescore-kaldi-rnnlm-pruned-pipe.cc

// Copyright 2017 Johns Hopkins University (author: Daniel Povey)
//           2017 Hainan Xu
//           2020 Airenas Vaièiûnas (VDU)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "fstext/fstext-lib.h"
#include "rnnlm/rnnlm-lattice-rescoring.h"
#include "lm/const-arpa-lm.h"
#include "util/common-utils.h"
#include "nnet3/nnet-utils.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "lat/compose-lattice-pruned.h"
#include <sys/stat.h>

using namespace kaldi;
typedef kaldi::int32 int32;
typedef kaldi::int64 int64;
using fst::SymbolTable;
using fst::VectorFst;
using fst::StdArc;
using fst::ReadFstKaldi;

//--------------------------------------------------------
struct rescoreData {
  std::string pipe_in_str;
  int32 max_ngram_order = 3;
  BaseFloat lm_scale = 0.5;
  BaseFloat acoustic_scale = 0.1;

  rnnlm::RnnlmComputeStateComputationOptions opts;
  ComposeLatticePrunedOptions compose_opts;

  const rnnlm::RnnlmComputeStateInfo *info;
  rnnlm::KaldiRnnlmDeterministicFst *lm_to_add_orig = nullptr;
  fst::ScaleDeterministicOnDemandFst *lm_to_subtract_det_scale = nullptr;

  ~rescoreData() {
    delete lm_to_add_orig;
    delete lm_to_subtract_det_scale;
  }
};

//--------------------------------------------------------
struct latData {
  std::string lattice_rspecifier;
  std::string lattice_wspecifier;
  std::string pipe_out;
};

//--------------------------------------------------------
bool pipeExists(std::string fileName) {
  struct stat fInfo;
  if (stat(fileName.c_str(), &fInfo) == 0) {
    return S_ISFIFO(fInfo.st_mode);
  }
  return false;
}
//--------------------------------------------------------
//--------------------------------------------------------
std::string readWord(const std::string &line, std::string &result) {
  result.clear();
  std::string::size_type pos_b = 0,
      pos_e = 0;
  for (pos_b = 0; pos_b < line.length() && std::isspace(line[pos_b]); pos_b++) {}

  if (pos_b >= line.length()) {
    return "";
  }
  char c = line[pos_b];
  if (c == '\"' || c == '\'') {
    pos_b++;
    for (pos_e = pos_b + 1; pos_e < line.length() && c != line[pos_e]; pos_e++) {}
  } else {
    for (pos_e = pos_b + 1; pos_e < line.length() && !std::isspace(line[pos_e]); pos_e++) {}
  }
  result = line.substr(pos_b, pos_e - pos_b);
  if (pos_e + 1 > line.length()) {
    return "";
  }
  return line.substr(pos_e + 1);
}

//--------------------------------------------------------
bool rescore(rescoreData &data, latData &ldata) {
  KALDI_LOG << "Rescoring";
  // Reads and writes as compact lattice.
  SequentialCompactLatticeReader compact_lattice_reader(ldata.lattice_rspecifier);
  CompactLatticeWriter compact_lattice_writer(ldata.lattice_wspecifier);

  int32 num_done = 0, num_err = 0;

  for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
    fst::DeterministicOnDemandFst <StdArc> *lm_to_add = new fst::ScaleDeterministicOnDemandFst(data.lm_scale, data.lm_to_add_orig);

    std::string key = compact_lattice_reader.Key();
    CompactLattice clat = compact_lattice_reader.Value();
    compact_lattice_reader.FreeCurrent();

    // Before composing with the LM FST, we scale the lattice weights
    // by the inverse of "lm_scale".  We'll later scale by "lm_scale".
    // We do it this way so we can determinize and it will give the
    // right effect (taking the "best path" through the LM) regardless
    // of the sign of lm_scale.
    if (data.acoustic_scale != 1.0) {
      fst::ScaleLattice(fst::AcousticLatticeScale(data.acoustic_scale), &clat);
    }
    TopSortCompactLatticeIfNeeded(&clat);

    fst::ComposeDeterministicOnDemandFst <StdArc> combined_lms(data.lm_to_subtract_det_scale, lm_to_add);

    // Composes lattice with language model.
    CompactLattice composed_clat;
    ComposeCompactLatticePruned(data.compose_opts, clat, &combined_lms, &composed_clat);

    data.lm_to_add_orig->Clear();

    if (composed_clat.NumStates() == 0) {
      // Something went wrong.  A warning will already have been printed.
      num_err++;
    } else {
      if (data.acoustic_scale != 1.0) {
        fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / data.acoustic_scale), &composed_clat);
      }
      compact_lattice_writer.Write(key, composed_clat);
      num_done++;
    }
    delete lm_to_add;
  }

  KALDI_LOG << "Overall, succeeded for " << num_done << " lattices, failed for " << num_err;
  return num_done > 0;
}

//--------------------------------------------------------
bool processPipeInput(rescoreData &data) {
  KALDI_LOG << "Reading pipe " << data.pipe_in_str;
  std::ifstream file;
  file.open(data.pipe_in_str);
  if (!file.is_open()) {
    KALDI_ERR << "Can't open " << data.pipe_in_str;
    return false;
  }
  std::string line;
  if (!std::getline(file, line)) {
    KALDI_WARN << "Can't read pipe";
    return true;
  }

  latData lData;

  line = readWord(line, lData.lattice_rspecifier);
  if (lData.lattice_rspecifier.empty()) {
    KALDI_WARN << "No lattice_rspecifier in pipe";
    return true;
  }
  line = readWord(line, lData.lattice_wspecifier);
  if (lData.lattice_wspecifier.empty()) {
    KALDI_WARN << "No lattice_wspecifier in pipe";
    return true;
  }
  line = readWord(line, lData.pipe_out);
  if (lData.pipe_out.empty()) {
    KALDI_WARN << "No pipe_out in pipe";
    return true;
  }
  KALDI_LOG << "lattice_rspecifier:        " << lData.lattice_rspecifier;
  KALDI_LOG << "lattice_wspecifier:        " << lData.lattice_wspecifier;
  KALDI_LOG << "pipe_out:                  " << lData.pipe_out;

  if (!pipeExists(lData.pipe_out)) {
    KALDI_WARN << "No file/pipe " << lData.pipe_out;
    return true;
  }
  std::ofstream o_pipe;
  o_pipe.open(lData.pipe_out, std::ios_base::app);
  if (!o_pipe.is_open()) {
    KALDI_WARN << "Can't open " << lData.pipe_out;
    return true;
  }

  o_pipe << "Start" << "\n";
  o_pipe.flush();
  try {
    bool res = rescore(data, lData);
    o_pipe << "End" << "\n";
    o_pipe << (res ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    o_pipe << e.what() << "\n";
    o_pipe << 1;
  }
  return true;
}

//--------------------------------------------------------
//--------------------------------------------------------
//--------------------------------------------------------
int main(int argc, char *argv[]) {
  try {
    const char *usage =
        "Rescores lattice with kaldi-rnnlm. \n"
        "This script preloads models and waits for rescore tasks from pipe.\n"
        "\n"
        "Usage: lattice-lmrescore-kaldi-rnnlm-pruned-pipe [options] \\\n"
        "             <old-lm-rxfilename> <embedding-file> \\\n"
        "             <raw-rnnlm-rxfilename> \\\n"
        "             <input-pipe-name>\n"
        " e.g.: lattice-lmrescore-kaldi-rnnlm-pruned --lm-scale=-1.0 fst_words.txt \\\n"
        "              --bos-symbol=1 --eos-symbol=2 \\\n"
        "              data/lang_test/G.fst word_embedding.mat \\\n"
        "              input.pipe\n\n"
        "Pipe file format: <lattice_rspecifier> <lattice_wspecifier> <end-pipe-name>\n";

    ParseOptions po(usage);

    rescoreData data;

    bool use_carpa = false;

    po.Register("lm-scale", &data.lm_scale, "Scaling factor for <lm-to-add>; its negative "
                                            "will be applied to <lm-to-subtract>.");
    po.Register("acoustic-scale", &data.acoustic_scale, "Scaling factor for acoustic "
                                                        "probabilities (e.g. 0.1 for non-chain systems); important because "
                                                        "of its effect on pruning.");
    po.Register("max-ngram-order", &data.max_ngram_order,
                "If positive, allow RNNLM histories longer than this to be identified "
                "with each other for rescoring purposes (an approximation that "
                "saves time and reduces output lattice size).");
    po.Register("use-const-arpa", &use_carpa, "If true, read the old-LM file "
                                              "as a const-arpa file as opposed to an FST file");

    data.opts.Register(&po);
    data.compose_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    if (data.opts.bos_index == -1 || data.opts.eos_index == -1) {
      KALDI_ERR << "must set --bos-symbol and --eos-symbol options";
    }
    if (data.acoustic_scale == 0.0) {
      KALDI_ERR << "Acoustic scale cannot be zero.";
    }

    std::string lm_to_subtract_rxfilename, word_embedding_rxfilename, rnnlm_rxfilename;

    lm_to_subtract_rxfilename = po.GetArg(1),
        word_embedding_rxfilename = po.GetArg(2);
    rnnlm_rxfilename = po.GetArg(3);
    data.pipe_in_str = po.GetArg(4);

    // for G.fst
    fst::BackoffDeterministicOnDemandFst <StdArc> *lm_to_subtract_det_backoff = NULL;
    VectorFst <StdArc> *lm_to_subtract_fst = NULL;

    // for G.carpa
    ConstArpaLm *const_arpa = NULL;
    fst::DeterministicOnDemandFst <StdArc> *carpa_lm_to_subtract_fst = NULL;

    KALDI_LOG << "Reading old LMs...";
    if (use_carpa) {
      const_arpa = new ConstArpaLm();
      ReadKaldiObject(lm_to_subtract_rxfilename, const_arpa);
      carpa_lm_to_subtract_fst = new ConstArpaLmDeterministicFst(*const_arpa);
      data.lm_to_subtract_det_scale = new fst::ScaleDeterministicOnDemandFst(-data.lm_scale, carpa_lm_to_subtract_fst);
    } else {
      lm_to_subtract_fst = fst::ReadAndPrepareLmFst(
          lm_to_subtract_rxfilename);
      lm_to_subtract_det_backoff =
          new fst::BackoffDeterministicOnDemandFst<StdArc>(*lm_to_subtract_fst);
      data.lm_to_subtract_det_scale = new fst::ScaleDeterministicOnDemandFst(-data.lm_scale, lm_to_subtract_det_backoff);
    }

    kaldi::nnet3::Nnet rnnlm;
    ReadKaldiObject(rnnlm_rxfilename, &rnnlm);

    KALDI_ASSERT(IsSimpleNnet(rnnlm));

    CuMatrix <BaseFloat> word_embedding_mat;
    ReadKaldiObject(word_embedding_rxfilename, &word_embedding_mat);

    const rnnlm::RnnlmComputeStateInfo info(data.opts, rnnlm, word_embedding_mat);
    data.info = &info;

    data.lm_to_add_orig = new rnnlm::KaldiRnnlmDeterministicFst(data.max_ngram_order, *data.info);

    int res = 0;
    while (true) {
      if (!processPipeInput(data)) {
        res = 1;
        break;
      };
    }
    delete lm_to_subtract_det_backoff;

    delete const_arpa;
    delete carpa_lm_to_subtract_fst;

    return res;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
