// nnet3bin/nnet3-latgen-faster-parallel-pipe.cc

// Copyright 2012-2020   Johns Hopkins University (author: Daniel Povey)
//                2014   Guoguo Chen

// Created from nnet3bin/nnet3-latgen-faster-parallel by Airenas Vaièiûnas (VDU) 2020
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


#include "base/timer.h"
#include "base/kaldi-common.h"
#include "decoder/decoder-wrappers.h"
#include "fstext/fstext-lib.h"
#include "hmm/transition-model.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"
#include "util/kaldi-thread.h"
#include "tree/context-dep.h"
#include "util/common-utils.h"
#include <fstream>
#include <sys/stat.h>

using namespace kaldi;
using namespace kaldi::nnet3;
typedef kaldi::int32 int32;
using fst::SymbolTable;
using fst::Fst;
using fst::StdArc;

//--------------------------------------------------------
struct genData {
  bool allow_partial = false;
  int32 online_ivector_period = 0;

  LatticeFasterDecoderConfig config;
  NnetSimpleComputationOptions decodable_opts;
  TaskSequencerConfig sequencer_config; // has --num-threads option

  std::string pipe_in_str;

  Fst <StdArc> *decode_fst = nullptr;
  fst::SymbolTable *word_syms = nullptr;

  TaskSequencer <DecodeUtteranceLatticeFasterClass> *sequencer;
  AmNnetSimple *am_nnet;
  TransitionModel *trans_model;

  ~genData() {
    delete decode_fst;
    delete word_syms;
  }
};

//--------------------------------------------------------
struct latData {
  std::string online_ivector_rspecifier;
  std::string feature_rspecifier;
  std::string lattice_wspecifier;
  std::string pipe_out;
};

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
bool decode(genData &data, latData &ldata) {
  KALDI_LOG << "Decoding";

  LatticeWriter lattice_writer;
  CompactLatticeWriter compact_lattice_writer;
  bool determinize = data.config.determinize_lattice;
  if (!(determinize ? compact_lattice_writer.Open(ldata.lattice_wspecifier)
                    : lattice_writer.Open(ldata.lattice_wspecifier))) {
    KALDI_ERR << "Could not open table for writing lattices: " << ldata.lattice_wspecifier;
    return false;
  }

  Int32VectorWriter words_writer("");
  Int32VectorWriter alignment_writer("");

  RandomAccessBaseFloatMatrixReader online_ivector_reader(ldata.online_ivector_rspecifier);

  double tot_like = 0.0;
  kaldi::int64 frame_count = 0;
  int num_success = 0, num_fail = 0;

  SequentialBaseFloatMatrixReader feature_reader(ldata.feature_rspecifier);
  Timer timer;
  timer.Reset();

  for (; !feature_reader.Done(); feature_reader.Next()) {
    std::string utt = feature_reader.Key();
    KALDI_LOG << "UTT " << utt;
    const Matrix <BaseFloat> &features(feature_reader.Value());
    if (features.NumRows() == 0) {
      KALDI_WARN << "Zero-length utterance: " << utt;
      num_fail++;
      continue;
    }
    const Matrix <BaseFloat> *online_ivectors = NULL;
    const Vector <BaseFloat> *ivector = NULL;
    if (!ldata.online_ivector_rspecifier.empty()) {
      if (!online_ivector_reader.HasKey(utt)) {
        KALDI_WARN << "No online iVector available for utterance " << utt;
        num_fail++;
        continue;
      } else {
        online_ivectors = &online_ivector_reader.Value(utt);
      }
    }

    LatticeFasterDecoder *decoder = new LatticeFasterDecoder(*data.decode_fst, data.config);

    DecodableInterface *nnet_decodable = new DecodableAmNnetSimpleParallel(
        data.decodable_opts, *data.trans_model, *data.am_nnet,
        features, ivector, online_ivectors,
        data.online_ivector_period);

    DecodeUtteranceLatticeFasterClass *task = new DecodeUtteranceLatticeFasterClass(
        decoder, nnet_decodable, // takes ownership of these two.
        *data.trans_model, data.word_syms, utt, data.decodable_opts.acoustic_scale,
        determinize, data.allow_partial, &alignment_writer, &words_writer,
        &compact_lattice_writer, &lattice_writer,
        &tot_like, &frame_count, &num_success, &num_fail, NULL);

    data.sequencer->Run(task);
  }
  data.sequencer->Wait();

  kaldi::int64 input_frame_count = frame_count * data.decodable_opts.frame_subsampling_factor;

  double elapsed = timer.Elapsed();
  KALDI_LOG << "Time taken " << elapsed
            << "s: real-time factor assuming 100 feature frames/sec is "
            << (data.sequencer_config.num_threads * elapsed * 100.0 / input_frame_count);
  KALDI_LOG << "Done " << num_success << " utterances, failed for " << num_fail;
  KALDI_LOG << "Overall log-likelihood per frame is "
            << (tot_like / frame_count) << " over "
            << frame_count << " frames.";
  return num_success > 0;
}

//--------------------------------------------------------
bool processPipeInput(genData &data) {
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

  line = readWord(line, lData.online_ivector_rspecifier);
  if (lData.online_ivector_rspecifier.empty()) {
    KALDI_WARN << "No online_ivector_rspecifier in pipe";
    return true;
  }
  line = readWord(line, lData.feature_rspecifier);
  if (lData.feature_rspecifier.empty()) {
    KALDI_WARN << "No feature_rspecifier in pipe";
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
  KALDI_LOG << "online_ivector_rspecifier: " << lData.online_ivector_rspecifier;
  KALDI_LOG << "feature_rspecifier:        " << lData.feature_rspecifier;
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
    bool res = decode(data, lData);
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
        "Generate lattices using nnet3 neural net model.  This version supports\n"
        "multiple decoding threads (using a shared decoding graph.)\n"
        "Usage: nnet3-latgen-faster-parallel-pipe [options] <nnet-in> <pipeName-to-read-input-parameters-from> \n"
        "Pipe file format: <online_ivector-rspecifier> <feature_rspecifier> <lattice_wspecifier> <end-pipe-name>\n";
    ParseOptions po(usage);

    genData data;

    std::string word_syms_filename;
    data.sequencer_config.Register(&po);
    data.config.Register(&po);
    data.decodable_opts.Register(&po);

    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &data.allow_partial,
                "If true, produce output even if end state was not reached.");
    po.Register("online-ivector-period", &data.online_ivector_period, "Number of frames "
                                                                      "between iVectors in matrices supplied to the --online-ivectors "
                                                                      "option");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        fst_in_str = po.GetArg(2);
    data.pipe_in_str = po.GetArg(3);

    TaskSequencer <DecodeUtteranceLatticeFasterClass> sequencer(data.sequencer_config);
    data.sequencer = &sequencer;

    if (word_syms_filename != "") {
      if (!(data.word_syms = fst::SymbolTable::ReadText(word_syms_filename))) {
        KALDI_ERR << "Could not read symbol table from file " << word_syms_filename;
      }
    }

    TransitionModel trans_model;
    AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
      SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet.GetNnet()));
      CollapseModel(CollapseModelConfig(), &(am_nnet.GetNnet()));
    }

    data.am_nnet = &am_nnet;
    data.trans_model = &trans_model;

    KALDI_LOG << "Init fst " << fst_in_str;
    data.decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
    KALDI_LOG << "initialized fst ";

    while (true) {
      if (!processPipeInput(data)) {
        return 1;
      };
    }
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
