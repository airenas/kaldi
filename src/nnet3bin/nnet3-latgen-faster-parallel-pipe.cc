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

//--------------------------------------------------------
struct genData{
  bool determinize;
  std::string pipe_in_str;
};

//--------------------------------------------------------
struct latData{
  std::string online_ivector_rspecifier;
  std::string feature_rspecifier;
  std::string lattice_wspecifier;
  std::string pipe_out;
};

//--------------------------------------------------------
//--------------------------------------------------------
std::string readWord(const std::string &line, std::string& result){
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
  result = line.substr(pos_b, pos_e-pos_b);
  if (pos_e + 1 > line.length()) {
    return "";
  }
  return line.substr(pos_e + 1);
}

//--------------------------------------------------------
bool processPipeInput(genData &data){
  KALDI_LOG << "Reading pipe " << data.pipe_in_str;
  std::ifstream file;
  file.open (data.pipe_in_str);
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
  if (lData.online_ivector_rspecifier.empty()){
    KALDI_WARN << "No online_ivector_rspecifier in pipe";
    return true;
  }
  line = readWord(line, lData.feature_rspecifier);
  if (lData.feature_rspecifier.empty()){
    KALDI_WARN << "No feature_rspecifier in pipe";
    return true;
  }
  line = readWord(line, lData.lattice_wspecifier);
  if (lData.lattice_wspecifier.empty()){
    KALDI_WARN << "No lattice_wspecifier in pipe";
    return true;
  }
  line = readWord(line, lData.pipe_out);
  if (lData.pipe_out.empty()){
    KALDI_WARN << "No pipe_out in pipe";
    return true;
  }
  KALDI_LOG << "online_ivector_rspecifier: " << lData.online_ivector_rspecifier;
  KALDI_LOG << "feature_rspecifier:        " << lData.feature_rspecifier;
  KALDI_LOG << "lattice_wspecifier:        " << lData.lattice_wspecifier;
  KALDI_LOG << "online_ivector_rspecifier: " << lData.pipe_out;

  return true;
}

//--------------------------------------------------------
//--------------------------------------------------------
//--------------------------------------------------------
int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::Fst;
    using fst::StdArc;

    const char *usage =
        "Generate lattices using nnet3 neural net model.  This version supports\n"
        "multiple decoding threads (using a shared decoding graph.)\n"
        "Usage: nnet3-latgen-faster-parallel-pipe [options] <nnet-in> <pipeName-to-read-input-parameters-from> \n"
        "Pipe file format: <online_ivector-rspecifier> <feature_rspecifier> <lattice_wspecifier> <end-pipe-name>\n";
    ParseOptions po(usage);

    Timer timer;
    bool allow_partial = false;
    TaskSequencerConfig sequencer_config; // has --num-threads option
    LatticeFasterDecoderConfig config;
    NnetSimpleComputationOptions decodable_opts;

    std::string word_syms_filename;
    std::string ivector_rspecifier,
        online_ivector_rspecifier,
        utt2spk_rspecifier;
    int32 online_ivector_period = 0;
    sequencer_config.Register(&po);
    config.Register(&po);
    decodable_opts.Register(&po);
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "If true, produce output even if end state was not reached.");
//    po.Register("online-ivectors", &online_ivector_rspecifier, "Rspecifier for "
//                "iVectors estimated online, as matrices.  If you supply this,"
//                " you must set the --online-ivector-period option.");
    po.Register("online-ivector-period", &online_ivector_period, "Number of frames "
                "between iVectors in matrices supplied to the --online-ivectors "
                "option");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    genData data;

    std::string model_in_filename = po.GetArg(1),
        fst_in_str = po.GetArg(2);
    data.pipe_in_str = po.GetArg(3);

    TaskSequencer<DecodeUtteranceLatticeFasterClass> sequencer(sequencer_config);
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


    data.determinize = config.determinize_lattice;
    //CompactLatticeWriter compact_lattice_writer;
//    LatticeWriter lattice_writer;
//    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
//           : lattice_writer.Open(lattice_wspecifier)))
//      KALDI_ERR << "Could not open table for writing lattices: "
//                 << lattice_wspecifier;
//
//    SequentialBaseFloatMatrixReader online_ivector_reader(
//        online_ivector_rspecifier);
//    RandomAccessBaseFloatVectorReaderMapped ivector_reader(
//        ivector_rspecifier, utt2spk_rspecifier);
//
//     fst::SymbolTable *word_syms = NULL;
//    if (word_syms_filename != "")
//      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
//        KALDI_ERR << "Could not read symbol table from file "
//                   << word_syms_filename;
//
//    double tot_like = 0.0;
//    kaldi::int64 frame_count = 0;
//    int num_success = 0, num_fail = 0;
//    bool nextIOnlineVector = false;
//
//    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
//      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
//
//      // Input FST is just one FST, not a table of FSTs.
//      Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
//      timer.Reset();
//
//      {
//        for (; !feature_reader.Done(); feature_reader.Next()) {
//          std::string utt = feature_reader.Key();
//          const Matrix<BaseFloat> &features (feature_reader.Value());
//          if (features.NumRows() == 0) {
//            KALDI_WARN << "Zero-length utterance: " << utt;
//            num_fail++;
//            continue;
//          }
//          const Matrix<BaseFloat> *online_ivectors = NULL;
//          const Vector<BaseFloat> *ivector = NULL;
//          if (!ivector_rspecifier.empty()) {
//            if (!ivector_reader.HasKey(utt)) {
//              KALDI_WARN << "No iVector available for utterance " << utt;
//              num_fail++;
//              continue;
//            } else {
//              ivector = &ivector_reader.Value(utt);
//            }
//          }
//          if (!online_ivector_rspecifier.empty()) {
//            if (feature_reader.Done()){
//              KALDI_WARN << "Closed online vector file";
//              num_fail++;
//              continue;
//            }
//            if (nextIOnlineVector){
//              online_ivector_reader.FreeCurrent();
//              online_ivector_reader.Next();
//            }
//            std::string oiv_utt = online_ivector_reader.Key();
//            if (oiv_utt != utt) {
//              KALDI_WARN << "Online vector key '"<<oiv_utt<<"' and utt key does not match " << utt;
//              num_fail++;
//              continue;
//            } else {
//              online_ivectors = &online_ivector_reader.Value();
//              nextIOnlineVector = true;
//            }
//            KALDI_LOG << "Online vector OK ";
//          }
//
//          LatticeFasterDecoder *decoder =
//              new LatticeFasterDecoder(*decode_fst, config);
//
//          DecodableInterface *nnet_decodable = new
//              DecodableAmNnetSimpleParallel(
//                  decodable_opts, trans_model, am_nnet,
//                  features, ivector, online_ivectors,
//                  online_ivector_period);
//
//          DecodeUtteranceLatticeFasterClass *task =
//              new DecodeUtteranceLatticeFasterClass(
//                  decoder, nnet_decodable, // takes ownership of these two.
//                  trans_model, word_syms, utt, decodable_opts.acoustic_scale,
//                  determinize, allow_partial, &alignment_writer, &words_writer,
//                   &compact_lattice_writer, &lattice_writer,
//                   &tot_like, &frame_count, &num_success, &num_fail, NULL);
//
//          sequencer.Run(task); // takes ownership of "task",
//                               // and will delete it when done.
//        }
//      }
//      sequencer.Wait(); // Waits for all tasks to be done.
//      delete decode_fst;
//    } else { // We have different FSTs for different utterances.
//      SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_in_str);
//      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
//      for (; !fst_reader.Done(); fst_reader.Next()) {
//        std::string utt = fst_reader.Key();
//        if (!feature_reader.HasKey(utt)) {
//          KALDI_WARN << "Not decoding utterance " << utt
//                     << " because no features available.";
//          num_fail++;
//          continue;
//        }
//        const Matrix<BaseFloat> &features = feature_reader.Value(utt);
//        if (features.NumRows() == 0) {
//          KALDI_WARN << "Zero-length utterance: " << utt;
//          num_fail++;
//          continue;
//        }
//
//        const Matrix<BaseFloat> *online_ivectors = NULL;
//        const Vector<BaseFloat> *ivector = NULL;
//        if (!ivector_rspecifier.empty()) {
//          if (!ivector_reader.HasKey(utt)) {
//            KALDI_WARN << "No iVector available for utterance " << utt;
//            num_fail++;
//            continue;
//          } else {
//            ivector = &ivector_reader.Value(utt);
//          }
//        }
//        if (!online_ivector_rspecifier.empty()) {
//          std::string oiv_utt = online_ivector_reader.Key();
//          if (oiv_utt != utt) {
//            KALDI_WARN << "Online vector key '"<<oiv_utt<<"' and utt key does not match " << utt;
//            num_fail++;
//            continue;
//          } else {
//            online_ivectors = &online_ivector_reader.Value();
//          }
//          KALDI_LOG << "Online vector OK 1";
//        }
//
//        // the following constructor takes ownership of the FST pointer so that
//        // it is deleted when 'decoder' is deleted.
//        LatticeFasterDecoder *decoder =
//            new LatticeFasterDecoder(config, fst_reader.Value().Copy());
//
//        DecodableInterface *nnet_decodable = new
//            DecodableAmNnetSimpleParallel(
//                decodable_opts, trans_model, am_nnet,
//                features, ivector, online_ivectors,
//                online_ivector_period);
//
//        DecodeUtteranceLatticeFasterClass *task =
//            new DecodeUtteranceLatticeFasterClass(
//                decoder, nnet_decodable, // takes ownership of these two.
//                trans_model, word_syms, utt, decodable_opts.acoustic_scale,
//                determinize, allow_partial, &alignment_writer, &words_writer,
//                &compact_lattice_writer, &lattice_writer,
//                &tot_like, &frame_count, &num_success, &num_fail, NULL);
//
//        sequencer.Run(task); // takes ownership of "task",
//        // and will delete it when done.
//      }
//      sequencer.Wait(); // Waits for all tasks to be done.
//    }
//
//    kaldi::int64 input_frame_count =
//        frame_count * decodable_opts.frame_subsampling_factor;
//
//    double elapsed = timer.Elapsed();
//    KALDI_LOG << "Time taken " << elapsed
//              << "s: real-time factor assuming 100 feature frames/sec is "
//              << (sequencer_config.num_threads * elapsed * 100.0 /
//                  input_frame_count);
//    KALDI_LOG << "Done " << num_success << " utterances, failed for "
//              << num_fail;
//    KALDI_LOG << "Overall log-likelihood per frame is "
//              << (tot_like / frame_count) << " over "
//              << frame_count << " frames.";
//
//    delete word_syms;
//    if (num_success != 0) return 0;
//    else return 1;

    while (true) {
      if (!processPipeInput(data)){
        return 1;
      };
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
