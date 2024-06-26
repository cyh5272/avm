/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#include <string>
#include <vector>
#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#include "config/aom_config.h"

#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/md5_helper.h"
#include "test/util.h"
#include "test/y4m_video_source.h"
#include "test/yuv_video_source.h"
#include "av1/encoder/firstpass.h"

namespace {
const size_t kFirstPassStatsSz = sizeof(FIRSTPASS_STATS);
class AVxFirstPassEncoderThreadTest
    : public ::libaom_test::CodecTestWith4Params<libaom_test::TestMode, int,
                                                 int, int>,
      public ::libaom_test::EncoderTest {
 protected:
  AVxFirstPassEncoderThreadTest()
      : EncoderTest(GET_PARAM(0)), encoder_initialized_(false),
        encoding_mode_(GET_PARAM(1)), set_cpu_used_(GET_PARAM(2)),
        tile_rows_(GET_PARAM(3)), tile_cols_(GET_PARAM(4)) {
    init_flags_ = AOM_CODEC_USE_PSNR;

    row_mt_ = 1;
    firstpass_stats_.buf = NULL;
    firstpass_stats_.sz = 0;
  }
  virtual ~AVxFirstPassEncoderThreadTest() { free(firstpass_stats_.buf); }

  virtual void SetUp() {
    InitializeConfig();
    SetMode(encoding_mode_);

    cfg_.g_lag_in_frames = 35;
    cfg_.rc_end_usage = AOM_Q;
  }

  virtual void BeginPassHook(unsigned int /*pass*/) {
    encoder_initialized_ = false;
    abort_ = false;
  }

  virtual void EndPassHook() {
    // For first pass stats test, only run first pass encoder.
    if (cfg_.g_pass == AOM_RC_FIRST_PASS) abort_ = true;
  }

  virtual void PreEncodeFrameHook(::libaom_test::VideoSource * /*video*/,
                                  ::libaom_test::Encoder *encoder) {
    if (!encoder_initialized_) {
      // Encode in 2-pass mode.
      SetTileSize(encoder);
      encoder->Control(AV1E_SET_ROW_MT, row_mt_);
      encoder->Control(AOME_SET_CPUUSED, set_cpu_used_);
      encoder->Control(AOME_SET_ENABLEAUTOALTREF, 1);
      encoder->Control(AOME_SET_ARNR_MAXFRAMES, 7);
      encoder->Control(AOME_SET_ARNR_STRENGTH, 5);
      encoder->Control(AV1E_SET_FRAME_PARALLEL_DECODING, 0);
      encoder->Control(AOME_SET_QP, 210);

      encoder_initialized_ = true;
    }
  }

  virtual void SetTileSize(libaom_test::Encoder *encoder) {
    encoder->Control(AV1E_SET_TILE_COLUMNS, tile_cols_);
    encoder->Control(AV1E_SET_TILE_ROWS, tile_rows_);
  }

  virtual void StatsPktHook(const aom_codec_cx_pkt_t *pkt) {
    const uint8_t *const pkt_buf =
        reinterpret_cast<uint8_t *>(pkt->data.twopass_stats.buf);
    const size_t pkt_size = pkt->data.twopass_stats.sz;

    // First pass stats size equals sizeof(FIRSTPASS_STATS)
    EXPECT_EQ(pkt_size, kFirstPassStatsSz)
        << "Error: First pass stats size doesn't equal kFirstPassStatsSz";

    firstpass_stats_.buf =
        realloc(firstpass_stats_.buf, firstpass_stats_.sz + pkt_size);
    memcpy((uint8_t *)firstpass_stats_.buf + firstpass_stats_.sz, pkt_buf,
           pkt_size);
    firstpass_stats_.sz += pkt_size;
  }

  bool encoder_initialized_;
  ::libaom_test::TestMode encoding_mode_;
  int set_cpu_used_;
  int tile_rows_;
  int tile_cols_;
  int row_mt_;
  aom_fixed_buf_t firstpass_stats_;
};

static void compare_fp_stats_md5(aom_fixed_buf_t *fp_stats) {
  // fp_stats consists of 2 set of first pass encoding stats. These 2 set of
  // stats are compared to check if the stats match.
  uint8_t *stats1 = reinterpret_cast<uint8_t *>(fp_stats->buf);
  uint8_t *stats2 = stats1 + fp_stats->sz / 2;
  ::libaom_test::MD5 md5_row_mt_0, md5_row_mt_1;

  md5_row_mt_0.Add(stats1, fp_stats->sz / 2);
  const char *md5_row_mt_0_str = md5_row_mt_0.Get();

  md5_row_mt_1.Add(stats2, fp_stats->sz / 2);
  const char *md5_row_mt_1_str = md5_row_mt_1.Get();

  // Check md5 match.
  ASSERT_STREQ(md5_row_mt_0_str, md5_row_mt_1_str)
      << "MD5 checksums don't match";
}

TEST_P(AVxFirstPassEncoderThreadTest, FirstPassStatsTest) {
  ::libaom_test::Y4mVideoSource video("niklas_1280_720_30.y4m", 0, 60);
  aom_fixed_buf_t firstpass_stats;
  size_t single_run_sz;

  // 5 encodes will be run:
  // 1. row_mt_=0 and threads=1
  // 2. row_mt_=1 and threads=1
  // 3. row_mt_=1 and threads=2
  // 4. row_mt_=1 and threads=4
  // 5. row_mt_=1 and threads=8

  // 4 comparisons will be made:
  // 1. Between run 1 and run 2.
  // 2. Between run 2 and run 3.
  // 3. Between run 3 and run 4.
  // 4. Between run 4 and run 5.

  // Test row_mt_: 0 vs 1 at single thread case(threads = 1)
  cfg_.g_threads = 1;

  row_mt_ = 0;
  init_flags_ = AOM_CODEC_USE_PSNR;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  row_mt_ = 1;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  firstpass_stats.buf = firstpass_stats_.buf;
  firstpass_stats.sz = firstpass_stats_.sz;
  single_run_sz = firstpass_stats_.sz / 2;

  // Compare to check if using or not using row-mt are bit exact.
  // Comparison 1 (between row_mt_=0 and row_mt_=1).
  ASSERT_NO_FATAL_FAILURE(compare_fp_stats_md5(&firstpass_stats));

  // Test single thread vs multiple threads
  row_mt_ = 1;

  cfg_.g_threads = 2;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  // offset to the 2nd and 3rd run.
  firstpass_stats.buf = reinterpret_cast<void *>(
      reinterpret_cast<uint8_t *>(firstpass_stats_.buf) + single_run_sz);

  // Compare to check if single-thread and multi-thread stats are bit exact.
  // Comparison 2 (between threads=1 and threads=2).
  ASSERT_NO_FATAL_FAILURE(compare_fp_stats_md5(&firstpass_stats));

  cfg_.g_threads = 4;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  // offset to the 3rd and 4th run
  firstpass_stats.buf = reinterpret_cast<void *>(
      reinterpret_cast<uint8_t *>(firstpass_stats_.buf) + single_run_sz * 2);

  // Comparison 3 (between threads=2 and threads=4).
  ASSERT_NO_FATAL_FAILURE(compare_fp_stats_md5(&firstpass_stats));

  cfg_.g_threads = 8;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  // offset to the 4th and 5th run.
  firstpass_stats.buf = reinterpret_cast<void *>(
      reinterpret_cast<uint8_t *>(firstpass_stats_.buf) + single_run_sz * 3);

  // Comparison 4 (between threads=4 and threads=8).
  compare_fp_stats_md5(&firstpass_stats);
}

class AVxEncoderThreadTest
    : public ::libaom_test::CodecTestWith5Params<libaom_test::TestMode, int,
                                                 int, int, int>,
      public ::libaom_test::EncoderTest {
 protected:
  AVxEncoderThreadTest()
      : EncoderTest(GET_PARAM(0)), encoder_initialized_(false),
        encoding_mode_(GET_PARAM(1)), set_cpu_used_(GET_PARAM(2)),
        tile_cols_(GET_PARAM(3)), tile_rows_(GET_PARAM(4)),
        row_mt_(GET_PARAM(5)) {
    init_flags_ = AOM_CODEC_USE_PSNR;
    aom_codec_dec_cfg_t cfg = aom_codec_dec_cfg_t();
    cfg.w = 1280;
    cfg.h = 720;
    decoder_ = codec_->CreateDecoder(cfg, 0);
    if (decoder_->IsAV1()) {
      decoder_->Control(AV1_SET_DECODE_TILE_ROW, -1);
      decoder_->Control(AV1_SET_DECODE_TILE_COL, -1);
    }

    size_enc_.clear();
    md5_dec_.clear();
    md5_enc_.clear();
  }
  virtual ~AVxEncoderThreadTest() { delete decoder_; }

  virtual void SetUp() {
    InitializeConfig();
    SetMode(encoding_mode_);

    cfg_.g_lag_in_frames = 5;
    cfg_.rc_end_usage = AOM_Q;
  }

  virtual void BeginPassHook(unsigned int /*pass*/) {
    encoder_initialized_ = false;
  }

  virtual void PreEncodeFrameHook(::libaom_test::VideoSource * /*video*/,
                                  ::libaom_test::Encoder *encoder) {
    if (!encoder_initialized_) {
      SetTileSize(encoder);
      encoder->Control(AOME_SET_CPUUSED, set_cpu_used_);
      encoder->Control(AV1E_SET_ROW_MT, row_mt_);
      encoder->Control(AOME_SET_ENABLEAUTOALTREF, 1);
      encoder->Control(AOME_SET_ARNR_MAXFRAMES, 7);
      encoder->Control(AOME_SET_ARNR_STRENGTH, 5);
      encoder->Control(AV1E_SET_FRAME_PARALLEL_DECODING, 0);
      encoder->Control(AOME_SET_QP, 210);
      encoder_initialized_ = true;
    }
  }

  virtual void SetTileSize(libaom_test::Encoder *encoder) {
    encoder->Control(AV1E_SET_TILE_COLUMNS, tile_cols_);
    encoder->Control(AV1E_SET_TILE_ROWS, tile_rows_);
  }

  virtual void FramePktHook(const aom_codec_cx_pkt_t *pkt
#if CONFIG_OUTPUT_FRAME_BASED_ON_ORDER_HINT
                            ,
                            ::libaom_test::DxDataIterator *dec_iter
#endif  // CONFIG_OUTPUT_FRAME_BASED_ON_ORDER_HINT
  ) {
    size_enc_.push_back(pkt->data.frame.sz);

    ::libaom_test::MD5 md5_enc;
    md5_enc.Add(reinterpret_cast<uint8_t *>(pkt->data.frame.buf),
                pkt->data.frame.sz);
    md5_enc_.push_back(md5_enc.Get());

    const aom_image_t *img;
    if (pkt->kind == AOM_CODEC_CX_FRAME_PKT) {
      const aom_codec_err_t res = decoder_->DecodeFrame(
          reinterpret_cast<uint8_t *>(pkt->data.frame.buf), pkt->data.frame.sz);
      if (res != AOM_CODEC_OK) {
        abort_ = true;
        ASSERT_EQ(AOM_CODEC_OK, res);
      }
      img = decoder_->GetDxData().Next();
#if CONFIG_OUTPUT_FRAME_BASED_ON_ORDER_HINT
    } else {
      assert(dec_iter != NULL);
      img = dec_iter->Peek();
#endif  // CONFIG_OUTPUT_FRAME_BASED_ON_ORDER_HINT
    }

    if (img) {
      ::libaom_test::MD5 md5_res;
      md5_res.Add(img);
      md5_dec_.push_back(md5_res.Get());
    }
  }

  void DoTest() {
    ::libaom_test::YUVVideoSource video(
        "niklas_640_480_30.yuv", AOM_IMG_FMT_I420, 640, 480, 30, 1, 15, 21);

    if (row_mt_ == 0) {
      // Encode using single thread.
      cfg_.g_threads = 1;
      init_flags_ = AOM_CODEC_USE_PSNR;
      ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
      std::vector<size_t> single_thr_size_enc;
      std::vector<std::string> single_thr_md5_enc;
      std::vector<std::string> single_thr_md5_dec;
      single_thr_size_enc = size_enc_;
      single_thr_md5_enc = md5_enc_;
      single_thr_md5_dec = md5_dec_;
      size_enc_.clear();
      md5_enc_.clear();
      md5_dec_.clear();

      // Encode using multiple threads.
      cfg_.g_threads = 4;
      ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
      std::vector<size_t> multi_thr_size_enc;
      std::vector<std::string> multi_thr_md5_enc;
      std::vector<std::string> multi_thr_md5_dec;
      multi_thr_size_enc = size_enc_;
      multi_thr_md5_enc = md5_enc_;
      multi_thr_md5_dec = md5_dec_;
      size_enc_.clear();
      md5_enc_.clear();
      md5_dec_.clear();

      // Check that the vectors are equal.
      ASSERT_EQ(single_thr_size_enc, multi_thr_size_enc);
      ASSERT_EQ(single_thr_md5_enc, multi_thr_md5_enc);
      ASSERT_EQ(single_thr_md5_dec, multi_thr_md5_dec);

      DoTestMaxThreads(&video, single_thr_size_enc, single_thr_md5_enc,
                       single_thr_md5_dec);
    } else if (row_mt_ == 1) {
      // Encode using multiple threads row-mt enabled.
      cfg_.g_threads = 2;
      ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
      std::vector<size_t> multi_thr2_row_mt_size_enc;
      std::vector<std::string> multi_thr2_row_mt_md5_enc;
      std::vector<std::string> multi_thr2_row_mt_md5_dec;
      multi_thr2_row_mt_size_enc = size_enc_;
      multi_thr2_row_mt_md5_enc = md5_enc_;
      multi_thr2_row_mt_md5_dec = md5_dec_;
      size_enc_.clear();
      md5_enc_.clear();
      md5_dec_.clear();

      // Disable threads=3 test for now to reduce the time so that the nightly
      // test would not time out.
      // cfg_.g_threads = 3;
      // ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
      // std::vector<size_t> multi_thr3_row_mt_size_enc;
      // std::vector<std::string> multi_thr3_row_mt_md5_enc;
      // std::vector<std::string> multi_thr3_row_mt_md5_dec;
      // multi_thr3_row_mt_size_enc = size_enc_;
      // multi_thr3_row_mt_md5_enc = md5_enc_;
      // multi_thr3_row_mt_md5_dec = md5_dec_;
      // size_enc_.clear();
      // md5_enc_.clear();
      // md5_dec_.clear();
      // Check that the vectors are equal.
      // ASSERT_EQ(multi_thr3_row_mt_size_enc, multi_thr2_row_mt_size_enc);
      // ASSERT_EQ(multi_thr3_row_mt_md5_enc, multi_thr2_row_mt_md5_enc);
      // ASSERT_EQ(multi_thr3_row_mt_md5_dec, multi_thr2_row_mt_md5_dec);

      cfg_.g_threads = 4;
      ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
      std::vector<size_t> multi_thr4_row_mt_size_enc;
      std::vector<std::string> multi_thr4_row_mt_md5_enc;
      std::vector<std::string> multi_thr4_row_mt_md5_dec;
      multi_thr4_row_mt_size_enc = size_enc_;
      multi_thr4_row_mt_md5_enc = md5_enc_;
      multi_thr4_row_mt_md5_dec = md5_dec_;
      size_enc_.clear();
      md5_enc_.clear();
      md5_dec_.clear();

      // Check that the vectors are equal.
      ASSERT_EQ(multi_thr4_row_mt_size_enc, multi_thr2_row_mt_size_enc);
      ASSERT_EQ(multi_thr4_row_mt_md5_enc, multi_thr2_row_mt_md5_enc);
      ASSERT_EQ(multi_thr4_row_mt_md5_dec, multi_thr2_row_mt_md5_dec);

      DoTestMaxThreads(&video, multi_thr2_row_mt_size_enc,
                       multi_thr2_row_mt_md5_enc, multi_thr2_row_mt_md5_dec);
    }
  }

  virtual void DoTestMaxThreads(::libaom_test::YUVVideoSource *video,
                                const std::vector<size_t> ref_size_enc,
                                const std::vector<std::string> ref_md5_enc,
                                const std::vector<std::string> ref_md5_dec) {
    // This value should be kept the same as MAX_NUM_THREADS
    // in aom_thread.h
    cfg_.g_threads = 64;
    ASSERT_NO_FATAL_FAILURE(RunLoop(video));
    std::vector<size_t> multi_thr_max_row_mt_size_enc;
    std::vector<std::string> multi_thr_max_row_mt_md5_enc;
    std::vector<std::string> multi_thr_max_row_mt_md5_dec;
    multi_thr_max_row_mt_size_enc = size_enc_;
    multi_thr_max_row_mt_md5_enc = md5_enc_;
    multi_thr_max_row_mt_md5_dec = md5_dec_;
    size_enc_.clear();
    md5_enc_.clear();
    md5_dec_.clear();

    // Check that the vectors are equal.
    ASSERT_EQ(ref_size_enc, multi_thr_max_row_mt_size_enc);
    ASSERT_EQ(ref_md5_enc, multi_thr_max_row_mt_md5_enc);
    ASSERT_EQ(ref_md5_dec, multi_thr_max_row_mt_md5_dec);
  }

  bool encoder_initialized_;
  ::libaom_test::TestMode encoding_mode_;
  int set_cpu_used_;
  int tile_cols_;
  int tile_rows_;
  int row_mt_;
  ::libaom_test::Decoder *decoder_;
  std::vector<size_t> size_enc_;
  std::vector<std::string> md5_enc_;
  std::vector<std::string> md5_dec_;
};

TEST_P(AVxEncoderThreadTest, EncoderResultTest) {
  cfg_.large_scale_tile = 0;
  decoder_->Control(AV1_SET_TILE_MODE, 0);
  DoTest();
}

class AVxEncoderThreadTestLarge : public AVxEncoderThreadTest {};

TEST_P(AVxEncoderThreadTestLarge, EncoderResultTest) {
  cfg_.large_scale_tile = 0;
  decoder_->Control(AV1_SET_TILE_MODE, 0);
  DoTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AVxFirstPassEncoderThreadTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AVxEncoderThreadTest);

// Test cpu_used 0, 1, 3 and 5.
// TODO(urvang): Once https://gitlab.com/AOMediaCodec/avm/-/issues/79 is fixed,
// change last parameter back to (0, 1) to re-enable unit tests with row-mt = 1.
AV1_INSTANTIATE_TEST_SUITE(AVxEncoderThreadTestLarge,
                           ::testing::Values(::libaom_test::kOnePassGood),
                           ::testing::Values(0, 1, 3, 5),
                           ::testing::Values(1, 6), ::testing::Values(1, 6),
                           ::testing::Values(0));

class AVxEncoderThreadLSTest : public AVxEncoderThreadTest {
  virtual void SetTileSize(libaom_test::Encoder *encoder) {
    encoder->Control(AV1E_SET_TILE_COLUMNS, tile_cols_);
    encoder->Control(AV1E_SET_TILE_ROWS, tile_rows_);
  }

  virtual void DoTestMaxThreads(::libaom_test::YUVVideoSource *video,
                                const std::vector<size_t> ref_size_enc,
                                const std::vector<std::string> ref_md5_enc,
                                const std::vector<std::string> ref_md5_dec) {
    (void)video;
    (void)ref_size_enc;
    (void)ref_md5_enc;
    (void)ref_md5_dec;
  }
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AVxEncoderThreadLSTest);

TEST_P(AVxEncoderThreadLSTest, EncoderResultTest) {
  cfg_.large_scale_tile = 1;
  decoder_->Control(AV1_SET_TILE_MODE, 1);
  decoder_->Control(AV1D_EXT_TILE_DEBUG, 1);
  DoTest();
}

class AVxEncoderThreadLSTestLarge : public AVxEncoderThreadLSTest {};

TEST_P(AVxEncoderThreadLSTestLarge, EncoderResultTest) {
  cfg_.large_scale_tile = 1;
  decoder_->Control(AV1_SET_TILE_MODE, 1);
  decoder_->Control(AV1D_EXT_TILE_DEBUG, 1);
  DoTest();
}

AV1_INSTANTIATE_TEST_SUITE(AVxEncoderThreadLSTestLarge, GOODQUALITY_TEST_MODES,
                           ::testing::Range(1, 3), ::testing::Values(0, 6),
                           ::testing::Values(0, 6), ::testing::Values(1));
}  // namespace
