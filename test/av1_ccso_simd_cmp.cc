/*
 * Copyright (c) 2022, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"

#include "av1/common/enums.h"
#include "av1/common/ccso.h"

#include "test/acm_random.h"
#include "test/function_equivalence_test.h"
#include "test/register_state_check.h"

using libaom_test::ACMRandom;
using libaom_test::FunctionEquivalenceTest;

namespace {

//////////////////////////////////////////////////////////////////////////////
// ccso_filter_block_hbd_wo_buf
//////////////////////////////////////////////////////////////////////////////

typedef void (*CCSO_WO_BUF)(const uint16_t *src_y, uint16_t *dst_yuv,
                            const int x, const int y, const int pic_width,
                            const int pic_height, int *src_cls,
                            const int8_t *offset_buf, const int src_y_stride,
                            const int dst_stride, const int y_uv_hscale,
                            const int y_uv_vscale, const int thr,
                            const int neg_thr, const int *src_loc,
                            const int max_val, const int blk_size,
                            const bool isSingleBand, const uint8_t shift_bits
#if CONFIG_CCSO_EDGE_CLF
                            ,
                            const int edge_clf
#endif
#if CONFIG_CCSO_BO_ONLY_OPTION
                            ,
                            const uint8_t ccso_bo_only
#endif
);
typedef libaom_test::FuncParam<CCSO_WO_BUF> TestFuncsCCSO_WO_BUF;

template <typename F>
class CCSOFilterTest : public FunctionEquivalenceTest<F> {
 public:
  static const int kIterations = 10000;
  static const int kMaxWidth =
      (MAX_SB_SIZE << 1) * 5;  // * 5 to cover longer strides
  static const int kMaxHeight = (MAX_SB_SIZE << 1) * 3;
  static const int kBufSize = kMaxWidth * kMaxHeight;
  static const int kMaxMaskWidth = 2 * MAX_SB_SIZE;
  static const int kMaxMaskSize = kMaxMaskWidth;

  virtual ~CCSOFilterTest() {}

  virtual void Execute() = 0;

  void Common() {
    // we just test whether block level filter generate same results
    y_uv_hscale_ = this->rng_(2);
    y_uv_vscale_ = y_uv_hscale_;
    pic_width_ = (MAX_SB_SIZE * 2) >> y_uv_hscale_;
    pic_height_ = (MAX_SB_SIZE * 2) >> y_uv_vscale_;
    blk_size_ = pic_width_;

    // AVX2 fetch 16 elements, chroma 4:2:0 case 32
    src_y_stride_ =
        this->rng_(kMaxWidth + 1 - 32) + 32 + (CCSO_PADDING_SIZE << 1);
    dst_stride_ = this->rng_(kMaxWidth + 1 - 32) + 32;

    filter_sup_ = this->rng_(6);
    derive_ccso_sample_pos(src_loc_, src_y_stride_, filter_sup_);

    const uint8_t quant_sz[4] = { 16, 8, 32, 64 };
    thr_ = quant_sz[this->rng_(4)];
    neg_thr_ = -1 * thr_;

    const uint8_t shift_bits_a[2] = { 8, 10 };
    shift_bits_ = shift_bits_a[this->rng_(2)];
    max_val_ = (1 << shift_bits_) - 1;
    isSingleBand_ = this->rng_(2);

    Execute();
  }

  uint16_t dst_ref_[kBufSize];
  uint16_t dst_tst_[kBufSize];
  int dst_stride_;

  uint16_t src_y_[kBufSize];
  int src_y_stride_;

  int8_t offset_buf_[CCSO_BAND_NUM * 16];
  uint8_t mask_[kMaxMaskSize];

  int src_loc_[2];

  int pic_width_;
  int pic_height_;
  int blk_size_;
  uint8_t filter_sup_;

  int y_uv_vscale_;
  int y_uv_hscale_;
  int thr_;
  int neg_thr_;
  int max_val_;
  bool isSingleBand_;
  uint8_t shift_bits_;
#if CONFIG_CCSO_EDGE_CLF
  int edge_clf_;
#endif
};

class CCSOWOBUFTest : public CCSOFilterTest<CCSO_WO_BUF> {
 protected:
  void Execute() {
    params_.ref_func(src_y_, dst_ref_, 0, 0, pic_width_, pic_height_, src_cls_,
                     offset_buf_, src_y_stride_, dst_stride_, y_uv_hscale_,
                     y_uv_vscale_, thr_, neg_thr_, src_loc_, max_val_,
                     blk_size_, isSingleBand_, shift_bits_
#if CONFIG_CCSO_EDGE_CLF
                     ,
                     edge_clf_
#endif
#if CONFIG_CCSO_BO_ONLY_OPTION
                     ,
                     0
#endif
    );

    ASM_REGISTER_STATE_CHECK(params_.tst_func(
        src_y_, dst_tst_, 0, 0, pic_width_, pic_height_, src_cls_, offset_buf_,
        src_y_stride_, dst_stride_, y_uv_hscale_, y_uv_vscale_, thr_, neg_thr_,
        src_loc_, max_val_, blk_size_, isSingleBand_, shift_bits_
#if CONFIG_CCSO_EDGE_CLF
        ,
        edge_clf_
#endif
#if CONFIG_CCSO_EDGE_CLF
        ,
        0
#endif
        ));

    for (int r = 0; r < blk_size_; ++r) {
      for (int c = 0; c < blk_size_; ++c) {
        ASSERT_EQ(dst_ref_[r * dst_stride_ + c], dst_tst_[r * dst_stride_ + c]);
      }
    }
  }
  int src_cls_[2];
};

TEST_P(CCSOWOBUFTest, RandomValues) {
  for (int iter = 0; iter < kIterations && !HasFatalFailure(); ++iter) {
    const int hi = 1 << 10;
    for (int i = 0; i < kBufSize; ++i) {
      dst_ref_[i] = 0;
      dst_tst_[i] = 0;
      src_y_[i] = rng_(hi);
    }
    const int ccso_offset[8] = { -10, -7, -3, -1, 0, 1, 3, 7 };

    for (int i = 0; i < CCSO_BAND_NUM * 16; i++) {
      offset_buf_[i] = ccso_offset[rng_(8)];
    }

    Common();
  }
}
//////////////////////////////////////////////////////////////////////////////
// ccso_filter_block_hbd_with_buf
//////////////////////////////////////////////////////////////////////////////
typedef void (*CCSO_With_BUF)(const uint16_t *src_y, uint16_t *dst_yuv,
                              const uint8_t *src_cls0, const uint8_t *src_cls1,
                              const int src_y_stride, const int dst_stride,
                              const int ccso_stride, const int x, const int y,
                              const int pic_width, const int pic_height,
                              const int8_t *offset_buf, const int blk_size,
                              const int y_uv_hscale, const int y_uv_vscale,
                              const int max_val, const uint8_t shift_bits
#if CONFIG_CCSO_BO_ONLY_OPTION
                              ,
                              const uint8_t ccso_bo_only
#endif
);
typedef libaom_test::FuncParam<CCSO_With_BUF> TestFuncsCCSO_With_BUF;

class CCSOWITHBUFTest : public CCSOFilterTest<CCSO_With_BUF> {
 protected:
  void Execute() {
    ccso_stride_ = src_y_stride_ - (CCSO_PADDING_SIZE << 1);
    params_.ref_func(src_y_, dst_ref_, src_cls0_, src_cls1_, src_y_stride_,
                     dst_stride_, ccso_stride_, 0, 0, pic_width_, pic_height_,
                     offset_buf_, blk_size_, y_uv_hscale_, y_uv_vscale_,
                     max_val_, shift_bits_
#if CONFIG_CCSO_BO_ONLY_OPTION
                     ,
                     0
#endif
    );

    ASM_REGISTER_STATE_CHECK(params_.tst_func(
        src_y_, dst_tst_, src_cls0_, src_cls1_, src_y_stride_, dst_stride_,
        ccso_stride_, 0, 0, pic_width_, pic_height_, offset_buf_, blk_size_,
        y_uv_hscale_, y_uv_vscale_, max_val_, shift_bits_
#if CONFIG_CCSO_BO_ONLY_OPTION
        ,
        0
#endif
        ));

    for (int r = 0; r < blk_size_; ++r) {
      for (int c = 0; c < blk_size_; ++c) {
        ASSERT_EQ(dst_ref_[r * dst_stride_ + c], dst_tst_[r * dst_stride_ + c]);
      }
    }
  }
  uint8_t src_cls0_[kBufSize];
  uint8_t src_cls1_[kBufSize];
  int ccso_stride_;
};

TEST_P(CCSOWITHBUFTest, RandomValues) {
  for (int iter = 0; iter < kIterations && !HasFatalFailure(); ++iter) {
    const int hi = 1 << 10;
    for (int i = 0; i < kBufSize; ++i) {
      dst_ref_[i] = 0;
      dst_tst_[i] = 0;
      src_cls0_[i] = rng_(3);
      src_cls1_[i] = rng_(3);
      src_y_[i] = rng_(hi);
    }
    const int ccso_offset[8] = { -10, -7, -3, -1, 0, 1, 3, 7 };

    for (int i = 0; i < CCSO_BAND_NUM * 16; i++) {
      offset_buf_[i] = ccso_offset[rng_(8)];
    }

    Common();
  }
}
//////////////////////////////////////////////////////////////////////////////
// ccso_derive_src_block_avx2
//////////////////////////////////////////////////////////////////////////////
typedef void (*CCSO_Derive_Src)(const uint16_t *src_y, uint8_t *const src_cls0,
                                uint8_t *const src_cls1, const int src_y_stride,
                                const int ccso_stride, const int x, const int y,
                                const int pic_width, const int pic_height,
                                const int y_uv_hscale, const int y_uv_vscale,
                                const int thr, const int neg_thr,
                                const int *src_loc, const int blk_size
#if CONFIG_CCSO_EDGE_CLF
                                ,
                                const int edge_clf
#endif
);
typedef libaom_test::FuncParam<CCSO_Derive_Src> TestFuncsCCSO_Derive_Src;

class CCSODeriveSrcTest : public CCSOFilterTest<CCSO_Derive_Src> {
 protected:
  void Execute() {
    ccso_stride_ = src_y_stride_ - (CCSO_PADDING_SIZE << 1);
    params_.ref_func(src_y_, src_cls0_ref, src_cls1_ref, src_y_stride_,
                     ccso_stride_, 0, 0, pic_width_, pic_height_, y_uv_hscale_,
                     y_uv_vscale_, thr_, neg_thr_, src_loc_, blk_size_
#if CONFIG_CCSO_EDGE_CLF
                     ,
                     edge_clf_
#endif
    );

    ASM_REGISTER_STATE_CHECK(params_.tst_func(
        src_y_, src_cls0_tst, src_cls1_tst, src_y_stride_, ccso_stride_, 0, 0,
        pic_width_, pic_height_, y_uv_hscale_, y_uv_vscale_, thr_, neg_thr_,
        src_loc_, blk_size_
#if CONFIG_CCSO_EDGE_CLF
        ,
        edge_clf_
#endif
        ));

    for (int r = 0; r < blk_size_; ++r) {
      for (int c = 0; c < blk_size_; ++c) {
        ASSERT_EQ(src_cls0_ref[(r << y_uv_vscale_) * ccso_stride_ +
                               (c << y_uv_hscale_)],
                  src_cls0_tst[(r << y_uv_vscale_) * ccso_stride_ +
                               (c << y_uv_hscale_)]);
        ASSERT_EQ(src_cls1_ref[(r << y_uv_vscale_) * ccso_stride_ +
                               (c << y_uv_hscale_)],
                  src_cls1_tst[(r << y_uv_vscale_) * ccso_stride_ +
                               (c << y_uv_hscale_)]);
      }
    }
  }
  uint8_t src_cls0_ref[kBufSize];
  uint8_t src_cls1_ref[kBufSize];
  uint8_t src_cls0_tst[kBufSize];
  uint8_t src_cls1_tst[kBufSize];
  int ccso_stride_;
};

TEST_P(CCSODeriveSrcTest, RandomValues) {
  for (int iter = 0; iter < kIterations && !HasFatalFailure(); ++iter) {
    const int hi = 1 << 10;
    for (int i = 0; i < kBufSize; ++i) {
      dst_ref_[i] = 0;
      dst_tst_[i] = 0;
      src_cls0_ref[i] = 0;
      src_cls1_ref[i] = 0;
      src_cls0_tst[i] = 0;
      src_cls1_tst[i] = 0;
      src_y_[i] = rng_(hi);
    }

    Common();
  }
}

//////////////////////////////////////////////////////////////////////////////
// compute_distortion_block_avx2
//////////////////////////////////////////////////////////////////////////////
typedef uint64_t (*CCSO_Dist_Block)(const uint16_t *org, const int org_stride,
                                    const uint16_t *rec16, const int rec_stride,
                                    const int x, const int y,
                                    const int log2_filter_unit_size,
                                    const int height, const int width);

typedef libaom_test::FuncParam<CCSO_Dist_Block> TestFuncsCCSO_Dist_Block;

class CCSODistBlockTest : public CCSOFilterTest<CCSO_Dist_Block> {
 protected:
  void Execute() {
    org_ = src_y_;
    org_stride_ = src_y_stride_;
    rec16_ = dst_ref_;
    rec_stride_ = src_y_stride_;
    log2_filter_unit_size_ = 1 - y_uv_hscale_ + 7;
    height_ = pic_height_;
    width_ = pic_width_;
    uint64_t ref = params_.ref_func(org_, org_stride_, rec16_, rec_stride_, 0,
                                    0, log2_filter_unit_size_, height_, width_);
    uint64_t tst;
    ASM_REGISTER_STATE_CHECK(
        tst = params_.tst_func(org_, org_stride_, rec16_, rec_stride_, 0, 0,
                               log2_filter_unit_size_, height_, width_));
    ASSERT_EQ(ref, tst);
  }
  uint16_t *org_;
  int org_stride_;
  uint16_t *rec16_;
  int rec_stride_;
  int log2_filter_unit_size_;
  int height_;
  int width_;
};

TEST_P(CCSODistBlockTest, RandomValues) {
  for (int iter = 0; iter < kIterations && !HasFatalFailure(); ++iter) {
    const int hi = 1 << 10;
    for (int i = 0; i < kBufSize; ++i) {
      dst_ref_[i] = rng_(hi);
      src_y_[i] = rng_(hi);
    }

    Common();
  }
}

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, CCSODistBlockTest,
    ::testing::Values(TestFuncsCCSO_Dist_Block(compute_distortion_block_c,
                                               compute_distortion_block_avx2)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, CCSODeriveSrcTest,
    ::testing::Values(TestFuncsCCSO_Derive_Src(ccso_derive_src_block_c,
                                               ccso_derive_src_block_avx2)));

INSTANTIATE_TEST_SUITE_P(AVX2, CCSOWITHBUFTest,
                         ::testing::Values(TestFuncsCCSO_With_BUF(
                             ccso_filter_block_hbd_with_buf_c,
                             ccso_filter_block_hbd_with_buf_avx2)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, CCSOWOBUFTest,
    ::testing::Values(TestFuncsCCSO_WO_BUF(ccso_filter_block_hbd_wo_buf_c,
                                           ccso_filter_block_hbd_wo_buf_avx2)));
#endif  // HAVE_AVX2

}  // namespace
