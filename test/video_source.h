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
#ifndef AOM_TEST_VIDEO_SOURCE_H_
#define AOM_TEST_VIDEO_SOURCE_H_

#if defined(_WIN32)
#undef NOMINMAX
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <string>
#include "test/acm_random.h"
#include "aom/aom_encoder.h"

namespace libaom_test {

// Helper macros to ensure LIBAOM_TEST_DATA_PATH is a quoted string.
// These are undefined right below GetDataPath
// NOTE: LIBAOM_TEST_DATA_PATH MUST NOT be a quoted string before
// Stringification or the GetDataPath will fail at runtime
#define TO_STRING(S) #S
#define STRINGIFY(S) TO_STRING(S)

// A simple function to encapsulate cross platform retrieval of test data path
static std::string GetDataPath() {
  const char *const data_path = getenv("LIBAOM_TEST_DATA_PATH");
  if (data_path == NULL) {
#ifdef LIBAOM_TEST_DATA_PATH
    // In some environments, we cannot set environment variables
    // Instead, we set the data path by using a preprocessor symbol
    // which can be set from make files
    return STRINGIFY(LIBAOM_TEST_DATA_PATH);
#else
    return ".";
#endif
  }
  return data_path;
}

// Undefining stringification macros because they are not used elsewhere
#undef TO_STRING
#undef STRINGIFY

inline FILE *OpenTestDataFile(const std::string &file_name) {
  const std::string path_to_source = GetDataPath() + "/" + file_name;
  return fopen(path_to_source.c_str(), "rb");
}

static FILE *GetTempOutFile(std::string *file_name) {
  file_name->clear();
#if defined(_WIN32)
  char fname[MAX_PATH];
  char tmppath[MAX_PATH];
  if (GetTempPathA(MAX_PATH, tmppath)) {
    // Assume for now that the filename generated is unique per process
    if (GetTempFileNameA(tmppath, "lvx", 0, fname)) {
      file_name->assign(fname);
      return fopen(fname, "wb+");
    }
  }
  return NULL;
#else
  char name_template[] = "/tmp/libaomtest.XXXXXX";
  const int fd = mkstemp(name_template);
  *file_name = name_template;
  return fdopen(fd, "wb+");
#endif
}

class TempOutFile {
 public:
  TempOutFile() { file_ = GetTempOutFile(&file_name_); }
  ~TempOutFile() {
    CloseFile();
    if (!file_name_.empty()) {
      EXPECT_EQ(0, remove(file_name_.c_str()));
    }
  }
  FILE *file() { return file_; }
  const std::string &file_name() { return file_name_; }

 protected:
  void CloseFile() {
    if (file_) {
      fclose(file_);
      file_ = NULL;
    }
  }
  FILE *file_;
  std::string file_name_;
};

// Abstract base class for test video sources, which provide a stream of
// aom_image_t images with associated timestamps and duration.
class VideoSource {
 public:
  virtual ~VideoSource() {}

  // Prepare the stream for reading, rewind/open as necessary.
  virtual void Begin() = 0;

  // Advance the cursor to the next frame
  virtual void Next() = 0;

  // Get the current video frame, or NULL on End-Of-Stream.
  virtual aom_image_t *img() const = 0;

  // Get the presentation timestamp of the current frame.
  virtual aom_codec_pts_t pts() const = 0;

  // Get the current frame's duration
  virtual unsigned long duration() const = 0;

  // Get the timebase for the stream
  virtual aom_rational_t timebase() const = 0;

  // Get the current frame counter, starting at 0.
  virtual unsigned int frame() const = 0;

  // Get the current file limit.
  virtual unsigned int limit() const = 0;
};

class DummyVideoSource : public VideoSource {
 public:
  DummyVideoSource()
      : img_(NULL), limit_(100), width_(80), height_(64),
        format_(AOM_IMG_FMT_I420) {
    ReallocImage();
  }

  virtual ~DummyVideoSource() { aom_img_free(img_); }

  virtual void Begin() {
    frame_ = 0;
    FillFrame();
  }

  virtual void Next() {
    ++frame_;
    FillFrame();
  }

  virtual aom_image_t *img() const { return (frame_ < limit_) ? img_ : NULL; }

  // Models a stream where Timebase = 1/FPS, so pts == frame.
  virtual aom_codec_pts_t pts() const { return frame_; }

  virtual unsigned long duration() const { return 1; }

  virtual aom_rational_t timebase() const {
    const aom_rational_t t = { 1, 30 };
    return t;
  }

  virtual unsigned int frame() const { return frame_; }

  virtual unsigned int limit() const { return limit_; }

  void set_limit(unsigned int limit) { limit_ = limit; }

  void SetSize(unsigned int width, unsigned int height) {
    if (width != width_ || height != height_) {
      width_ = width;
      height_ = height;
      ReallocImage();
    }
  }

  void SetImageFormat(aom_img_fmt_t format) {
    if (format_ != format) {
      format_ = format;
      ReallocImage();
    }
  }

 protected:
  virtual void FillFrame() {
    if (img_) memset(img_->img_data, 0, raw_sz_);
  }

  void ReallocImage() {
    aom_img_free(img_);
    img_ = aom_img_alloc(NULL, format_, width_, height_, 32);
    raw_sz_ = ((img_->w + 31) & ~31) * img_->h * img_->bps / 8;
  }

  aom_image_t *img_;
  size_t raw_sz_;
  unsigned int limit_;
  unsigned int frame_;
  unsigned int width_;
  unsigned int height_;
  aom_img_fmt_t format_;
};

class RandomVideoSource : public DummyVideoSource {
 public:
  RandomVideoSource(int seed = ACMRandom::DeterministicSeed())
      : rnd_(seed), seed_(seed) {}

 protected:
  // Reset the RNG to get a matching stream for the second pass
  virtual void Begin() {
    frame_ = 0;
    rnd_.Reset(seed_);
    FillFrame();
  }

  // 15 frames of noise, followed by 15 static frames. Reset to 0 rather
  // than holding previous frames to encourage keyframes to be thrown.
  virtual void FillFrame() {
    if (img_) {
      if (frame_ % 30 < 15)
        for (size_t i = 0; i < raw_sz_; ++i) img_->img_data[i] = rnd_.Rand8();
      else
        memset(img_->img_data, 0, raw_sz_);
    }
  }

  ACMRandom rnd_;
  int seed_;
};

// Abstract base class for test video sources, which provide a stream of
// decompressed images to the decoder.
class CompressedVideoSource {
 public:
  virtual ~CompressedVideoSource() {}

  virtual void Init() = 0;

  // Prepare the stream for reading, rewind/open as necessary.
  virtual void Begin() = 0;

  // Advance the cursor to the next frame
  virtual void Next() = 0;

  virtual const uint8_t *cxdata() const = 0;

  virtual size_t frame_size() const = 0;

  virtual unsigned int frame_number() const = 0;
};

}  // namespace libaom_test

#endif  // AOM_TEST_VIDEO_SOURCE_H_
