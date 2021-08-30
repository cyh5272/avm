/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

The CNN restoration utility is developed by Google initially and contributed
to the Alliance for open Media for demonstrating usage of tensorflow lite
models to restore frames, and for experiments related to super-resolution.

For questions and technical support, please contact debargha@google.com

SOURCE
#######

The only source file is:
cnn_restore_y4m.c


BUILD
#####
Configure cmake with -DCONFIG_TENSORFLOW_LITE=1 and make. From a clean build
directory run:

cmake -DCONFIG_TENSORFLOW_LITE=1 path/to/aom && make -j32

This will produce the cnn_restore_y4m utility binary in the build directory.


UTILITY
#######
The utility produced by the source above is cnn_restore_y4m, which
can be used to restore the y-channel of an input y4m video. Currently
the models supported are meant to be used after down-compress-up
processing where the down/up ratio is one of {2:1, 3:2, 4:3, 5:4, 6:5, 7:6}.
Further for each ratio three compression levels are supported:
{no compression, light compression, heavy compression}. Thus a total
of 18 models can be used.

Usage:
  ./cnn_restore_y4m
      <y4m_input>
      <num_frames>
      <upsampling_ratio>
          in form <p>:<q>[:<c>] where <p>/<q> is the upsampling
          ratio with <p> greater than <q>.
          <c> is optional compression level in [0 - 6]
              0: no compression (default)
              1: light inter compression
              2: medium inter compression
              3: heavy inter compression
              4: light intra compression
              5: medium intra compression
              6: heavy intra compression
      <y4m_output>

Examples:
  ./cnn_restore_y4m /tmp/my_downup.y4m 128 4:3 /tmp/my_downup_sr.y4m

    Restore 128 frames of /tmp/my_downup.y4m into /tmp/my_downup_sr.y4m assuming
    the source was downsampled by 4:3, not compressed, upsampled by 4:3 to
    get back to the original resolution.

  ./cnn_restore_y4m /tmp/my_downup.y4m 128 3:2:1 /tmp/my_downup_sr.y4m

    Restore 128 frames of /tmp/my_downup.y4m into /tmp/my_downup_sr.y4m assuming
    the source was downsampled by 3:2, compressed lightlyi in inter mode,
    upsampled by 3:2 to get back to the original resolution.]

  ./cnn_restore_y4m /tmp/my_downup.y4m 128 5:4:6 /tmp/my_downup_sr.y4m

    Restore 128 frames of /tmp/my_downup.y4m into /tmp/my_downup_sr.y4m assuming
    the source was downsampled by 5:4, compressed heavily in intra mode,
    upsampled by 5:4 to get back to the original resolution.]


Note, the utility lanczos_resample_y4m built from source in tools/lanczos
can be conveniently combined with this utility and aomenc/aomdec to produce
a full out of loop down-compress-up-superresolution pipeline.
See the sample scripts:

  tools/lanczos/lanczos_downup_sr.sh and
  tools/lanczos/lanczos_downcompup_sr.sh

for down-up-superresolution and down-compress-up-superresolution pipelines
respectively.
