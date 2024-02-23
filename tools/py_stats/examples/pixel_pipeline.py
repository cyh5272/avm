#!/usr/bin/env python3
## Copyright (c) 2023, Alliance for Open Media. All rights reserved
##
## This source code is subject to the terms of the BSD 3-Clause Clear License and the
## Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear License was
## not distributed with this source code in the LICENSE file, you can obtain it
## at aomedia.org/license/software-license/bsd-3-c-c/.  If the Alliance for Open Media Patent
## License 1.0 was not distributed with this source code in the PATENTS file, you
## can obtain it at aomedia.org/license/patent-license/.
##
from __future__ import annotations

from functools import partial
import pathlib
import tempfile

from absl import app
from absl import flags
from absl import logging
from avm_stats.extract_proto import *
from avm_stats.frame_visualizations import *
from avm_stats.proto_helpers import *
from avm_stats.yuv_tools import *
import matplotlib.pyplot as plt

_STREAM = flags.DEFINE_string("stream", None, "Path to AVM stream.")
flags.mark_flag_as_required("stream")

_SOURCE = flags.DEFINE_string(
    "source", None, "Path to source YUV/Y4M before encoding (optional)."
)

_EXTRACT_PROTO_BIN = flags.DEFINE_string(
    "extract_proto_bin", None, "Path to extract_proto binary."
)
flags.mark_flag_as_required("extract_proto_bin")

_FRAME = flags.DEFINE_integer(
    "frame", 0, "Frame number to visualize (defaults to the first frame)."
)
_PLANE = flags.DEFINE_enum(
    "plane", "y", ["y", "u", "v"], "Plane to visualize (defaults to luma: 'y')"
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = pathlib.Path(tmp_dir)
    stream_path = pathlib.Path(_STREAM.value)
    extract_proto_path = pathlib.Path(_EXTRACT_PROTO_BIN.value)
    yuv_path = _SOURCE.value or None
    frames = extract_and_load_protos(
        extract_proto_path=extract_proto_path,
        stream_path=stream_path,
        output_path=tmp_path,
        skip_if_output_already_exists=False,
        yuv_path=yuv_path,
        frame_limit=_FRAME.value,
    )
    seq = list(frames)
    num_frames = len(seq)
    logging.info(f"Loaded {num_frames} frame protos.")

  if yuv_path:
    visualizations = [
        OriginalYuvLayer,
        PredictionYuvLayer,
        partial(ResidualYuvLayer, show_relative=False),
        partial(ResidualYuvLayer, show_relative=True),
        PrefilteredYuvLayer,
        partial(FilterDeltaYuvLayer, show_relative=False),
        partial(FilterDeltaYuvLayer, show_relative=True),
        ReconstructionYuvLayer,
        partial(DistortionYuvLayer, show_relative=False),
        partial(DistortionYuvLayer, show_relative=True),
    ]
  # If original YUV is not available, skip the visualizations that depend on it.
  else:
    visualizations = [
        PredictionYuvLayer,
        partial(ResidualYuvLayer, show_relative=False),
        partial(ResidualYuvLayer, show_relative=True),
        PrefilteredYuvLayer,
        partial(FilterDeltaYuvLayer, show_relative=False),
        partial(FilterDeltaYuvLayer, show_relative=True),
        ReconstructionYuvLayer,
    ]

  subplot_cols = 5
  subplot_rows = 2
  fig, axes = plt.subplots(subplot_rows, subplot_cols)

  plane = Plane[_PLANE.value.upper()]
  for i in range(subplot_cols * subplot_rows):
    axes_row = i // subplot_cols
    axes_col = i % subplot_cols
    ax = axes[axes_row][axes_col]
    if i < len(visualizations):
      visualizations[i](plane=plane).show(seq[_FRAME.value], ax)
    else:
      fig.delaxes(ax)
  plt.show()


if __name__ == "__main__":
  app.run(main)
