#!/usr/bin/env bash
set -euo pipefail
cd /work
DS=/opt/nvidia/deepstream/deepstream
OUTPUT="scripts/cap_interactive.build.$$"
trap 'rm -f "$OUTPUT"' EXIT
g++ -std=c++17 -O3 scripts/interactive.cpp -I"$DS/sources/includes" \
  -I/usr/local/cuda-12.1/targets/x86_64-linux/include \
  $(pkg-config --cflags --libs gstreamer-1.0) -L"$DS/lib" -Wl,-rpath,"$DS/lib" \
  -lnvdsgst_meta -lnvds_meta -o "$OUTPUT"
mv -f "$OUTPUT" scripts/cap_interactive
