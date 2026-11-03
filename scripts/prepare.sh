#!/usr/bin/env bash
set -euo pipefail
BASE=/media/dezzex/4tb_ssd/camex-capacity-20261015
NEW=/media/dezzex/4tb_ssd/camex-150-200-tuning-20261015
test ! -e "$NEW/SETUP_COMPLETE"
mkdir -p "$NEW"/{models,logs,runs,results}
sha256sum "$BASE/run.sh" "$BASE/scripts/run_interactive.py" "$BASE/scripts/interactive.cpp" "$BASE/scripts/cap_interactive" > "$NEW/previous_before.sha256"
cp -a "$BASE/models/." "$NEW/models/"
cp -a "$BASE/run.sh" "$NEW/run.sh"
cp -al "$BASE/datasets" "$NEW/datasets"
cp -al "$BASE/source_profiles" "$NEW/source_profiles"
docker run -d --name camex-tune150200-20261015 --gpus all --memory 48g --memory-swap 48g --pids-limit 8192 \
  --network host -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -v "$NEW:/work" -v "$NEW/datasets:/work/datasets:ro" -v "$NEW/source_profiles:/work/source_profiles:ro" \
  camex-bench-base:20261015 sleep infinity
chown -R dezzex:dezzex "$NEW/scripts" "$NEW/models" "$NEW/logs" "$NEW/runs" "$NEW/results"
chown dezzex:dezzex "$NEW"
touch "$NEW/SETUP_COMPLETE"
