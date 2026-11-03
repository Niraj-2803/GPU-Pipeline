CAMEX 150 and 200 input follow-up
16 October 2026 JST

The 150-input and 200-input 30-second headless tests achieved 10 FPS per input
with detection and per-camera NvDCF tracking. These tests use an explicit
synchronized file-playback start after every decoder has initialized.
Initialization time and the single immutable PTS/DTS offset are in result.json.
This corrects the offline player's startup method; it does not accelerate the
model or prove native RTSP capture-to-alert latency. Earlier startup backlog
tests must not be described as the GPU's absolute capacity limit.

Prepared workstation command, 200 inputs, headless:

/media/dezzex/4tb_ssd/camex-150-200-tuning-20261015/run.sh \
  -n 200 -res 2K --source-fps 10 --codec h265 --precision int8-head \
  -track yes -out no --analysis-fps 10 --sync-start --warmup 8 --seconds 30

Use -n 150 for the 150-input check.

To record the tested 5-FPS preview while AI stays at 10 FPS:

/media/dezzex/4tb_ssd/camex-150-200-tuning-20261015/run.sh \
  -n 200 -res 2K --source-fps 10 --codec h265 --precision int8-head \
  -track yes -out yes --mosaic-fps 5 --analysis-fps 10 --sync-start \
  --warmup 8 --seconds 10

The runner automatically starts its experiment container and requests sudo
when needed. It prints per-second FPS, GPU compute, decoder/encoder usage,
VRAM, temperature and total CPU utilization. Each run prints its unique folder.
Run one test at a time, with the same approved company workloads paused.
The --sync-start and --mosaic-fps options belong only to this tuning folder.

All 50 distinct simulated camera files are independently decoded, repeated
three times for 150 inputs or four times for 200. Files are H.265 2560x1440
at 10 FPS, upscaled from FHD. Detector is YOLOv8s640 with experimental INT8
backbone and FP16 head. Tracker uses original NvDCF perf, 640x384, batch 8,
one tracker instance. All 200 inputs produced person tracks. IDs are local
to each camera. These are short file tests, not tracking-accuracy benchmarks.

The original 100-input and 9-FPS runners were hash-verified unchanged.
The isolated experiment containers were stopped after testing. Existing
company services remain paused as previously requested.

Contents:
- Follow-up Word report with utilization, limits and repeat commands.
- Annotated video: AI at 10 FPS, recorded preview at 5 FPS.
- Raw recordings, including the 10-FPS mosaic that missed the delay limit.
- Results.csv, all_results.json and individual performance/resource records.
- Raw exact settings, source assignments, engine hashes and stage profile.
- Reproduction source scripts. Large models and datasets are not in this ZIP;
  the commands use the prepared workstation and its existing assets.

Do not overwrite old results. The new runner writes unique timestamped folders.
Batch 16 did not improve the selected setup. Four tracker groups performed
worse and were not adopted. An initial 5-FPS preview test deferred tiler setup;
the final version warms the tiler before measurement and is recorded separately.
