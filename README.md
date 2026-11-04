# CAMEX CPU and GPU Detection and Tracking Reference

Two video-processing backends share the **same source configuration and per-frame JSON output**:

| Backend | Decode | Person detector | Tracker |
|---|---|---|---|
| GPU | NVIDIA hardware decode | YOLOv8s TensorRT, INT8 backbone and FP16 head | NvDCF |
| CPU | FFmpeg/PyAV software decode | OpenVINO; optional `person-detection-0202` INT8 | ByteTrack |

Verified local profiles: **50 Full HD inputs at 10 FPS on an RTX 3080**, and **five Full HD inputs at 10 FPS on a Ryzen 7 2700 using the CPU-specific detector**. These are synthetic RTSP replay measurements, not a complete CAMEX product capacity guarantee. The fast CPU detector is a **different model**, not an accuracy-equivalent version of YOLOv8s. The unchanged YOLO CPU path remains available.

The 100-input GPU stress case missed the selected latency limit; ten CPU inputs missed the FPS target. Do not quote either as a qualified configuration. Read [results and limitations](docs/RESULTS.md) before sizing a deployment.

## Start here

1. [Install the common tools](#1-get-the-code-and-install-the-common-tools).
2. [Get the exact demo data](#2-get-the-demo-data).
3. Choose [CPU setup](#3-cpu-openvino) or [GPU setup](#4-gpu-deepstream-and-tensorrt).
4. Use [real cameras](#5-connect-real-rtsp-cameras) or [looped RTSP replay](#6-run-the-verified-rtsp-replay-profiles).
5. [Inspect results](#7-results-and-verification), then read [CAMEX integration](docs/INTEGRATION.md).

[GPU video](https://github.com/Niraj-2803/GPU-Pipeline/releases/download/pipeline-reference-20261105/gpu50_video.mp4) · [CPU video](https://github.com/Niraj-2803/GPU-Pipeline/releases/download/pipeline-reference-20261105/cpu5_video.mp4) · [Release assets](https://github.com/Niraj-2803/GPU-Pipeline/releases/tag/pipeline-reference-20261105)

## 1. Get the code and install the common tools

Verified platform: Ubuntu 22.04 x86-64, Python 3.10. Use a virtual environment, not the system Python packages.
Online setup needs access to PyPI, the model/download URLs, Docker Hub and
NVIDIA's container registry. Offline machines need those wheels, archives and
container images staged in advance.

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip ffmpeg libgl1 libglib2.0-0 curl
git clone --branch pipeline-reference-20261105 https://github.com/Niraj-2803/GPU-Pipeline.git
cd GPU-Pipeline
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
export CAMEX_PYTHON="$PWD/.venv/bin/python"
```

`requirements-lock-ubuntu.txt` records the full clean-verification environment.
Use it instead of `requirements-cpu.txt` when reproducing the exact tested
Ubuntu/Python package set; other platforms still need their own validation.

The command uses the frozen release tag for reproducibility. Both backends are
included; examples specify `--backend` to remove ambiguity. For a working PR
checkout, use either delivery branch named below instead of the tag.
The CPU branch/tag defaults to OpenVINO; the GPU branch defaults to DeepStream.

Review branches: `codex/cpu-pipeline-delivery-20261105` and `codex/gpu-pipeline-delivery-20261105`. Their PR targets are the existing `CPU-Pipeline_Development` and `GPU-Pipeline-Development` branches. Until merged, cloning an old development branch will **not** get this update. After merge, those development branches provide ongoing updates; the tag remains the fixed experiment version.

Use **`run_pipeline.sh`**. The original `run.sh`, `scripts/`, old engines and historical results are retained only as legacy prepared-Dell experiments. They are not the portable entrypoint.

## 2. Get the demo data

Checksum-verified release archives contain synthetic airport clips, their source configuration, the exact person-output ONNX used by the benchmark, and model-source/license notices. No staff photos or private camera credentials are included.

```bash
# Five distinct 60-second FHD/H.265/10-FPS clips for the CPU example.
.venv/bin/python tools/fetch_demo.py --profile mini

# Fifty distinct clips for the published GPU workload.
.venv/bin/python tools/fetch_demo.py --profile full
```

These print `data/mini/sources.json` and `data/full/sources.json`. `assets/demo_assets.json` records URLs, sizes and SHA-256 hashes; extracted files are also verified.
Use only the bundle needed for your example: mini for the CPU quickstart, full
for the 50-feed GPU test. The full bundle also works for CPU tests when its
source path is substituted.

For an offline archive downloaded from the release page:

```bash
.venv/bin/python tools/fetch_demo.py --profile full --archive /path/to/camex-demo-full50.tar.gz
```

`-n` above the source count repeats inputs under distinct suffixed IDs. This is a load test, not evidence of that many distinct physical cameras. Use the full archive, not five repeated clips, when reproducing the recorded 50-view workload.

## 3. CPU OpenVINO

CPU inference needs no NVIDIA GPU, CUDA or Docker. Docker is needed only for the optional local RTSP replay helper.

```bash
.venv/bin/python tools/fetch_cpu_model.py --output-dir models/cpu-person
```

The helper downloads the official Open Model Zoo model and verifies both XML and BIN against published SHA-384 values. Keep both files together. The SSD adapter applies its required **raw BGR, 512x512 resize**, without division by 255.

Run a short file smoke test:

```bash
RUN="runs/cpu-file-$(date +%Y%m%dT%H%M%S)"
./run_pipeline.sh --backend openvino \
  --sources data/mini/sources.json -n 5 -res FHD --source-fps 10 --analysis-fps 10 \
  --model models/cpu-person/person-detection-0202.xml --precision int8 \
  --detector ssd --ssd-class-id 0 --threshold 0.5 \
  --threads 8 --cpu-streams 2 --requests 4 --performance-hint THROUGHPUT \
  -track yes -out no --warmup 8 --seconds 30 --run-dir "$RUN"
.venv/bin/python tools/check_result.py "$RUN" --seconds 30 --metadata
```

To test the original YOLOv8s on CPU instead:

```bash
./run_pipeline.sh --backend openvino --sources data/mini/sources.json -n 1 \
  --model data/mini/models/yolov8s-person.onnx --precision fp32 --detector yolo \
  --analysis-size 640 --threshold 0.25 --threads 8 --requests 2 \
  --warmup 8 --seconds 30 -out no
```

That unchanged model did not meet the five-camera target on our Ryzen host. Do not substitute models without rechecking quality. See [model preparation](docs/MODELS.md).

## 4. GPU DeepStream and TensorRT

### Host prerequisites

The host needs a compatible NVIDIA driver, Docker Engine and NVIDIA Container Toolkit. Verify the driver first:

```bash
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,driver_version --format=csv
```

If it fails, ask the administrator to install the appropriate driver and reboot during a maintenance window. This project does not change host drivers.

On a clean Ubuntu host:

```bash
sudo apt-get install -y docker.io
sudo usermod -aG docker "$USER"
```

Log out and back in. Docker-group membership grants elevated host access. Install NVIDIA Container Toolkit using the [official instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then configure Docker during an approved maintenance window:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

Do not restart Docker casually on a shared server.

### Build the runtime image

```bash
docker build -f Dockerfile.deepstream -t camex-cmp-deepstream:7.0-review .
```

The image pins DeepStream 7.0/TensorRT 8.6 by digest. It is not a universal runtime for every GPU generation. Check compatibility before using another card. `--gpu-id` selects the **nvidia-smi index**; the runner maps CUDA PCI order explicitly on mixed-GPU hosts.

### Prepare calibration and build the INT8 engine

```bash
.venv/bin/python tools/prepare_calibration.py \
  --sources data/full/sources.json --views 50 --output-dir data/calibration

# Set this from nvidia-smi. It was 1 on our mixed-GPU test host.
GPU=0
.venv/bin/python tools/build_engine.py \
  --model data/full/models/yolov8s-person.onnx \
  --precision int8 --calibration-dir data/calibration \
  --output-dir models/local-int8 --gpu-id "$GPU"
```

The result is `models/local-int8/yolov8s-int8-head-b16.engine`. The recipe uses an INT8 backbone, FP16 detection head, MinMax calibration, and a 1/8/16 batch profile. Here “head” means final network layers, not a head-only object detector. Calibration uses 500 frames at 0,4,...,36 seconds across 50 views.

Build on the target GPU with this runtime. Never rename/copy an old RTX 5000 or Jetson engine and assume compatibility. Keep the generated log and provenance JSON. Calibration and tactic selection can take several minutes; build time is not inference time.

Optional FP16 baseline, in a separate output directory:

```bash
.venv/bin/python tools/build_engine.py \
  --model data/full/models/yolov8s-person.onnx --precision fp16 \
  --opt-batch 8 --output-dir models/local-fp16 --gpu-id "$GPU"
```

## 5. Connect real RTSP cameras

Create an untracked `local.rtsp.json` with the actual camera settings:

```json
{
  "schema_version": "1.0",
  "sources": [
    {"camera_id":"entrance","uri":"rtsp://USER:PASSWORD@CAMERA_IP/STREAM_PATH","width":1920,"height":1080,"fps":10},
    {"camera_id":"lobby","uri":"rtsp://USER:PASSWORD@SECOND_CAMERA/STREAM_PATH","width":1920,"height":1080,"fps":10}
  ]
}
```

```bash
chmod 600 local.rtsp.json
```

Use administrator-provided URLs; vendor paths vary. URL-encode special characters in credentials. Check reachability and decode with FFprobe from this Ubuntu host. Never commit credentials or paste private logs into public issues.

Point either backend's `--sources` to this file. IDs must be unique and use letters, digits, `_` or `-`. Relative video paths resolve relative to the configuration file. GPU inputs must share one resolution and encoded FPS per run: FHD or 2560x1440, at 10 or 30 FPS. Use separate workers for different GPU source profiles. H.264/H.265 RTSP are handled by the decode stack; qualification used H.265.

`--analysis-fps 10` does not reconfigure a camera. A 30-FPS camera has a different decode workload. `-res` and `--source-fps` validate settings; they do not resize/re-encode input.
For the published workload, configure a camera-side 10-FPS analytics stream.
Merely reducing inference to 10 FPS does not reproduce 10-FPS input decoding.

## 6. Run the verified RTSP replay profiles

The helper copies compressed files into independent localhost-only RTSP publishers, waits for actual decoded frames, and loops the clips. It does not render Unity scenes or simulate WAN conditions. Docker is required for this helper, even with CPU inference.

### CPU profile: five feeds

Terminal A, from the repository root:

```bash
.venv/bin/python tools/replay_rtsp.py --sources data/mini/sources.json \
  -n 5 --port 29554 --seconds 1800 --output local-replay-cpu
```

Wait for `RTSP READY` and leave Terminal A running. In Terminal B:

```bash
cd /path/to/GPU-Pipeline
export CAMEX_PYTHON="$PWD/.venv/bin/python"
RUN="runs/cpu-rtsp-$(date +%Y%m%dT%H%M%S)"
./run_pipeline.sh --backend openvino --sources local-replay-cpu/sources.json \
  -n 5 --analysis-fps 10 --model models/cpu-person/person-detection-0202.xml \
  --precision int8 --detector ssd --ssd-class-id 0 --threshold 0.5 \
  --threads 8 --cpu-streams 2 --requests 4 --performance-hint THROUGHPUT \
  --warmup 15 --seconds 600 -out no --run-dir "$RUN"
.venv/bin/python tools/check_result.py "$RUN" --seconds 600 --metadata
```

### GPU profile: 50 feeds

Stop any previous replay with Ctrl+C before reusing its port. Terminal A:

```bash
.venv/bin/python tools/replay_rtsp.py --sources data/full/sources.json \
  -n 50 --port 29554 --seconds 1800 --output local-replay-gpu
```

Wait for `RTSP READY`. Terminal B:

```bash
cd /path/to/GPU-Pipeline
export CAMEX_PYTHON="$PWD/.venv/bin/python"
GPU=0
RUN="runs/gpu-rtsp-$(date +%Y%m%dT%H%M%S)"
./run_pipeline.sh --backend deepstream --sources local-replay-gpu/sources.json \
  -n 50 --analysis-fps 10 --model data/full/models/yolov8s-person.onnx \
  --engine models/local-int8/yolov8s-int8-head-b16.engine --precision int8 \
  --gpu-id "$GPU" --batch 8 --mux-size native --mux-timeout-us 40000 \
  --decode-surfaces 2 --startup-timeout 120 --reconnect-seconds 30 \
  --warmup 15 --seconds 600 -track yes -out no --run-dir "$RUN"
.venv/bin/python tools/check_result.py "$RUN" --seconds 600 \
  --max-frame-age-ms 300 --metadata
```

Start with fewer feeds on untested hardware. Fifty is a verified profile on the stated host, not a universal maximum. Run one timed workload at a time and record other activity on the host.

### Record or stop

For an annotated preview, replace `-out no` with `-out yes --mosaic-fps 10`, use `--seconds 45` and a new run directory. Open `mosaic.mp4` after completion. Recording adds tiling, overlay and encoding costs and needs its own result check. A preview's playback FPS is not proof of inference FPS. An unfinished `mosaic.partial.mp4` is not valid evidence.

Ctrl+C stops the inference run. Stop Terminal A separately to stop its publishers and named relay. The tools leave unrelated services, firewall settings and containers alone.

## 7. Results and verification

Each run creates a unique folder:

| File | Contents |
|---|---|
| `settings.json` | Requested settings, redacted sources, runtime/source hashes |
| `metadata.jsonl` | One event per processed frame, including empty detections |
| `summary.json` | Actual FPS, P50/P95, errors, resources and native result |
| `per_camera.csv` | Camera-level counts, FPS and track counts |
| `per_second.jsonl` | Interval rates revealing stalls or bursts |
| `resources.jsonl` | CPU/RAM and selected-GPU utilization/memory |
| `native.log` | GPU diagnostics; may contain private camera URLs |
| `mosaic.mp4` | Optional finalized annotated video |

A zero exit code means normal completion, **not necessarily a passed FPS target**. The legacy `summary.throughput_check` uses 95% of target. The explicit release check above uses a stricter 9.8-FPS minimum at a 10-FPS target, 300-ms processing P95, completed duration and no runtime/schema/output-drop error. The GPU command also checks pipeline frame-age P95. Neither timing is physical exposure-to-alert latency.

Review [measurement definitions](docs/MEASUREMENTS.md) and [the results](docs/RESULTS.md). Report the slowest camera, not just aggregate FPS. Track IDs/counts are not cross-camera identity or tracking-accuracy scores.

## 8. Integrate with CAMEX

Read [the integration guide](docs/INTEGRATION.md) and [metadata contract](docs/CONTRACT.md). Both backends write the same schema, so projection/counting/persistence need no backend-specific parser.

```bash
.venv/bin/python examples/consume_metadata.py "$RUN/metadata.jsonl" --follow
.venv/bin/python examples/camex_adapter.py "$RUN/metadata.jsonl" --follow
```

The adapter emits native-pixel boxes, bottom-center footpoints and camera-scoped tracking keys. Replace its output with the existing CAMEX projection/event callback. It does not invent calibration or implement the full analytics product. Heavy database/network/model calls belong in the separate consumer, not the inference callback.

## 9. Settings and tests

| Setting | Meaning |
|---|---|
| `--analysis-fps` | Analysis target, not camera configuration |
| `--threshold`, `--nms` | Confidence and YOLO NMS; model-specific |
| `--batch` | GPU inference batch; stay inside the engine profile |
| `--mux-size`, `--mux-timeout-us` | GPU intermediate size and batch timing |
| `--decode-surfaces` | Extra decode buffers; less memory can reduce stability |
| `--startup-timeout`, `--reconnect-seconds` | GPU startup/reconnect timing |
| `--threads`, `--cpu-streams`, `--requests` | CPU scheduling/concurrency |
| `--performance-hint`, `--cpu-batch` | CPU latency/throughput mode and YOLO batching |
| `--detector ssd` | Explicit CPU SSD adapter; CPU batch must be 1 |
| `--analysis-size` | YOLO input size; requires a matching export |
| `--vram-reserve-mib` | Free-memory reserve guard for GPU runs |

Do not silently change detector, confidence, resolution or analysis rate and call it the same benchmark.

```bash
.venv/bin/python -m pytest -q tests
```

The full suite requires the CPU requirements even on a GPU host. Optional CPU quantization and independent model export are in [model preparation](docs/MODELS.md); they are not needed with the release model files.

## 10. Troubleshooting and boundaries

- **Old code/commands:** choose a delivery branch and `run_pipeline.sh`, not the legacy Dell runner.
- **Missing input/model:** run the download helpers and inspect the verified configuration paths.
- **Docker permission denied:** complete the group/login step; do not weaken file permissions.
- **No NVIDIA runtime:** fix driver/Container Toolkit before engine building.
- **Registry authentication or download failure:** check network access and the registry's current access requirements; do not put access tokens in this repository.
- **Engine mismatch:** rebuild on the actual GPU/runtime; do not rename an old engine.
- **RTSP not ready:** verify URL, credentials, TCP access, codec and keyframes. Keep private logs private. The replay helper retries transient startup errors.
- **Slow startup:** all GPU sources must produce fresh output before warmup. Do not hide missing cameras by reducing the expected count.
- **Memory or latency limit:** reduce inputs, inspect other workloads and compare buffering. Never kill another team's processes automatically.
- **CPU limit:** more threads do not guarantee more FPS. Measure the intended model on the actual CPU.

Scope is person detection and camera-local tracking, not Face ID, cross-camera identity, full CAMEX analytics, production recording, high availability or an SLA. Synthetic tests do not prove field accuracy or multi-year reliability.

## Licensing

Third-party code, runtimes and weights retain their licenses. The YOLO reference derives from Ultralytics YOLOv8: read the included source notice and [licensing guidance](https://www.ultralytics.com/license) before proprietary/commercial integration. OpenVINO/Open Model Zoo and NVIDIA runtime terms also apply. This reference is not a grant of closed-source deployment rights.
