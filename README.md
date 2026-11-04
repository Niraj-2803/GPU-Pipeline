# CAMEX Portable Detection and Tracking Reference

This local optimization review adds opt-in profiles described in
[optimization review notes](docs/OPTIMIZATION_REVIEW.md). The baseline setup
below remains available. The faster CPU SSD model is not an accuracy-equivalent
replacement for YOLOv8s and has not been approved for integration.

Two implementations share source configuration, commands and JSON metadata:

- **DeepStream:** NVIDIA hardware decoding, TensorRT YOLOv8s, NvDCF tracking.
- **CPU:** PyAV/FFmpeg software decoding, OpenVINO CPU YOLOv8s, ByteTrack tracking.

Branch roles: `GPU-Pipeline-Development` is the GPU reference;
`CPU-Pipeline_Development` adds the CPU implementation. Select the CPU branch
before using `--backend openvino`. The supplied local review versions have not
yet been published to those shared branches.

This is a reference for integration, not the complete CAMEX product. It does
not perform Face ID, cross-camera identification, queue analytics or alerts.
Track IDs belong to one camera and one run. Different trackers and precision
modes can produce different boxes and IDs despite the same output schema.

The original `run.sh` and `scripts/` remain as the legacy prepared-Dell
benchmark. **Use `run_pipeline.sh` for this portable reference.** The original
TensorRT engines in this repository must not be assumed compatible with another
GPU or TensorRT version.

## 1. Host prerequisites

The verified host is Ubuntu 22.04 x86-64. See the delivered comparison report
for exact hardware, versions, measured results and limitations.
The local evidence summary is in [validation notes](docs/VALIDATION.md).

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip ffmpeg libgl1 libglib2.0-0
git clone https://github.com/Niraj-2803/GPU-Pipeline.git
cd GPU-Pipeline
# Select the reviewed GPU or CPU branch/review commit supplied with the release.
python3 -m venv .venv
.venv/bin/pip install -r requirements-cpu.txt
export CAMEX_PYTHON="$PWD/.venv/bin/python"
```

The CPU installation does not require Docker, CUDA or an NVIDIA GPU.
`supervision==0.27.0` is deliberately pinned because its ByteTrack API differs
from newer releases. The CPU backend explicitly compiles for `CPU`, not AUTO.
Do not replace these pins without rerunning the tests and agreement checks.

For GPU execution, install the Ubuntu-compatible NVIDIA driver, Docker Engine
and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
On an existing production machine, coordinate driver changes, reboots and
Docker restarts with its owner; none are part of the runner.

For a clean Ubuntu host, Docker can be installed with `sudo apt-get install
docker.io`. Add your account with `sudo usermod -aG docker "$USER"`, then log out
and back in. After installing the NVIDIA toolkit from the linked official
instructions, configure it with `sudo nvidia-ctk runtime configure --runtime=docker`
and restart Docker with `sudo systemctl restart docker` during the approved
maintenance window. These administrator steps are not required again for each run.

Verify GPU access before proceeding:

```bash
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total --format=csv
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
docker build -f Dockerfile.deepstream -t camex-cmp-deepstream:7.0-review .
```

The image pins DeepStream 7.0/TensorRT 8.6 through a digest. GPU compatibility
must be checked for each deployment. `--gpu-id` selects the **nvidia-smi index**.
The runner maps this to CUDA's PCI order and sets the same device throughout
decode, inference, tracking and encoding. This matters on mixed-GPU hosts.

## 2. Model preparation

Use the **review model bundle**, containing `yolov8s-person.onnx`, for exact
reproduction of the delivered results. Its SHA-256 is:

```text
75b32bc520027b73d9b44acbb167ea5f78bf9750f5415e4f0985291cea73f7de
```

Place it at `models/yolov8s-person.onnx`, or supply its actual path with
`--model`. The model expects RGB NCHW float32 values divided by 255, with a
640x640 centered, zero-padded letterbox. Its output is `[batch,8400,6]`:
`x1,y1,x2,y2,person_score,max_class_score`. This is **not** an arbitrary stock
YOLO ONNX layout. A mismatched export can run but produce invalid detections.

For an independent export, follow [model export instructions](docs/MODELS.md).
New exports and INT8 conversions require a new agreement report; do not attach
the old benchmark figures to different model files. Confirm the model's
commercial license separately from this reference code.

Build the GPU engine on the target GPU:

```bash
.venv/bin/python tools/build_engine.py \
  --model models/yolov8s-person.onnx \
  --output-dir models/local-gpu --precision fp16 --gpu-id 0
```

Use `--precision fp32` and a separate output directory for a FP32 engine.
The builder does not overwrite an existing engine. Retain its log, provenance
JSON and model/engine hashes. Engine build time is not inference time.

## 3. Configure files or RTSP cameras

Copy `examples/sources.json` to `local.sources.json`. Supply real paths and
the **actual encoded** width, height and FPS for each source. Relative paths
are relative to the configuration file. Files are probed before the run.

```json
{"schema_version":"1.0","sources":[
  {"camera_id":"entrance","uri":"/data/C01.mp4","width":1920,"height":1080,"fps":10},
  {"camera_id":"lobby","uri":"/data/C02.mp4","width":1920,"height":1080,"fps":10}
]}
```

For real cameras, use `rtsp://user:password@camera-address/path` instead of the
file path. Get the stream path and credentials from the camera administrator;
vendor paths vary. Use TCP, confirm connectivity from this host with FFprobe
and select a camera-side 10-FPS analytics stream where possible.

Keep credentials in an untracked file such as `local.rtsp.json`, with mode
`600`. Never commit it. The saved settings redact RTSP user info and query
parameters, but native failure logs can contain URLs and must remain private.

`-n` selects the number of independent inputs. If it exceeds the configured
sources, the runner repeats them under distinct IDs such as `C01_r2`. This is
explicit load testing, not evidence of that many distinct physical cameras.

`-res FHD` validates 1920x1080. `-res 2K` validates 2560x1440. Neither option
resizes a source or changes its encoded FPS. Record any upscaling in the source
configuration with `"is_upscaled": true`.

## 4. Run the same workflow on each backend

GPU example:

```bash
./run_pipeline.sh --backend deepstream --sources local.sources.json \
  -n 5 -res FHD --source-fps 10 --analysis-fps 10 \
  --model models/yolov8s-person.onnx \
  --engine models/local-gpu/yolov8s-fp16-b16.engine --precision fp16 \
  --gpu-id 0 --batch 4 -track yes -out no --warmup 8 --seconds 30
```

CPU example:

```bash
./run_pipeline.sh --backend openvino --sources local.sources.json \
  -n 1 -res FHD --source-fps 10 --analysis-fps 10 \
  --model models/yolov8s-person.onnx --precision fp32 \
  --threads 8 --requests 2 -track yes -out no --warmup 8 --seconds 30
```

Start with one source, then increase gradually. CPU-only support does not
promise GPU-level throughput. Use `-track no` to isolate detection. Use
`-out yes --mosaic-fps 5` to record an annotated MP4; `--mosaic-fps 10` requests
a 10-FPS preview. Recording adds CPU/GPU work and is a separate test mode.

CPU tuning controls are `--threads`, `--requests`, `--cpu-streams 1|2|4` and
`--hyperthreads`. Start from the tested defaults. More threads or streams are
not necessarily faster; keep each tuning run and its effective OpenVINO config.
An explicitly lower `--analysis-fps 5` profile is a different workload and must
not be reported as achieving a 10-FPS target.

The terminal reports progress every second and prints a unique results folder.
Check `summary.json`, not just a preview's nominal FPS. `throughput_check`
requires every camera to reach 95% of the requested analysis rate with no
reported run, schema or metadata-drop error. Latency remains a separate metric.
The displayed preview may repeat frames and is not a throughput counter.

Press Ctrl+C to stop the current runner. The GPU wrapper stops only its named
test container. It never stops the host's unrelated containers or services.

## 5. RTSP replay without physical cameras

Use the exact same file configuration to publish distinct local RTSP feeds:

```bash
.venv/bin/python tools/replay_rtsp.py --sources local.sources.json \
  -n 5 --port 29554 --seconds 1800 --output local-replay
```

After `RTSP READY`, run either backend with `--sources local-replay/sources.json`.
This uses CPU-light stream copying, not re-encoding. The relay binds only to
127.0.0.1. Close the replay command with Ctrl+C when done; it stops its publishers
and its named relay container. Use a different unused port for another replay.
Direct file tests on both backends require sufficiently long files; use this looping
RTSP replay for a ten-minute confirmation rather than silently resetting PTS.

## 6. Optional CPU INT8 optimization

Keep the FP32 baseline. Quantization is a separate quality/performance trade-off:

```bash
.venv/bin/pip install -r requirements-optimization.txt
.venv/bin/python tools/quantize_cpu.py --model models/yolov8s-person.onnx \
  --sources local.sources.json --views 20 --output models/cpu-int8/person.xml
.venv/bin/python tools/evaluate_agreement.py --reference models/yolov8s-person.onnx \
  --candidate models/cpu-int8/person.xml --sources local.sources.json \
  --views 20 --output quality/cpu-int8
```

The default retains a floating-point detection head. `--quantize-head` tests a
separate variant including supported head operations. Calibration uses sampled
frames at 0,10,20,40,50 seconds. Evaluation uses 5,25,45 seconds. Source clips must
be at least 51 seconds long. Agreement with FP32 is **not labeled accuracy**.
The example gate is 95% precision and recall relative to the FP32 detections.
Use a separate real-world labeled test before deployment.

After evaluating, use `--backend openvino --model models/cpu-int8/person.xml
--precision int8`. The metadata format stays the same.

## 7. Integrate into CAMEX

Read [the metadata contract](docs/CONTRACT.md). Each run writes `metadata.jsonl`:

```bash
.venv/bin/python examples/consume_metadata.py runs/RUN_FOLDER/metadata.jsonl --follow
```

Replace `on_frame` in that example with the existing projection/counting/event
consumer. Use native-frame coordinates and the matching calibration version.
Use `(run_id,camera_id,tracking_epoch,track_id)` as a tracking key. The IDs are not global IDs.
Keep heavy database, network and additional-model calls outside the tracking
callback. Metadata queues are bounded and report drops instead of blocking AI.

## 8. Tests and repeatable benchmarks

```bash
.venv/bin/python -m pytest -q tests
.venv/bin/python tools/benchmark.py --backend openvino \
  --sources local.sources.json --model models/yolov8s-person.onnx \
  --precision fp32 --counts 1,2,5 --repeat 3 --output benchmark/cpu
```

For GPU, use `--backend deepstream`, its engine path, precision and GPU index.
Measure one run at a time. Keep the same sources, model, thresholds and rate.
Record tracker/precision differences and any background workload. Report all
failures, not only the fastest run. See [measurement definitions](docs/MEASUREMENTS.md).

## Troubleshooting

- **Missing videos:** MP4 files are intentionally not in Git. Supply the review
  dataset separately or use your authorized files/RTSP cameras.
- **Missing Docker image:** run the Docker build command above. The portable
  runner does not depend on `camex-bench-base:20261015` or an old named container.
- **Engine incompatibility:** rebuild on the actual GPU/runtime. Do not rename
  an engine made for another GPU.
- **Mixed-GPU startup stall:** check the nvidia-smi index, PCI mapping and the
  selected device in the engine build log. Do not assume CUDA index order.
- **Out of GPU memory:** reduce simultaneous inputs/batch and inspect other GPU
  workloads. Do not kill another team's processes automatically.
- **Low FPS:** distinguish decoder, model, tracker and encoder load. Lowering
  analysis FPS does not necessarily lower the camera's encoded/decode FPS.
- **RTSP reconnection:** the GPU source requests periodic reconnects; CPU uses
  bounded read timeouts and retries. Validate behavior on the actual camera.
- **Many RTSP sources fail during startup:** the replay tool waits for an actual
  decoded frame from every publisher before printing `RTSP READY`. The GPU
  adapter assigns distinct source IDs and allows 15 seconds of inactivity
  before reconnecting, avoiding a reconnect loop during multi-source startup.
- **Recording does not finalize:** the live GPU output branch receives its own
  EOS after the measurement window. Only a finalized `mosaic.mp4` is review
  evidence; a `mosaic.partial.mp4` from a failed run is not.
- **No results at all:** check `native.log` for GPU errors, source reachability,
  decoded frame dimensions, and whether any camera missed startup.

Original experiment notes remain in `delivery/README.txt`. They describe the
older Dell test only, not this machine or the complete production pipeline.
