# CAMEX integration

## Boundary between this reference and the product

```text
Physical camera RTSP OR local synthetic RTSP replay
                    |
        GPU DeepStream OR CPU OpenVINO
        decode -> preprocess -> detect -> track
                    |
              metadata.jsonl
                    |
        CAMEX adapter in a separate process
                    |
   existing calibration/projection -> analytics -> storage/API/UI
```

This repository owns decode, person detection, camera-local tracking and metadata export. CAMEX retains camera configuration, authorization, calibration, multi-camera association, business rules, alert delivery, storage and UI. The included adapter does not implement those product services.

The Python launcher/configuration/monitor run in the host virtual environment.
The GPU worker runs in the DeepStream Docker image; the CPU worker runs in the
host virtual environment. If the CAMEX backend is containerized, mount the
chosen run directory read-only into that backend or forward events through
its existing internal transport. Restrict file access to the service account.

## Input contract

The source JSON contains `schema_version: "1.0"` and a nonempty `sources` list.
See [the source schema](sources.schema.json). The runtime also checks uniqueness,
file availability and matching dimensions/rate; JSON schema alone is not enough.

| Field | Meaning |
|---|---|
| `camera_id` | Unique stable application ID; letters, digits, `_`, `-`, maximum 64 characters |
| `uri` | RTSP URL or local file path; relative paths resolve against the JSON file |
| `width`, `height` | Actual encoded dimensions, even positive integers |
| `fps` | Actual encoded source FPS, not desired inference FPS |
| `is_upscaled` | Optional declaration that the source was upscaled |

One camera should use the same ID in CAMEX and this configuration. For a real deployment, list each camera once and use its true stream settings. Increasing `-n` beyond the list length intentionally repeats sources for load testing and is not a camera discovery mechanism.

## Output contract

Read [CONTRACT.md](CONTRACT.md) and the machine-readable [JSON schema](metadata.schema.json). Each complete line in `metadata.jsonl` is one processed frame, including frames with no detections. The two backends use identical field names and coordinate units.

| Field | Integration use |
|---|---|
| `schema_version`, `type` | Version check before parsing |
| `run_id` | New on every process run; do not join identities across restarts |
| `camera_id`, `frame_id` | Source and frame association |
| `source_pts_ns`, `timestamp_domain` | Media association within the stated clock domain |
| `tracking_epoch` | Changes after a long processing gap; prevents joining reused track IDs |
| `width`, `height` | Native decoded frame dimensions |
| `objects[].bbox_xyxy` | `[left, top, right, bottom]` in native pixels, not model or mosaic pixels |
| `objects[].track_id` | String, preserving large IDs; null when unconfirmed or tracking disabled |
| `objects[].confidence` | Detector confidence; `-1` can mean unavailable for GPU tracker predictions |
| `processing_ms` | Post-decode processing age, not physical camera-to-alert latency |

The identity key is `(run_id, camera_id, tracking_epoch, track_id)`. It is not a person identity across cameras. CPU ByteTrack and GPU NvDCF can associate boxes differently; shared JSON does not imply identical IDs or tracking accuracy.

Media timestamps are not UTC exposure timestamps. GPU RTSP uses pipeline PTS and CPU RTSP uses stream-relative PTS. Do not calculate camera-to-alert latency by subtracting these from wall-clock UTC. Introduce a synchronized capture timestamp or visual clock for that separate test.

This export contains metadata, **not original frames or crop bytes**. If CAMEX
needs pixels for face recognition, appearance models or evidence recording,
design a bounded crop/frame-sharing branch or a DeepStream secondary-model
stage. Do not silently reintroduce full CPU decoding of every feed. Pixel
transport and those additional models are not included in this release.

## Minimal consumer

Start the pipeline, then run the adapter separately:

```bash
.venv/bin/python examples/camex_adapter.py runs/YOUR_RUN/metadata.jsonl --follow
```

`to_projection_requests(frame)` returns ordinary dictionaries. For a box `[200,100,320,480]`, it emits a bottom-center footpoint `[260,480]`. Pass that pixel point to the existing calibration for the same `camera_id`; attach your calibration-version ID to the resulting event. Bottom-center is an estimate, not a validated physical ground-contact point.

```python
from examples.camex_adapter import to_projection_requests

def on_frame(frame):
    for detection in to_projection_requests(frame):
        # Replace with CAMEX's existing queue/callback.
        submit_to_camex(detection)
```

`submit_to_camex` is deliberately a placeholder for the application's own function, not a dependency provided here. The command-line adapter prints JSON so it can also feed an existing service via stdin or an IPC bridge. The diagnostic `examples/consume_metadata.py` prints counts only.

## Switching backends

Keep the source JSON and consumer unchanged. Start a new run with `--backend deepstream` or `--backend openvino` and the correct model options. Restarting creates a new `run_id`; clear live associations from the old run. See README for exact commands and model-specific preprocessing. A TensorRT engine cannot be loaded by OpenVINO, and the CPU SSD model is not a YOLO engine.

## Operational responsibilities

- Keep database writes, web requests and additional model calls outside inference callbacks. The exporter uses a bounded queue and reports dropped metadata rather than blocking the entire pipeline.
- Follow only complete newline-terminated records. The sample consumer rejects a truncated final line when not following a live file.
- Monitor every camera, FPS, errors, reconnects, output age and `metadata_dropped`; a healthy average can hide one stalled source.
- Use backpressure, deduplication, retry and retention policies in the CAMEX consumer. File output alone is not a durable event-delivery service.
- Keep private source configs and native logs access-controlled. The default output directory is ignored by Git, but exported logs still need review before sharing.
- Define application health separately from process health. Exit code zero does not certify throughput; use `tools/check_result.py` and your deployment acceptance tests.
- Add service supervision/container orchestration only after choosing resource limits and validating the complete workload. This reference does not install system services or change the existing platform.
