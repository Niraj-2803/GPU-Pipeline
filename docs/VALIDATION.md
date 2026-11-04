# Local validation on 4 November 2026

Review candidate, not yet published to the shared branches.

Ubuntu 22.04; Ryzen 7 2700 (8 cores / 16 threads), about 24 GB RAM; RTX 3080 10 GB. DeepStream 7.0 / TensorRT 8.6; OpenVINO 2026.4.1. Synthetic H.265 10-FPS airport sources. The GPU and CPU use NvDCF and ByteTrack respectively.

All rates below are measured; the one-camera CPU trials below target are retained.

| Profile | Cameras / analysis target | Seconds | Slowest FPS | P95 processing ms | CPU mean % | GPU mean % | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| GPU FP16 FHD RTSP | 20 / 10 FPS | 600 | 10.00 | 21.4 | 12.3 | 26.2 | PASS |
| GPU FP16 2K file | 20 / 10 FPS | 30 | 10.00 | 24.4 | 6.8 | 19.0 | PASS |
| CPU FP32 FHD file | 1 / 10 FPS | 30 | 7.90 | 228.7 | 52.0 | 0.0 | BELOW_TARGET_OR_ERROR |
| CPU INT8 FHD file | 1 / 10 FPS | 30 | 9.47 | 203.7 | 51.6 | 0.0 | BELOW_TARGET_OR_ERROR |
| CPU INT8 FHD file | 2 / 10 FPS | 30 | 4.90 | 304.4 | 56.1 | 0.0 | BELOW_TARGET_OR_ERROR |
| CPU INT8 2K file | 1 / 10 FPS | 30 | 9.43 | 208.5 | 52.1 | 0.0 | BELOW_TARGET_OR_ERROR |
| CPU INT8 FHD RTSP | 2 / 5 FPS | 600 | 4.97 | 352.1 | 89.2 | 0.0 | PASS |

## Correctness and scope

- 28 automated tests passed, including source validation, metadata, tracking gaps and the consumer.
- Final long-run metadata was checked against the native-coordinate contract. Every hundredth frame was also checked against the full JSON schema.
- GPU FP16 agreement: 737/738 reference detections matched on 60 frames.
- CPU INT8 agreement: 729/738 reference detections matched; 729/737 candidate detections matched.
- These are sampled timestamps from the same 20 synthetic clips, not independent field accuracy.
- Both adapters recovered from an intentional eight-second RTSP publisher interruption; tracking epochs separated long-gap tracks.
- Multi-source startup, mixed-GPU device ordering and video finalization regressions found during testing were corrected.
- The evidence bundle retains all failed/aborted diagnostics separately from the final runs.

## Reproduction

Use README sections 1-5 for setup, models, file input and looping RTSP input. For the GPU long test use 20 sources, FP16, 10-FPS analysis, 600 seconds, 8-second warmup and `-out no`. For the CPU long test use two sources, the supplied full INT8 model, `--threads 16 --cpu-streams 2 --hyperthreads`, 5-FPS analysis and the same duration/warmup. Source input remains 10 FPS.

A separate 45-second recording test was run for each backend. Recording results and videos are in the review evidence, not included in headless timing.

Do not extrapolate these counts to another CPU/GPU, additional models or the complete CAMEX product. Client footage, tracking ground truth, end-to-end camera-to-alert latency, sustained multi-day operation and product integration remain deployment acceptance work.
