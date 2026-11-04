# Optimization review candidate

This branch is local and is not approved for integration yet. The baseline
README still describes the original portable reference. Do not interpret
the optional CPU model as accuracy-equivalent to YOLOv8s.

## GPU profile

The qualified 50-feed profile uses YOLOv8s at 640x640, a newly calibrated
TensorRT INT8 backbone with FP16 detection-head layers, inference batch 8,
native-resolution mux buffers and a 40000-us batching timeout. Input is
Full HD H.265 at 10 FPS. It uses the same metadata contract and NvDCF.

The 100-feed stress configuration uses two workers of 50 on one GPU,
960x540 intermediate mux buffers, no extra decoder surfaces and staged
startup. It sustained nearly 10 FPS per input but exceeded this review's
300-ms frame-age P95 limit. It is not the recommended capacity.

## Optional CPU profile

`--backend openvino --detector ssd` selects the explicit SSD adapter, with
the Open Model Zoo `person-detection-0202` model. The model uses BGR pixels
resized to 512x512, without the YOLO normalization/letterbox. Output boxes
are mapped to native input coordinates. The tested profile uses threshold
0.5, eight inference threads, two streams, four requests and throughput mode.

The unchanged YOLOv8s CPU model did not meet five cameras at 10 FPS on this
Ryzen 7 2700. The faster SSD profile met the five-camera speed target but has
different detections and needs client-footage quality approval. The existing
YOLO path remains available; there is no automatic model downgrade.

Both the GPU 50-feed profile and CPU 5-feed profile completed a 600-second
RTSP replay confirmation. Separate preview runs are not included in those
headless timing results. The final GPU startup check requires all sources to
remain fresh before warmup, instead of accepting a single historical frame.

## Review boundaries

- See the accompanying `REVIEW.md` and raw summaries for exact results.
- Input/output schema remains camera-local metadata version 1.0.
- No Face ID, cross-camera identity, complete CAMEX workload or availability
  SLA is included in these figures.
- Final team-facing README and publishing are held for Abdul's review.
