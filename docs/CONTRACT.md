# Metadata Contract 1.0

Both backends emit one JSON object per processed frame in `metadata.jsonl`.
The file is newline-delimited UTF-8 JSON. Each line is self-contained.

```json
{
  "schema_version":"1.0", "type":"frame", "run_id":"20261104T120000_000000Z",
  "backend":"openvino", "camera_id":"entrance", "frame_id":42, "tracking_epoch":0,
  "pts_ns":4200000000, "source_pts_ns":4200000000, "timestamp_domain":"media",
  "monotonic_s":12345.67, "elapsed_s":2.2, "width":1920, "height":1080,
  "processing_ms":83.5, "tracker":"bytetrack",
  "objects":[{"class_id":0,"label":"person","confidence":0.89,
    "track_id":"7","bbox_xyxy":[200.0,100.0,320.0,480.0]}]
}
```

- `width`, `height` and `bbox_xyxy` refer to the original decoded image, not the
  detector input (640x640 for GPU YOLO, 512x512 for the fast CPU SSD profile)
  or mosaic tile. Bounds are clipped to the image.
- IDs are strings to preserve 64-bit tracker IDs through JavaScript consumers.
  `track_id: null` means no confirmed association for that detection.
- `tracking_epoch` increments after a processing gap longer than two seconds.
  Use `(run_id,camera_id,tracking_epoch,track_id)` as the key. A reconnect or
  long interruption must not join a reused tracker ID to an old track. Older
  1.0 samples without the field can be read with an epoch of zero.
- GPU NvDCF can refine/predict boxes. Its detector confidence can be `-1` when
  unavailable. Do not treat that sentinel as a probability.
- `frame_id` and `source_pts_ns` are local to a source/run. A source restart or
  differing tracker implementation must not create a cross-camera identity.
- File `timestamp_domain: media` means relative media PTS. GPU RTSP uses pipeline
  running PTS; CPU RTSP uses stream-relative PTS. They are **not UTC exposure
  timestamps** and cannot establish physical-camera end-to-end latency.
- `processing_ms` is observed post-decode/pacing to completed metadata, including
  applicable preprocessing/queue/inference/tracking work. The summary separately
  preserves legacy GPU presentation-age measurements.
- `run_id` resolves to `settings.json`, which records source configuration,
  precision, model hashes and the selected backend. Integrators should add their
  calibration/configuration version alongside this record.

`summary.json` and `per_camera.csv` contain aggregate results. `resources.jsonl`
contains time-stamped host/GPU measurements. `metadata_dropped` reports output
backpressure. A throughput pass is not a correctness or tracking-accuracy pass.

These adapters intentionally avoid introducing a new database or broker. A
production consumer can translate this contract to its existing transport and
must define persistence, access controls, event retries and retention itself.
