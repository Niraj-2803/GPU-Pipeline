# Measurement Definitions

Run one timed workload at a time after a recorded warmup. Archive the entire run
folder, model hashes, source manifest, runtime versions and hardware inventory.

| Metric | Meaning |
|---|---|
| Source FPS | Encoded stream rate supplied by the source; not a runtime promise |
| Analyzed FPS | Unique processed frames during the measurement window |
| Minimum FPS | Slowest camera's average over that complete window |
| Processing P50/P95 | Paced/received decoded frame to completed metadata |
| Native GPU P95 | Legacy presentation age in pipeline time; not physical alert latency |
| Metadata dropped | Frames omitted by a full metadata-output queue |
| CPU percentage | Whole-host utilization, normalized across logical CPUs |
| GPU percentage | Selected device compute utilization, separate from video engines |
| Decoder/encoder | Selected device's fixed-function video-engine activity |
| VRAM/RAM | Device used memory and host memory; shared services can contribute |

The CPU and GPU use different trackers, so end-to-end FPS is a complete-pipeline
comparison. Run `-track no` for a detector-only comparison. Precision conversions
must retain their own agreement/accuracy results. CPU metadata timing includes
CPU color conversion and resizing; GPU resizing stays on its GPU path.

Local RTSP replay verifies networking, demux, decode and the source adapter. It
does not reproduce real-camera exposure clocks, network loss or every vendor's
reconnect behavior. Source packet timestamps alone are not UTC timestamps.

For a physical-camera latency test, add a synchronized visual clock/event at
capture and measure its appearance/alert at the receiver. Keep that result
separate from the processing-age metrics above.

The current throughput gate requires every camera to average at least 95% of
the target analysis FPS, with no run/schema/metadata-drop error. Also review
one-second rates and P95 latency. A bounded latest-frame queue may preserve
freshness by dropping work; that must never be described as processing all
input frames. Report the CPU `dropped_analysis_frames` counter.

No capacity here includes facial recognition, cross-camera re-ID, all CAMEX
business models, production recording, high availability or a customer SLA.
