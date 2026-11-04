# Model Files and Reproduction

## Exact reference

The original benchmark used a person-output YOLOv8s ONNX, rather than a generic
stock YOLO export. The local review model bundle contains that file, with this
SHA-256:

`75b32bc520027b73d9b44acbb167ea5f78bf9750f5415e4f0985291cea73f7de`

The same ONNX feeds TensorRT and the OpenVINO FP32 baseline. CPU quantization
adds its own provenance file. Save both the XML and BIN for an OpenVINO model.
Save the engine, engine build log and JSON manifest for TensorRT. Engine files
are tied to the GPU/runtime and must be built on the deployment hardware.

The output has a row for each proposal: x1,y1,x2,y2 in 640x640 letterbox pixels,
person score, and maximum class score. Retain a row only when person score is
at least 0.25 and is also a maximum-class score. Then run IoU 0.45 NMS, maximum
300 detections. Inverse letterboxing yields native source coordinates.

## Optional export from stock weights

This path is an alternative, not a byte-identical reproduction of the supplied
ONNX. Validate the resulting model before assigning the old benchmark numbers
to it. Use a separate export environment so model conversion dependencies do
not change the runtime environment.

```bash
python3 -m venv .venv-export
.venv-export/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv-export/bin/pip install -r requirements-export.txt
.venv-export/bin/python tools/export_person_model.py \
  --weights yolov8s.pt --output models/yolov8s-person-exported.onnx
```

Ultralytics downloads the named weights when they are not already available.
Use its [official YOLOv8 model source](https://docs.ultralytics.com/models/yolov8/).
For offline installation, stage the weights and Python wheels ahead of time.
The export wrapper converts stock center/size boxes to xyxy and preserves the
person/max-class columns expected by both backends.

The export path was exercised in the local review and its output matched all
410 reference detections on a 30-frame agreement check. This does not imply a
byte-identical ONNX or accuracy equivalence on all scenes. The exporter keeps
its Ultralytics settings inside the output folder, not in the user's shared
settings directory.

Model weights and inference libraries have their own license terms. Check
commercial deployment rights before incorporation into a client product.

## Quality checks

`tools/evaluate_agreement.py` compares detections against FP32 using the same
preprocessed tensors, threshold and NMS. Calibration samples and evaluation
timestamps are separate. This detects regressions but does not establish real
world accuracy: use independently labeled footage for that claim.
The delivered 60-frame holdout uses different timestamps from the same 20
synthetic clips, not an independent camera/site dataset.

Treat GPU FP16 and CPU INT8 as different precision modes. Report their accuracy
or agreement results alongside their throughput. Do not conflate a change of
tracker, precision, source FPS or input dimensions with a hardware speedup.
