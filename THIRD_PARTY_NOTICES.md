# Third party notices

This document does not change the project owner's license for CAMEX code.
Dependencies and model weights retain their own licenses and terms.

- Ultralytics YOLOv8 code and reference weights: https://github.com/ultralytics/ultralytics/tree/v8.3.40 and https://www.ultralytics.com/license . The demo archive includes the upstream AGPL-3.0 license and a model-source notice. Check obligations or obtain appropriate commercial rights before proprietary integration.
- OpenVINO runtime and Open Model Zoo models: https://github.com/openvinotoolkit/openvino and https://github.com/openvinotoolkit/open_model_zoo . The CPU-model helper downloads pinned upstream files and verifies their published checksums; it does not grant additional rights.
- NVIDIA DeepStream, TensorRT, CUDA and the NGC base image: NVIDIA license terms apply. See the base image's license notices and https://docs.nvidia.com/metropolis/deepstream/ .
- ByteTrack implementation via Supervision: https://github.com/roboflow/supervision . Review the pinned package's license and upstream attribution.
- FFmpeg/PyAV, OpenCV and MediaMTX: retain their respective upstream licenses; codec distribution requirements depend on the build and deployment.

The included video data is synthetic benchmark material, not real staff or client CCTV footage. Do not add private camera data, credentials or logs to a public release.
