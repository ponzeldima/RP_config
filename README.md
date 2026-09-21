# Raspberry Pi Model Benchmark

Every model has one self-contained directory:

```text
models/
  custom_yolov8n_fpv/
    model.yaml             # classes, input size, thresholds, artifact paths
    source_metadata.yaml   # original metadata retained with the export
    model.pt               # Ultralytics/PyTorch artifact
    model.onnx             # ONNX artifact
    imx/                   # all IMX500 converter output
      packerOut.zip
      labels.txt
      pack/network.rpk
```

The folder name is the model identifier. Results use it too:

```text
runs/custom_yolov8n_fpv/2026-09-09_12-00-00/
  run_config.yaml
  run.log
  summary.json
  system.csv
  detections.csv
  recording.mp4            # only when --record is used
```

Install the runtime needed by the selected backend on the Raspberry Pi:

install venv

```bash
python -m venv .venv --system-site-packages
```

```bash
python3 -m pip install psutil pyyaml opencv-python ultralytics onnxruntime openvino flask
sudo apt install imx500-all imx500-tools
python3 -m pip install modlib
# Hailo AI HAT+ (install HailoRT/hailo_platform from Hailo's Raspberry Pi deb packages)
sudo apt install hailo-all
```

Examples:

```bash
# Discover selectable models by their folder name.
python3 benchmark.py --list-models

# ONNX inference on the Raspberry Pi with the IMX500 used as a regular camera.
python3 benchmark.py v8n_640_fpv_public_datasts_310826 --backend onnx --source camera --record --stream

# Original PT model on the Raspberry Pi, no browser stream or recording.
python3 benchmark.py v8n_640_fpv_public_datasts_310826 --backend pt --source camera

# OpenVINO XML/BIN inference on the Raspberry Pi.
python3 benchmark.py v8n_640_fpv_public_datasts_310826 --backend openvino --source camera

# AI inference on the IMX500 camera.
python3 benchmark.py custom_yolov8n_fpv --backend imx --source camera --record --stream

# AI inference on a Hailo-8 AI HAT+.
python3 benchmark.py v8n_640_fpv_public_datasts_310826 --backend hailo --source camera

# Loop an input video for 60 seconds.
python3 benchmark.py custom_yolov8n_fpv --backend onnx --source test.mp4
```

For the included YOLOv8n PT model, a single PyTorch CPU thread measured faster
than 2 or 4 threads on Raspberry Pi 5. It is the default; compare alternatives
with `--torch-threads 2` or `--torch-threads 4` and keep the value in each run's
`run_config.yaml` when reporting results.

`summary.json` reports end-to-end FPS, inference-latency percentiles, confidence and
the percentage of measured frames in which each class was detected. `system.csv`
records CPU, RAM, CPU frequency, temperature, and Raspberry Pi throttling flags once
per second, plus `inference_busy_percent`: the share of that second spent inside the
backend's `detect()` call, which approximates accelerator load (Hailo AI HAT+, IMX500,
CPU/GPU, etc.) since these devices expose no direct utilization counter. mAP, precision,
and recall require a ground-truth dataset and are not estimated from a live camera stream.