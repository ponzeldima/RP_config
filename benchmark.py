#!/usr/bin/env python3
"""Run reproducible YOLO benchmarks on a Raspberry Pi or IMX500 AI Camera."""

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import psutil
import yaml


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
RUNS_DIR = PROJECT_DIR / "runs"


@dataclass
class Detection:
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: Optional[int] = None


@dataclass
class ModelSpec:
    name: str
    directory: Path
    classes: list[str]
    input_size: tuple[int, int]
    confidence_threshold: float
    nms_threshold: float
    artifacts: dict[str, Path]

    @classmethod
    def load(cls, model_name: str) -> "ModelSpec":
        directory = (MODELS_DIR / model_name).resolve()
        manifest_path = directory / "model.yaml"
        if MODELS_DIR.resolve() not in directory.parents or not manifest_path.is_file():
            raise ValueError(f"Model '{model_name}' was not found at {manifest_path}")

        with manifest_path.open(encoding="utf-8") as manifest_file:
            data = yaml.safe_load(manifest_file) or {}

        input_size = data.get("input_size", [640, 640])
        if not isinstance(input_size, list) or len(input_size) != 2:
            raise ValueError("model.yaml input_size must be [width, height]")

        artifacts = {
            name: directory / relative_path
            for name, relative_path in (data.get("artifacts") or {}).items()
        }
        classes = data.get("classes") or []
        if not classes:
            raise ValueError("model.yaml must define at least one class")

        return cls(
            name=directory.name,
            directory=directory,
            classes=classes,
            input_size=(int(input_size[0]), int(input_size[1])),
            confidence_threshold=float(data.get("confidence_threshold", 0.25)),
            nms_threshold=float(data.get("nms_threshold", 0.45)),
            artifacts=artifacts,
        )

    def artifact(self, backend: str) -> Path:
        path = self.artifacts.get(backend)
        if path is None:
            raise ValueError(f"Model '{self.name}' has no '{backend}' artifact")
        if not path.is_file():
            raise FileNotFoundError(f"Missing model artifact: {path}")
        return path


class SystemMonitor:
    THROTTLING_FLAGS = {
        0: "under_voltage",
        1: "arm_frequency_capped",
        2: "throttled",
        3: "soft_temperature_limit",
        16: "under_voltage_since_boot",
        17: "arm_frequency_capped_since_boot",
        18: "throttled_since_boot",
        19: "soft_temperature_limit_since_boot",
    }

    def __init__(self, output_path: Path, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.last_sample = 0.0
        self.samples = 0
        self.busy_ms = 0.0
        self.file = output_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "timestamp_s", "cpu_percent", "ram_percent", "ram_used_mb",
                "cpu_frequency_mhz", "temperature_c", "throttled_hex", "throttled_flags",
                "inference_busy_percent",
            ],
        )
        self.writer.writeheader()
        psutil.cpu_percent(None)

    @staticmethod
    def temperature_c() -> Optional[float]:
        thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
        try:
            return int(thermal_path.read_text().strip()) / 1000
        except (FileNotFoundError, ValueError, OSError):
            return None

    @classmethod
    def throttling(cls) -> tuple[str, str]:
        try:
            output = subprocess.check_output(
                ["vcgencmd", "get_throttled"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            match = re.search(r"0x([0-9a-fA-F]+)", output)
            if match is None:
                return "unknown", ""
            value = int(match.group(1), 16)
            flags = [name for bit, name in cls.THROTTLING_FLAGS.items() if value & (1 << bit)]
            return f"0x{value:x}", ",".join(flags)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return "unavailable", ""

    def record_inference_time(self, latency_ms: float) -> None:
        self.busy_ms += latency_ms

    def sample_if_due(self, elapsed_s: float) -> None:
        if elapsed_s - self.last_sample < self.interval_s:
            return
        interval_elapsed_s = elapsed_s - self.last_sample
        self.last_sample = elapsed_s
        memory = psutil.virtual_memory()
        frequency = psutil.cpu_freq()
        throttled_hex, throttled_flags = self.throttling()
        busy_percent = min(100.0, 100 * self.busy_ms / (interval_elapsed_s * 1000)) if interval_elapsed_s else 0.0
        self.busy_ms = 0.0
        self.writer.writerow({
            "timestamp_s": f"{elapsed_s:.3f}",
            "cpu_percent": f"{psutil.cpu_percent(None):.1f}",
            "ram_percent": f"{memory.percent:.1f}",
            "ram_used_mb": f"{memory.used / 1024 / 1024:.1f}",
            "cpu_frequency_mhz": f"{frequency.current:.1f}" if frequency else "",
            "temperature_c": f"{self.temperature_c():.1f}" if self.temperature_c() is not None else "",
            "throttled_hex": throttled_hex,
            "throttled_flags": throttled_flags,
            "inference_busy_percent": f"{busy_percent:.1f}",
        })
        self.file.flush()
        self.samples += 1

    def close(self) -> None:
        self.file.close()


class RunLogger:
    def __init__(self, output_dir: Path, model: ModelSpec, backend: str):
        self.output_dir = output_dir
        self.model = model
        self.backend = backend
        self.frames = 0
        self.detected_frames = 0
        self.latencies_ms: list[float] = []
        self.class_stats = {
            name: {"detections": 0, "frames": 0, "confidence_total": 0.0, "confidence_max": 0.0}
            for name in model.classes
        }
        self.detections_file = (output_dir / "detections.csv").open("w", newline="", encoding="utf-8")
        self.detections_writer = csv.DictWriter(
            self.detections_file,
            fieldnames=[
                "timestamp_s", "frame", "class_id", "class_name", "confidence",
                "x1", "y1", "x2", "y2", "track_id",
            ],
        )
        self.detections_writer.writeheader()

    def add_frame(self, elapsed_s: float, latency_ms: float, detections: list[Detection]) -> None:
        self.frames += 1
        self.latencies_ms.append(latency_ms)
        frame_classes: set[int] = set()
        for detection in detections:
            if not 0 <= detection.class_id < len(self.model.classes):
                continue
            class_name = self.model.classes[detection.class_id]
            stats = self.class_stats[class_name]
            stats["detections"] += 1
            stats["confidence_total"] += detection.confidence
            stats["confidence_max"] = max(stats["confidence_max"], detection.confidence)
            frame_classes.add(detection.class_id)
            self.detections_writer.writerow({
                "timestamp_s": f"{elapsed_s:.3f}", "frame": self.frames,
                "class_id": detection.class_id, "class_name": class_name,
                "confidence": f"{detection.confidence:.4f}",
                "x1": f"{detection.x1:.1f}", "y1": f"{detection.y1:.1f}",
                "x2": f"{detection.x2:.1f}", "y2": f"{detection.y2:.1f}",
                "track_id": detection.track_id if detection.track_id is not None else "",
            })
        if detections:
            self.detected_frames += 1
        for class_id in frame_classes:
            self.class_stats[self.model.classes[class_id]]["frames"] += 1

    def summary(self, elapsed_s: float, system_samples: int) -> dict:
        latencies = np.asarray(self.latencies_ms, dtype=float)
        per_class = {}
        for class_name, stats in self.class_stats.items():
            per_class[class_name] = {
                "detections": stats["detections"],
                "frames_seen": stats["frames"],
                "frames_seen_percent": round(100 * stats["frames"] / self.frames, 2) if self.frames else 0,
                "mean_confidence": round(stats["confidence_total"] / stats["detections"], 4)
                if stats["detections"] else None,
                "max_confidence": round(stats["confidence_max"], 4) if stats["detections"] else None,
            }
        return {
            "model": self.model.name,
            "backend": self.backend,
            "measurement_seconds": round(elapsed_s, 3),
            "frames": self.frames,
            "end_to_end_fps": round(self.frames / elapsed_s, 3) if elapsed_s else 0,
            "frames_with_detections_percent": round(100 * self.detected_frames / self.frames, 2)
            if self.frames else 0,
            "inference_latency_ms": {
                "mean": round(float(latencies.mean()), 3) if len(latencies) else None,
                "p50": round(float(np.percentile(latencies, 50)), 3) if len(latencies) else None,
                "p95": round(float(np.percentile(latencies, 95)), 3) if len(latencies) else None,
            },
            "per_class": per_class,
            "system_samples": system_samples,
            "accuracy_note": "Ground-truth annotations are required for mAP, precision, and recall.",
        }

    def close(self) -> None:
        self.detections_file.close()


class VideoRecorder:
    def __init__(self, output_path: Optional[Path], fps: float):
        self.output_path = output_path
        self.fps = fps
        self.writer: Optional[cv2.VideoWriter] = None

    def write(self, frame: np.ndarray) -> None:
        if self.output_path is None:
            return
        if self.writer is None:
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.output_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Cannot open video writer: {self.output_path}")
        self.writer.write(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()


class MjpegServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.condition = threading.Condition()
        self.frame: Optional[bytes] = None
        self.version = 0
        self.server = None

    def start(self) -> None:
        try:
            from flask import Flask, Response
            from werkzeug.serving import make_server
        except ImportError as error:
            raise RuntimeError("Browser streaming requires Flask: python3 -m pip install flask") from error

        app = Flask(__name__)

        @app.route("/")
        def index():
            def generate():
                version = -1
                while True:
                    with self.condition:
                        self.condition.wait_for(lambda: self.version != version, timeout=10)
                        if self.frame is None:
                            continue
                        jpeg = self.frame
                        version = self.version
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

        self.server = make_server(self.host, self.port, app, threaded=True)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def publish(self, frame: np.ndarray) -> None:
        if self.server is None:
            return
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return
        with self.condition:
            self.frame = encoded.tobytes()
            self.version += 1
            self.condition.notify_all()

    def close(self) -> None:
        if self.server is not None:
            self.server.shutdown()


class HostBackend:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class PtBackend(HostBackend):
    def __init__(
        self, model_path: Path, confidence: float, input_size: tuple[int, int], torch_threads: int
    ):
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("PT backend requires ultralytics: python3 -m pip install ultralytics") from error
        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(1)
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.input_size = input_size
        self.torch_threads = torch_threads

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model(frame, conf=self.confidence, imgsz=self.input_size, verbose=False)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.cpu()
        return [
            Detection(int(class_id), float(score), *map(float, box))
            for box, score, class_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
        ]


class OnnxBackend(HostBackend):
    def __init__(self, model_path: Path, model: ModelSpec, confidence: float):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("ONNX backend requires onnxruntime: python3 -m pip install onnxruntime") from error
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.width, self.height = model.input_size
        self.class_count = len(model.classes)
        self.confidence = confidence
        self.nms_threshold = model.nms_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        image_height, image_width = frame.shape[:2]
        scale = min(self.width / image_width, self.height / image_height)
        resized_width = round(image_width * scale)
        resized_height = round(image_height * scale)
        pad_x = (self.width - resized_width) // 2
        pad_y = (self.height - resized_height) // 2
        resized = cv2.resize(frame, (resized_width, resized_height))
        letterboxed = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        letterboxed[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
        blob = cv2.dnn.blobFromImage(letterboxed, 1 / 255.0, swapRB=True)
        output = np.squeeze(self.session.run(None, {self.input_name: blob})[0])
        if output.ndim != 2:
            raise RuntimeError(f"Unsupported ONNX output shape: {output.shape}")

        # Standard Ultralytics YOLOv8 export is [4 + classes, predictions],
        # for example [6, 8400] for a two-class model. It is not an Nx6 NMS output.
        raw_yolov8 = output.shape[0] == 4 + self.class_count and output.shape[1] > output.shape[0]
        if raw_yolov8:
            output = output.T

        candidates: list[tuple[int, float, float, float, float, float, float]] = []
        if not raw_yolov8 and output.shape[1] == 6:
            for x1, y1, x2, y2, score, class_id in output:
                if score >= self.confidence:
                    candidates.append((int(class_id), float(score), x1, y1, x2, y2, 1.0))
        elif raw_yolov8:
            for row in output:
                scores = row[4:]
                class_id = int(np.argmax(scores))
                score = float(scores[class_id])
                if score < self.confidence:
                    continue
                cx, cy, width, height = row[:4]
                candidates.append((class_id, score, cx - width / 2, cy - height / 2, width, height, 0.0))
        else:
            raise RuntimeError(f"Unsupported ONNX output shape: {output.shape}")

        boxes, scores, mapped = [], [], []
        for class_id, score, first, second, third, fourth, xyxy in candidates:
            if xyxy:
                x1, y1, x2, y2 = first, second, third, fourth
                width, height = x2 - x1, y2 - y1
            else:
                x1, y1, width, height = first, second, third, fourth
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            width /= scale
            height /= scale
            boxes.append([x1, y1, width, height])
            scores.append(score)
            mapped.append((class_id, score, x1, y1, x1 + width, y1 + height))

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, self.nms_threshold)
        return [Detection(*mapped[int(index)]) for index in np.asarray(indices).reshape(-1)] if len(indices) else []


class OpenVinoBackend(HostBackend):
    def __init__(self, model_path: Path, model: ModelSpec, confidence: float):
        try:
            from openvino import Core
        except ImportError as error:
            raise RuntimeError(
                "OpenVINO backend requires openvino: python3 -m pip install openvino"
            ) from error
        compiled_model = Core().compile_model(str(model_path), "CPU")
        self.input_layer = compiled_model.input(0)
        input_shape = list(self.input_layer.shape)
        if len(input_shape) != 4 or any(int(dimension) <= 0 for dimension in input_shape[2:]):
            raise RuntimeError(f"Unsupported OpenVINO input shape: {input_shape}")
        self.compiled_model = compiled_model
        self.output_layer = compiled_model.output(0)
        self.height, self.width = map(int, input_shape[2:])
        self.class_count = len(model.classes)
        self.confidence = confidence
        self.nms_threshold = model.nms_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        image_height, image_width = frame.shape[:2]
        scale = min(self.width / image_width, self.height / image_height)
        resized_width = round(image_width * scale)
        resized_height = round(image_height * scale)
        pad_x = (self.width - resized_width) // 2
        pad_y = (self.height - resized_height) // 2
        resized = cv2.resize(frame, (resized_width, resized_height))
        letterboxed = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        letterboxed[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
        blob = cv2.dnn.blobFromImage(letterboxed, 1 / 255.0, swapRB=True)
        output = np.squeeze(self.compiled_model({self.input_layer: blob})[self.output_layer])
        if output.ndim != 2:
            raise RuntimeError(f"Unsupported OpenVINO output shape: {output.shape}")

        raw_yolov8 = output.shape[0] == 4 + self.class_count and output.shape[1] > output.shape[0]
        if not raw_yolov8:
            raise RuntimeError(f"Unsupported OpenVINO output shape: {output.shape}")

        boxes, scores, mapped = [], [], []
        for row in output.T:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < self.confidence:
                continue
            cx, cy, width, height = row[:4]
            x1 = (cx - width / 2 - pad_x) / scale
            y1 = (cy - height / 2 - pad_y) / scale
            width /= scale
            height /= scale
            boxes.append([x1, y1, width, height])
            scores.append(score)
            mapped.append((class_id, score, x1, y1, x1 + width, y1 + height))

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, self.nms_threshold)
        return [Detection(*mapped[int(index)]) for index in np.asarray(indices).reshape(-1)] if len(indices) else []


class HailoBackend(HostBackend):
    """Runs .hef models compiled with baked-in NMS on a Hailo-8 AI HAT+ via HailoRT."""

    def __init__(self, model_path: Path, model: ModelSpec, confidence: float):
        try:
            from hailo_platform import (
                HEF, ConfigureParams, FormatType, HailoStreamInterface,
                InferVStreams, InputVStreamParams, OutputVStreamParams, VDevice,
            )
        except ImportError as error:
            raise RuntimeError(
                "Hailo backend requires the HailoRT Python API (hailo_platform)"
            ) from error

        self.hef = HEF(str(model_path))
        self.exit_stack = ExitStack()
        self.target = self.exit_stack.enter_context(VDevice())
        configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        network_group_params = self.network_group.create_params()
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        # The compiled HEF expects raw uint8 pixels; it already bakes in its own normalization.
        input_vstreams_params = InputVStreamParams.make(
            self.network_group, quantized=True, format_type=FormatType.UINT8
        )
        output_vstreams_params = OutputVStreamParams.make(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        self.infer_pipeline = self.exit_stack.enter_context(
            InferVStreams(self.network_group, input_vstreams_params, output_vstreams_params)
        )
        self.exit_stack.enter_context(self.network_group.activate(network_group_params))
        self.width, self.height = model.input_size
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[Detection]:
        image_height, image_width = frame.shape[:2]
        scale = min(self.width / image_width, self.height / image_height)
        resized_width = round(image_width * scale)
        resized_height = round(image_height * scale)
        pad_x = (self.width - resized_width) // 2
        pad_y = (self.height - resized_height) // 2
        resized = cv2.resize(frame, (resized_width, resized_height))
        letterboxed = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        letterboxed[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        input_data = {self.input_vstream_info.name: np.expand_dims(rgb, axis=0)}
        outputs = self.infer_pipeline.infer(input_data)

        # Baked-in NMS output: one array per class, rows are [y1, x1, y2, x2, score] normalized to 0-1.
        per_class_boxes = next(iter(outputs.values()))[0]
        detections = []
        for class_id, boxes in enumerate(per_class_boxes):
            for row in np.asarray(boxes).reshape(-1, 5):
                y1, x1, y2, x2, score = row
                if score < self.confidence:
                    continue
                detections.append(Detection(
                    class_id, float(score),
                    (x1 * self.width - pad_x) / scale, (y1 * self.height - pad_y) / scale,
                    (x2 * self.width - pad_x) / scale, (y2 * self.height - pad_y) / scale,
                ))
        return detections

    def close(self) -> None:
        self.exit_stack.close()


class FrameSource:
    def frames(self) -> Iterable[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class CameraSource(FrameSource):
    def __init__(self, size: tuple[int, int], fps: float):
        try:
            from picamera2 import Picamera2
        except ImportError as error:
            raise RuntimeError("Camera source requires Picamera2") from error
        self.camera = Picamera2()
        # Without an explicit FrameDurationLimits control, Picamera2 falls back to the
        # sensor mode's default duration, which is often capped around 30 fps.
        frame_duration_us = int(1_000_000 / fps)
        config = self.camera.create_video_configuration(
            main={"size": size, "format": "RGB888"},
            controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
        )
        self.camera.configure(config)
        self.camera.start()

    def frames(self) -> Iterable[np.ndarray]:
        while True:
            yield cv2.cvtColor(self.camera.capture_array(), cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        self.camera.stop()
        self.camera.close()


class VideoSource(FrameSource):
    def __init__(self, path: Path, loop: bool):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.loop = loop

    def frames(self) -> Iterable[np.ndarray]:
        while True:
            ok, frame = self.capture.read()
            if ok:
                yield frame
            elif self.loop:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                return

    def close(self) -> None:
        self.capture.release()


class ImageDirectorySource(FrameSource):
    def __init__(self, path: Path, loop: bool):
        self.images = sorted(
            image for image in path.iterdir() if image.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not self.images:
            raise ValueError(f"No images found in {path}")
        self.loop = loop

    def frames(self) -> Iterable[np.ndarray]:
        while True:
            for image_path in self.images:
                frame = cv2.imread(str(image_path))
                if frame is not None:
                    yield frame
            if not self.loop:
                return


def annotate(frame: np.ndarray, detections: list[Detection], model: ModelSpec, fps: float) -> np.ndarray:
    output = frame.copy()
    for detection in detections:
        if not 0 <= detection.class_id < len(model.classes):
            continue
        cv2.rectangle(output, (int(detection.x1), int(detection.y1)), (int(detection.x2), int(detection.y2)), (0, 220, 0), 2)
        label = f"{model.classes[detection.class_id]} {detection.confidence:.2f}"
        if detection.track_id is not None:
            label = f"#{detection.track_id} {label}"
        cv2.putText(output, label, (int(detection.x1), max(18, int(detection.y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    cv2.putText(output, f"FPS: {fps:.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
    return output


def create_output_dir(model_name: str) -> Path:
    output_dir = RUNS_DIR / model_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def available_models() -> list[str]:
    if not MODELS_DIR.is_dir():
        return []
    return sorted(path.name for path in MODELS_DIR.iterdir() if (path / "model.yaml").is_file())


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("benchmark")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / "run.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def build_source(source_value: str, size: tuple[int, int], loop: bool, fps: float) -> FrameSource:
    if source_value == "camera":
        return CameraSource(size, fps)
    path = Path(source_value).expanduser()
    if path.is_dir():
        return ImageDirectorySource(path, loop)
    if path.is_file():
        return VideoSource(path, loop)
    raise ValueError("--source must be 'camera', a video file, or an image directory")


def build_host_backend(
    backend: str, model: ModelSpec, confidence: float, torch_threads: int
) -> HostBackend:
    if backend == "pt":
        return PtBackend(model.artifact("pt"), confidence, model.input_size, torch_threads)
    if backend == "onnx":
        return OnnxBackend(model.artifact("onnx"), model, confidence)
    if backend == "openvino":
        return OpenVinoBackend(model.artifact("openvino"), model, confidence)
    if backend == "hailo":
        return HailoBackend(model.artifact("hailo"), model, confidence)
    raise ValueError(f"Unsupported host backend: {backend}")


def run_host_benchmark(args: argparse.Namespace, model: ModelSpec, output_dir: Path, logger: logging.Logger) -> dict:
    backend = build_host_backend(args.backend, model, args.confidence, args.torch_threads)
    source = build_source(args.source, args.camera_size, args.loop_source, args.camera_fps)
    monitor = SystemMonitor(output_dir / "system.csv")
    run_logger = RunLogger(output_dir, model, args.backend)
    recorder = VideoRecorder(output_dir / "recording.mp4" if args.record else None, args.record_fps)
    stream = MjpegServer(args.stream_host, args.stream_port) if args.stream else None
    if stream:
        stream.start()
        logger.info("Browser stream: http://%s:%s", args.stream_host, args.stream_port)

    frame_iterator = iter(source.frames())
    try:
        warmup_until = time.monotonic() + args.warmup
        while time.monotonic() < warmup_until:
            backend.detect(next(frame_iterator))

        started_at = time.monotonic()
        for frame in frame_iterator:
            elapsed_s = time.monotonic() - started_at
            if elapsed_s >= args.duration or (args.max_frames and run_logger.frames >= args.max_frames):
                break
            inference_started = time.monotonic()
            detections = backend.detect(frame)
            latency_ms = (time.monotonic() - inference_started) * 1000
            elapsed_s = time.monotonic() - started_at
            run_logger.add_frame(elapsed_s, latency_ms, detections)
            monitor.record_inference_time(latency_ms)
            monitor.sample_if_due(elapsed_s)
            annotated = annotate(frame, detections, model, run_logger.frames / elapsed_s if elapsed_s else 0)
            recorder.write(annotated)
            if stream:
                stream.publish(annotated)
        elapsed_s = time.monotonic() - started_at
        monitor.sample_if_due(elapsed_s + monitor.interval_s)
        return run_logger.summary(elapsed_s, monitor.samples)
    finally:
        source.close()
        recorder.close()
        monitor.close()
        run_logger.close()
        backend.close()
        if stream:
            stream.close()


def run_imx_benchmark(args: argparse.Namespace, model: ModelSpec, output_dir: Path, logger: logging.Logger) -> dict:
    if args.source != "camera":
        raise ValueError("The IMX500 backend supports only --source camera")
    try:
        from modlib.apps import Annotator, BYTETracker
        from modlib.devices import AiCamera
        from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
        from modlib.models.post_processors import pp_od_yolo_ultralytics
    except ImportError as error:
        raise RuntimeError("IMX backend requires modlib and IMX500 system packages") from error

    class CustomImxModel(Model):
        def __init__(self):
            super().__init__(
                model_file=str(model.artifact("imx")), model_type=MODEL_TYPE.CONVERTED,
                color_format=COLOR_FORMAT.RGB, preserve_aspect_ratio=False,
            )

        def post_process(self, output_tensors):
            return pp_od_yolo_ultralytics(output_tensors)

    class TrackerArgs:
        track_thresh = 0.25
        track_buffer = 30
        match_thresh = 0.8
        aspect_ratio_thresh = 3.0
        min_box_area = 1.0
        mot20 = False

    device = AiCamera(frame_rate=args.camera_fps)
    device.deploy(CustomImxModel())
    tracker = BYTETracker(TrackerArgs())
    annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.5)
    monitor = SystemMonitor(output_dir / "system.csv")
    run_logger = RunLogger(output_dir, model, "imx")
    recorder = VideoRecorder(output_dir / "recording.mp4" if args.record else None, args.record_fps)
    stream = MjpegServer(args.stream_host, args.stream_port) if args.stream else None
    if stream:
        stream.start()
        logger.info("Browser stream: http://%s:%s", args.stream_host, args.stream_port)

    try:
        with device as camera:
            frame_iterator = iter(camera)
            warmup_until = time.monotonic() + args.warmup
            while time.monotonic() < warmup_until:
                next(frame_iterator)
            started_at = time.monotonic()
            for frame in frame_iterator:
                elapsed_s = time.monotonic() - started_at
                if elapsed_s >= args.duration or (args.max_frames and run_logger.frames >= args.max_frames):
                    break
                inference_started = time.monotonic()
                raw = frame.detections
                filtered = raw[raw.confidence >= args.confidence] if raw is not None else raw
                tracked = tracker.update(frame, filtered)
                latency_ms = (time.monotonic() - inference_started) * 1000
                detections = []
                labels = []
                for box, score, class_id, track_id in tracked:
                    box = np.asarray(box).reshape(-1)
                    detections.append(Detection(int(class_id), float(score), *map(float, box[:4]), int(track_id)))
                    labels.append(f"#{track_id} {model.classes[int(class_id)]}: {score:.2f}")
                elapsed_s = time.monotonic() - started_at
                run_logger.add_frame(elapsed_s, latency_ms, detections)
                monitor.record_inference_time(latency_ms)
                monitor.sample_if_due(elapsed_s)
                annotator.annotate_boxes(frame, tracked, labels=labels)
                annotated = frame.image
                cv2.putText(annotated, f"FPS: {frame.fps:.1f} DPS: {frame.dps:.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
                recorder.write(annotated)
                if stream:
                    stream.publish(annotated)
            elapsed_s = time.monotonic() - started_at
            monitor.sample_if_due(elapsed_s + monitor.interval_s)
            return run_logger.summary(elapsed_s, monitor.samples)
    finally:
        recorder.close()
        monitor.close()
        run_logger.close()
        if stream:
            stream.close()


def parse_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", value)
    if not match:
        raise argparse.ArgumentTypeError("Camera size must be WIDTHxHEIGHT, for example 1280x720")
    return int(match.group(1)), int(match.group(2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="Folder name below models/")
    parser.add_argument("--list-models", action="store_true", help="List model folders with model.yaml")
    parser.add_argument("--backend", choices=("pt", "onnx", "openvino", "imx", "hailo"))
    parser.add_argument("--source", default="camera", help="camera, video path, or image directory")
    parser.add_argument("--duration", type=float, default=30, help="Measurement duration in seconds")
    parser.add_argument("--warmup", type=float, default=10, help="Warmup duration in seconds")
    parser.add_argument("--max-frames", type=int, help="Optional measurement frame limit")
    parser.add_argument("--confidence", type=float, help="Override model confidence threshold")
    parser.add_argument(
        "--torch-threads", type=int, default=1,
        help="PyTorch CPU threads for the PT backend; 1 is fastest for the included YOLOv8n test",
    )
    parser.add_argument("--camera-size", type=parse_size, default=(1280, 720))
    parser.add_argument("--camera-fps", type=int, default=60, help="Requested capture rate for --source camera")
    parser.add_argument("--no-loop-source", action="store_false", dest="loop_source", help="Do not repeat file sources")
    parser.set_defaults(loop_source=True)
    parser.add_argument("--record", action="store_true", help="Write annotated recording.mp4")
    parser.add_argument("--record-fps", type=float, default=16)
    parser.add_argument("--stream", action="store_true", help="Publish annotated MJPEG stream")
    parser.add_argument("--stream-host", default="0.0.0.0")
    parser.add_argument("--stream-port", type=int, default=5000)
    args = parser.parse_args()
    if not args.list_models and (not args.model or not args.backend):
        parser.error("model and --backend are required unless --list-models is used")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.list_models:
            models = available_models()
            print("\n".join(models) if models else "No models found in models/")
            return 0
        if args.duration <= 0 or args.warmup < 0 or args.torch_threads <= 0:
            raise ValueError("--duration and --torch-threads must be positive; --warmup cannot be negative")
        model = ModelSpec.load(args.model)
        args.confidence = args.confidence if args.confidence is not None else model.confidence_threshold
        output_dir = create_output_dir(model.name)
        logger = setup_logging(output_dir)
        with (output_dir / "run_config.yaml").open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(vars(args), config_file, sort_keys=False)
        logger.info("Model: %s | backend: %s | output: %s", model.name, args.backend, output_dir)
        summary = run_imx_benchmark(args, model, output_dir, logger) if args.backend == "imx" else run_host_benchmark(args, model, output_dir, logger)
        with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)
        logger.info("Done: %.2f FPS across %s frames", summary["end_to_end_fps"], summary["frames"])
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Benchmark interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())