#!/usr/bin/env python3
"""Record the Raspberry Pi camera indefinitely with a browser preview/status page."""

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Camera Recorder</title>
  <style>
    :root { color-scheme: light; font-family: "DejaVu Sans", sans-serif; background: #edf1ef; color: #172521; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; padding: 28px; }
    main { max-width: 1100px; margin: 0 auto; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-bottom: 18px; }
    h1 { margin: 0; font-size: 24px; font-weight: 650; }
    #status { display: inline-flex; align-items: center; gap: 9px; font-size: 14px; font-weight: 700; }
    #light { width: 11px; height: 11px; border-radius: 50%; background: #78847f; }
    #status.active #light { background: #d94232; box-shadow: 0 0 0 4px #d9423222; }
    figure { margin: 0; background: #101816; border-radius: 5px; overflow: hidden; min-height: 220px; display: grid; place-items: center; }
    img { width: 100%; height: auto; display: block; }
    .controls { display: flex; align-items: center; gap: 10px; padding: 16px 0; }
    button { border: 0; border-radius: 4px; padding: 11px 16px; background: #146b54; color: white; font: inherit; font-weight: 650; cursor: pointer; }
    button.stop { background: #273630; }
    button:disabled { opacity: .42; cursor: default; }
    #file { margin-left: auto; color: #52615b; font-size: 13px; overflow-wrap: anywhere; text-align: right; }
    @media (max-width: 560px) { body { padding: 18px; } header { align-items: flex-start; flex-direction: column; } .controls { flex-wrap: wrap; } #file { flex-basis: 100%; margin: 4px 0 0; text-align: left; } }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Camera recorder</h1>
      <div id="status"><span id="light"></span><span id="label">Connecting</span></div>
    </header>
    <figure><img src="/stream" alt="Live camera preview"></figure>
    <div class="controls">
      <button id="start" onclick="setRecording(true)">Start recording</button>
      <button id="stop" class="stop" onclick="setRecording(false)">Stop recording</button>
      <span id="file"></span>
    </div>
  </main>
  <script>
    async function refresh() {
      try {
        const state = await (await fetch('/api/status')).json();
        document.getElementById('status').classList.toggle('active', state.recording);
        document.getElementById('label').textContent = state.recording ? 'RECORDING' : 'NOT RECORDING';
        document.getElementById('start').disabled = state.recording;
        document.getElementById('stop').disabled = !state.recording;
        document.getElementById('file').textContent = state.recording ? `Saving to ${state.file}` : '';
      } catch (_) {
        document.getElementById('label').textContent = 'Camera service unavailable';
      }
    }
    async function setRecording(active) {
      await fetch(active ? '/api/record/start' : '/api/record/stop', { method: 'POST' });
      refresh();
    }
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>"""


class CameraRecorder:
    def __init__(self, output_dir: Path, size: tuple[int, int], fps: float, start_recording: bool):
        try:
            from picamera2 import Picamera2
            from picamera2.encoders import H264Encoder
            from picamera2.outputs import FfmpegOutput
        except ImportError as error:
            raise RuntimeError("Camera recorder requires Picamera2 and its H.264 encoder") from error

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.H264Encoder = H264Encoder
        self.FfmpegOutput = FfmpegOutput
        self.lock = threading.Condition()
        self.stop_requested = threading.Event()
        self.latest_jpeg: Optional[bytes] = None
        self.frame_version = 0
        self.recording_requested = False
        self.recording_path: Optional[Path] = None
        self.encoder = None
        self.error: Optional[str] = None

        self.camera = Picamera2()
        sensor_modes = self.camera.sensor_modes
        preview_scale = min(640 / size[0], 480 / size[1], 1.0)
        preview_size = (
            max(2, int(size[0] * preview_scale) // 2 * 2),
            max(2, int(size[1] * preview_scale) // 2 * 2),
        )
        config = self.camera.create_video_configuration(
            main={"size": size, "format": "YUV420"},
            lores={"size": preview_size, "format": "RGB888"},
        )
        self.camera.configure(config)
        sensor_size = self.camera.camera_configuration()["sensor"]["output_size"]
        mode_fps = [mode["fps"] for mode in sensor_modes if mode["size"] == sensor_size]
        if mode_fps:
            max_fps = max(mode_fps)
            if self.fps > max_fps:
                print(f"Camera mode supports at most {max_fps:.2f} fps; using {max_fps:.2f} fps")
                self.fps = max_fps
        frame_duration_us = int(1_000_000 / self.fps)
        self.camera.set_controls({"FrameDurationLimits": (frame_duration_us, frame_duration_us)})
        self.camera.start()
        time.sleep(1)
        if start_recording:
            self.set_recording(True)

    def _new_path(self) -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        return self.output_dir / f"recording_{stamp}.mp4"

    def set_recording(self, active: bool, resume_preview: bool = True) -> None:
        with self.lock:
            if active == self.recording_requested:
                return
            if active:
                path = self._new_path()
                encoder = self.H264Encoder(bitrate=12_000_000, framerate=int(round(self.fps)))
                encoder.threads = 4
                self.camera.start_recording(encoder, self.FfmpegOutput(str(path)), name="main")
                self.encoder = encoder
                self.recording_path = path
                self.error = None
                self.recording_requested = True
            else:
                if self.encoder is not None:
                    self.camera.stop_recording()
                    self.encoder = None
                    if resume_preview:
                        self.camera.start()
                self.recording_path = None
                self.recording_requested = False
            self.lock.notify_all()

    def status(self) -> dict:
        with self.lock:
            return {
                "recording": self.recording_requested,
                "file": self.recording_path.name if self.recording_path else None,
                "error": self.error,
            }

    def frames(self):
        while not self.stop_requested.is_set():
            with self.lock:
                if self.stop_requested.is_set():
                    break
                frame = self.camera.capture_array("lores")
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.lock:
                    self.latest_jpeg = encoded.tobytes()
                    self.frame_version += 1
                    self.lock.notify_all()

    def close(self) -> None:
        self.stop_requested.set()
        self.set_recording(False, resume_preview=False)
        with self.lock:
            self.lock.notify_all()
        self.camera.stop()
        self.camera.close()


def create_app(recorder: CameraRecorder):
    from flask import Flask, Response, jsonify

    app = Flask(__name__)

    @app.get("/")
    def index():
        return PAGE

    @app.get("/api/status")
    def status():
        return jsonify(recorder.status())

    @app.post("/api/record/start")
    def start_recording():
        recorder.set_recording(True)
        return jsonify(recorder.status())

    @app.post("/api/record/stop")
    def stop_recording():
        recorder.set_recording(False)
        return jsonify(recorder.status())

    @app.get("/stream")
    def stream():
        def generate():
            version = -1
            while not recorder.stop_requested.is_set():
                with recorder.lock:
                    recorder.lock.wait_for(
                        lambda: recorder.frame_version != version or recorder.stop_requested.is_set(),
                        timeout=10,
                    )
                    if recorder.stop_requested.is_set():
                        return
                    jpeg = recorder.latest_jpeg
                    version = recorder.frame_version
                if jpeg is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = map(int, value.lower().split("x", 1))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Camera size must be WIDTHxHEIGHT, for example 1280x720") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Camera width and height must be positive")
    return width, height


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--camera-size", type=parse_size, default=(2028, 1520))
    parser.add_argument("--camera-fps", type=float, default=30)
    parser.add_argument("--start-stopped", action="store_true", help="Start with preview on but recording off")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    if args.camera_fps <= 0:
        parser.error("--camera-fps must be positive")

    try:
        from flask import Flask
        from werkzeug.serving import make_server
    except ImportError as error:
        parser.error("Install the browser dependencies with: python3 -m pip install flask")

    recorder = CameraRecorder(args.output_dir, args.camera_size, args.camera_fps, not args.start_stopped)
    server = make_server(args.host, args.port, create_app(recorder), threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Camera interface: http://localhost:{args.port}/")
    print(f"Saving recordings in: {recorder.output_dir.resolve()}")
    try:
        recorder.frames()
    except KeyboardInterrupt:
        print("Stopping camera recorder")
    finally:
        server.shutdown()
        recorder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())