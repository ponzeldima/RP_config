import cv2
import numpy as np
from flask import Flask, Response

# Стрім/трекінг для власної YOLOv8n моделі на IMX500 (AI Camera) через офіційну
# бібліотеку modlib (pip install modlib). Вбудований пост-процесинг
# imx500_mobilenet_ssd.json тут не підходить - він розрахований лише на COCO-класи
# базової MobileNet SSD, а не на кастомні класи ['bird', 'cow', 'fpv'].
#
# Перед запуском:
# 1) Сконвертувати модель у формат IMX500 (на ПК, де встановлено ultralytics):
#      from ultralytics import YOLO
#      YOLO("best.pt").export(format="imx", data="best.yaml")
#    Після конвертації покласти артефакти у models/<назва-моделі>/imx/.
# 2) Скопіювати теку models/<назва-моделі>/ на Raspberry Pi, поруч зі скриптом.
# 3) На Pi встановити залежності: sudo apt install imx500-all imx500-tools
#      python3 -m pip install modlib flask opencv-python
from modlib.apps import Annotator, BYTETracker
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_od_yolo_ultralytics

MODEL_DIR = "models/custom_yolov8n_fpv/imx"
CONFIDENCE_THRESHOLD = 0.25

app = Flask(__name__)


class CustomYOLOv8n(Model):
    """Власна YOLOv8n модель (bird, cow, fpv), сконвертована для IMX500."""

    def __init__(self):
        super().__init__(
            model_file=f"{MODEL_DIR}/packerOut.zip",
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(f"{MODEL_DIR}/labels.txt", dtype=str, delimiter="\n")

    def post_process(self, output_tensors):
        return pp_od_yolo_ultralytics(output_tensors)


class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


print("1. Ініціалізація AI камери та завантаження власної моделі...")
device = AiCamera(frame_rate=16)
model = CustomYOLOv8n()
device.deploy(model)

tracker = BYTETracker(BYTETrackerArgs())
annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.5)

print("2. Стрімінг з трекінгом успішно запущено!")
print("Відкрийте http://192.168.3.192:5000 у браузері на Windows.")


def generate_frames():
    frame_idx = 0
    with device as stream:
        for frame in stream:
            frame_idx += 1
            raw = frame.detections

            # Діагностика: раз в секунду показує, чи модель взагалі щось "бачить"
            # (нижче порогу відсікання), щоб відрізнити баг від заниженої впевненості моделі.
            if frame_idx % 30 == 0:
                max_conf = float(raw.confidence.max()) if raw is not None and len(raw) else 0.0
                print(f"[debug] FPS: {frame.fps:.1f}, DPS: {frame.dps:.1f}, сирих детекцій: {len(raw) if raw is not None else 0}, макс. впевненість: {max_conf:.2f}")

            detections = raw[raw.confidence > CONFIDENCE_THRESHOLD] if raw is not None else raw
            # Трекер додає стабільний track_id для кожного об'єкта між кадрами
            detections = tracker.update(frame, detections)

            labels = [
                f"#{track_id} {model.labels[class_id]}: {score:0.2f}"
                for _, score, class_id, track_id in detections
            ]
            annotator.annotate_boxes(frame, detections, labels=labels)
            cv2.putText(frame.image, f"FPS: {frame.fps:.1f}  DPS: {frame.dps:.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # frame.image для VGA-показу вже в BGR (за форматом modlib), конвертація в RGB2BGR тут була зайвою
            # і саме вона "перефарбовувала" кольори
            ret, buffer = cv2.imencode('.jpg', frame.image)
            jpeg_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')


@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)

