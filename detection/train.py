from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

model = YOLO("yolo11n.pt")

model.train(
    data=str(ROOT / "data" / "grocery" / "data.yaml"),
    epochs=50,
    imgsz=640,
    batch=16,
    workers=4,
    project=str(ROOT / "models"),
    name="visioncart",
    pretrained=True,
    cache=True
)