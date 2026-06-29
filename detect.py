from ultralytics import YOLO
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

# Load model only once
model = YOLO(ROOT / "models" / "best.pt")


def detect_products(image_path, conf=0.4):
    """
    Detect grocery products from an image.

    Args:
        image_path (str or Path): Path to image.
        conf (float): Confidence threshold.

    Returns:
        list: Unique detected product categories.
    """

    results = model(image_path, conf=conf)

    detected_products = []

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            detected_products.append(label)

    basket = sorted(set(detected_products))

    # Save basket
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "basket.json", "w") as f:

        json.dump(

            {

                "basket": basket

            },

            f,

            indent=4

        )

    return basket


if __name__ == "__main__":

    image = (

        ROOT

        / "data"

        / "grocery"

        / "test"

        / "images"

        / "BEANS0012_png.rf.229ae741be8dcb706500fdf0af7ff2c7.jpg"

    )

    basket = detect_products(image)

    print("\nDetected Basket")

    print("----------------")

    for item in basket:

        print(item)
