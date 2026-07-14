# Adam Scicluna, 2026, for RPV pipeline. Mask2Former server wrapper.

import os
from typing import List, Optional

import numpy as np
import cv2

from .mask2former_predictor import Mask2FormerPredictor
from .server_wrapper import ServerMixin, host_model, send_request, str_to_image, image_to_str


class Mask2Former:
    """Serve Mask2Former to produce a masked image (stuff removed) for YOLO-E."""

    def __init__(
        self,
        version: str = "ade20k",
        backbone: str = "r50",
        mode: str = "semantic",
        mask_type: str = "blur",
        max_test_size: Optional[int] = None,
        stuff_classes: Optional[List[str]] = None,
    ) -> None:
        # Allow overriding max_test_size via env (optional)
        env_max = os.environ.get("MASK2FORMER_MAX_TEST_SIZE")
        if env_max:
            try:
                max_test_size = int(env_max)
            except ValueError:
                pass

        self.mask_type = mask_type
        self.predictor = Mask2FormerPredictor(
            version=version,
            backbone=backbone,
            mode=mode,
            max_test_size=max_test_size,
            always_stuff_class_names=stuff_classes or [],
        )
        self._num_steps = 0  # For logging purposes; incremented on each mask_image call

    def mask_image(self, image: np.ndarray) -> np.ndarray:
        """Run Mask2Former and mask out stuff pixels using the configured fill mode."""
        # # Save the image passed to this function to see if it is RGB or BGR (for debugging purposes)
        # # We want the image going into Mask2Former to be RGB
        # path_to_save = f"/home/student/scicluna-rpv/RPV-SemNav/running-outputs/mask2former_input_images_bgr/step_{self._num_steps}.jpg"
        # os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        # cv2.imwrite(path_to_save, image)
        # print(f"img channel means (0,1,2): {image[:,:,0].mean():.1f}, {image[:,:,1].mean():.1f}, {image[:,:,2].mean():.1f}")
        # self._num_steps += 1

        preds = self.predictor.predict(image)
        masked = self.predictor.mask_stuff(image, preds, fill_mode=self.mask_type)
        return masked


class Mask2FormerClient:
    def __init__(self, port: int = 12181):
        self.url = f"http://localhost:{port}/mask2former"

    def mask_image(self, image_numpy: np.ndarray) -> np.ndarray:
        """Send an image, receive a masked image (np.ndarray)."""
        response = send_request(self.url, image=image_numpy)
        masked_str = response["masked_image"]
        masked_img = str_to_image(masked_str)
        return masked_img


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12181)
    parser.add_argument("--version", type=str, default=os.environ.get("MASK2FORMER_VERSION", "ade20k"))
    parser.add_argument("--backbone", type=str, default=os.environ.get("MASK2FORMER_BACKBONE", "r50"))
    parser.add_argument("--mode", type=str, default=os.environ.get("MASK2FORMER_MODE", "semantic"))
    parser.add_argument("--mask_type", type=str, default=os.environ.get("MASK2FORMER_MASK_TYPE", "blur"))
    parser.add_argument(
        "--stuff_classes",
        type=str,
        default=os.environ.get("MASK2FORMER_STUFF_CLASSES", "door,window,curtain"),
        help="Comma-separated extra class names to treat as stuff (optional)",
    )
    args = parser.parse_args()

    stuff_classes = [c.strip() for c in args.stuff_classes.split(",") if c.strip()] if args.stuff_classes else []

    print("Loading Mask2Former model...")

    class Mask2FormerServer(ServerMixin, Mask2Former):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            masked = self.mask_image(image)
            # Use a quality score of 100, since this will be passed to YOLOE for detection (preserve detail)
            return {"masked_image": image_to_str(masked, quality=100)}

    mask2former = Mask2FormerServer(
        version=args.version,
        backbone=args.backbone,
        mode=args.mode,
        mask_type=args.mask_type,
        stuff_classes=stuff_classes,
    )
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(mask2former, name="mask2former", port=args.port)
