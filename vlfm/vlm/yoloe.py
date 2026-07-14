# Adam Scicluna, 2026, for RPV pipeline. Based on yolov7.py.

from typing import List, Optional

import numpy as np
import os
import cv2

from vlfm.vlm.lvis_classes import load_lvis_class_names
from vlfm.vlm.coco_classes import COCO_CLASSES
from .server_wrapper import ServerMixin, host_model, send_request, str_to_image, image_to_str

from ultralytics import YOLO

class YOLOE:
    def __init__(self, weights: str, conf_threshold: float = 0.3, device: Optional[str] = None):
        self._num_steps = 0  # For logging purposes; incremented on each predict call
        """Initialize YOLO-E (Ultralytics) for detection over masked images."""
        if device is None:
            device = "cuda" if YOLO(weights).overrides.get("device", None) is None else None  # YOLO handles device
        self.yoloe_model = YOLO(weights)
        # Send model to device if specified (Ultralytics handles string device internally)
        self.yoloe_model.to(device)

        self.conf_threshold = conf_threshold

        # Set LVIS text prompts for prompt-based weights; safe for prompt-free too
        try:
            # If LVIS_CLASSES exist, otherwise use COCO_CLASSES
            lvis_classes = load_lvis_class_names("data/lvis.yaml", include_all_names=True)
            print(f"YOLOE CLIENT: Loaded {len(lvis_classes)} LVIS class names for YOLOE")
            if len(lvis_classes) > 0:
                print(f"YOLOE CLIENT: Setting YOLOE classes to LVIS classes")
                self.yoloe_model.set_classes(lvis_classes, self.yoloe_model.get_text_pe(lvis_classes))
            else:
                print(f"YOLOE CLIENT: No LVIS class names found, defaulting to COCO classes")
                self.yoloe_model.set_classes(COCO_CLASSES, self.yoloe_model.get_text_pe(COCO_CLASSES))
        except Exception as e:
            # If the loaded weights are prompt-free, set_classes may be unsupported; ignore
            print(f"YOLOE CLIENT: Failed to load LVIS classes or set classes for YOLOE; proceeding without setting classes. Error: {e}")

    def predict(
        self,
        image: np.ndarray,
        conf_thres: Optional[float] = None,
    ) -> dict:
        """
        Run detection on an (already masked) RGB image and return labels plus optional box info.
        YOLOE-26 is designed to not require NMS post-processing, so raw predictions can be returned directly.

        Returns dict with keys:
          labels: list[str]
          boxes: list[[x1,y1,x2,y2]] (pixel coordinates)
          scores: list[float]
          class_ids: list[int]
        """
        # # Save the image passed to this function to see if it is RGB or BGR (for debugging purposes)
        # # We want the image going into Mask2Former to be RGB
        # path_to_save = f"/home/student/scicluna-rpv/RPV-SemNav/running-outputs/yoloe_input_images_bgr/step_{self._num_steps}.jpg"
        # os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        # cv2.imwrite(path_to_save, image)
        # print(f"img channel means (0,1,2): {image[:,:,0].mean():.1f}, {image[:,:,1].mean():.1f}, {image[:,:,2].mean():.1f}")


        conf = self.conf_threshold if conf_thres is None else conf_thres
        results = self.yoloe_model.predict(image, conf=conf, verbose=False)
        result = results[0]

        boxes: List[List[float]] = []
        scores: List[float] = []
        class_ids: List[int] = []
        labels: List[str] = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            scores = result.boxes.conf.cpu().numpy().tolist()
            class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
            labels = [result.names[i] for i in class_ids]

        resp = {
            "labels": labels,
            "boxes": boxes,
            "scores": scores,
            "class_ids": class_ids,
            "unique_labels": list(set(labels)),
        }

        # path_to_save = f"/home/student/scicluna-rpv/RPV-SemNav/running-outputs/yoloe_detections/in-model_step_{self._num_steps}.jpg"
        # os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        # if result.plot() is not None:
        #     cv2.imwrite(path_to_save, result.plot())
        self._num_steps += 1

        try:
            # Optional visualization payload; encode to string to keep JSON serializable
            resp["image_detections"] = image_to_str(result.plot(), quality=90)
        except Exception:
            pass

        return resp


class YOLOEClient:
    def __init__(self, port: int = 12184):
        self.url = f"http://localhost:{port}/yoloe"

    def predict(self, image_numpy: np.ndarray) -> dict:
        """Return the raw dict from the server containing labels/boxes/scores/class_ids."""
        response = send_request(self.url, image=image_numpy)
        return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12184)
    args = parser.parse_args()

    print("Loading model...")

    class YOLOEServer(ServerMixin, YOLOE):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(image)

    yoloe = YOLOEServer("checkpoints/yoloe-26x-seg.pt")
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(yoloe, name="yoloe", port=args.port)
