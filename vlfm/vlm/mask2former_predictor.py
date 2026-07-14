# Author: Adam Scicluna
# Date: 12 Feb, 2026
# Reusable Mask2Former predictor class supporting panoptic and semantic modes.
# Wraps model loading, config setup, metadata patching, and timed inference.

import os
import sys
import time
import cv2
import torch
import numpy as np
from PIL import Image

# Resolve paths relative to this module's location.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_MASK2FORMER_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, "../..", "Mask2Former"))
if _MASK2FORMER_ROOT not in sys.path:
    sys.path.insert(0, _MASK2FORMER_ROOT)

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import Visualizer, ColorMode

from mask2former import add_maskformer2_config
import mask2former.data  # registers ADE20K / COCO panoptic dataset metadata


# ---- Predefined model configurations ---- #
# Maps (version, backbone, mode) -> (config_path_relative_to_Mask2Former, checkpoint_filename)
# When mode is not in the key, the config is shared between panoptic and semantic.
MODEL_CONFIGS = {
    # --- Panoptic models --- #
    ("ade20k", "swin-l", "panoptic"): (
        "configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml",
        "ade20k-swinl_model_final_e0c58e.pkl",
    ),
    ("ade20k", "r50", "panoptic"): (
        "configs/ade20k/panoptic-segmentation/maskformer2_R50_bs16_160k.yaml",
        "ade20k-r50_model_final_5c90d4.pkl",
    ),
    ("coco", "swin-l", "panoptic"): (
        "configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
        "coco-swinl_model_final_f07440.pkl",
    ),
    ("coco", "r50", "panoptic"): (
        "configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        "coco-r50_model_final_94dc52.pkl",
    ),
    # --- ADE20K semantic-specific models (higher mIoU, lower resolution) --- #
    ("ade20k", "swin-l", "semantic"): (
        "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
        "ade20k-semseg-swinl_model_final_6b4a3a.pkl",
    ),
    ("ade20k", "r50", "semantic"): (
        "configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml",
        "ade20k-semseg-r50_model_final_500878.pkl",
    ),
}


class Mask2FormerPredictor:
    """Reusable Mask2Former predictor (panoptic or semantic mode).

    Usage:
        from mask2former_predictor import Mask2FormerPredictor

        # Panoptic mode (default) — segments with thing/stuff labels
        model = Mask2FormerPredictor(version="ade20k", backbone="swin-l")
        results = model.predict("path/to/image.jpg")

        # Semantic mode — per-pixel class labels, lighter compute
        model = Mask2FormerPredictor(version="ade20k", backbone="r50", mode="semantic")
        results = model.predict("path/to/image.jpg")
        stuff_mask = model.get_stuff_mask(results)  # bool mask of stuff pixels

    Modes:
        - "panoptic": enables all three heads (semantic + instance + panoptic).
          Predictions contain 'panoptic_seg', 'sem_seg', 'instances'.
        - "semantic": enables only the semantic head (cheapest).
          Predictions contain 'sem_seg' (C, H, W) logits tensor.
    """

    def __init__(
        self,
        version: str = "ade20k",
        backbone: str = "r50",
        *,
        mode: str = "semantic",
        config_file: str | None = None,
        model_weights: str | None = None,
        mask2former_root: str | None = None,
        max_test_size: int | None = None,
        always_stuff_class_names: list[str] | None = None,
    ):
        """Initialise the Mask2Former predictor.

        Args:
            version: Dataset the model was trained on ("ade20k" or "coco").
            backbone: Backbone architecture ("swin-l", "r50", etc.).
            mode: "panoptic" (all heads) or "semantic" (semantic head only).
            config_file: Override — absolute path to a YAML config.
            model_weights: Override — absolute path to a .pkl checkpoint.
            mask2former_root: Override — path to the cloned Mask2Former repo.
            max_test_size: Cap the longest test edge (pixels) to save GPU memory.
                           E.g. 640 or 800. None = use config defaults.
        """
        if mode not in ("panoptic", "semantic"):
            raise ValueError(f"mode must be 'panoptic' or 'semantic', got {mode!r}")
        self.mode = mode
        self.version = version
        self.backbone = backbone
        self._m2f_root = mask2former_root or _MASK2FORMER_ROOT

        # Resolve config and weights
        if config_file is None or model_weights is None:
            # Try mode-specific key first, then fall back to panoptic config
            key = (version.lower(), backbone.lower(), self.mode)
            fallback_key = (version.lower(), backbone.lower(), "panoptic")
            if key not in MODEL_CONFIGS and fallback_key not in MODEL_CONFIGS:
                available = sorted({(v, b) for v, b, _ in MODEL_CONFIGS})
                raise ValueError(
                    f"No predefined config for (version={version!r}, backbone={backbone!r}). "
                    f"Available (version, backbone) combos: {available}.  "
                    f"Or pass explicit config_file + model_weights."
                )
            chosen_key = key if key in MODEL_CONFIGS else fallback_key
            rel_cfg, rel_ckpt = MODEL_CONFIGS[chosen_key]
            config_file = config_file or os.path.join(self._m2f_root, rel_cfg)
            model_weights = model_weights or os.path.join(self._m2f_root, "checkpoints", rel_ckpt)

        self._config_file = config_file
        self._model_weights = model_weights
        self._max_test_size = max_test_size

        # Build Detectron2 config
        self.cfg = self._build_cfg()

        # Build predictor (loads model weights)
        self.predictor = DefaultPredictor(self.cfg)

        # Fix metadata so the visualizer works with Mask2Former's unified class IDs
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        self._patch_metadata()

        # Build a lookup: is class i a "thing" (True) or "stuff" (False)?
        # Source this from the dataset category definitions.
        self._isthing = self._build_isthing_lookup()
        # Global override: treat specified class names as 'stuff' (isthing=False)
        if always_stuff_class_names:
            name_to_id = {name.lower(): idx for idx, name in enumerate(self.class_names)}
            for cname in always_stuff_class_names:
                idx = name_to_id.get(cname.lower())
                if idx is not None:
                    self._isthing[idx] = False

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def class_names(self) -> list[str]:
        """Full list of class names in the unified ID space."""
        return self.metadata.stuff_classes

    def predict(self, image_or_path, *, timed: bool = False):
        """Run inference on a single image.

        Args:
            image_or_path: BGR numpy array or path to an image file.
            timed: If True, return (predictions, elapsed_seconds).

        Returns:
            predictions dict (or tuple if timed=True).
        """
        image = self._load_image(image_or_path)
        if timed:
            self._sync()
            t0 = time.perf_counter()
        preds = self.predictor(image)
        if timed:
            self._sync()
            elapsed = time.perf_counter() - t0
            return preds, elapsed
        return preds

    def predict_batch(self, images_or_paths, *, timed: bool = False):
        """Run inference on a list of images.

        Returns:
            list of predictions (or list of (predictions, elapsed) if timed).
        """
        return [self.predict(img, timed=timed) for img in images_or_paths]

    def get_segment_label(self, seg_info: dict) -> str:
        """Human-readable label for a segment_info dict (panoptic mode)."""
        cat_id = seg_info["category_id"]
        name = self.class_names[cat_id]
        tag = "T" if seg_info.get("isthing", False) else "S"
        score = seg_info.get("score")
        label = f"{name} ({tag})"
        if score is not None:
            label += f" {score:.0%}"
        return label

    def get_semantic_seg(self, predictions) -> np.ndarray:
        """Extract per-pixel class IDs from predictions.

        Works in both modes:
            - semantic: uses 'sem_seg' logits directly.
            - panoptic: uses 'sem_seg' logits (always available when SEMANTIC_ON=True).

        Returns:
            (H, W) numpy int array of class IDs.
        """
        sem_seg = predictions["sem_seg"]  # (C, H, W) logits tensor
        return sem_seg.argmax(dim=0).cpu().numpy().astype(int)

    def get_stuff_mask(self, predictions) -> np.ndarray:
        """Return a boolean mask where True = stuff pixel.

        Useful for masking out background / structural regions before passing
        the image to an object detector like YOLO-E.

        Args:
            predictions: dict returned by predict().

        Returns:
            (H, W) bool numpy array. True for stuff, False for thing.
        """
        class_ids = self.get_semantic_seg(predictions)  # (H, W)
        # Vectorised lookup: build bool array for all class IDs
        isthing_arr = np.array(self._isthing, dtype=bool)  # length N_classes
        return ~isthing_arr[class_ids]  # True where stuff

    def mask_stuff(
        self,
        image_or_path,
        predictions,
        *,
        fill_value: int = 0,
        fill_mode: str = "binary"  # Options: "binary", "mean", "blur", "noise"
    ) -> np.ndarray:
        """
        Mask stuff pixels in an image, with options for fill method and extra classes.

        Args:
            image_or_path: BGR numpy array or path to an image file.
            predictions: dict returned by predict().
            fill_mode: How to fill masked regions: "binary", "mean", "blur", or "noise".

        Returns:
            BGR numpy array with stuff regions (and extra classes) masked out.
        """
        image = self._load_image(image_or_path).copy()
        stuff_mask = self.get_stuff_mask(predictions)
        # sem_seg output resolution may differ from input — resize mask if needed
        if stuff_mask.shape[:2] != image.shape[:2]:
            stuff_mask = cv2.resize(
                stuff_mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        if fill_mode == "binary":
            image[stuff_mask] = fill_value
        elif fill_mode == "mean":
            # Compute mean color of unmasked area
            if np.any(~stuff_mask):
                mean_color = image[~stuff_mask].mean(axis=0).astype(np.uint8)
            else:
                mean_color = np.array([0, 0, 0], dtype=np.uint8)
            image[stuff_mask] = mean_color
        elif fill_mode == "blur":
            # Blur the whole image, then copy blurred pixels to masked area
            blurred = cv2.GaussianBlur(image, (31, 31), 0)
            image[stuff_mask] = blurred[stuff_mask]
        elif fill_mode == "noise":
            # Fill masked area with random noise
            noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)
            image[stuff_mask] = noise[stuff_mask]
        else:
            raise ValueError(f"Unknown fill_mode: {fill_mode}. Choose from 'binary', 'mean', 'blur', 'noise'.")
        return image
    

    def visualise(self, image_or_path, predictions, *, alpha: float = 0.5):
        """Draw segmentation results on an image.

        In panoptic mode: draws per-segment masks with thing/stuff tags.
        In semantic mode: draws per-pixel class colours with class names.

        Args:
            image_or_path: BGR numpy array or path to an image file.
            predictions: dict returned by predict().
            alpha: Mask transparency.

        Returns:
            RGB numpy array with overlaid masks + labels.
        """
        image = self._load_image(image_or_path)
        image_rgb = image[:, :, ::-1]
        all_colors = self.metadata.stuff_colors

        if self.mode == "panoptic":
            visualizer = Visualizer(image_rgb, self.metadata, instance_mode=ColorMode.IMAGE)
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            panoptic_cpu = panoptic_seg.to("cpu")

            for seg in segments_info:
                mask = (panoptic_cpu == seg["id"]).numpy().astype(bool)
                if mask.sum() == 0:
                    continue
                cat_id = seg["category_id"]
                color = [x / 255 for x in all_colors[cat_id]]
                label = self.get_segment_label(seg)
                visualizer.draw_binary_mask(
                    mask, color=color, edge_color=(1, 1, 1), text=label, alpha=alpha,
                )
        else:  # semantic
            visualizer = Visualizer(image_rgb, self.metadata, instance_mode=ColorMode.IMAGE)
            class_ids = self.get_semantic_seg(predictions)
            # Resize if sem_seg resolution differs from image
            if class_ids.shape[:2] != image_rgb.shape[:2]:
                class_ids = cv2.resize(
                    class_ids.astype(np.float32),
                    (image_rgb.shape[1], image_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(int)
            unique_ids = np.unique(class_ids)
            for cid in unique_ids:
                mask = (class_ids == cid).astype(bool)
                if mask.sum() == 0:
                    continue
                color = [x / 255 for x in all_colors[cid]]
                tag = "T" if self._isthing[cid] else "S"
                label = f"{self.class_names[cid]} ({tag})"
                visualizer.draw_binary_mask(
                    mask, color=color, edge_color=(1, 1, 1), text=label, alpha=alpha,
                )

        return visualizer.get_output().get_image()  # RGB

    def save_visualisation(self, image_or_path, predictions, output_path, **kwargs):
        """Visualise and save to disk (as BGR JPEG/PNG)."""
        vis_rgb = self.visualise(image_or_path, predictions, **kwargs)
        cv2.imwrite(output_path, vis_rgb[:, :, ::-1])

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _build_cfg(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(self._config_file)
        cfg.MODEL.WEIGHTS = self._model_weights
        # Enable heads based on mode
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True  # always needed
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = (self.mode == "panoptic")
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = (self.mode == "panoptic")
        # Cap test resolution to fit in GPU memory if requested
        if self._max_test_size is not None:
            cfg.INPUT.MIN_SIZE_TEST = min(cfg.INPUT.MIN_SIZE_TEST, self._max_test_size)
            cfg.INPUT.MAX_SIZE_TEST = self._max_test_size
        print(f'Max test size for Mask2Former: {cfg.INPUT.MAX_SIZE_TEST}')
        cfg.freeze()
        return cfg

    def _patch_metadata(self):
        """Align thing_classes/colors to the full unified class list.

        Mask2Former outputs category_ids in the unified N-class space, but
        detectron2's visualizer indexes thing_classes which only has the
        'thing' subset.  Overwrite with the full list so indices match.

        For semantic-only datasets (ade20k_sem_seg_*), the metadata may lack
        stuff_colors entirely, so we source colours from the panoptic category
        definitions.
        """
        # Ensure stuff_colors exist (sem_seg datasets don't register them)
        if not hasattr(self.metadata, "stuff_colors") or not self.metadata.stuff_colors:
            colors = self._get_category_colors()
            if colors:
                self.metadata.stuff_colors = colors

        all_classes = self.metadata.stuff_classes
        all_colors = self.metadata.stuff_colors
        try:
            delattr(self.metadata, "thing_classes")
        except AttributeError:
            pass
        try:
            delattr(self.metadata, "thing_colors")
        except AttributeError:
            pass
        self.metadata.thing_classes = all_classes
        self.metadata.thing_colors = all_colors

    def _get_category_colors(self) -> list[list[int]]:
        """Get per-class RGB colours from the dataset category definitions."""
        version = self.version.lower()
        if "ade20k" in version:
            from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES as CATS
        elif "coco" in version:
            from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES as CATS
        else:
            return []
        return [cat["color"] for cat in CATS]

    def _build_isthing_lookup(self) -> list[bool]:
        """Build a per-class boolean list: True = thing, False = stuff.

        Sources the 'isthing' flag from the dataset category definitions
        (ADE20K_150_CATEGORIES or COCO_CATEGORIES).
        """
        version = self.version.lower()
        if "ade20k" in version:
            from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES as CATS
        elif "coco" in version:
            from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES as CATS
        else:
            # Fallback: assume all classes are things
            return [True] * len(self.class_names)
        return [bool(cat.get("isthing", 0)) for cat in CATS]


    # @staticmethod
    # def _load_image(image_or_path):
    #     if isinstance(image_or_path, (str, os.PathLike)):
    #         img = cv2.imread(str(image_or_path))
    #         if img is None:
    #             raise FileNotFoundError(f"Could not read image: {image_or_path}")
    #         return img
    #     return image_or_path


    @staticmethod
    def _load_image(image_or_path) -> np.ndarray:
        """
        Returns a BGR numpy array regardless of input type
        """
        if isinstance(image_or_path, str):
            return cv2.imread(image_or_path)
        
        elif isinstance(image_or_path, np.ndarray):
            # Habitat gives RGBA (H, W, 4) — drop alpha and convert to BGR
            if image_or_path.shape[2] == 4:
                image_or_path = image_or_path[..., :3]          # drop alpha
                return cv2.cvtColor(image_or_path, cv2.COLOR_RGB2BGR)
            # Already BGR (e.g. from cv2.imread) — pass through
            return image_or_path
        
        elif isinstance(image_or_path, Image.Image):
            # PIL is always RGB — convert to BGR for OpenCV
            img = np.array(image_or_path.convert("RGB"))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        else:
            raise TypeError(f"Unsupported image type: {type(image_or_path)}")
        

    @staticmethod
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
