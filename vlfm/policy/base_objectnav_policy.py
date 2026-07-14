# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import Tensor

from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import get_fov, rho_theta

from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.lvis_classes import load_lvis_class_names

# Import clients for new vision models in pipeline
from vlfm.vlm.clip import CLIPClient
from vlfm.vlm.mask2former import Mask2FormerClient
from vlfm.vlm.sam3 import SAM3Client
from vlfm.vlm.yoloe import YOLOEClient

# This may be needed
from vlfm.vlm.detections import ObjectDetections

# from vlfm.vlm.blip2 import BLIP2Client
# from vlfm.vlm.grounding_dino import GroundingDINOClient, ObjectDetections
# from vlfm.vlm.sam import MobileSAMClient
# from vlfm.vlm.yolov7 import YOLOv7Client

try:
    from habitat_baselines.common.tensor_dict import TensorDict

    from vlfm.policy.base_policy import BasePolicy
except Exception:

    class BasePolicy:  # type: ignore
        pass


class BaseObjectNavPolicy(BasePolicy):
    _target_object: str = ""
    _policy_info: Dict[str, Any] = {}
    _object_masks: Union[np.ndarray, Any] = None  # set by ._update_object_map()
    _stop_action: Union[Tensor, Any] = None  # MUST BE SET BY SUBCLASS
    _observations_cache: Dict[str, Any] = {}
    _non_coco_caption = ""
    _load_yolo: bool = True

    def __init__(
        self,
        pointnav_policy_path: str,
        depth_image_shape: Tuple[int, int],
        pointnav_stop_radius: float,
        object_map_erosion_size: float,
        visualize: bool = True,
        compute_frontiers: bool = True,
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        agent_radius: float = 0.18,
        obstacle_map_area_threshold: float = 1.5,
        hole_area_thresh: int = 100000,
        use_vqa: bool = False,
        vqa_prompt: str = "Is this ",
        coco_threshold: float = 0.8,
        non_coco_threshold: float = 0.4,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Must pass arguments due to "action_space" being mandatory in Habitat v0.3.X
        super().__init__(*args, **kwargs) 
        # self._debug = False
        # self._object_detector = GroundingDINOClient(port=int(os.environ.get("GROUNDING_DINO_PORT", "12181")))
        # self._coco_object_detector = YOLOv7Client(port=int(os.environ.get("YOLOV7_PORT", "12184")))
        # self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "12183")))
        # self._use_vqa = use_vqa
        # if use_vqa:
        #     self._vqa = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "12185")))

        # Initialise new vision models in pipeline
        self._object_masker = Mask2FormerClient(port=int(os.environ.get("MASK2FORMER_PORT", "12181")))
        self._object_detector = YOLOEClient(port=int(os.environ.get("YOLOE_PORT", "12184")))
        self._object_segmenter = SAM3Client(port=int(os.environ.get("SAM3_PORT", "12183")))
        self._clip = CLIPClient(port=int(os.environ.get("CLIP_PORT", "12182")))     # Maybe should be called _vqa

        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(erosion_size=object_map_erosion_size)
        self._depth_image_shape = tuple(depth_image_shape)
        self._pointnav_stop_radius = pointnav_stop_radius
        self._visualize = visualize
        # print(f'BaseObjectNavPolicy -> Visualise? = {self._visualize}??????????????????????????????')
        self._vqa_prompt = vqa_prompt
        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold

        self._num_steps = 0
        self._steps_between_detections = 3
        self._navigate_redetect_interval = 3
        self._navigate_step_counter = 0
        self._did_reset = False
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._called_stop = False
        self._compute_frontiers = compute_frontiers
        if compute_frontiers:
            self._obstacle_map = ObstacleMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
            )
        # Flag to do either object-object or object-room-object RPV cooccurrence
        self._direct_object_object: bool = os.environ.get("DIRECT_OBJECT_OBJECT", "false").lower() in ("true", "1", "yes") # Default to RPV mode

    def _reset(self) -> None:
        self._target_object = ""
        self._pointnav_policy.reset()
        self._object_map.reset()
        self._last_goal = np.zeros(2)
        self._num_steps = 0
        self._done_initializing = False
        self._called_stop = False
        if self._compute_frontiers:
            self._obstacle_map.reset()
        self._did_reset = True
        self._mode = "initialize"

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., spinning in place to get a good view of the scene).
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        """
        self._pre_step(observations, masks)

        object_map_rgbd = self._observations_cache["object_map_rgbd"]

        # Skip the detection/segmentation pipeline when already navigating to a
        # confirmed goal. The goal is derived from the object map, which persists
        # across steps, so we can reuse it without rerunning vision.
        skip_vision = getattr(self, "_mode", "") == "navigate" and self._object_map.has_object(self._target_object)

        if skip_vision:
            detections = [{}]
            # Lightweight SAM3-only re-detection every N navigate steps
            self._navigate_step_counter += 1
            if self._navigate_step_counter % self._navigate_redetect_interval == 0:
                print(
                    f"Navigate mode: redetecting {self._target_object} every {self._navigate_redetect_interval} steps (SAM3)"
                )
                for (rgb, depth, tf, min_d, max_d, fx_val, fy_val) in object_map_rgbd:
                    self._redetect_target_object(rgb, depth, tf, min_d, max_d, fx_val, fy_val)
        else:
            # Allow subclasses (e.g., ITMPolicyV2) to consume detections/segments before mode selection
            # In explore mode, only get detections every "steps between detection" steps (default 3, unneccesary to run every step)
            if getattr(self, "_mode", "") == "explore" and self._num_steps % self._steps_between_detections != 0:
                detections = [{}]
            else:
                detections = [
                    self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
                    for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
            ]
            # Still perform segmentation every step, to look for the target object
            self._get_object_signals(detections)

        
        robot_xy = self._observations_cache["robot_xy"]
        goal = self._get_target_object_location(robot_xy)

        if not self._done_initializing:  # Initialize
            mode = "initialize"
            pointnav_action = self._initialize()
        elif goal is None:  # Haven't found target object yet
            mode = "explore"
            pointnav_action = self._explore(observations)
        else:
            mode = "navigate"
            pointnav_action = self._pointnav(goal[:2], stop=True)

        self._mode = mode

        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
        self._policy_info.update(self._get_policy_info(detections[0]))
        self._num_steps += 1

        self._observations_cache = {}
        self._did_reset = False

        return pointnav_action, rnn_hidden_states

    def _get_object_signals(self, detections: List[Dict[str, Any]]) -> None:
        """Hook for subclasses to run logic after object map update and before mode selection."""
        return

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if not self._did_reset and masks[0] == 0:
            self._reset()
            self._target_object = observations["objectgoal"]
        try:
            self._cache_observations(observations)
        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration
        self._policy_info = {}

    def _initialize(self) -> Tensor:
        raise NotImplementedError

    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        if self._object_map.has_object(self._target_object):
            return self._object_map.get_best_object(self._target_object, position)
        else:
            return None

    def _get_policy_info(self, detections: Dict[str, Any]) -> Dict[str, Any]:
        if self._object_map.has_object(self._target_object):
            target_point_cloud = self._object_map.get_target_cloud(self._target_object)
        else:
            target_point_cloud = np.array([])
        policy_info = {
            "target_object": self._target_object.split("|")[0],
            "gps": str(self._observations_cache["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache["robot_heading"]),
            "target_detected": self._object_map.has_object(self._target_object),
            "target_point_cloud": target_point_cloud,
            "nav_goal": self._last_goal,
            "stop_called": self._called_stop,
            # don't render these on egocentric images when making videos:
            "render_below_images": [
                "target_object",
            ],
        }

        if not self._visualize:
            return policy_info

        annotated_depth = self._observations_cache["object_map_rgbd"][0][1] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if self._object_masks.sum() > 0:
            # If self._object_masks isn't all zero, get the object segmentations and
            # draw them on the rgb and depth images
            contours, _ = cv2.findContours(self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_frame = detections.get("annotated_frame") if isinstance(detections, dict) else None
            base_frame = annotated_frame if annotated_frame is not None else self._observations_cache["object_map_rgbd"][0][0]
            annotated_rgb = cv2.drawContours(base_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = self._observations_cache["object_map_rgbd"][0][0]
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map.visualize(), cv2.COLOR_BGR2RGB)

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        return policy_info


    def _get_object_detections(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Mask "stuff" (Mask2Former) → YOLO-E ("prompt generator") → labels (plus boxes/scores). 
        Ensure target classes are included as SAM3 prompts
        """
        # Habitat observations are RGB; for consistency with prior testing, feed Mask2Former and YOLOE predictors BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # SKIP_MASK2FORMER: bypass Mask2Former masking when using COCO-only classes
        if os.environ.get("SKIP_MASK2FORMER", "") == "1":
            masked_img = img
        else:
            # Mask "stuff" with Mask2Former
            masked_img = self._object_masker.mask_image(img)
        # Run YOLO-E on the masked image to get detections (unique labels, boxes, scores) as prompts
        det = self._object_detector.predict(masked_img)  # dict: labels, boxes, scores, class_ids, unique_labels

        # Ensure target classes are included in unique labels returned by YOLO-E for SAM3 prompts
        target_classes = [c for c in self._target_object.split("|") if c]
        unique_labels = det.get("unique_labels", [])
        for prompt in target_classes:
            if prompt not in unique_labels:
                unique_labels.append(prompt)

        # Return the detections and an annotated image (decoded if provided)
        annotated_frame = None
        if det.get("image_detections"):
            try:
                annotated_frame = str_to_image(det["image_detections"])
            except Exception:
                annotated_frame = None

        #print(f"GET OBJECT DETECTIONS: \nDetected labels: {det.get('labels', [])} \nUnique labels: {unique_labels} \nTarget classes: {target_classes}")
        # path_to_save = f"/home/student/scicluna-rpv/RPV-SemNav/running-outputs/yoloe_detections_rgb/step_{self._num_steps}.jpg"
        # os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        # if annotated_frame is not None:
        #     cv2.imwrite(path_to_save, annotated_frame)
        #     print(f"Saved YOLO-E detections to {path_to_save}")
        # else:
        #     print("No annotated frame returned from YOLO-E to save!")


        return {
            "labels": det.get("labels", []),
            "boxes": det.get("boxes", []),
            "scores": det.get("scores", []),
            "class_ids": det.get("class_ids", []),
            "unique_labels": unique_labels,
            "annotated_frame": annotated_frame,  # Plot YOLO-E detections on masked image from M2F
        }

        

    def _pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)
            self._last_goal = goal
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info["rho_theta"] = np.array([rho, theta])
        if rho < self._pointnav_stop_radius and stop:
            self._called_stop = True
            return self._stop_action
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        return action


    def _redetect_target_object(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> None:
        """Lightweight SAM3-only re-detection of the target during navigate mode.

        Skips YOLO-E / Mask2Former — just prompts SAM3 with the target label,
        then updates the object map and explored cone if a mask is found.
        """
        target_label = self._target_object.split("|")[0]
        if not target_label:
            return

        sam3_out = self._object_segmenter.segment_masks(rgb, [target_label])
        masks = sam3_out.get("masks", [])
        sam_labels = sam3_out.get("labels", [])
        height, width = rgb.shape[:2]

        # Rebuild _object_masks so the visualization overlay reflects the
        # latest segmentation rather than the stale initial detection.
        self._object_masks = np.zeros((height, width), dtype=np.uint8)

        for m_idx, mask in enumerate(masks):
            label = sam_labels[m_idx] if m_idx < len(sam_labels) else None
            if label != target_label:
                continue
            mask_np = np.array(mask, dtype=bool)
            if mask_np.shape[:2] != (height, width):
                mask_np = cv2.resize(
                    mask_np.astype(np.uint8), (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            # Update visualization mask
            self._object_masks[mask_np] = 1
            self._object_map.update_map(
                self._target_object,
                depth,
                mask_np.astype(np.uint8),
                tf_camera_to_episodic,
                min_depth, max_depth, fx, fy,
            )

        cone_fov = get_fov(fx, depth.shape[1])
        self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)

    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> Dict[str, Any]:
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                SAM3 segmentation to extract better object point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.

        Returns:
            Dict: The segmentations made from the SAM3 segmentation model.
        """
        #print(f"SCICLUNA UPDATE OBJECT MAP FUNCTION")


        detections = self._get_object_detections(rgb)
        unique_labels = detections.get("unique_labels", [])
        labels_for_prompts = list(unique_labels)  # working copy

        #print(f"Object detections for SAM3 prompts: {labels_for_prompts}")

        # Ensure the primary target is included as a prompt
        target_label = self._target_object.split("|")[0]
        if target_label and target_label not in labels_for_prompts:
            labels_for_prompts.append(target_label)

        # print(f"Target label: {target_label} | Final labels for SAM3 prompts: {labels_for_prompts}")

        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)

        # If depth is missing, try to infer (optional, same as original behavior)
        if np.array_equal(depth, np.ones_like(depth)) and unique_labels:
            depth = self._infer_depth(rgb, min_depth, max_depth)
            obs = list(self._observations_cache["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache["object_map_rgbd"][0] = tuple(obs)

        # SAM3 segmentation using detected labels as prompts
        sam3_out = self._object_segmenter.segment_masks(rgb, labels_for_prompts)
        masks = sam3_out.get("masks", [])
        sam_boxes = sam3_out.get("boxes", [])
        sam_scores = sam3_out.get("scores", [])
        sam_labels = sam3_out.get("labels", [])

        # print(f"SAM3 labels: {sam_labels}")
        # print(f"SAM3 scores: {sam_scores}")

        segments: List[Dict[str, Any]] = []

        # Update map with SAM3 masks and cache segment metadata for downstream scoring
        for m_idx, mask in enumerate(masks):
            mask_np = np.array(mask, dtype=bool)
            if mask_np.shape[:2] != (height, width):
                mask_np = cv2.resize(mask_np.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)

            # Compute pixel centroid for this mask (downstream can project if needed)
            coords = np.argwhere(mask_np)
            centroid = coords.mean(axis=0).tolist() if coords.size > 0 else [0.0, 0.0]

            label = sam_labels[m_idx] if m_idx < len(sam_labels) else None
            is_target = label == target_label

            if is_target:
                # Only add to the object map (and mark contours) when the segment matches the target label
                self._object_masks[mask_np] = 1

                self._object_map.update_map(
                    self._target_object,
                    depth,
                    mask_np.astype(np.uint8),
                    tf_camera_to_episodic,
                    min_depth,
                    max_depth,
                    fx,
                    fy,
                )

            segments.append(
                {
                    "label": label,
                    "box": sam_boxes[m_idx] if m_idx < len(sam_boxes) else None,
                    "score": sam_scores[m_idx] if m_idx < len(sam_scores) else None,
                    "mask": mask_np,
                    "centroid_px": centroid,
                    "is_target": is_target,
                }
            )



        cone_fov = get_fov(fx, depth.shape[1])
        self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)

        # Cache segment metadata for downstream consumers (e.g., value map scoring)
        self._policy_info["segments"] = segments

        return {
            "labels": labels_for_prompts,
            "sam_masks": masks,
            "sam_boxes": sam_boxes,
            "sam_scores": sam_scores,
            "sam_labels": sam_labels,
            "segments": segments,
            "annotated_frame": detections.get("annotated_frame", rgb),
        }


    def _cache_observations(self, observations: "TensorDict") -> None:
        """Extracts the rgb, depth, and camera transform from the observations.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """
        raise NotImplementedError

    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """Infers the depth image from the rgb image.

        Args:
            rgb (np.ndarray): The rgb image to infer the depth from.

        Returns:
            np.ndarray: The inferred depth image.
        """
        raise NotImplementedError


@dataclass
class VLFMConfig:
    name: str = "HabitatITMPolicy"
    text_prompt: str = "Seems like there is a target_object ahead."
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    depth_image_shape: Tuple[int, int] = (224, 224)
    pointnav_stop_radius: float = 0.9
    use_max_confidence: bool = False
    object_map_erosion_size: int = 5
    exploration_thresh: float = 0.0
    obstacle_map_area_threshold: float = 1.5  # in square meters
    min_obstacle_height: float = 0.61
    max_obstacle_height: float = 0.88
    hole_area_thresh: int = 100000
    use_vqa: bool = False
    vqa_prompt: str = "Is this "
    coco_threshold: float = 0.8
    non_coco_threshold: float = 0.4
    agent_radius: float = 0.18

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        # This returns all the fields listed above, except the name field
        return [f.name for f in fields(VLFMConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="vlfm_config_base", node=VLFMConfig())
