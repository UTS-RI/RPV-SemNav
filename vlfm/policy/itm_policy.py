# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold

# CLIP
from vlfm.vlm.clip import CLIPClient
from vlfm.vlm.room_types import ROOM_TYPES

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"


class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._itm = CLIPClient(port=int(os.environ.get("CLIP_PORT", "12182")))
        self._room_types = ROOM_TYPES
        self._text_prompt = text_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map,  # Always pass obstacle map so FMM can use the free-space mask
        )
        self._acyclic_enforcer = AcyclicEnforcer()

    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        os.environ["DEBUG_INFO"] = ""
        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("Sticking to last point.")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

        return best_frontier, best_value

    def _get_policy_info(self, detections: Dict[str, Any]) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        # Render two obstacle maps: clean (no markers) and with detection overlays
        centroids = self._value_map.get_object_centroids()
        if self._compute_frontiers:
            # Clean obstacle map (no detection markers)
            policy_info["obstacle_map"] = cv2.cvtColor(
                self._obstacle_map.visualize(),
                cv2.COLOR_BGR2RGB,
            )
            # Obstacle map with numbered detection markers
            if centroids:
                policy_info["obstacle_map_detections"] = cv2.cvtColor(
                    self._obstacle_map.visualize(object_centroids=centroids),
                    cv2.COLOR_BGR2RGB,
                )

        # Add detection legend as overlay text (readable at any scale)
        if centroids:
            lines = [f"  {i+1}: {o.get('label','?')} ({o.get('score',0):.2f})"
                     for i, o in enumerate(centroids)]
            det_text = "detections: " + " | ".join(lines)
            policy_info["render_below_images"].append("detections")
            policy_info["detections"] = det_text

        return policy_info


    def _update_value_map(self) -> None:
        #print(f"SCICLUNA UPDATE VALUE MAP: Updating value map with ITM scores for target '{self._target_object}'")

        # ---- Process NEW segments (only present on detection steps) ----
        # Segments are consumed once and cleared so stale centroids are not
        # re-projected with a newer camera pose on subsequent steps.
        segments = self._policy_info.pop("segments", [])

        if segments:
            target_label = self._target_object.split("|")[0]

            query_labels = [seg.get("label") for seg in segments]
            if any(query_labels):
                # ---- CLIP scoring ----
                if self._direct_object_object:
                    # Direct object-object scoring (no RPV or softmax, just similarities)
                    segment_clip_resp = self._itm.cooccurrence(
                        query=target_label,
                        candidates=query_labels
                    )
                    segment_similarities = segment_clip_resp.get("similarities", []) or []
                    print(f"SCICLUNA UPDATE VALUE MAP: Direct object-object CLIP similarities for target '{target_label}': {segment_similarities}")
                    #print(f'Type of segment_similarities: {type(segment_similarities)}, Length: {len(segment_similarities)}')
                
                    assert len(segments) == len(segment_similarities), (
                        f"Number of segments ({len(segments)}) and CLIP similarity responses "
                        f"({len(segment_similarities)}) must match."
                    )

                    for idx, seg in enumerate(segments):
                        # Call it "dot_product" for consistency with RPV mode, even though it's just a similarity score
                        seg["dot_product"] = segment_similarities[idx][0] if isinstance(segment_similarities[idx], list) else segment_similarities[idx] 
                        print(f"SCICLUNA UPDATE VALUE MAP: Segment {idx} label '{seg.get('label', 'unknown')}' similarity score: {seg['dot_product']}")

                else:
                    # Object-room-object RPV scoring
                    target_clip_resp = self._itm.cooccurrence(query=target_label, candidates=self._room_types)
                    target_room_probs = target_clip_resp.get("probabilities", []) or []
                    #print(f"SCICLUNA UPDATE VALUE MAP: CLIP probabilities for target '{target_label}' over room types: {target_room_probs}")    
                    

                    segment_clip_resp = self._itm.cooccurrence(
                        query=query_labels,
                        candidates=self._room_types,
                        target_prob_dist=target_room_probs,
                    )
                    segment_probs = segment_clip_resp.get("probabilities", []) or []
                    segment_dot_products = segment_clip_resp.get("dot_products", []) or []
                    #print(f"SCICLUNA UPDATE VALUE MAP: CLIP probabilities for segments over room types: {segment_probs}, with segments: {[seg.get('label', 'unknown') for seg in segments]}")
                    #print(f"SCICLUNA UPDATE VALUE MAP: CLIP dot products for segments with target '{target_label}': {segment_dot_products}")

                    assert (
                        len(segments) == len(segment_dot_products) == len(segment_probs)
                    ), (
                        f"Number of segments ({len(segments)}), CLIP probability responses "
                        f"({len(segment_probs)}), and CLIP dot product responses "
                        f"({len(segment_dot_products)}) must all match."
                    )

                    for idx, seg in enumerate(segments):
                        seg["probability"] = segment_probs[idx]
                        seg["dot_product"] = segment_dot_products[idx]

                # ---- Project each segment centroid to episodic world coords ----
                if "object_map_rgbd" not in self._observations_cache:
                    print("No object_map_rgbd in cache, cannot project centroids.")
                    raise Exception("No object_map_rgbd in cache")

                rgb_obs, depth_obs, tf_cam, min_d, max_d, fx, fy = self._observations_cache["object_map_rgbd"][0]

                signals_placed = 0
                for seg in segments:
                    dot_prod = seg.get("dot_product")
                    if dot_prod is None or dot_prod <= 0:
                        continue

                    centroid_px = seg.get("centroid_px")
                    if centroid_px is None:
                        continue

                    mask = seg.get("mask")  # boolean mask for robust depth
                    world_xy = self._project_centroid_to_world(
                        centroid_px, depth_obs, tf_cam, min_d, max_d, fx, fy, mask=mask
                    )
                    if world_xy is None:
                        continue

                    label = seg.get("label", "unknown")
                    score = float(dot_prod)

                    print(f"SCICLUNA UPDATE VALUE MAP: Adding object signal for '{label}' at {world_xy} with score {score:.4f} (Target: '{target_label}')")


                    self._value_map.add_object_signal(world_xy, score, label)
                    signals_placed += 1

        # ---- Always recompute FMM / free-space every step ----
        # Even when no new segments were added, the obstacle map gains new
        # free space each step, so dirty objects must be re-propagated.
        if self._value_map._tracked_objects:
            self._value_map.recompute_value_map()
            self._value_map.apply_free_space_mask()

        # Always update agent trajectory for visualisation
        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )


    # Centroid back-projection
    @staticmethod
    def _project_centroid_to_world(
        centroid_px: list,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        mask: np.ndarray = None,
    ) -> Union[np.ndarray, None]:
        """Back-project an image-space centroid into episodic world (x, y).

        Uses the **median depth over the object mask** when available,
        falling back to the single centroid pixel otherwise.

        Args:
            centroid_px: [row, col] in image pixels (from np.argwhere mean).
            depth: The normalised [0,1] depth image.
            tf_camera_to_episodic: 4×4 camera→episodic transform.
            min_depth / max_depth: Depth range in metres.
            fx, fy: Camera focal lengths in pixels.
            mask: Optional boolean mask of the detected object.

        Returns:
            2-element array [x, y] in episodic metres, or None on failure.
        """
        row, col = int(centroid_px[0]), int(centroid_px[1])
        h, w = depth.shape[:2]
        row = np.clip(row, 0, h - 1)
        col = np.clip(col, 0, w - 1)

        # --- Robust depth: median over mask pixels, fall back to centroid ---
        if mask is not None and mask.shape[:2] == (h, w) and mask.any():
            mask_depths = depth[mask]
            mask_depths_m = mask_depths * (max_depth - min_depth) + min_depth
            valid = (mask_depths_m > min_depth * 1.01) & (mask_depths_m < max_depth * 0.99)
            if valid.sum() > 0:
                z_m = float(np.median(mask_depths_m[valid]))
            else:
                z_norm = depth[row, col]
                z_m = z_norm * (max_depth - min_depth) + min_depth
        else:
            z_norm = depth[row, col]
            z_m = z_norm * (max_depth - min_depth) + min_depth

        # Reject invalid depths
        if z_m <= min_depth * 1.01 or z_m >= max_depth * 0.99:
            return None

        cx, cy = w / 2.0, h / 2.0
        x_cam = (col - cx) * z_m / fx
        y_cam = (row - cy) * z_m / fy

        # The transform tf_camera_to_episodic expects the camera-frame
        # convention used in get_point_cloud():
        #   axis-0 = forward (depth),  axis-1 = left,  axis-2 = up
        # Convert from pinhole (x=right, y=down, z=forward):
        point_cam = np.array([z_m, -x_cam, -y_cam, 1.0])

        point_world = tf_camera_to_episodic @ point_cam

        if point_world[3] == 0:
            raise Exception("Homogeneous coordinate is zero, cannot project to world coordinates.")
            return None

        return point_world[:2] / point_world[3]


    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError


class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        reduce_fn = self._vis_reduce_fn if len(self._text_prompt.split(PROMPT_SEPARATOR)) > 1 else None
        return self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=reduce_fn)


class ITMPolicyV2(BaseITMPolicy):
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _get_object_signals(self, detections: List[Dict[str, Any]]) -> None:
        """
        Run value-map update after detections/segmentation are populated

        Currently not doing anything with detections input, since update_value_map
        has access to self.policy_info which contains the segments and their labels/other info
        """
        self._update_value_map()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        reduce_fn = self._vis_reduce_fn if len(self._text_prompt.split(PROMPT_SEPARATOR)) > 1 else None
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=reduce_fn)
        return sorted_frontiers, sorted_values


class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=self._reduce_values)

        return sorted_frontiers, sorted_values

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return [v[0] for v in values]
