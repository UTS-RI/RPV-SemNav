from dataclasses import dataclass
from typing import Any

import numpy as np
from habitat import registry
from habitat.config.default_structured_configs import (
    MeasurementConfig,
)
from habitat.core.embodied_task import Measure
from habitat.core.simulator import Simulator
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@registry.register_measure
class AgentEpisodeDistance(Measure):
    cls_uuid: str = "agent_episode_distance"

    def __init__(self, sim: Simulator, config: DictConfig, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        self._previous_position: np.ndarray = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return AgentEpisodeDistance.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        self._previous_position = self._sim.get_agent_state().position
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        current_position = self._sim.get_agent_state().position
        self._metric += float(np.linalg.norm(current_position - self._previous_position))
        self._previous_position = current_position


@dataclass
class AgentEpisodeDistanceMeasurementConfig(MeasurementConfig):
    type: str = AgentEpisodeDistance.__name__


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements.agent_episode_distance",
    group="habitat/task/measurements",
    name="agent_episode_distance",
    node=AgentEpisodeDistanceMeasurementConfig,
)