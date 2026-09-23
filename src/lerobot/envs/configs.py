# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import abc
import importlib
from dataclasses import dataclass, field
from enum import Enum

import draccus
import gymnasium as gym
from gymnasium.envs.registration import registry as gym_registry

from lerobot.configs import PolicyFeature
from lerobot.processor import PolicyProcessorPipeline


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)
    max_parallel_tasks: int = 1
    disable_env_checker: bool = True

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    def package_name(self) -> str:
        """Package name to import if environment not found in gym registry"""
        return f"gym_{self.type}"

    @property
    def gym_id(self) -> str:
        """ID string used in gym.make() to instantiate the environment"""
        return f"{self.package_name}/{self.task}"

    @property
    @abc.abstractmethod
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()

    def create_envs(
        self,
        n_envs: int,
        use_async_envs: bool = False,
    ) -> dict[str, dict[int, gym.vector.VectorEnv]]:
        """Create {suite: {task_id: VectorEnv}}.

        Default: single-task env via gym.make(). Multi-task benchmarks override.
        AsyncVectorEnv is the default for n_envs > 1; auto-downgraded to Sync for n_envs=1.
        """
        env_cls = gym.vector.AsyncVectorEnv if (use_async_envs and n_envs > 1) else gym.vector.SyncVectorEnv

        if self.gym_id not in gym_registry:
            print(f"gym id '{self.gym_id}' not found, attempting to import '{self.package_name}'...")
            try:
                importlib.import_module(self.package_name)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"Package '{self.package_name}' required for env '{self.type}' not found. "
                    f"Please install it or check PYTHONPATH."
                ) from e

            if self.gym_id not in gym_registry:
                raise gym.error.NameNotFound(
                    f"Environment '{self.gym_id}' not registered even after importing '{self.package_name}'."
                )

        def _make_one():
            return gym.make(self.gym_id, disable_env_checker=self.disable_env_checker, **self.gym_kwargs)

        extra_kwargs: dict = {}
        if env_cls is gym.vector.AsyncVectorEnv:
            extra_kwargs["context"] = "forkserver"
        try:
            from gymnasium.vector import AutoresetMode

            vec = env_cls(
                [_make_one for _ in range(n_envs)], autoreset_mode=AutoresetMode.SAME_STEP, **extra_kwargs
            )
        except ImportError:
            vec = env_cls([_make_one for _ in range(n_envs)], **extra_kwargs)
        return {self.type: {0: vec}}

    def get_env_processors(self):
        """Return (preprocessor, postprocessor) for this env. Default: identity."""
        return PolicyProcessorPipeline(steps=[]), PolicyProcessorPipeline(steps=[])


@dataclass
class HubEnvConfig(EnvConfig):
    """Base class for environments that delegate creation to a hub-hosted make_env.

    Hub environments download and execute remote code from the HF Hub.
    The hub_path points to a repository containing an env.py with a make_env function.
    """

    hub_path: str | None = None  # required: e.g., "username/repo" or "username/repo@branch:file.py"

    @property
    def gym_kwargs(self) -> dict:
        # Not used for hub environments - the hub's make_env handles everything
        return {}


class G1EndEffector(str, Enum):
    """What the G1's arms carry, by hardware name.

    Shared with UnitreeG1Config, which owns the robot-level flag.
    """

    DUMMY = "dummy"  # bare wrists
    DEX1 = "dex1"  # parallel grippers
    DEX3 = "dex3"  # three-finger hands

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise ValueError(f"`end_effector` is expected to be in {list(cls)}, but {value} is provided.")


@EnvConfig.register_subclass("unitree_g1_mujoco")
@dataclass
class UnitreeG1MujocoEnv(HubEnvConfig):
    """Config for the MuJoCo simulation of the Unitree G1.

    The end effector selects the MuJoCo model, and with it the finger actuators and the
    wrist cameras that exist: "dummy" for bare wrists, "dex1" for the parallel grippers,
    "dex3" for the three-finger hands.

    The sim renders either its cameras or its window, never both: the offscreen contexts
    are bound to the thread that creates them, which is not the one driving the viewer.
    `onscreen` is therefore resolved from `publish_images` unless it is set explicitly.
    """

    hub_path: str = "lerobot/unitree-g1-mujoco"
    end_effector: G1EndEffector = G1EndEffector.DEX1
    publish_images: bool = True
    camera_port: int = 5555
    onscreen: bool | None = None

    def __post_init__(self) -> None:
        self.end_effector = G1EndEffector(self.end_effector)
        if self.onscreen is None:
            self.onscreen = not self.publish_images
        elif self.onscreen and self.publish_images:
            raise ValueError(
                "The G1 sim cannot publish camera images and open its viewer in the same "
                "process. Set publish_images=False to watch the window, or onscreen=False "
                "to take the image stream."
            )
