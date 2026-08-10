#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import RobotConfig
    from .robot import Robot
    from .utils import make_robot_from_config

__all__ = ["Robot", "RobotConfig", "make_robot_from_config"]


def __getattr__(name: str) -> Any:
    """Load hardware abstractions only when their public exports are requested.

    Lightweight subpackages such as ``lerobot.robots.alohamini.kinematics`` do
    not require the configuration and hardware dependency stack.
    """
    if name == "RobotConfig":
        from .config import RobotConfig

        return RobotConfig
    if name == "Robot":
        from .robot import Robot

        return Robot
    if name == "make_robot_from_config":
        from .utils import make_robot_from_config

        return make_robot_from_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
