# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.controllers import BaseController
import numpy as np
import torch

# changed to return tensor of joint_velocities
class DifferentialController(BaseController):
    """Controller uses unicycle model for a diffrential drive

        Args:
            name (str): [description]
            wheel_radius (float): Radius of left and right wheels in cms
            wheel_base (float): Distance between left and right wheels in cms
        """

    def __init__(
        self,
        name: str,
        wheel_radius: float,
        wheel_base: float,
        max_linear_speed: float = 1.0e20,
        max_angular_speed: float = 1.0e20,
        max_wheel_speed: float = 1.0e20,
    ) -> None:
        super().__init__(name)
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.max_wheel_speed = max_wheel_speed

        assert self.max_linear_speed >= 0
        assert self.max_angular_speed >= 0
        assert self.max_wheel_speed >= 0

    def forward(self, command: torch.FloatTensor) -> ArticulationAction:
        """[summary]

        Args:
            command (np.ndarray): [description]

        Returns:
            ArticulationAction: [description]
        """
        if isinstance(command, list):
            command = np.array(command)
        if command.shape[0] != 2:
            raise Exception("command should be of length 2")

        # limit vehical speed
        command = np.clip(
            command,
            a_min=[-self.max_linear_speed, -self.max_angular_speed],
            a_max=[self.max_linear_speed, self.max_angular_speed],
        )
        # calculate wheel speed
        joint_velocities = np.array([0.0, 0.0])
        joint_velocities[0] = ((2 * command[0]) - (command[1] * self.wheel_base)) / (2 * self.wheel_radius)
        joint_velocities[1] = ((2 * command[0]) + (command[1] * self.wheel_base)) / (2 * self.wheel_radius)

        joint_velocities = np.clip(
            joint_velocities,
            a_min=[-self.max_wheel_speed, -self.max_wheel_speed],
            a_max=[self.max_wheel_speed, self.max_wheel_speed],
        )
        return torch.tensor(joint_velocities)

    def reset(self) -> None:
        """[summary]
        """
        return
