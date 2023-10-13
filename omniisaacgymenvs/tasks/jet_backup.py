from fileinput import close
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.controllers.differential_controller import DifferentialController
from omniisaacgymenvs.robots.articulations.jetbot import Jetbot
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import GeometryPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationActions
from omni.isaac.range_sensor import _range_sensor
from omni.isaac.core.utils.rotations import quat_to_euler_angles
import omni.kit.commands
from pxr import Gf
from pathlib import Path
import numpy as np
import torch
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale
import math
import omni.replicator.isaac as dr
from omni.isaac.sensor import LidarRtx
from gym import spaces
from omni.isaac.core.objects import DynamicSphere, DynamicCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.render_product import create_hydra_texture
import omni.replicator.core as rep
"""
TODO:
- add variables like episode length and collision range to config
- use @torch.jit.script to speed up functions that get called every step
- clean up code
"""


class JetbotTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._jetbot_positions = torch.tensor([0.0, 0.0, 0.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500
        
        self.collision_range = 0.11 # 0.11 or 0.20

        self.ranges_count = 360
        self._num_observations = self.ranges_count + 2 # +2 for angle and distance (polar coords)
        self._num_actions = 2

        self._diff_controller = DifferentialController(name="simple_control",wheel_radius=0.0325, wheel_base=0.1125)

        RLTask.__init__(self, name, env)

        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # init tensors that need to be set to correct device
        self.prev_goal_distance = torch.zeros(self._num_envs).to(self._device)
        self.prev_heading = torch.zeros(self._num_envs).to(self._device)
        self.target_position = torch.tensor([0, -1.5, 0.0]).to(self._device)
        self.obstacle_position = torch.tensor([1.4, 0, 0.05]).to(self._device)

        return

    def set_up_scene(self, scene) -> None:
        self.get_jetbot()
        self.add_target()
        self.add_obstacle()
        self.get_home()
        RLTask.set_up_scene(self, scene)
        self._home = ArticulationView(prim_paths_expr="/World/envs/.*/Home/home", name="home_view")
        self._jetbots = ArticulationView(prim_paths_expr="/World/envs/.*/Jetbot/jetbot", name="jetbot_view")
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/Target/target", name="targets_view")
        self._obstacles = RigidPrimView(prim_paths_expr="/World/envs/.*/Obstacle/obstacle", name="obstacles_view")
        self._targets._non_root_link = True
        scene.add(self._jetbots)
        scene.add(self._targets)
        # scene.add(self._obstacles)
       
        return

    def get_jetbot(self):
        jetbot = Jetbot(prim_path=self.default_zero_env_path + "/Jetbot/jetbot", name="Jetbot", translation=self._jetbot_positions)
        self._sim_config.apply_articulation_settings("Jetbot", get_prim_at_path(jetbot.prim_path), self._sim_config.parse_actor_config("Jetbot"))
        result, lidar = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path=self.default_zero_env_path + "/Jetbot/jetbot/chassis/Lidar/Lidar",
            parent=None,
            min_range=0.10,
            max_range=20.0,     
            draw_points=False,
            draw_lines=False,
            horizontal_fov=360.0,
            vertical_fov=30.0,
            horizontal_resolution=360/self.ranges_count, # 5
            vertical_resolution=4.0,
            rotation_rate=0.0,
            high_lod=False,
            yaw_offset=0.0,
            enable_semantics=False,
        )
        lidar.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, 0.0, 0.015))


    def add_target(self):
        target = DynamicCuboid(prim_path=self.default_zero_env_path + "/Target/target",
            name="target",
            position=self.target_position,
            scale=np.array([.1, .1, .1]),
            color=np.array([.125,.82,0.22]))
        
        self._sim_config.apply_articulation_settings("target", get_prim_at_path(target.prim_path),
                                                     self._sim_config.parse_actor_config("target"))
        target.set_collision_enabled(False)
        
        


    def add_obstacle(self):
        radius = 0.1
        color = torch.tensor([1, 0, 0])
        obstacle = DynamicSphere(
            prim_path=self.default_zero_env_path + "/Obstacle/obstacle",
            translation=self.obstacle_position,
            name="obstacle",
            radius=radius,
            color=color)
        self._sim_config.apply_articulation_settings("obstacle", get_prim_at_path(obstacle.prim_path),
                                                     self._sim_config.parse_actor_config("obstacle"))
        obstacle.set_collision_enabled(True)

    def get_home(self):
        current_working_dir = Path.cwd()
        asset_path = str(current_working_dir.parent) + "/assets/jetbot"

        add_reference_to_stage(
            usd_path=asset_path + "/obstacles.usd",
            prim_path= self.default_zero_env_path + "/Home/home"
        )



    # part of this could use jit
    def get_observations(self) -> dict:
        """Return lidar ranges and polar coordinates as observations to RL agent."""
        self.ranges = torch.zeros((self._num_envs, self.ranges_count)).to(self._device)


        for i in range(self._num_envs):
            np_ranges = self.lidarInterface.get_linear_depth_data(self._lidarpaths[i]).squeeze()
            self.ranges[i] = torch.tensor(np_ranges)
        
        #print(self.ranges.shape)

        self.positions, self.rotations = self._jetbots.get_world_poses()
        self.target_positions, _ = self._targets.get_world_poses()
        # print("*********************************************")
        # print(self.positions, self.rotations)
        yaws = []
        for rot in self.rotations:
            yaws.append(quat_to_euler_angles(rot)[2])
        yaws = torch.tensor(yaws).to(self._device)

        #print("position", self.position)
        #print("yaw", yaws)
        #print("target pos", self.target_pos)
        goal_angles = torch.atan2(self.target_positions[:,1] - self.positions[:,1], self.target_positions[:,0] - self.positions[:,0])

        self.headings = goal_angles - yaws
        self.headings = torch.where(self.headings > math.pi, self.headings - 2 * math.pi, self.headings)
        self.headings = torch.where(self.headings < -math.pi, self.headings + 2 * math.pi, self.headings)

        self.goal_distances = torch.linalg.norm(self.positions - self.target_positions, dim=1).to(self._device)

        to_target = self.target_positions - self.positions
        to_target[:, 2] = 0.0

        self.prev_potentials[:] = self.potentials.clone()
        self.potentials[:] = -torch.norm(to_target, p=2, dim=-1) / self.dt

        obs = torch.hstack((self.ranges, self.headings.unsqueeze(1), self.goal_distances.unsqueeze(1)))
        self.obs_buf[:] = obs
        # print(self.positions[0])

        observations = {
            self._jetbots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        """Perform actions to move the robot."""

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        actions = actions.to(self._device)

        indices = torch.arange(self._jetbots.count, dtype=torch.int32, device=self._device)
        # self._cartpoles.set_joint_efforts(forces, indices=indices)
        
        controls = torch.zeros((self._num_envs, 2))
        # print("************************************")
        # print(actions[0])
        for i in range(self._num_envs):
            controls[i] = self._diff_controller.forward([0.4*actions[i][0].item()+0.05, actions[i][1].item()])
        # print(controls[0])
        self._jetbots.set_joint_velocity_targets(controls, indices=indices)

    def reset_idx(self, env_ids):
        # """Resetting the environment at the beginning of episode."""
        # num_resets = len(env_ids)

        # self.goal_reached = torch.zeros(self._num_envs, device=self._device)
        # self.collisions = torch.zeros(self._num_envs, device=self._device)

        # # apply resets

        # root_pos = self.initial_root_pos.clone()
        # root_pos[env_ids, 0] += torch_rand_float(-1.5, 1.5, (num_resets, 1), device=self._device).view(-1)
        # root_pos[env_ids, 1] += torch_rand_float(-1.5, 1.5, (num_resets, 1), device=self._device).view(-1)
        # root_pos[env_ids, 2] = 0

        # root_vel = torch.zeros((num_resets, 6), device=self._device)
        # self._jetbots.set_world_poses(root_pos[env_ids], self.initial_root_rot[env_ids].clone(), indices=env_ids)
        # self._jetbots.set_velocities(root_vel, indices=env_ids)

        # self.reset_buf[env_ids] = 0
        # self.progress_buf[env_ids] = 0
        """Resetting the environment at the beginning of episode."""
        num_resets = len(env_ids)

        self.goal_reached = torch.zeros(self._num_envs, device=self._device)
        self.collisions = torch.zeros(self._num_envs, device=self._device)

        # apply resets
        root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
        root_vel = torch.zeros((num_resets, 6), device=self._device)
        #  + torch.tensor([-0.2, -0.3, 0.0], device=self._device)
        self._jetbots.set_world_poses(root_pos, root_rot, indices=env_ids)
        self._jetbots.set_velocities(root_vel, indices=env_ids)

        target_pos = self.initial_target_pos[env_ids] 
        
        self._targets.set_world_poses(target_pos, indices=env_ids)

        to_target = target_pos - self.initial_root_pos[env_ids]
        to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def post_reset(self):
        """This is run when first starting the simulation before first episode."""
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
        jetbot_paths = self._jetbots.prim_paths
        self._lidarpaths = [path + "/chassis/Lidar/Lidar" for path in jetbot_paths]

        # get some initial poses
        self.initial_root_pos, self.initial_root_rot = self._jetbots.get_world_poses()
        self.initial_target_pos, _ = self._targets.get_world_poses()
        #self.target_pos, _ = self._targets.get_world_poses()

        self.dt = 1.0 / 60.0
        self.potentials = torch.tensor([-1000.0 / self.dt], dtype=torch.float32, device=self._device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        # randomize all envs
        indices = torch.arange(self._jetbots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

        if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self)

    # could use jit
    def calculate_metrics(self) -> None:
        """Calculate rewards for the RL agent."""
        rewards = torch.zeros_like(self.rew_buf)

        closest_ranges, indices = torch.min(self.ranges, 1)
        self.collisions = torch.where(closest_ranges < self.collision_range, 1.0, 0.0).to(self._device)

        closer_to_goal = torch.where(self.goal_distances < self.prev_goal_distance, 1, -1)
        self.prev_goal_distance = self.goal_distances
        self.goal_reached = torch.where(self.goal_distances < 0.2, 1, 0).to(self._device)


        closer_to_heading = torch.where(torch.abs(self.headings) < torch.abs(self.prev_heading), 1, 0)
        correct_heading = torch.where(torch.abs(self.headings) < 0.2, 1, 0)
        heading_bonus = torch.where(torch.logical_or(correct_heading, closer_to_heading), 1, -1)

        self.prev_heading = self.headings

        progress_reward = self.potentials - self.prev_potentials

        episode_end = torch.where(self.progress_buf >= self._max_episode_length - 1, 1.0, 0.0)
        #print(episode_end)
        # print(self.positions[1])
        # print(self.target_positions[1])
        # print(self.goal_distances[1])
        # print(self.goal_reached[1].bool())

        rewards -= 20 * self.collisions
        rewards -= 20 * episode_end
        rewards += closer_to_goal * 0.01
        rewards += closer_to_heading * 0.01
        #rewards += heading_bonus * 0.005
        rewards += 0.1 * progress_reward
        rewards += 20 * self.goal_reached

        self.rew_buf[:] = rewards

    # could use jit
    def is_done(self) -> None:
        """Flags the environnments in which the episode should end."""
        #self.reset_buf[:] = torch.zeros(self._num_envs)
        resets = torch.where(self.progress_buf >= self._max_episode_length - 1, 1.0, self.reset_buf.double())
        resets = torch.where(self.collisions.bool(), 1.0, resets.double())
        resets = torch.where(self.goal_reached.bool(), 1.0, resets.double())
        self.reset_buf[:] = resets
