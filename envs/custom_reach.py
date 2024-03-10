from panda_gym.envs.core import Task, RobotTaskEnv
from panda_gym.utils import distance
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
import random
import gym
import cv2
import gym.spaces
import gym.utils.seeding
import numpy as np
import torch
import pybullet
import pybullet_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomReach(Task):
    def __init__(
            self,
            sim,
            get_ee_position,
            reward_type="sparse",
            distance_threshold=0.05,
            goal_range=0.3,
            table_color=None,
            plane_color=None
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal = np.zeros(3)
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, goal_range / 2])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range / 2])
        self.table_color = table_color
        self.plane_color = plane_color
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def get_obs(self) -> np.ndarray:
        return np.array([])

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4, rgba=self.plane_color)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3, rgba=self.table_color)
        self.sim.create_sphere(
            body_name="target",
            radius=0.05,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=None)  # np.array([0.1, 0.9, 0.1, 0.3]),

    #  )

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def set_task(self, task):
        self.goal = task

    def get_task(self):
        return self.goal

    def sample_task(self):
        r = 0.15
        theta = random.random() * (np.pi) - (np.pi / 2)
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        z = 0.15 / 2
        return np.array([x, y, z])

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray):
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            r = float(d < self.distance_threshold)
        else:
            r = -d
        return r


class CustomReachEnv(RobotTaskEnv):
    """Robotic task goal env, as the junction of a task and a robot.
    Args:
        robot (PyBulletRobot): The robot.
        task (task): The task.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee",
                 table_color=None, plane_color=None) -> None:
        self.sim = PyBullet(render=render)

        robot = Panda(self.sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = CustomReach(self.sim, reward_type=reward_type, get_ee_position=robot.get_ee_position,
                           table_color=table_color, plane_color=plane_color)
        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.robot = robot
        self.task = task
        self.task_dim = 3
        obs = self.reset()  # required for init; seed can be changed later
        observation_shape = obs['image'].shape
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, shape=observation_shape, dtype=np.uint8)
            })
        self.action_space = self.robot.action_space
        self.compute_reward = self.task.compute_reward
        self._saved_goal = dict()

    def _get_obs(self):
        observation = self.render('rgb_array', 64, 64)
        if self.task.is_success(self.task.get_achieved_goal(), self.task.get_goal()):
            observation[-20:, :, :] = 1

        # observation = np.transpose(observation, axes=[2, 0, 1])
        return observation

    def reset_model(self):
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        return {"image": self._get_obs(), "is_terminal": False, "is_first": True}

    def reset(self):
        return self.reset_model()

    def sample_task(self):
        return self.task.sample_task()

    def get_task(self):
        return self.task.get_task()

    def set_task(self, task):
        self.task.set_task(task)

    def save_state(self) -> int:
        state_id = self.sim.save_state()
        self._saved_goal[state_id] = self.task.goal
        return state_id

    def restore_state(self, state_id: int) -> None:
        self.sim.restore_state(state_id)
        self.task.goal = self._saved_goal[state_id]

    def remove_state(self, state_id: int) -> None:
        self._saved_goal.pop(state_id)
        self.sim.remove_state(state_id)

    def step(self, action: np.ndarray):
        act = action
        self.robot.set_action(act)
        self.sim.step()
        obs = self._get_obs()
        done = False
        info = {"is_success": self.task.is_success(self.task.get_achieved_goal(), self.task.get_goal())}
        reward = self.task.compute_reward(self.task.get_achieved_goal(), self.task.get_goal(), info)# - 0.1*np.linalg.norm(action)
        assert isinstance(reward, float)  # needed for pytype checking

        return (
            {"image": obs, "is_terminal": done, "is_first": False},
            reward,
            done,
            {},
        )

    def close(self) -> None:
        self.sim.close()

    def render(
            self,
            mode: str = 'rgb_array',
            width: int = 720,
            height: int = 480,
            target_position=None,
            distance: float = 1.4,
            yaw: float = 45,
            pitch: float = -30,
            roll: float = 0,
    ):
        """Render.
        If mode is "human", make the rendering real-time. All other arguments are
        unused. If mode is "rgb_array", return an RGB array of the scene.
        Args:
            mode (str): "human" of "rgb_array". If "human", this method waits for the time necessary to have
                a realistic temporal rendering and all other args are ignored. Else, return an RGB array.
            width (int, optional): Image width. Defaults to 720.
            height (int, optional): Image height. Defaults to 480.
            target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
                Defaults to [0., 0., 0.].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (int, optional): Rool of the camera. Defaults to 0.
        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        target_position = target_position if target_position is not None else np.zeros(3)
        return self.sim.render(
            mode,
            width=width,
            height=height,
            target_position=target_position,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )
