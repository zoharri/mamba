import io
import random
from typing import Dict

import cv2
import gym
import numpy as np
from PIL import Image
from gym import Env
from gym import spaces
from matplotlib import pyplot as plt
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points


class PointEnvBarrier(gym.Env):
    """
    2D point robot must navigate outside a ball, with door at different locations
    over a semi-circle
     - tasks sampled from unit half-circle
     - reward is 1 outside the ball and 0 inside
    """

    def __init__(self,
                 n_tasks=2,
                 modify_init_state_dist=True,
                 **kwargs):

        self.step_count = 0
        self.modify_init_state_dist = modify_init_state_dist

        # np.random.seed(1337)
        self.radius = 1.0   # radius of barrier
        self.door_width_angle = np.pi / 8   # 8 full doors with no "intersection" to cover semi-circle
        self.door_width = np.sqrt(1+1-2*np.cos(self.door_width_angle/2))

        # construct sphere barrier
        self._barrier = Point(0, 0).buffer(1.0).difference(Point(0, 0).buffer(0.999))

        self.reset_task()
        self.task_dim = 1
        self.observation_space = gym.spaces.Dict(
            {
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            })
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

    def sample_task(self):
        angle = np.random.uniform(0, np.pi)
        return {"goal_angle": np.array([angle])}

    def set_task(self, task):
        self._angle = task["goal_angle"]

    def get_task(self):
        return {"goal_angle": self._angle}

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        return task

    def reset_model(self):
        self.step_count = 0
        if self.modify_init_state_dist:     # at random inside upper half-ball
            # self._state = np.array([np.random.uniform(-1.5, 1.5), np.random.uniform(-0.5, 1.5)])
            random_angle = np.random.uniform(-np.pi, np.pi)
            self._state = np.random.uniform(0, self.radius) * np.array([np.sin(random_angle), np.cos(random_angle)])
        else:
            self._state = np.array([0, 0])
        return {"state": self._get_obs(), "is_terminal": False, "is_first": True}

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action), action

        lookahead_state = self._state + 0.1 * action

        # check intersection of path with circle
        path = LineString([tuple(self._state), tuple(lookahead_state)])
        intersection = path.intersection(self._barrier)
        if not intersection.is_empty:
            # find nearest intersection point
            nearest_intersection = nearest_points(intersection, Point(self._state))[0]
            intersection_point = np.array([nearest_intersection.x, nearest_intersection.y])
            intersection_angle = np.arctan2(intersection_point[1], intersection_point[0])
            if (self._angle - (self.door_width_angle / 2)) <= intersection_angle \
                    <= (self._angle + (self.door_width_angle / 2)):
                self._state = lookahead_state
            else:
                self._state = (lookahead_state / np.linalg.norm(lookahead_state)) * 0.99
                # self._state = self._state + 0.999 * (intersection_point - self._state)
        else:
            self._state = lookahead_state

        reward = self.reward(self._state)

        done = False
        return (
            {"state": self._get_obs(), "is_terminal": done, "is_first": False},
            reward,
            done,
            {},
        )

    def reward(self, state):
        return (np.linalg.norm(state) > self.radius).astype(np.float32)

    def is_goal_state(self):
        return np.linalg.norm(self._state) > self.radius

    def draw_map(self, state, episode_idx, goal_pos):
        ax = plt.gca()
        x, y = goal_pos[0], goal_pos[1]
        plt.axis('scaled')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 2)
        plt.xticks([])
        plt.yticks([])
        circle = plt.Circle((x, y), radius=self.door_width, alpha=0.3)
        circle1 = plt.Circle((0, 0), 1, edgecolor='black', facecolor='w')
        ax.add_artist(circle1)
        ax.add_artist(circle)
        circle2 = plt.Circle((state[0], state[1]), 0.05, edgecolor='red', facecolor='red')
        ax.add_artist(circle2)
        plt.title("Episode {}".format(episode_idx))
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        map_image = Image.open(buf)
        plt.close()
        return np.array(map_image)

    def state2image(self, state: np.array, episode_idx: int, task: Dict):
        if isinstance(task, dict):
            goal_angle = task["goal_angle"]
            goal_pos = np.array([np.cos(goal_angle), np.sin(goal_angle)])
        else:
            goal_pos = []
        return self.draw_map(state["state"], episode_idx, goal_pos)
