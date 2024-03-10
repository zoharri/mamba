import random
from typing import Dict

import cv2
import gym
import numpy as np
from gym import Env
from gym import spaces


def semi_circle_goal_sampler():
    r = 1.0
    angle = random.uniform(0, np.pi)
    goal = r * np.array((np.cos(angle), np.sin(angle)))
    return goal


def circle_goal_sampler():
    r = 1.0
    angle = random.uniform(0, 2 * np.pi)
    goal = r * np.array((np.cos(angle), np.sin(angle)))
    return goal


def semi_circle_edges_goal_sampler():
    r = 1.0
    sec_size = np.pi / 9
    section = random.randint(0, 2)
    if section == 0:
        angle = random.uniform(0, sec_size)
    elif section == 1:
        angle = random.uniform(sec_size, np.pi - sec_size)
    else:
        angle = random.uniform(np.pi - sec_size, np.pi)

    goal = r * np.array((np.cos(angle), np.sin(angle)))
    return goal


def goal_to_angle(goal):
    return np.rad2deg(np.arctan2(goal[1], goal[0]))


GOAL_SAMPLERS = {
    'semi-circle-edge': semi_circle_edges_goal_sampler,
    'semi-circle': semi_circle_goal_sampler,
    'circle': circle_goal_sampler,
}


class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, goal_sampler=None, num_episodes=1):
        if callable(goal_sampler):
            self.goal_sampler = goal_sampler
        elif isinstance(goal_sampler, str):
            self.goal_sampler = GOAL_SAMPLERS[goal_sampler]
        elif goal_sampler is None:
            self.goal_sampler = semi_circle_goal_sampler
        else:
            raise NotImplementedError(goal_sampler)

        self.num_episodes = num_episodes
        self.task_dim = 2
        self.observation_space = gym.spaces.Dict(
            {
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            })
        # we convert the actions from [-1, 1] to [-0.1, 0.1] in the step() function
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self._is_half = goal_sampler == 'semi-circle-edge' or goal_sampler == 'semi-circle'

    def sample_task(self):
        goal = self.goal_sampler()
        return {"goal_pos": goal}

    def set_task(self, task):
        self._goal = task["goal_pos"]

    def get_task(self):
        return {"goal_pos": self._goal}

    def reset_model(self):
        self._state = np.zeros(2)
        return {"state": self._get_obs(), "is_terminal": False, "is_first": True}

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action), action
        done = False
        self._state = self._state + 0.1 * action
        reward = - np.linalg.norm(self._state - self._goal, ord=2)

        return (
            {"state": self._get_obs(), "is_terminal": done, "is_first": False},
            reward,
            done,
            {},
        )

    @staticmethod
    def draw_map(state: np.array, curr_episode: int, goal: np.array, goal_radius: float, is_half: bool):
        map_size = 120
        middle = map_size // 2
        radius = 50
        map_image = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255

        if not is_half:
            map_image = cv2.circle(map_image, (middle, middle), radius, (0, 0, 0), 1)
        else:
            map_image = cv2.ellipse(map_image, (middle, middle), (radius, radius), 0, 0, 180, (0, 0, 0), 1)

        map_image = cv2.circle(map_image, (int(goal[0] * radius + middle), int(goal[1] * radius + middle)),
                               int(goal_radius * radius), (0, 255, 0), -1)
        map_image = cv2.circle(map_image, (int(state[0] * radius + middle), int(state[1] * radius + middle)), 3,
                               (0, 0, 255), -1)
        map_image = map_image[:middle * 4 // 5:-1, :, :].copy()

        cv2.putText(map_image, f"{curr_episode}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return map_image


class SparsePointEnv(PointEnv):
    """ Reward is L2 distance given only within goal radius """

    def __init__(self, goal_radius=2, goal_sampler='semi-circle'):
        super().__init__(goal_sampler=goal_sampler)
        self._goal_radius = goal_radius / 10

    def sample_task(self):
        goal = self.goal_sampler()
        return {"goal_pos": goal, "goal_radius": self._goal_radius}

    def set_task(self, task):
        self._goal = task["goal_pos"]
        self._goal_radius = task["goal_radius"]

    def get_task(self):
        return {"goal_pos": self._goal, "goal_radius": self._goal_radius}

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self._goal_radius).astype(np.float32)
        r = r * mask
        return r

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self._goal_radius:
            sparse_reward += 1
        return (
            {"state": self._get_obs(), "is_terminal": done, "is_first": False},
            sparse_reward,
            done,
            {})

    def state2image(self, state: np.array, episode_idx: int, task: Dict):
        if isinstance(task, dict):
            goal_pos = task['goal_pos']
            goal_radius = task['goal_radius']
        else:
            goal_pos = []
        return PointEnv.draw_map(state["state"], episode_idx, goal_pos, goal_radius, self._is_half)
