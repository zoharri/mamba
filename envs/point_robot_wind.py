import random

import numpy as np

from envs.point_robot import SparsePointEnv


class SparsePointWindEnv(SparsePointEnv):
    """ Reward is L2 distance given only within goal radius """

    def __init__(self, goal_radius=2, goal_sampler='semi-circle', wind_force=0.1):
        super().__init__(goal_radius, goal_sampler)
        self._wind_force = wind_force

    def step(self, action):
        wind_vec = np.array([random.gauss(0, self._wind_force) for _ in range(self.action_space.shape[0])])
        return super().step(action + wind_vec)
