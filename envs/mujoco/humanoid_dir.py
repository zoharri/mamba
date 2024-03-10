from typing import Dict

import cv2
import numpy as np
from gym.envs.mujoco import HumanoidEnv as HumanoidEnv
from gym import spaces

import random


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))


class HumanoidDirEnv(HumanoidEnv):

    def __init__(self):
        self.goal_direction = self.sample_task()["direction"]
        self.task_dim = 1
        self.action_scale = 1  # Mujoco environment initialization takes a step,

        super(HumanoidDirEnv, self).__init__()

        # Override action space to make it range from  (-1, 1)
        assert (self.action_space.low == -self.action_space.high).all()
        self.action_scale = self.action_space.high[0]
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=self.action_space.shape)  # Overriding original action_space which is (-0.4, 0.4, shape = (17, ))

    def step(self, action):
        pos_before = np.copy(mass_center(self.model, self.sim)[:2])

        rescaled_action = action * self.action_scale  # Scale the action from (-1, 1) to original.
        self.do_simulation(rescaled_action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[:2]

        alive_bonus = 5.0
        data = self.sim.data
        goal_direction = (np.cos(self.goal_direction), np.sin(self.goal_direction))
        lin_vel_cost = 0.25 * np.sum(goal_direction * (pos_after - pos_before)) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos

        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        # done = False

        return (
            {"state": self._get_obs(), "is_terminal": np.array(done), "is_first": np.array(False)},
            reward,
            done,
            {},
        )

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    # def get_all_task_idx(self):
    #     return range(len(self.tasks))

    # def reset_task(self, idx):
    #     self._task = self.tasks[idx]
    #     self._goal = self._task['goal'] # assume parameterization of task by single vector

    def sample_task(self):
        return {"direction": self.sample_tasks(1)[0]}

    def set_task(self, task):
        self.goal_direction = task["direction"]

    def get_task(self):
        return {"direction": np.array([self.goal_direction])}

    def reset_model(self):
        super().reset_model()
        ob = self._get_obs()
        return {"state": ob, "is_terminal": False, "is_first": True}

    def reset(self):
        return super().reset()

    @staticmethod
    def state2image(state: np.array, episode_idx: int, task: Dict):
        goal_direction = task["direction"]
        direction_vec = (np.cos(goal_direction), np.sin(goal_direction))
        map_size = 420
        middle = map_size // 2
        radius = 50
        map_image = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255
        # Draw arrow for goal direction
        map_image = cv2.arrowedLine(map_image, (middle, middle),
                                    (middle + int(direction_vec[0] * radius), int(direction_vec[0] * radius)),
                                    (0, 255, 0), 1, cv2.LINE_AA, 0, 0.3)
        map_image = cv2.circle(map_image, (
            int(state["state"][0] * radius + middle), int(state["state"][1] * radius + middle)), 3,
                               (0, 0, 255), -1)

        cv2.putText(map_image, f"{episode_idx}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return map_image

    def sample_tasks(self, num_tasks):
        return [random.uniform(0., 2.0 * np.pi) for _ in range(num_tasks)]
