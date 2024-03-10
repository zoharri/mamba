import collections
from typing import Dict, Union

import cv2
import numpy as np
from abc import ABC, abstractmethod
from dm_control.composer.variation import distributions
from dm_control.rl import control
from dm_control.suite import quadruped, reacher, base
from dm_control import suite  # leave this import, it is needed in dmc.py
from dm_control.suite.utils import randomizers


class HyperXGoalDist(distributions.Distribution):
    __slots__ = ()

    def __init__(self, single_sample=False, theta=1, radius=1):
        super().__init__(single_sample=single_sample)
        self._theta = theta
        self._radius = radius

    def _callable(self, random_state):
        def ret_call(*args, **kwargs):
            a = random_state.random() * 2 * np.pi * self._theta
            r = 3 * self._radius * random_state.random() ** 0.5
            return np.array([r * np.cos(a), r * np.sin(a)])

        return ret_call


class DMCMetaEnv(control.Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_model(self):
        self._task.reset_model(self.physics)

    def get_loc(self):
        self._task.get_loc(self.physics)

    def get_task(self):
        return self._task.get_task()

    def set_task(self, task):
        self._task.set_task(task)

    def sample_task(self):
        return self._task.sample_task()

    def get_observation(self, physics):
        return self._task.get_observation(physics)

    def state2image(self, state: np.array, episode_idx: int, task: Dict):
        return self._task.state2image(state, episode_idx, task)


@quadruped.SUITE.add()
def goalmeta(random=None, environment_kwargs=None):
    """Returns the metagoal task."""
    xml_string = quadruped.make_model(floor_size=quadruped._DEFAULT_TIME_LIMIT * quadruped._WALK_SPEED)
    physics = quadruped.Physics.from_xml_string(xml_string, quadruped.common.ASSETS)
    task = QuadrupedGoalMeta(is_meta=True, goal_sampler=HyperXGoalDist(theta=environment_kwargs.pop("theta"),
                                                                       radius=environment_kwargs.pop("radius")),
                             goal_radius=0.75, random=random, dense_reward=environment_kwargs.pop("dense_reward"),
                             num_goals=environment_kwargs.pop("num_goals"))
    environment_kwargs = environment_kwargs or {}
    return DMCMetaEnv(physics, task,
                      control_timestep=quadruped._CONTROL_TIMESTEP,
                      **environment_kwargs)


@quadruped.SUITE.add()
def goal(random=None, environment_kwargs=None):
    """Returns the goal task."""
    xml_string = quadruped.make_model(floor_size=quadruped._DEFAULT_TIME_LIMIT * quadruped._WALK_SPEED)
    physics = quadruped.Physics.from_xml_string(xml_string, quadruped.common.ASSETS)
    task = QuadrupedGoalMeta(is_meta=False, goal_sampler=HyperXGoalDist(theta=environment_kwargs.pop("theta"),
                                                                        radius=environment_kwargs.pop("radius")),
                             goal_radius=0.75, random=random, dense_reward=environment_kwargs.pop("dense_reward"),
                             num_goals=environment_kwargs.pop("num_goals"))
    environment_kwargs = environment_kwargs or {}
    return DMCMetaEnv(physics, task,
                      control_timestep=quadruped._CONTROL_TIMESTEP,
                      **environment_kwargs)


@reacher.SUITE.add()
def goalmeta(random=None, environment_kwargs=None):
    """Returns the goal task."""
    physics = reacher.Physics.from_xml_string(*reacher.get_model_and_assets())
    task = ReacherGoalMeta(is_meta=True, goal_sampler=HyperXGoalDist(theta=environment_kwargs.pop("theta"),
                                                                     radius=0.065 * environment_kwargs.pop("radius")),
                           goal_radius=0.05, random=random, dense_reward=environment_kwargs.pop("dense_reward"),
                           num_goals=environment_kwargs.pop("num_goals"))
    environment_kwargs = environment_kwargs or {}
    return DMCMetaEnv(physics, task, **environment_kwargs)


@reacher.SUITE.add()
def goal(random=None, environment_kwargs=None):
    """Returns the goal task."""
    physics = reacher.Physics.from_xml_string(*reacher.get_model_and_assets())
    task = ReacherGoalMeta(is_meta=False, goal_sampler=HyperXGoalDist(theta=environment_kwargs.pop("theta"),
                                                                      radius=0.065 * environment_kwargs.pop("radius")),
                           goal_radius=0.05, random=random, dense_reward=environment_kwargs.pop("dense_reward"),
                           num_goals=environment_kwargs.pop("num_goals"))
    environment_kwargs = environment_kwargs or {}
    return DMCMetaEnv(physics, task, **environment_kwargs)


class AbstractGoalMeta(ABC):
    def __init__(self, is_meta, goal_sampler, random=None, goal_radius: float = -1, dense_reward: bool = False,
                 num_goals: int = 1):
        self.goal_sampler = goal_sampler
        self.goal_radius = goal_radius
        self._is_meta = is_meta
        self._dense_reward = dense_reward
        self._num_goals = num_goals

        self._all_goal_poses = self._sample_goal()

        self.goal_pos = self._all_goal_poses[0]
        self._curr_goal_idx = 0

        super().__init__(random=random)

    def _sample_goal(self):
        if self._is_meta:
            return [self.goal_sampler() for _ in range(self._num_goals)]
        else:
            # goals are fixed between -1 and 1
            return [[2 * i / self._num_goals - 1, 0] for i in range(self._num_goals)]

    def initialize_episode(self, physics):
        self.reset_model(physics)

    def get_task(self):
        return {"goal_poses": self._all_goal_poses, "goal_radius": self.goal_radius}

    def set_task(self, task: Dict):
        self._all_goal_poses = task["goal_poses"]
        self.goal_radius = task["goal_radius"]

    def sample_task(self):
        return {"goal_poses": self._sample_goal(), "goal_radius": self.goal_radius}

    @abstractmethod
    def reset_model(self, physics):
        self.goal_pos = self._all_goal_poses[0]
        self._curr_goal_idx = 0

    @abstractmethod
    def get_observation(self, physics):
        raise NotImplementedError

    def get_reward(self, physics):
        """Returns a reward to the agent."""

        dist = self._self_to_goal_distance(physics)
        if self._dense_reward is True:
            goal_reward = -dist
        else:
            goal_reward = np.maximum(1.0 - dist / self.goal_radius, 0.0).item()

        if dist < self.goal_radius and self._curr_goal_idx < self._num_goals - 1:
            self._curr_goal_idx += 1
            self.goal_pos = self._all_goal_poses[self._curr_goal_idx]

        reward = goal_reward + self.get_extra_reward(physics)
        return reward

    def get_extra_reward(self, physics):
        return 0

    def _self_to_goal_distance(self, physics):
        """Returns horizontal distance from the quadruped torso to the goal."""
        self_to_goal = (self.get_loc(physics)[:2]
                        - self.goal_pos)
        return np.linalg.norm(self_to_goal)

    def get_loc(self, physics):
        raise NotImplementedError

    @staticmethod
    def state2image(state: np.array, episode_idx: int, task: Union[Dict, int]):
        state = state["state"]
        goal_poses = task["goal_poses"]
        goal_radius = task["goal_radius"]
        map_size = 420
        middle = map_size // 2
        radius = 50 * 0.75 / goal_radius
        map_image = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255
        for i, goal_pos in enumerate(goal_poses):
            map_image = cv2.circle(map_image.copy(),
                                   (int(goal_pos[0] * radius + middle), int(goal_pos[1] * radius + middle)),
                                   int(goal_radius * radius), (0, 255, 0), -1)
            map_image = cv2.putText(map_image.copy(), f"{i}", (int(goal_pos[0] * radius + middle), int(goal_pos[1] * radius + middle)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 0),
                                    1)
        map_image = cv2.circle(map_image.copy(), (int(state[0] * radius + middle), int(state[1] * radius + middle)), 3,
                               (0, 0, 255), -1)

        map_image = cv2.putText(map_image.copy(), f"{episode_idx}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                                1)
        return map_image


class QuadrupedGoalMeta(AbstractGoalMeta, base.Task):

    def __init__(self, is_meta, goal_sampler, random=None, goal_radius: float = -1, dense_reward: bool = False,
                 num_goals: int = 1):
        AbstractGoalMeta.__init__(self, is_meta=is_meta, goal_sampler=goal_sampler, random=random,
                                  goal_radius=goal_radius, dense_reward=dense_reward, num_goals=num_goals)
        quadruped.base.Task.__init__(self, random=random)

    def reset_model(self, physics):
        super().reset_model(physics)
        orientation = np.zeros(4)
        quadruped._find_non_contacting_height(physics, orientation)
        base.Task.initialize_episode(self, physics)

    def get_observation(self, physics):
        """Returns an observation to the agent."""
        common_obs = quadruped._common_observations(physics)
        loc = self.get_loc(physics)
        common_obs.update({"state": loc[:2], "state_z": loc[2:]})
        return common_obs

    def get_extra_reward(self, physics):
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(physics.named.data.cfrc_ext, -1, 1)))
        ctrl_cost = .01 * np.square(physics.data.ctrl).sum()
        survive_reward = 0
        return - ctrl_cost - contact_cost + survive_reward

    def get_loc(self, physics):
        return physics.named.data.xpos['torso']


class ReacherGoalMeta(AbstractGoalMeta, base.Task):
    def __init__(self, is_meta, goal_sampler, random=None, goal_radius: float = -1, dense_reward: bool = False,
                 num_goals: int = 1):
        AbstractGoalMeta.__init__(self, is_meta=is_meta, goal_sampler=goal_sampler, random=random,
                                  goal_radius=goal_radius, dense_reward=dense_reward, num_goals=num_goals)
        base.Task.__init__(self, random=random)

    def reset_model(self, physics):
        super().reset_model(physics)
        physics.named.model.geom_size['target', 0] = self.goal_radius
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        base.Task.initialize_episode(self, physics)

    def get_observation(self, physics):
        """Returns an observation to the agent."""
        obs = collections.OrderedDict()
        # obs['position'] = physics.position()
        # obs['to_target'] = physics.finger_to_target()
        obs['velocity'] = physics.velocity()
        obs.update({"state": self.get_loc(physics)})
        return obs

    def get_extra_reward(self, physics):
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(physics.named.data.cfrc_ext, -1, 1)))
        ctrl_cost = .01 * np.square(physics.data.ctrl).sum()
        survive_reward = 0
        return - ctrl_cost - contact_cost + survive_reward

    def get_loc(self, physics):
        return physics.named.data.geom_xpos['finger', :2]
