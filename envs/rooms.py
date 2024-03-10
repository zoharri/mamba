import itertools
import random

import gym
import numpy as np
import torch
from gym import spaces


class RoomNaviNew(gym.Env):
    def __init__(self, num_cells=3, corridor_len=3, num_rooms=3):
        super(RoomNaviNew, self).__init__()

        self.seed()
        self.num_cells = num_cells  # should be even!
        self.corridor_len = corridor_len
        self.num_rooms = num_rooms
        self.num_states = num_rooms * (self.num_cells ** 2) + 2 * self.corridor_len

        self.width = self.num_rooms * self.num_cells + (
                self.num_rooms - 1) * self.corridor_len
        self.height = self.num_cells

        self.step_count = 0
        self.room_length = self.num_cells + self.corridor_len  # length between the middle of one room to the next one
        # self.observation_space = spaces.Box(low=0, high=self.width - 1, shape=(2,))
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(low=0, high=self.width - 1, shape=(2,)),
            })
        self._action_space = spaces.Discrete(5)  # noop, up, right, down, left
        self.task_dim = 2 * self.num_rooms  # location of goal 1, goal 2, goal 3
        self.num_tasks = 4 ** self.num_rooms  # loc of goal 1 (4 options) x loc of goal 2 (4 options) x loc of goal 3 (4 options)

        # goals can be anywhere except on possible starting states and immediately around it
        self.offset = (self.num_cells - 1) // 2
        self.rooms_centers_x = np.arange(0, self.num_rooms) * self.room_length + self.offset
        self.possible_goals = [[[room_center_x - self.offset, self.offset],
                                [room_center_x - self.offset, -self.offset],
                                [room_center_x + self.offset, self.offset],
                                [room_center_x + self.offset, -self.offset]] for room_center_x in self.rooms_centers_x]

        self.possible_states = []
        for i, room_center_x in enumerate(self.rooms_centers_x):
            # add current room states
            self.possible_states += itertools.product(range(room_center_x - self.offset,
                                                            room_center_x + self.offset + 1),
                                                      range(-self.offset, self.offset + 1))
            if i == self.num_rooms - 1:
                # last room doesn't have a corridor
                break
            # add corridor states
            self.possible_states += itertools.product(range(room_center_x + self.offset + self.corridor_len // 2,
                                                            room_center_x + self.offset + self.corridor_len + 1),
                                                      [0])

        self.index_matrix = torch.zeros((self.height, self.width)).long()
        for i, (x, y) in enumerate(self.possible_states):
            self.index_matrix[y + self.height // 2,
            x] = i

        self.starting_state = [0, 0]
        # reset the environment state
        self._env_state = np.array(self.starting_state)
        # reset the goal
        self._goals = self.reset_model()

    def normalize_obs(self, obs):
        obs = (obs + self.num_cells // 2 + 1) * 2 / np.array([self.width, self.num_cells]) - 1
        return obs

    def unnormalize_obs(self, obs):
        obs = ((obs + 1) * np.array([self.width, self.num_cells]) / 2) - 1 - self.num_cells // 2
        return np.round(obs).astype(int)

    @property
    def action_space(self):
        space = self._action_space
        space.discrete = True
        return space

    def sample_task(self):
        return np.array([random.choice(curr_possible_goals) for curr_possible_goals in self.possible_goals]).flatten()

    def set_task(self, task=None):
        self.goals = task
        self.reached_goals = [False] * self.num_rooms

    def get_task(self):
        return self.goals.copy()

    def reset_model(self):
        self.step_count = 0
        self.reached_goals = [False] * self.num_rooms
        self._env_state = np.array(self.starting_state)

        return {"state": self.normalize_obs(self._env_state.copy()), "is_terminal": False, "is_first": True}

    def reset(self):
        return self.reset_model()

    def state_transition(self, action):
        """
        Moving the agent between states
        """

        # -- CASE: INSIDE CORRIDOR --

        inside_cooridor = self.room_length - self.corridor_len <= self._env_state[0] % self.room_length
        inside_x = 0 <= self._env_state[0] < self.width
        inside_y = -self.num_cells + self.offset < self._env_state[1] < self.num_cells - self.offset
        if inside_cooridor and inside_x and inside_y:
            if action in [1, 3]:  # up or down
                # cannot walk into walls
                return self.normalize_obs(self._env_state)
            else:
                # left and right is always possible
                if action == 2:  # right
                    self._env_state[0] = self._env_state[0] + 1
                elif action == 4:  # left
                    self._env_state[0] = self._env_state[0] - 1
                return self.normalize_obs(self._env_state)

        # -- CASE: INSIDE ROOM --

        # execute action to see where we'd end up
        new_env_state = self._env_state.copy()
        if action == 1:  # up
            new_env_state[1] = new_env_state[1] + 1
        elif action == 2:  # right
            new_env_state[0] = new_env_state[0] + 1
        elif action == 3:  # down
            new_env_state[1] = new_env_state[1] - 1
        elif action == 4:  # left
            new_env_state[0] = new_env_state[0] - 1

        walked_into_corridor = self.room_length - self.corridor_len <= new_env_state[0] % self.room_length and \
                               new_env_state[0] < self.width

        # check if this is a valid spot
        inside_x = 0 <= new_env_state[0] < self.width
        inside_y = -self.num_cells + self.offset < new_env_state[1] < self.num_cells - self.offset
        if walked_into_corridor and inside_x and inside_y:
            if new_env_state[1] == 0:
                self._env_state = new_env_state
        else:
            if inside_x and inside_y:
                self._env_state = new_env_state

        return self.normalize_obs(self._env_state)

    def step(self, action):

        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action[0]
        # assert self.action_space.contains(action)

        done = False

        # perform state transition
        state = self.state_transition(action)

        # check if maximum step limit is reached
        self.step_count += 1

        # compute reward
        reward = -0.1
        for i in range(self.num_rooms):
            reached_prev_goal = True if i == 0 else self.reached_goals[i - 1]
            curr_goal = self.goals.reshape(-1, 2)[i]
            # Don't turn off the last goal
            if self._env_state[0] == curr_goal[0] and self._env_state[1] == curr_goal[1] and (
                    not self.reached_goals[i] or i == self.num_rooms - 1) and reached_prev_goal:
                self.reached_goals[i] = True
                reward = 1.0

        task = self.get_task()
        info = {'task': task}

        # return state, reward, done, info
        return (
            {"state": state, "is_terminal": done, "is_first": False},
            reward,
            done,
            {},
        )

    def obs_to_state_idx(self, cell):
        if isinstance(cell, list) or isinstance(cell, tuple):
            cell = np.array(cell)
        if isinstance(cell, np.ndarray):
            cell = torch.from_numpy(cell)
        cell = cell.long()
        cell_shape = cell.shape
        if len(cell_shape) > 2:
            cell = cell.reshape(-1, cell.shape[-1])
        indices = self.index_matrix[cell[:, 1], cell[:, 0]]
        indices = indices.reshape(cell_shape[:-1])
        return indices

    def state_idx_to_obs(self, idx):
        return self.possible_states[idx]

    def task_to_id(self, goals):
        # Does this work?
        combinations = list(itertools.product(self.possible_goals))
        combinations = np.array(combinations).reshape(-1, 4)
        mask = np.sum(np.abs(combinations - goals), axis=-1) == 0

        classes = torch.arange(0, len(mask))[mask].item()

        return classes

    def state2image(self, state, episode_idx, task):
        goals = task.reshape(-1, 2)
        img = np.zeros((self.height + 1, self.width, 3))
        # draw corridor
        for x in range(self.width):
            for y in range(-self.offset, self.offset + 1):
                # check if this is a valid spot
                # - corridors
                inside_corridor = self.room_length - self.corridor_len <= x % self.room_length
                if inside_corridor:
                    if y != 0:
                        facecolor = 'white'
                    else:
                        facecolor = 'black'
                else:
                    facecolor = 'black'
                if facecolor == 'white':
                    img[y + self.offset, x] = [255, 255, 255]
                else:
                    img[y + self.offset, x] = [0, 0, 0]
        # draw agent
        pixels_state = self.unnormalize_obs(state["state"])
        img[pixels_state[1] + self.offset, pixels_state[0]] = [255, 0, 0]
        for i in range(0, len(goals)):
            curr_color = [0, 0, 255]
            img[goals[i, 1] + self.offset, goals[i, 0]] = curr_color

        # draw episode number
        img[self.height, :episode_idx] = [0, 255, 0]
        img[self.height, episode_idx:] = [255, 255, 255]
        return img

