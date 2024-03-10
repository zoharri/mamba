import datetime
import gym
import numpy as np
import uuid


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()

    def action_space(self):
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.discrete = True
        return space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    @property
    def observation_space(self):
        spaces = self.env.observation_space.spaces
        if "reward" not in spaces:
            spaces["reward"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        return gym.spaces.Dict(spaces)

    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = value

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "reward" not in obs:
            obs["reward"] = reward
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if "reward" not in obs:
            obs["reward"] = 0.0
        return obs


class SelectAction(gym.Wrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])


class TimeAugmentedState(gym.Wrapper):
    def __init__(self, env, max_time_steps=-1):
        super().__init__(env)
        self._time = 0
        self._max_time_steps = max_time_steps

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._time += 1
        obs["time_step"] = self._time / self._max_time_steps if self._max_time_steps > 0 else self._time
        return obs, reward, done, info

    @property
    def observation_space(self):
        spaces = self.env.observation_space.spaces
        if "time_Step" not in spaces:
            spaces["time_step"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        return gym.spaces.Dict(spaces)

    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = value

    def reset(self):
        obs = self.env.reset()
        self._time = 0
        obs["time_step"] = self._time
        return obs


class MetaLearningEnv(gym.Wrapper):
    def __init__(self, env, num_meta_episodes, max_episode_length):
        super().__init__(env)
        mandatory_functions = ["get_task", "set_task", "sample_task", "reset_model"]
        for function in mandatory_functions:
            if not hasattr(env, function):
                raise ValueError(f"Cannot use meta learning mode, the environment does not have a {function} function.")

        self._num_meta_episodes = num_meta_episodes
        self._max_episode_length = max_episode_length
        self._current_episode = 0
        self._current_step = 0

    def get_task(self):
        return self.env.get_task()

    def _set_task(self, task):
        task = task if task is not None else self.sample_task()
        self.env.set_task(task)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._current_step += 1
        done = done or self._current_step >= self._max_episode_length
        info["meta_episode_reset"] = done
        if done:
            self.reset_model()
            if self._current_episode < self._num_meta_episodes:
                done = False

        return obs, reward, done, info

    def reset(self, task=None):
        self._current_step = 0
        self._current_episode = 0
        self._set_task(task)
        return self.env.reset()

    def sample_task(self):
        return self.env.sample_task()

    def reset_model(self):
        self._current_step = 0
        self._current_episode += 1
        self.env.reset_model()

class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()
