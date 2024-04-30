import gym
import numpy as np
from envs.dmc_meta import suite


class DeepMindControl:
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, detach_image_from_obs=False, small_state_space=False, environment_kwargs={}):
        domain, task = name.split("_", 1)
        self._detach_image_from_obs = detach_image_from_obs
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if isinstance(domain, str):
            self._env = suite.load(domain, task, environment_kwargs=environment_kwargs)
        else:
            assert task is None
            self._env = domain()

        self._env.reset()

        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self.reward_range = [-np.inf, np.inf]
        self._state_keys_to_ignore = ["force_torque"] if small_state_space else []

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            if key in self._state_keys_to_ignore:
                continue
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        if not self._detach_image_from_obs:
            spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0

        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items() if key not in self._state_keys_to_ignore}
        if not self._detach_image_from_obs:
            obs["image"] = self.render()
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()

        done = time_step.last()

        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items() if key not in self._state_keys_to_ignore}
        if not self._detach_image_from_obs:
            obs["image"] = self.render()
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        scene_image = self._env.physics.render(*self._size, camera_id=self._camera)

        return scene_image

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            return getattr(self._env, item)
