import numpy as np
import gym
import crafter
import envs.crafter_description as descriptor

class Crafter():

  def __init__(self, task, size=(64, 64), outdir=None, seed=None):
    assert task in ('reward', 'noreward')
    self._env = crafter.Env(size=size, reward=(task == 'reward'), seed=seed)
    self._size = size
    if outdir:
      self._env = crafter.Recorder(
          self._env, outdir,
          save_stats=True,
          save_video=False,
          save_episode=False,
      )
    self._achievements = crafter.constants.achievements.copy()
    self._done = True
    self.__step = 0

  @property
  def observation_space(self):
    spaces = {}
    spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
    spaces["log_reward"] = gym.spaces.Box(-np.inf, np.inf,shape=(1,), dtype=np.float32)
    spaces.update({
        f'log_achievement_{k}': gym.spaces.Box(-np.inf, np.inf,shape=(1,), dtype=np.float32)
        for k in self._achievements})
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def reset(self):
    self._done = False
    image = self._env.reset()
    return self._obs(image, 0.0, {}, is_first=True)

  def step(self, action):
    if len(action.shape) >= 1:
        action = np.argmax(action)
    self.__step += 1
    image, reward, self._done, info = self._env.step(action)
    curPos = "\n" + "=="*15 + "Step: {}, Reward: {}".format(self.__step, reward) + "=="*15 + "\n"
    # desc0, desc1 = descriptor.describe_frame(info)
    # with open("./descriptions.txt", "a+") as myfile:
    #     myfile.write(curPos)
    #     myfile.write(desc0 + "\n")
    #     myfile.write(desc1)

    reward = np.float32(reward)
    return self._obs(
        image, reward, info,
        is_last=self._done,
        is_terminal=info['discount'] == 0), reward, self._done, info

  def _obs(
      self, image, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    log_achievements = {
        f'log_achievement_{k}': info['achievements'][k] if info else 0
        for k in self._achievements}
    return dict(
        image=image,
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        log_reward=np.float32(info['reward'] if info else 0.0),
        **log_achievements,
    )

  def render(self):
    return self._env.render()