import os
import pathlib
import pickle
import cv2
import imageio
import numpy as np
import torch
from time import sleep
from PIL import Image, ImageDraw, ImageFont
import envs.wrappers as wrappers
from dreamer import Dreamer
from envs.custom_reach import CustomReachEnv
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
class DummyLogger():
    def get_agent_frames(self):
        return 0


def get_action(obs, agent_state):
    obs = {k: np.array([obs[k]]) for k in obs}
    action, agent_state = agent(obs, np.array([done]), agent_state, training=False)
    action = {k: action[k][0].detach().cpu().numpy() for k in action}
    return action, agent_state


def draw_text_on_image(image, reward, episode):
    width, height = image.size[0], image.size[1]
    draw = ImageDraw.Draw(image)
    text = "Reached Goal!" if reward > 0 else "Searching for Goal..."
    font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=40)
    text_width, text_height = font.getsize(text)
    text_x = (width - text_width) // 2
    text_y = 0
    text_color = (0, 255, 0) if reward > 0 else (255, 0, 0)
    draw.text((text_x, text_y), text, font=font, fill=text_color)
    draw.text((0, 60), f"Episode: {episode}", font=font, fill=(0, 0, 0))
    return image


def load_env_and_agent(config_dir):
    config_dir = pathlib.Path(config_dir).expanduser()
    with open(os.path.join(config_dir, 'config.txt'), 'rb') as f:
        config = pickle.load(f)
    suite, task = config.task.split("_", 1)
    assert suite == "panda-reach", RuntimeError("Only panda-reach is supported")
    with suppress_stdout():
        base_env = CustomReachEnv()
    env = base_env
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    env = wrappers.RewardObs(env)
    max_time_steps = config.max_episode_length * config.num_meta_episodes
    env = wrappers.TimeAugmentedState(env, max_time_steps)
    env = wrappers.MetaLearningEnv(env, config.num_meta_episodes, config.max_episode_length)
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config,
        DummyLogger(),
        None,
    ).to(config.device)
    agent.load_state_dict(torch.load(config_dir / "best_model.pt"), strict=False)
    return agent, env, base_env, config.max_episode_length


exp_dir = "assets/panda_reach_exp"
out_vis_name = "panda_reach.gif"
agent, env, base_env, max_episode_length = load_env_and_agent(exp_dir)

imgs_list = []
done = False
episode = 0
time_s = 0
while not done:
    # You can also set the task manually if you want to test a specific task
    task = env.sample_task()
    obs = env.reset(task)
    print(f"The sampled task: {env.get_task()}")
    done = False
    agent_state = None
    while not done:
        action, agent_state = get_action(obs, agent_state)
        obs, reward, done, infos = env.step(action)
        pil_image = Image.fromarray(base_env.render('rgb_array'))

        resulting_image = draw_text_on_image(pil_image, reward, episode)
        imgs_list.append(imageio.core.util.Array(np.array(resulting_image)))
        print(f"Reward: {reward}")
        sleep(0.01)
        time_s += 1
        if time_s % max_episode_length == 0:
            episode += 1

imageio.mimsave(out_vis_name, imgs_list, duration=1 / 3, loop=0)
