import collections
import io
import os
import pathlib
import re
import time
from typing import Tuple, Optional

import wandb

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd


def to_np(x):
    return x.detach().cpu().numpy()


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class RequiresGrad:
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)


class Logger:
    def __init__(self, step, config):
        wandb.init(
            project="dreamer-v3",
            name=config.exp_label,
            config=config,
            mode="disabled",  # for debug don't capture wandb
        )
        self._config = config
        self._last_step = None
        self._last_time = None
        self.step = step

    def get_agent_frames(self):
        return self.step // self._config.action_repeat

    def scalar(self, name, value):
        wandb.log({name: value, "env_step": self.step}, commit=False)

    def image(self, name, value):
        wandb.log({name: wandb.Image(value), "env_step": self.step})

    def video(self, name, value):
        wandb.log({
            name: wandb.Video(value[0].transpose(0, 3, 1, 2)),
            "env_step": self.step,
        })

    def write(self, fps=False):
        metrics = {'agent_frames': self.get_agent_frames()}
        if fps:
            metrics["fps"] = self._compute_fps(self.step)
        wandb.log(metrics, commit=True)

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration


def simulate(agent, envs, tasks, cache, directory, logger, is_eval=False, limit=None, state2image=None,
             num_meta_episodes=1):
    is_meta = num_meta_episodes > 1
    total_env_steps = 0
    completed_tasks = []
    env_index_to_ongoing_tasks = {}
    tasks_left = list(tasks)
    obs = [None] * len(envs)
    partial_episodes = [None] * len(envs)
    env_ids = [None] * len(envs)
    agent_state = None
    done = np.ones(len(envs), bool)
    eval_lengths = []
    eval_scores = []
    score = None
    logged_video = False
    if is_meta:
        scores_per_episode = [[] for _ in range(num_meta_episodes)]
    while len(completed_tasks) < len(tasks):
        # reset envs if necessary
        reset_results, reset_indices = [], []
        for env_index in range(len(envs)):
            if done[env_index]:
                if len(tasks_left) == 0:
                    # reset with some dummy task
                    reset_results.append(envs[env_index].reset())
                else:
                    # reset with a task
                    task = tasks_left.pop()
                    if task is not None:
                        reset_results.append(envs[env_index].reset(task))
                    else:
                        reset_results.append(envs[env_index].reset())
                    env_index_to_ongoing_tasks[env_index] = task
                reset_indices.append(env_index)
        reset_results = [r() for r in reset_results]
        for env_index, reset_result in zip(reset_indices, reset_results):
            t = reset_result.copy()
            t = {k: convert(v) for k, v in t.items()}
            # action will be added to transition in add_to_cache
            t["reward"] = 0.0
            t["discount"] = 1.0
            if is_meta:
                t["meta_episode_reset"] = False
            partial_episodes[env_index] = [t]
            env_ids[env_index] = envs[env_index].id
            # replace obs with done by initial state
            obs[env_index] = reset_result

        # step agents
        obs = {k: np.stack([obs[env_index][k] for env_index in range(len(envs))]) for k in obs[0]}
        action, agent_state = agent(obs, done, agent_state)

        if isinstance(action, dict):
            action = [
                {k: np.array(action[k][i].detach().cpu()) for k in action}
                for i in range(len(envs))
            ]
        else:
            action = np.array(action)
        assert len(action) == len(envs)
        # step envs
        results = [e.step(a) for e, a in zip(envs, action)]
        results = [r() for r in results]
        obs, reward, done, infos = zip(*results)
        obs = list(obs)
        done = np.array(done)

        for env_index, (a, result) in enumerate(zip(action, results)):
            o, r, d, info = result
            o = {k: convert(v) for k, v in o.items()}
            transition = o.copy()
            if isinstance(a, dict):
                transition.update(a)
            else:
                transition["action"] = a
            transition["reward"] = r
            transition["discount"] = info.get("discount", np.array(1 - float(d)))
            transition["meta_episode_reset"] = info.get("meta_episode_reset", np.array(1 - float(d)))
            partial_episodes[env_index].append(transition)

        # handle done events
        if done.any():
            completed_env_indices = [index for index, d in enumerate(done) if d]
            for env_index in completed_env_indices:
                episode = partial_episodes[env_index]
                env_id = env_ids[env_index]

                # if the task was one that was given (reset with a task),
                if env_index in env_index_to_ongoing_tasks:
                    # we denote as completed
                    task = env_index_to_ongoing_tasks[env_index]
                    completed_tasks.append((task, episode))
                    total_env_steps += len(episode) - 1
                    del env_index_to_ongoing_tasks[env_index]

                    # then we need to add it to the cache
                    for t in episode:
                        add_to_cache(cache, env_id, t)

                    # and do the required simulations
                    save_episodes(directory, {env_id: cache[env_id]})
                    if is_eval:
                        score, length, score_per_episode = _simulator_helper_eval(env_id, cache[env_id],
                                                                                   cache, num_meta_episodes)
                        eval_scores.append(score)
                        eval_lengths.append(length)
                        if is_meta:
                            for j in range(len(scores_per_episode)):
                                scores_per_episode[j].append(score_per_episode[j])
                        if not logged_video:
                            video = None
                            if "image" in cache[env_id].keys():
                                video = np.array(cache[env_id]["image"])
                            elif state2image is not None:
                                if is_meta:
                                    meta_episodes_indices = np.where(cache[env_id]["meta_episode_reset"])[0]
                                    meta_episodes_indices = [0] + list(meta_episodes_indices)
                                    meta_episode_numbers = np.repeat(np.arange(num_meta_episodes),
                                                                     np.diff(meta_episodes_indices))
                                else:
                                    meta_episode_numbers = np.ones((len(list(cache[env_id].values())[0]), 1))
                                curr_cache_reordered = [dict(zip(cache[env_id], t)) for t in
                                                        zip(*cache[env_id].values())]
                                video = np.array(
                                    [state2image(s, t, task) for t, s in
                                     zip(meta_episode_numbers, curr_cache_reordered)])

                            if video is not None:
                                logger.video(f"eval_policy", video[None])
                            logged_video = True
                    else:
                        _simulator_helper_train(logger, cache[env_id], cache, limit)

                partial_episodes[env_index] = []
                env_ids[env_index] = None

    if is_eval:
        score = sum(eval_scores) / len(eval_scores)
        length = sum(eval_lengths) / len(eval_lengths)
        if is_meta:
            for j in range(num_meta_episodes):
                score_per_episode[j] = sum(scores_per_episode[j]) / len(scores_per_episode[j])
        episode_num = len(eval_scores)
        log_step = logger.step

        if is_meta:
            for j in range(num_meta_episodes):
                logger.scalar(f"eval_return/episode{str(j)}", score_per_episode[j])
            if num_meta_episodes == 2:
                logger.scalar(f"eval_return/diff",
                              score_per_episode[1] - score_per_episode[0])
            logger.scalar(f"eval_return/sum", score)
        else:
            logger.scalar(f"eval_return", score)
        logger.scalar(f"eval_length", length)
        logger.scalar(
            f"eval_episodes", episode_num
        )
        logger.write()
        # keep only last item for saving memory. this cache is used for video_pred later
        while len(cache) > 1:
            # FIFO
            cache.popitem(last=False)
    return completed_tasks, total_env_steps, score


def _simulator_helper_train(logger, episode, cache, limit):
    length = len(episode["reward"]) - 1
    score = float(np.array(episode["reward"]).sum())
    step_in_dataset = erase_over_episodes(cache, limit)
    logger.scalar(f"dataset_size", step_in_dataset)
    logger.scalar(f"train_return", score)
    logger.scalar(f"train_length", length)
    logger.scalar(f"train_episodes", len(cache))
    logger.write()


def _simulator_helper_eval(env_id, episode, cache, num_meta_episodes):
    length = len(episode["reward"]) - 1
    score = float(np.array(episode["reward"]).sum())
    score_per_episode = None
    is_meta = num_meta_episodes > 1
    if is_meta:
        if "meta_episode_reset" not in cache[env_id].keys():
            raise ValueError(
                f"observations must have meta_episode_reset key when num_meta_episodes > 1")
        if np.array(cache[env_id]["meta_episode_reset"]).sum() != num_meta_episodes:
            raise ValueError(
                f"had more episodes resets than num_meta_episodes")

        meta_episodes_indices = np.where(cache[env_id]["meta_episode_reset"])[0]
        meta_episodes_indices = [0] + list(meta_episodes_indices)
        score_per_episode = []
        for j in range(len(meta_episodes_indices) - 1):
            score_per_episode.append(np.array(cache[env_id]["reward"]).astype(np.float64)[
                                     meta_episodes_indices[j]:meta_episodes_indices[j + 1]].sum())

    return score, length, score_per_episode


def add_to_cache(cache, id, transition):
    if id not in cache:
        cache[id] = dict()
        for key, val in transition.items():
            cache[id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[id]:
                # fill missing data(action, etc.) at second time
                cache[id][key] = [convert(0 * val)]
                cache[id][key].append(convert(val))
            else:
                cache[id][key].append(convert(val))


def erase_over_episodes(cache, dataset_size):
    step_in_dataset = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
        if (
                not dataset_size
                or step_in_dataset + (len(ep["reward"]) - 1) <= dataset_size
        ):
            step_in_dataset += len(ep["reward"]) - 1
        else:
            del cache[key]
    return step_in_dataset


def convert(value, precision=32):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)


def save_episodes(directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    for filename, episode in episodes.items():
        length = len(episode["reward"])
        filename = directory / f"{filename}-{length}.npz"
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())
    return True


def from_generator(generator, batch_size):
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(next(generator))
        data = {}
        for key in batch[0].keys():
            data[key] = []
            for i in range(batch_size):
                data[key].append(batch[i][key])
            data[key] = np.stack(data[key], 0)
        yield data


def sample_episodes(episodes, length, seed=0, sample_first:bool=False):
    random = np.random.RandomState(seed)
    while True:
        size = 0
        ret = None
        p = np.array(
            [len(next(iter(episode.values()))) for episode in episodes.values()]
        )
        p = p / np.sum(p) # p is the relative size of each episode in the buffer, but nothing forces the episodes to remain in the same order
        while size < length:
            episode = random.choice(list(episodes.values()), p=p)
            total = len(next(iter(episode.values()))) # size of the episode
            # make sure at least one transition included
            if total < 2:
                continue
            index = 0 if sample_first else int(random.randint(0, total - 1))
            current_ret = {
                k: v[index: min(index + length - size, total)] for k, v in episode.items()
            }
            if "is_first" in current_ret:
                current_ret["is_first"][0] = True

            if not ret:
                ret = current_ret
            else:
                ret = {k: np.append(ret[k], v, axis=0) for k, v in current_ret.items()}
            size = len(next(iter(ret.values())))
        yield ret


def load_episodes(directory, limit=None, reverse=True):
    directory = pathlib.Path(directory).expanduser()
    episodes = collections.OrderedDict()
    total = 0
    if reverse:
        for filename in reversed(sorted(directory.glob("*.npz"))):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            # extract only filename without extension
            episodes[str(os.path.splitext(os.path.basename(filename))[0])] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    else:
        for filename in sorted(directory.glob("*.npz")):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            episodes[str(filename)] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    return episodes


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class DiscDist:
    def __init__(
            self,
            logits,
            low=-20.0,
            high=20.0,
            transfwd=symlog,
            transbwd=symexp,
            device="cuda",
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255).to(device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
                F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
                + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        return self._dist.mean

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return log_probs0 * (1 - x) + log_probs1 * x


class UnnormalizedHuber(torchd.normal.Normal):
    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(
                torch.sqrt((event - self.mean) ** 2 + self._threshold ** 2)
                - self._threshold
        )

    def mode(self):
        return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(torchd.Transform):
    def __init__(self, validate_args=False, name="tanh"):
        super().__init__()

    def _forward(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = torch.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = torch.math.log(2.0)
        return 2.0 * (log2 - x - torch.softplus(-2.0 * x))


def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        # (inputs, pcont) -> (inputs[index], pcont[index])
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([outputs, last], dim=-1)
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.flip(outputs, [1])
    outputs = torch.unbind(outputs, dim=0)
    return outputs


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    if axis != 0:
        dims = list(range(len(reward.shape)))
        dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns


class Optimizer:
    def __init__(
            self,
            name,
            parameters,
            lr,
            eps=1e-4,
            clip=None,
            wd=None,
            wd_pattern=r".*",
            opt="adam",
            use_amp=False,
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "nadam": lambda: NotImplemented(f"{opt} is not implemented"),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, params, retain_graph=False):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = norm.item()
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start):
    last = start
    outputs = []
    for index in range(inputs[0].shape[0]):
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        last_as_list = [last] if isinstance(last, dict) else last
        for i, _last in enumerate(last_as_list):
            if isinstance(_last, dict):
                # we save a dic with list of tensors per value
                if index == 0:
                    outputs.append({key: [] for key in _last.keys()})
                for key, value in _last.items():
                    outputs[i][key].append(value.clone().unsqueeze(0))
            else:
                # we save a list of tensors
                if index == 0:
                    outputs.append([])
                outputs[i].append(_last.clone().unsqueeze(0))
    for i in range(len(outputs)):
        if isinstance(outputs[i], dict):
            for key in outputs[i].keys():
                outputs[i][key] = torch.cat(outputs[i][key], dim=0)
        else:
            outputs[i] = torch.cat(outputs[i], dim=0)
    return outputs


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until


def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clip(torch.Tensor([step / duration]), 0, 1)[0]
            return (1 - mix) * initial + mix * final
        match = re.match(r"warmup\((.+),(.+)\)", string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = torch.clip(step / warmup, 0, 1)
            return scale * value
        match = re.match(r"exp\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        match = re.match(r"horizon\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clip(step / duration, 0, 1)
            horizon = (1 - mix) * initial + mix * final
            return 1 - 1 / horizon
        raise NotImplementedError(string)


def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def tensorstats(tensor, prefix=None):
    metrics = {
        "mean": to_np(torch.mean(tensor)),
        "std": to_np(torch.std(tensor)),
        "min": to_np(torch.min(tensor)),
        "max": to_np(torch.max(tensor)),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def window_data_repeat(data: torch.Tensor, window_size: int, sequence_indices: Optional[torch.Tensor], shift_left:bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
    # note, data should be a continuous sequence, i.e. no skipping elements by sampling
    assert window_size % 2 == 1, "window_size must be odd"
    # data: (batch_size, seq_len, embed_size)
    padding = (window_size - 1) // 2
    padding_data = data[:, 0, :].unsqueeze(1).repeat(1, padding, 1)
    # create masks, same size as the data, expect the last dimension is 1
    padding_mask = torch.zeros(list(padding_data.shape[:2]) + [1], dtype=padding_data.dtype, device=padding_data.device)
    data_mask = torch.ones(list(data.shape[:2]) + [1], dtype=data.dtype, device=data.device)
    # concatenate the masks to the data (adding a feature in the last dimension)
    padding_data = torch.cat([padding_mask, padding_data], dim=-1)
    data = torch.cat([data_mask, data], dim=-1)
    # concatenate the data with padding
    padded_embed = torch.cat([padding_data, data, padding_data], dim=1)  # (batch_size, seq_len + 2*padding, embed_size + 1)
    if shift_left:
        # Shift the padded data to the left by one, and correcting the mask
        padded_embed[:, padding, 0] = 0
        shifted_padded_embed = padded_embed[:, 1:]
        padded_embed = torch.cat([shifted_padded_embed, padded_embed[:, -1:]], dim=1)
    # unfold the padded data to get the windowed data
    windowed_embed = padded_embed.unfold(1, window_size, 1).permute(0, 1, 3, 2)  # (batch_size, seq_len, window_size, embed_size + 1)
    # subsample the sequences
    if sequence_indices is not None:
        batch_indices = np.arange(data.shape[0])[:, np.newaxis]
        windowed_embed = windowed_embed[batch_indices, sequence_indices]
    windowed_embed = windowed_embed.reshape(-1, windowed_embed.shape[1] * windowed_embed.shape[2], windowed_embed.shape[3])  # (batch_size, seq_len * window_size, embed_size + 1)
    # split the windowed data into data and mask
    windowed_data = windowed_embed[:, :, 1:]
    windowed_mask = windowed_embed[:, :, :1]

    return windowed_data, windowed_mask
