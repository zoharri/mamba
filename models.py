import copy
import time

import torch
from torch import nn
import numpy as np

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_input_layers,
            config.dyn_output_layers,
            config.dyn_rec_depth,
            config.dyn_shared,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_temp_post,
            config.dyn_min_std,
            config.dyn_cell,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
            config.rnn_detach_every
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter

        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        if config.reconstruction_window > 0:
            self.heads["multi_decoder"] = networks.MultiDecoder(
                feat_size + self.embed_size, shapes, **config.decoder
            )
        reward_mlp_shape = (255,) if config.reward_head == "symlog_disc" else []
        self.heads["reward"] = networks.MLP(
            feat_size,  # pytorch version
            reward_mlp_shape,
            config.reward_layers,
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head,
            outscale=0.0,
            device=config.device,
        )
        if config.reconstruction_window > 0:
            self.heads["multi_reward"] = networks.MLP(
                feat_size + self.embed_size,  # pytorch version
                reward_mlp_shape,
                config.reward_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.reward_head,
                outscale=0.0,
                device=config.device,
            )

        self.heads["cont"] = networks.MLP(
            feat_size,  # pytorch version
            [],
            config.cont_layers,
            config.units,
            config.act,
            config.norm,
            dist="binary",
            device=config.device,
        )
        if config.reconstruction_window > 0:
            self.heads["multi_cont"] = networks.MLP(
                feat_size + self.embed_size,  # pytorch version
                [],
                config.cont_layers,
                config.units,
                config.act,
                config.norm,
                dist="binary",
                device=config.device,
            )
        self._config.grad_heads = list(self._config.grad_heads)
        new_grad_heads = []
        if config.reconstruction_window > 0:
            for name in config.grad_heads:
                new_grad_heads.append("multi_" + name)
        self._config.grad_heads.extend(new_grad_heads)
        self._config.grad_heads = tuple(self._config.grad_heads)

        for name in self._config.grad_heads:
            assert name in self.heads, name

        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        self._scales = dict(reward=config.reward_scale, cont=config.cont_scale)

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        time_metrics = {}
        data_preprocess_time = time.time()
        data = self.preprocess(data)
        self._add_wm_timing(time_metrics, 'data_preprocess', time.time() - data_preprocess_time)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embedding_time = time.time()
                embed = self.encoder(data)
                self._add_wm_timing(time_metrics, 'embedding', time.time() - embedding_time)
                rssm_time = time.time()
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                self._add_wm_timing(time_metrics, 'rssm', time.time() - rssm_time)
                kl_free = tools.schedule(self._config.kl_free, self._step)
                dyn_scale = tools.schedule(self._config.dyn_scale, self._step)
                rep_scale = tools.schedule(self._config.rep_scale, self._step)
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )

                get_features_time = time.time()
                feat = self.dynamics.get_feat(post)
                self._add_wm_timing(time_metrics, 'get_features', time.time() - get_features_time)

                losses, mses = self._compute_prediction_losses(feat, embed, data, time_metrics)

            optimizer_time = time.time()
            metrics = self._model_opt(sum(losses.values()) + kl_loss, self.parameters())
            self._add_wm_timing(time_metrics, 'optimizer', time.time() - optimizer_time)
            self._add_wm_timing(time_metrics, 'total', time.time() - data_preprocess_time, use_counter=False)
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        for name, mse in mses.items():
            metrics[f"{name}_mse"] = to_np(mse)
        metrics.update(time_metrics)
        with torch.cuda.amp.autocast(self._use_amp):
            prior_ent = self.dynamics.get_dist(prior).entropy()
            post_ent = self.dynamics.get_dist(post).entropy()
            metrics["prior_ent"] = to_np(torch.mean(prior_ent))
            metrics["post_ent"] = to_np(torch.mean(post_ent))
            context = dict(
                embed=embed,
                feat=feat,
                kl=kl_value,
                postent=post_ent,
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    @staticmethod
    def _add_wm_timing(time_metrics, name, t, use_counter=True):
        if use_counter:
            name = f'{len(time_metrics)}_{name}'
        name = f'wm_{name}_time'
        time_metrics[name] = t

    def get_uncertainty_measure(self, features):
        reward_dist = self.heads["reward"](features)
        reward_logits = reward_dist.logits
        logits_pairs_mean = (reward_logits[:, :, 1:] + reward_logits[:, :, :-1]) / 2
        max_pair = logits_pairs_mean.max(dim=-1)[0]
        return (logits_pairs_mean.sum(dim=-1) - max_pair).unsqueeze(-1)

    def _compute_prediction_losses(self, feat, embed, data, time_metrics):
        preds = {}
        sequence_indices = None

        sample_features_time = time.time()
        if self._config.hidden_states_subsample != -1:
            # randomly subsample a minibatch of hidden states
            batch_size, batch_length = feat.shape[:2]

            sequence_indices = np.array([np.random.choice(
                np.arange(batch_length),
                self._config.hidden_states_subsample,
                replace=True,
            ) for _ in range(batch_size)]).reshape(batch_size, self._config.hidden_states_subsample)
            batch_indices = np.arange(batch_size)[:, np.newaxis]
            feat = feat[batch_indices, sequence_indices]

        self._add_wm_timing(time_metrics, 'sample_features', time.time() - sample_features_time)

        repeat_data_time = time.time()
        windowed_embed = None
        if self._config.reconstruction_window > 0:
            windowed_embed, _ = tools.window_data_repeat(embed, self._config.reconstruction_window, sequence_indices)
        self._add_wm_timing(time_metrics, 'repeat_data', time.time() - repeat_data_time)

        feature_to_prediction_time = time.time()
        for name, head in self.heads.items():
            grad_head = name in self._config.grad_heads
            curr_feat = feat if grad_head else feat.detach()
            if "multi" in name:
                feat_repeat = curr_feat.repeat(1, self._config.reconstruction_window, 1)
                feat_and_embed_input = torch.cat([feat_repeat, windowed_embed], dim=-1)
                pred = head(feat_and_embed_input)
            else:
                pred = head(curr_feat)

            if type(pred) is dict:
                preds.update({f"{name},{k}": v for k, v in pred.items()})
            else:
                preds[name] = pred
        self._add_wm_timing(time_metrics, 'feature_to_prediction', time.time() - feature_to_prediction_time)

        compute_loss_time = time.time()
        losses, mses = {}, {}
        for name, pred in preds.items():
            if ',' in name:
                head_name, key_name = name.split(",")
            else:
                head_name, key_name = None, name
            if head_name is not None and head_name.startswith("multi_"):
                curr_data, masking = tools.window_data_repeat(data[key_name], self._config.reconstruction_window,
                                                              sequence_indices, shift_left=True)
            elif head_name is None and key_name.startswith("multi_"):
                curr_data, masking = tools.window_data_repeat(data[key_name[6:]], self._config.reconstruction_window,
                                                              sequence_indices, shift_left=True)
            else:
                curr_data = data[key_name]
                if sequence_indices is not None:
                    batch_indices = np.arange(curr_data.shape[0])[:, np.newaxis]
                    curr_data = curr_data[batch_indices, sequence_indices]
                masking = None
            # negative log likelihood loss
            like = pred.log_prob(curr_data)
            mse = (pred.mode().detach() - curr_data).square().sum(-1).sqrt()
            if masking is not None:
                # if there is masking we apply it:
                # note, we don't have to count the mean with exactly the same number of elements as the masking,
                # because the window size if fixed so the number of zeroed elements is fixed, therefore it is
                # like multiplying by a constant which could be adjusted by the scale.
                like = like * masking.resize_as(like, )
                mse = mse * masking.resize_as(mse, )
            loss = -torch.mean(like) * self._scales.get(name, 1.0)
            losses[name] = loss
            mses[name] = torch.mean(mse)

        self._add_wm_timing(time_metrics, 'compute_loss', time.time() - compute_loss_time)
        return losses, mses

    def preprocess(self, obs):
        obs = obs.copy()
        if "image" in obs.keys():
            obs["image"] = torch.Tensor(obs["image"]) / 255.0 - 0.5
        # (batch_size, batch_length) -> (batch_size, batch_length, 1)
        if "reward" in obs:
            obs["reward"] = torch.Tensor(obs["reward"]).unsqueeze(-1)
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        if "is_terminal" in obs:
            # this label is necessary to train cont_head
            obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        else:
            raise ValueError('"is_terminal" was not found in observation.')
        if "time_step" in obs:
            obs["time_step"] = torch.Tensor(obs["time_step"]).unsqueeze(-1)

        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
                :6
                ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6] + 0.5
        model = model + 0.5
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, logger, world_model, stop_grad_actor=True, reward=None):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._logger = logger
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor

        self._reward = reward

        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.ActionHead(
            feat_size,
            config.num_actions,
            config.actor_layers,
            config.units,
            config.act,
            config.norm,
            config.actor_dist,
            config.actor_init_std,
            config.actor_min_std,
            config.actor_max_std,
            config.actor_temp,
            outscale=1.0,
            unimix_ratio=config.action_unimix_ratio,
        )
        value_mlp_shape = (255,) if config.value_head == "symlog_disc" else []
        self.value = networks.MLP(
            feat_size,
            value_mlp_shape,
            config.value_layers,
            config.units,
            config.act,
            config.norm,
            config.value_head,
            outscale=0.0,
            device=config.device,
        )
        if config.slow_value_target:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw,
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.value_lr,
            config.ac_opt_eps,
            config.value_grad_clip,
            **kw,
        )
        if self._config.reward_EMA:
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
            self,
            start,
            objective=None,
    ):
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}

        with (tools.RequiresGrad(self.actor)):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)

                # this target is not scaled
                target, weights, base, actor_ent, state_ent = self._compute_target(
                    imag_feat, imag_state, imag_action, reward,
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_state,
                    imag_action,
                    target,
                    actor_ent,
                    state_ent,
                    weights,
                    base,
                )
                metrics.update(mets)
                value_input = imag_feat[:-1].detach()

        if self._config.log_imagined_horizon_effective_length:
            for q in [0.9, 0.2, 0.1, 0.01]:
                metrics[f'imagination_horizon_above_{q}'] = to_np(
                    torch.sum(torch.greater(weights, q)) / weights.shape[1])
        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                # slow is flag to indicate whether slow_target is used for lambda-return
                if self._config.slow_value_target:
                    slow_target = self._slow_value(value_input)
                    value_loss = value_loss - value.log_prob(
                        slow_target.mode().detach()
                    )
                if self._config.value_decay:
                    value_loss += self._config.value_decay * value.mode()
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor_dist in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach() if self._stop_grad_actor else feat
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        return feats, states, actions

    def _compute_target(
            self, imag_feat, imag_state, imag_action, reward
    ):
        if "cont" in self._world_model.heads:
            discount = self._config.discount * self._world_model.heads["cont"](imag_feat).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        actor_ent = None
        if self._config.future_entropy and self._config.actor_entropy() > 0:
            actor_ent = self.actor(imag_feat).entropy()
            reward += self._config.actor_entropy() * actor_ent
        state_ent = None
        if self._config.future_entropy and self._config.actor_state_entropy() > 0:
            state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
            reward += self._config.actor_state_entropy() * state_ent
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        target = torch.stack(target, dim=1)
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1], actor_ent, state_ent

    def _compute_actor_loss(
            self,
            imag_feat,
            imag_state,
            imag_action,
            target,
            actor_ent,
            state_ent,
            weights,
            base,
    ):
        metrics = {}
        inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
        policy = self.actor(inp)
        actor_ent = policy.entropy()
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        # Q-val for actor is not transformed using symlog
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            values = self.reward_ema.values
            metrics["EMA_005"] = to_np(values[0])
            metrics["EMA_095"] = to_np(values[1])
        else:
            adv = target - base

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                    policy.log_prob(imag_action)[:-1][:, :, None]
                    * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                    policy.log_prob(imag_action)[:-1][:, :, None]
                    * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix()
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        if not self._config.future_entropy and (self._config.actor_entropy() > 0):
            actor_entropy = self._config.actor_entropy() * actor_ent[:-1][:, :, None]
            actor_target += actor_entropy
        if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
            state_entropy = self._config.actor_state_entropy() * state_ent[:-1]
            actor_target += state_entropy
            metrics["actor_state_entropy"] = to_np(torch.mean(state_entropy))
        actor_loss = -torch.mean(weights[:-1] * actor_target)
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.slow_value_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
