import einops
import hydra
import numpy as np
from collections import deque

import torch
from torch import nn

from torchvision import transforms as T

from agent.networks.rgb_modules import BaseEncoder, ResnetEncoder
import utils

from agent.networks.policy_head import DeterministicHead
from agent.networks.gpt import GPT, GPTConfig, CrossAttention
from agent.networks.mlp import MLP


class Actor(nn.Module):
    def __init__(
        self,
        repr_dim,
        act_dim,
        hidden_dim,
        policy_type="gpt",
        policy_head="deterministic",
        num_feat_per_step=1,
    ):
        super().__init__()

        self._policy_type = policy_type
        self._policy_head = policy_head
        self._repr_dim = repr_dim
        self._act_dim = act_dim
        self._num_feat_per_step = num_feat_per_step

        self._action_token = nn.Parameter(torch.randn(1, 1, 1, repr_dim))

        # GPT model
        if policy_type == "gpt":
            self._policy = GPT(
                GPTConfig(
                    block_size=65,  # 50,  # 51,  # 50,
                    input_dim=repr_dim,
                    output_dim=hidden_dim,
                    n_layer=8,
                    n_head=4,
                    n_embd=hidden_dim,
                    dropout=0.1,  # 0.6, #0.1,
                )
            )
        else:
            raise NotImplementedError
        self._action_head = DeterministicHead(
            hidden_dim, self._act_dim, hidden_size=hidden_dim, num_layers=2
        )
        self.apply(utils.weight_init)

    def forward(
        self,
        obs,
        num_prompt_feats,
        stddev,
        action=None,
        cluster_centers=None,
        mask=None,
    ):
        B, T, D = obs.shape
        if self._policy_type == "mlp":
            if T * D < self._repr_dim:
                gt_num_time_steps = (
                    self._repr_dim // D - num_prompt_feats
                ) // self._num_feat_per_step
                num_repeat = (
                    gt_num_time_steps
                    - (T - num_prompt_feats) // self._num_feat_per_step
                )
                initial_obs = obs[
                    :, num_prompt_feats : num_prompt_feats + self._num_feat_per_step
                ]
                initial_obs = initial_obs.repeat(1, num_repeat, 1)
                obs = torch.cat(
                    [obs[:, :num_prompt_feats], initial_obs, obs[:, num_prompt_feats:]],
                    dim=1,
                )
                B, T, D = obs.shape
            obs = obs.view(B, 1, T * D)
            features = self._policy(obs)
        elif self._policy_type == "gpt":
            # insert action token at each self._num_feat_per_step interval
            prompt = obs[:, :num_prompt_feats]
            obs = obs[:, num_prompt_feats:]
            obs = obs.view(B, -1, self._num_feat_per_step, obs.shape[-1])
            action_token = self._action_token.repeat(B, obs.shape[1], 1, 1)
            obs = torch.cat([obs, action_token], dim=-2).view(B, -1, D)
            obs = torch.cat([prompt, obs], dim=1)

            if mask is not None:
                mask = torch.cat([mask, torch.ones(B, 1).to(mask.device)], dim=1)
                mask = mask.view(B, -1, 1, self._num_feat_per_step + 1)
                base_mask = torch.ones(
                    B,
                    mask.shape[1],
                    self._num_feat_per_step + 1,
                    self._num_feat_per_step + 1,
                ).to(mask.device)
                base_mask[:, :, -1:] = mask

            # get action features
            features = self._policy(obs, mask=base_mask if mask is not None else None)
            features = features[:, num_prompt_feats:]
            num_feat_per_step = self._num_feat_per_step + 1  # +1 for action token
            features = features[:, num_feat_per_step - 1 :: num_feat_per_step]

        # action head
        pred_action = self._action_head(
            features,
            stddev,
            **{"cluster_centers": cluster_centers, "action_seq": action},
        )

        if action is None:
            return pred_action
        else:
            loss = self._action_head.loss_fn(
                pred_action,
                action,
                reduction="mean",
                **{"cluster_centers": cluster_centers},
            )
            return pred_action, loss[0] if isinstance(loss, tuple) else loss


class BCAgent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        hidden_dim,
        stddev_schedule,
        stddev_clip,
        use_tb,
        augment,
        encoder_type,
        policy_type,
        policy_head,
        pixel_keys,
        aux_keys,
        use_aux_inputs,
        train_encoder,
        norm,
        separate_encoders,
        temporal_agg,
        max_episode_len,
        num_queries,
        use_actions,
    ):
        self.device = device
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb
        self.augment = augment
        self.encoder_type = encoder_type
        self.policy_head = policy_head
        self.use_aux_inputs = use_aux_inputs
        self.norm = norm
        self.train_encoder = train_encoder
        self.separate_encoders = separate_encoders
        self.use_actions = use_actions  # only for the prompt

        # actor parameters
        self._act_dim = action_shape[0]

        # keys
        self.aux_keys = aux_keys
        self.pixel_keys = pixel_keys

        # action chunking params
        self.temporal_agg = temporal_agg
        self.max_episode_len = max_episode_len
        self.num_queries = num_queries if self.temporal_agg else 1

        # number of inputs per time step
        num_feat_per_step = len(self.pixel_keys)
        if use_aux_inputs:
            num_feat_per_step += len(self.aux_keys)

        # observation params
        if use_aux_inputs:
            aux_shape = {key: obs_shape[key] for key in self.aux_keys}
        obs_shape = obs_shape[self.pixel_keys[0]]

        # Track model size
        model_size = 0

        # encoder
        if self.separate_encoders:
            self.encoder = {}
        if self.encoder_type == "base":
            if self.separate_encoders:
                for key in self.pixel_keys:
                    self.encoder[key] = BaseEncoder(obs_shape).to(device)
                    self.repr_dim = self.encoder[key].repr_dim
                    model_size += sum(
                        p.numel()
                        for p in self.encoder[key].parameters()
                        if p.requires_grad
                    )
            else:
                self.encoder = BaseEncoder(obs_shape).to(device)
                self.repr_dim = self.encoder.repr_dim
                model_size += sum(
                    p.numel() for p in self.encoder.parameters() if p.requires_grad
                )
        elif self.encoder_type == "resnet":
            self.repr_dim = 512
            enc_fn = lambda: ResnetEncoder(
                obs_shape,
                512,
                cond_dim=None,
                cond_fusion="none",
            ).to(device)
            if self.separate_encoders:
                for key in self.pixel_keys:
                    self.encoder[key] = enc_fn()
                    model_size += sum(
                        p.numel()
                        for p in self.encoder[key].parameters()
                        if p.requires_grad
                    )
            else:
                self.encoder = enc_fn()
                model_size += sum(
                    p.numel() for p in self.encoder.parameters() if p.requires_grad
                )

        # projector for proprioceptive features
        if use_aux_inputs:
            self.aux_projector = nn.ModuleDict()
            for key in self.aux_keys:
                if key.startswith("digit"):
                    self.aux_projector[key] = ResnetEncoder(
                        obs_shape,
                        512,
                        cond_dim=None,
                        cond_fusion="none",
                    ).to(device)
                else:
                    self.aux_projector[key] = MLP(
                        aux_shape[key][0],
                        hidden_channels=[self.repr_dim, self.repr_dim],
                    ).to(device)
                    self.aux_projector[key].apply(utils.weight_init)
                model_size += sum(
                    p.numel()
                    for p in self.aux_projector[key].parameters()
                    if p.requires_grad
                )

        # projector for actions
        if self.use_actions:
            self.action_projector = MLP(
                self._act_dim, hidden_channels=[self.repr_dim, self.repr_dim]
            ).to(device)
            self.action_projector.apply(utils.weight_init)
            model_size += sum(
                p.numel() for p in self.action_projector.parameters() if p.requires_grad
            )

        # actor
        action_dim = (
            self._act_dim * self.num_queries if self.temporal_agg else self._act_dim
        )
        self.actor = Actor(
            self.repr_dim,
            action_dim,
            hidden_dim,
            policy_type,
            policy_head,
            num_feat_per_step,
        ).to(device)
        model_size += sum(p.numel() for p in self.actor.parameters() if p.requires_grad)

        print(f"Total number of parameters in the model: {model_size}")

        # optimizers
        # encoder
        if self.train_encoder:
            if self.separate_encoders:
                params = []
                for key in self.pixel_keys:
                    params += list(self.encoder[key].parameters())
            else:
                params = list(self.encoder.parameters())
            self.encoder_opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
            # self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(
            #     self.encoder_opt, step_size=15000, gamma=0.1
            # )
        # proprio
        if self.use_aux_inputs:
            self.aux_opt = torch.optim.AdamW(
                self.aux_projector.parameters(), lr=lr, weight_decay=1e-4
            )
            # self.proprio_scheduler = torch.optim.lr_scheduler.StepLR(
            #     self.proprio_opt, step_size=15000, gamma=0.1
            # )

        # action projector
        if self.use_actions:
            self.action_opt = torch.optim.AdamW(
                self.action_projector.parameters(), lr=lr, weight_decay=1e-4
            )
            # self.action_scheduler = torch.optim.lr_scheduler.StepLR(
            #     self.action_opt, step_size=15000, gamma=0.1
            # )
        # actor
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=lr, weight_decay=1e-4
        )
        # self.actor_scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.actor_opt, step_size=15000, gamma=0.1
        # )

        # augmentations
        if self.norm:
            if self.encoder_type == "small":
                MEAN = torch.tensor([0.0, 0.0, 0.0])
                STD = torch.tensor([1.0, 1.0, 1.0])
            elif self.encoder_type == "resnet" or self.norm:
                MEAN = torch.tensor([0.485, 0.456, 0.406])
                STD = torch.tensor([0.229, 0.224, 0.225])
            self.customAug = T.Compose([T.Normalize(mean=MEAN, std=STD)])

        # data augmentation
        if self.augment:
            # self.aug = utils.RandomShiftsAug(pad=4)
            self.test_aug = T.Compose([T.ToPILImage(), T.ToTensor()])
            self.digit_aug = T.Compose([T.ToTensor()])

        self.train()
        self.buffer_reset()

    def __repr__(self):
        return "bc"

    def train(self, training=True):
        self.training = training
        if training:
            if self.separate_encoders:
                for key in self.pixel_keys:
                    if self.train_encoder:
                        self.encoder[key].train(training)
                    else:
                        self.encoder[key].eval()
            else:
                if self.train_encoder:
                    self.encoder.train(training)
                else:
                    self.encoder.eval()
            if self.use_aux_inputs:
                self.aux_projector.train(training)
            if self.use_actions:
                self.action_projector.train(training)
            self.actor.train(training)
        else:
            if self.separate_encoders:
                for key in self.pixel_keys:
                    self.encoder[key].eval()
            else:
                self.encoder.eval()
            if self.use_aux_inputs:
                self.aux_projector.eval()
            if self.use_actions:
                self.action_projector.eval()
            self.actor.eval()

    def buffer_reset(self):
        self.observation_buffer = {}
        for key in self.pixel_keys:
            self.observation_buffer[key] = deque(maxlen=1)
        if self.use_aux_inputs:
            self.aux_buffer = {}
            for key in self.aux_keys:
                self.aux_buffer[key] = deque(maxlen=1)

        # temporal aggregation
        if self.temporal_agg:
            self.all_time_actions = torch.zeros(
                [
                    self.max_episode_len,
                    self.max_episode_len + self.num_queries,
                    self._act_dim,
                ]
            ).to(self.device)

    def clear_buffers(self):
        del self.observation_buffer
        if self.use_aux_inputs:
            del self.aux_buffer
        if self.temporal_agg:
            del self.all_time_actions

    def reinit_optimizers(self):
        if self.train_encoder:
            if self.separate_encoders:
                params = []
                for key in self.pixel_keys:
                    params += list(self.encoder[key].parameters())
            else:
                params = list(self.encoder.parameters())
            self.encoder_opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
            try:
                self.aux_cond_opt = torch.optim.AdamW(
                    self.aux_cond_cross_attn.parameters(),
                    lr=self.lr,
                    weight_decay=1e-4,
                )
            except AttributeError:
                print("Not optimizing aux_cond_cross_attn")
        if self.use_aux_inputs:
            self.aux_opt = torch.optim.AdamW(
                self.aux_projector.parameters(), lr=self.lr, weight_decay=1e-4
            )
        if self.use_actions:
            self.action_opt = torch.optim.AdamW(
                self.action_projector.parameters(), lr=self.lr, weight_decay=1e-4
            )
        params = list(self.actor.parameters())
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=self.lr, weight_decay=1e-4
        )

    def act(self, obs, prompt, norm_stats, step, global_step, eval_mode=False):
        if norm_stats is not None:

            def pre_process(aux_key, s_qpos):
                try:
                    return (s_qpos - norm_stats[aux_key]["min"]) / (
                        norm_stats[aux_key]["max"] - norm_stats[aux_key]["min"] + 1e-5
                    )
                except KeyError:
                    return (s_qpos - norm_stats[aux_key]["mean"]) / (
                        norm_stats[aux_key]["std"] + 1e-5
                    )

            # pre_process = lambda aux_key, s_qpos: (
            #     s_qpos - norm_stats[aux_key]["min"]
            # ) / (norm_stats[aux_key]["max"] - norm_stats[aux_key]["min"] + 1e-5)
            post_process = (
                lambda a: a
                * (norm_stats["actions"]["max"] - norm_stats["actions"]["min"])
                + norm_stats["actions"]["min"]
            )

        # lang projection
        lang_features = None

        # add to buffer
        features = []
        aux_features = []
        aux_cond_features = []
        if self.use_aux_inputs:
            # TODO: Add conditioning here
            for key in self.aux_keys:
                obs[key] = pre_process(key, obs[key])
                if key.startswith("digit"):
                    self.aux_buffer[key].append(
                        self.digit_aug(obs[key].transpose(1, 2, 0)).numpy()
                    )
                    aux_feat = torch.as_tensor(
                        np.array(self.aux_buffer[key]), device=self.device
                    ).float()
                    aux_feat = self.aux_projector[key](aux_feat)[None, :, :]
                else:
                    self.aux_buffer[key].append(obs[key])
                    aux_feat = torch.as_tensor(
                        np.array(self.aux_buffer[key]), device=self.device
                    ).float()
                    aux_feat = self.aux_projector[key](aux_feat[None, :, :])
                aux_features.append(aux_feat[0])

        for key in self.pixel_keys:
            self.observation_buffer[key].append(
                self.test_aug(obs[key].transpose(1, 2, 0)).numpy()
            )
            pixels = torch.as_tensor(
                np.array(self.observation_buffer[key]), device=self.device
            ).float()
            pixels = self.customAug(pixels) if self.norm else pixels
            # encoder
            pixels = (
                self.encoder[key](pixels)
                if self.separate_encoders
                else self.encoder(pixels)
            )
            features.append(pixels)

        features.extend(aux_features)
        features = torch.cat(features, dim=-1).view(-1, self.repr_dim)

        stddev = utils.schedule(self.stddev_schedule, global_step)
        action = self.actor(features.unsqueeze(0), 0, stddev)

        if eval_mode:
            action = action.mean
        else:
            action = action.sample()
        if self.temporal_agg:
            action = action.view(-1, self.num_queries, self._act_dim)
            self.all_time_actions[[step], step : step + self.num_queries] = action[-1:]
            actions_for_curr_step = self.all_time_actions[:, step]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
            action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            if norm_stats is not None:
                return post_process(action.cpu().numpy()[0])
            return action.cpu().numpy()[0]
        else:
            if norm_stats is not None:
                return post_process(action.cpu().numpy()[0, -1, :])
            return action.cpu().numpy()[0, -1, :]

    def update(self, expert_replay_iter, step):
        metrics = dict()
        batch = next(expert_replay_iter)
        data = utils.to_torch(batch, self.device)
        action = data["actions"].float()

        # features
        features = []
        aux_features = []
        if self.use_aux_inputs:
            for key in self.aux_keys:
                aux_feat = data[key].float()
                if "digit" in key:
                    shape = aux_feat.shape
                    # rearrange
                    aux_feat = einops.rearrange(aux_feat, "b t c h w -> (b t) c h w")
                    # augment
                    # aux_feat = self.aug(aux_feat) if self.augment else aux_feat
                    aux_feat = self.customAug(aux_feat) if self.norm else aux_feat
                    aux_feat = self.aux_projector[key](aux_feat)
                    aux_feat = einops.rearrange(
                        aux_feat, "(b t) d -> b t d", t=shape[1]
                    )
                else:
                    aux_feat = self.aux_projector[key](aux_feat)
                aux_features.append(aux_feat)
        for key in self.pixel_keys:
            pixel = data[key].float()
            shape = pixel.shape
            # rearrange
            pixel = einops.rearrange(pixel, "b t c h w -> (b t) c h w")
            # augment
            # pixel = self.aug(pixel) if self.augment else pixel
            pixel = self.customAug(pixel) if self.norm else pixel
            # encode
            if self.train_encoder:
                pixel = (
                    self.encoder[key](pixel)
                    if self.separate_encoders
                    else self.encoder(pixel)
                )
            else:
                with torch.no_grad():
                    pixel = (
                        self.encoder[key](pixel)
                        if self.separate_encoders
                        else self.encoder(pixel)
                    )
            pixel = einops.rearrange(pixel, "(b t) d -> b t d", t=shape[1])
            features.append(pixel)
        features.extend(aux_features)
        # concatenate
        features = torch.cat(features, dim=-1).view(
            action.shape[0], -1, self.repr_dim
        )  # (B, T * num_feat_per_step, D)

        # rearrange action
        if self.temporal_agg:
            action = einops.rearrange(action, "b t1 t2 d -> b t1 (t2 d)")

        # actor loss
        stddev = utils.schedule(self.stddev_schedule, step)
        _, actor_loss = self.actor(
            features,
            0,
            stddev,
            action,
            mask=None,
        )
        if self.train_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)

        if self.use_aux_inputs:
            self.aux_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss["actor_loss"].backward()
        if self.train_encoder:
            self.encoder_opt.step()
        try:
            self.aux_cond_opt.step()
        except AttributeError:
            pass
        if self.use_aux_inputs:
            self.aux_opt.step()
        if self.use_actions:
            self.action_opt.step()
        self.actor_opt.step()

        if self.use_tb:
            for key, value in actor_loss.items():
                metrics[key] = value.item()

        return metrics

    def save_snapshot(self):
        model_keys = ["actor", "encoder"]
        opt_keys = ["actor_opt"]
        if self.train_encoder:
            opt_keys += ["encoder_opt"]
        if self.use_aux_inputs:
            model_keys += ["aux_projector"]
            opt_keys += ["aux_opt"]
        if self.use_actions:
            model_keys += ["action_projector"]
            opt_keys += ["action_opt"]
        # models
        payload = {
            k: self.__dict__[k].state_dict() for k in model_keys if k != "encoder"
        }
        if "encoder" in model_keys:
            if self.separate_encoders:
                for key in self.pixel_keys:
                    payload[f"encoder_{key}"] = self.encoder[key].state_dict()
            else:
                payload["encoder"] = self.encoder.state_dict()
        # optimizers
        payload.update({k: self.__dict__[k] for k in opt_keys})

        others = [
            "use_aux_inputs",
            "aux_keys",
            "use_actions",
            "max_episode_len",
        ]
        payload.update({k: self.__dict__[k] for k in others})
        return payload

    def load_snapshot(self, payload, encoder_only=False, eval=False, load_opt=False):
        # models
        if encoder_only:
            model_keys = ["encoder"]
            payload = {"encoder": payload}
        else:
            model_keys = ["actor", "encoder"]
            if self.use_aux_inputs:
                model_keys += ["aux_projector"]
            if self.use_actions:
                model_keys += ["action_projector"]

        for k in model_keys:
            if k == "encoder" and self.separate_encoders:
                for key in self.pixel_keys:
                    self.encoder[key].load_state_dict(payload[f"encoder_{key}"])
            else:
                self.__dict__[k].load_state_dict(payload[k])

        if eval:
            self.train(False)
            return

        # if not eval
        if not load_opt:
            self.reinit_optimizers()
        else:
            opt_keys = ["actor_opt"]
            if self.train_encoder:
                opt_keys += ["encoder_opt"]
            if self.use_aux_inputs:
                opt_keys += ["aux_opt"]
            if self.use_actions:
                opt_keys += ["action_opt"]
            for k in opt_keys:
                self.__dict__[k] = payload[k]
        self.train(True)
