#!/usr/bin/env python3

import warnings
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path

import hydra
import torch
import numpy as np

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    if cfg.use_aux_inputs:
        for key in cfg.suite.aux_keys:
            obs_shape[key] = obs_spec[key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = (
        action_spec.shape
        if cfg.suite.action_type == "continuous"
        else action_spec.num_values
    )
    return hydra.utils.instantiate(cfg.agent)


class WorkspaceIL:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # load data
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.batch_size
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)
        self.stats = self.expert_replay_loader.dataset.stats

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.cfg.suite.task_make_fn.max_episode_len = (
            self.expert_replay_loader.dataset._max_episode_len
        )
        self.cfg.suite.task_make_fn.max_state_dim = (
            self.expert_replay_loader.dataset._max_state_dim
        )
        self.env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)

        # create agent
        self.agent = make_agent(
            self.env[0].observation_spec(), self.env[0].action_spec(), cfg
        )

        # TODO: Make this compatible with no eval case
        self.envs_till_idx = self.expert_replay_loader.dataset.envs_till_idx
        print(f"envs_till_idx: {self.expert_replay_loader.dataset.envs_till_idx}")

        # Discretizer for BeT

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def eval(self):
        self.agent.train(False)
        episode_rewards = []
        successes = []

        num_envs = (
            len(self.env) if self.cfg.suite.name == "calvin" else self.envs_till_idx
        )

        for env_idx in range(num_envs):
            print(f"evaluating env {env_idx}")
            episode, total_reward = 0, 0
            eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
            success = []

            while eval_until_episode(episode):
                time_step = self.env[env_idx].reset()
                self.agent.buffer_reset()
                step = 0

                # prompt
                if self.cfg.prompt != None and self.cfg.prompt != "intermediate_goal":
                    prompt = self.expert_replay_loader.dataset.sample_test(env_idx)
                else:
                    prompt = None

                if episode == 0:
                    self.video_recorder.init(self.env[env_idx], enabled=True)

                # plot obs with cv2
                while not time_step.last():
                    if self.cfg.prompt == "intermediate_goal":
                        prompt = self.expert_replay_loader.dataset.sample_test(
                            env_idx, step
                        )
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(
                            time_step.observation,
                            prompt,
                            self.stats,
                            step,
                            self.global_step,
                            eval_mode=True,
                        )
                    time_step = self.env[env_idx].step(action)
                    self.video_recorder.record(self.env[env_idx])
                    total_reward += time_step.reward
                    step += 1

                episode += 1
                success.append(time_step.observation["goal_achieved"])
            self.video_recorder.save(f"{self.global_step}_env{env_idx}.mp4")
            episode_rewards.append(total_reward / episode)
            successes.append(np.mean(success))

        for _ in range(len(self.env) - num_envs):
            episode_rewards.append(0)
            successes.append(0)

        with self.logger.log_and_dump_ctx(self.global_step, ty="eval") as log:
            for env_idx, reward in enumerate(episode_rewards):
                log(f"episode_reward_env{env_idx}", reward)
                log(f"success_env{env_idx}", successes[env_idx])
            log("episode_reward", np.mean(episode_rewards[:num_envs]))
            log("success", np.mean(successes))
            log("episode_length", step * self.cfg.suite.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

        self.agent.train(True)

    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.suite.num_train_steps, 1  # self.cfg.suite.action_repeat
        )
        log_every_step = utils.Every(
            self.cfg.suite.log_every_steps, 1  # self.cfg.suite.action_repeat
        )
        eval_every_step = utils.Every(
            self.cfg.suite.eval_every_steps, 1  # self.cfg.suite.action_repeat
        )
        save_every_step = utils.Every(
            self.cfg.suite.save_every_steps, 1  # self.cfg.suite.action_repeat
        )

        metrics = None
        while train_until_step(self.global_step):
            # try to evaluate
            if (
                self.cfg.eval
                and eval_every_step(self.global_step)
                and self.global_step > 0
            ):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval()

            # update
            metrics = self.agent.update(self.expert_replay_iter, self.global_step)
            self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # log
            if log_every_step(self.global_step):
                elapsed_time, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_frame, ty="train") as log:
                    log("total_time", total_time)
                    log("actor_loss", metrics["actor_loss"])
                    log("step", self.global_step)

            # save snapshot
            if save_every_step(self.global_step):
                self.save_snapshot()

            # Update scene
            if self.cfg.sequential_train:
                if (
                    self.global_step + 1
                ) % self.steps_till_next_scene == 0 and self.scene_idx < len(
                    self.scene_names
                ) - 1:
                    self.scene_idx = (self.scene_idx + 1) % len(self.scene_names)
                    self.envs_till_idx += len(
                        self.task_names[self.scene_names[self.scene_idx]]
                    )
                    self.expert_replay_loader.dataset.envs_till_idx = self.envs_till_idx
                    self.steps_till_next_scene = (
                        self.envs_till_idx * self.cfg.suite.num_train_steps_per_task
                    )
                    self.expert_replay_iter = iter(self.expert_replay_loader)

                    # self.agent.reinit_optimizers()

            self._global_step += 1

    def save_snapshot(self):
        snapshot_dir = self.work_dir / "snapshot"
        snapshot_dir.mkdir(exist_ok=True)
        snapshot = snapshot_dir / f"{self.global_step}.pt"
        self.agent.clear_buffers()
        keys_to_save = ["timer", "_global_step", "_global_episode", "stats"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open("wb") as f:
            torch.save(payload, f)

        self.agent.buffer_reset()

    def load_snapshot(self, snapshots, encoder_only=False):
        # bc
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        if "vqvae" in snapshots:
            with snapshots["vqvae"].open("rb") as f:
                payload = torch.load(f)
            agent_payload["vqvae"] = payload
        self.agent.load_snapshot(agent_payload, encoder_only=encoder_only, eval=False)
        # self.agent.load_snapshot_eval(agent_payload)


@hydra.main(config_path="cfgs", config_name="config", version_base=None)
def main(cfg):
    from train_bc import WorkspaceIL as W

    workspace = W(cfg)

    # Load weights
    if cfg.load_bc:
        # BC weight
        snapshots = {}
        bc_snapshot = Path(cfg.bc_weight)
        if not bc_snapshot.exists():
            raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
        print(f"loading bc weight: {bc_snapshot}")
        snapshots["bc"] = bc_snapshot
        # vqvae_snapshot = Path(cfg.vqvae_weight)
        # if vqvae_snapshot.exists():
        #     print(f"loading vqvae weight: {vqvae_snapshot}")
        #     snapshots["vqvae"] = vqvae_snapshot
        workspace.load_snapshot(snapshots)

    workspace.train()
    # workspace.eval()


if __name__ == "__main__":
    main()
