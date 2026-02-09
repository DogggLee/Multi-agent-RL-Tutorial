import csv
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml
import imageio.v2 as imageio
from torch.utils.tensorboard import SummaryWriter

from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from HAPPO import HAPPO_MPE


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
ENV_DIR = os.path.join(REPO_ROOT, "MADDPG_Continous", "envs")
sys.path.append(ENV_DIR)

import hunter_blocker_env


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_shared_policy_groups(agent_ids):
    shared = {}
    for agent_id in agent_ids:
        if agent_id.startswith("hunter_"):
            shared[agent_id] = "hunter"
        elif agent_id.startswith("blocker_"):
            shared[agent_id] = "blocker"
        elif agent_id.startswith("target_"):
            shared[agent_id] = "target"
        else:
            shared[agent_id] = agent_id
    return shared


def make_env(env_cfg, render_mode):
    return hunter_blocker_env.parallel_env(
        render_mode=render_mode,
        num_hunters=env_cfg["num_hunters"],
        num_blockers=env_cfg["num_blockers"],
        num_targets=env_cfg["num_targets"],
        num_obstacles=env_cfg["num_obstacles"],
        world_size=env_cfg["world_size"],
        capture_distance=env_cfg["capture_distance"],
        capture_steps=env_cfg["capture_steps"],
        hunter_speed=env_cfg["hunter_speed"],
        blocker_speed=env_cfg["blocker_speed"],
        target_speed=env_cfg["target_speed"],
        hunter_view_range=env_cfg["hunter_view_range"],
        blocker_view_range=env_cfg["blocker_view_range"],
        target_view_range=env_cfg["target_view_range"],
        max_cycles=env_cfg["episode_limit"],
        continuous_actions=True,
    )


def get_capture_flag(env):
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
    world = getattr(raw_env, "world", None)
    return bool(getattr(world, "capture", False))


class HunterBlockerRunner:
    def __init__(self, config, run_dir):
        self.config = config
        self.run_dir = run_dir
        self.env_cfg = config["env"]
        self.train_cfg = config["train"]
        self.validation_cfg = config["validation"]
        self.device = torch.device(self.train_cfg["device"])
        self.writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))

        self.env = make_env(self.env_cfg, self.env_cfg["render_mode"])
        self.env.reset(seed=self.train_cfg["seed"])

        self.dim_info = {
            agent_id: [
                self.env.observation_space(agent_id).shape[0],
                self.env.action_space(agent_id).shape[0],
            ]
            for agent_id in self.env.agents
        }
        self.agent_ids = list(self.env.agents)
        self.max_obs_dim = max(dim[0] for dim in self.dim_info.values())
        self.action_dim = self.dim_info[self.agent_ids[0]][1]

        args = SimpleNamespace(
            device=self.device,
            N=len(self.agent_ids),
            obs_dim=self.max_obs_dim,
            action_dim=self.action_dim,
            state_dim=sum(dim[0] for dim in self.dim_info.values()),
            episode_limit=self.env_cfg["episode_limit"],
            batch_size=self.train_cfg["batch_size"],
            mini_batch_size=self.train_cfg["mini_batch_size"],
            max_train_steps=self.train_cfg["episodes"] * self.env_cfg["episode_limit"],
            lr=self.train_cfg["lr"],
            gamma=self.train_cfg["gamma"],
            lamda=self.train_cfg["lamda"],
            epsilon=self.train_cfg["epsilon"],
            K_epochs=self.train_cfg["K_epochs"],
            entropy_coef=self.train_cfg["entropy_coef"],
            use_adv_norm=self.train_cfg["use_adv_norm"],
            use_reward_norm=self.train_cfg["use_reward_norm"],
            use_reward_scaling=self.train_cfg["use_reward_scaling"],
            add_agent_id=self.train_cfg["add_agent_id"],
            use_lr_decay=self.train_cfg["use_lr_decay"],
            use_orthogonal_init=self.train_cfg["use_orthogonal_init"],
            set_adam_eps=self.train_cfg["set_adam_eps"],
            act_dim=self.action_dim,
            shared_policy_groups=build_shared_policy_groups(self.agent_ids),
        )

        self.agent = HAPPO_MPE(args)
        self.agent.env = self.env
        self.agent.all_agents = self.agent_ids
        self.agent.dim_info = self.dim_info

        def get_obs_dims():
            return {agent_id: self.env.observation_space(agent_id).shape[0] for agent_id in self.agent_ids}

        self.env.get_obs_dims = get_obs_dims

        self.replay_buffer = ReplayBuffer(args)
        self.total_steps = 0

        self.reward_norm = None
        self.reward_scaling = None
        if args.use_reward_norm:
            self.reward_norm = Normalization(shape=args.N)
        elif args.use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=args.N, gamma=args.gamma)

    def _pad_obs_to_max_dim(self, obs_dict):
        obs_list = []
        for agent_id in self.agent_ids:
            if agent_id in obs_dict:
                obs = obs_dict[agent_id]
                if len(obs) < self.max_obs_dim:
                    padded_obs = np.zeros(self.max_obs_dim, dtype=np.float32)
                    padded_obs[: len(obs)] = obs
                    obs_list.append(padded_obs)
                else:
                    obs_list.append(obs)
            else:
                obs_list.append(np.zeros(self.max_obs_dim, dtype=np.float32))
        return np.array(obs_list, dtype=np.float32)

    def _get_global_state(self, obs_dict):
        state_parts = []
        for agent_id in self.agent_ids:
            if agent_id in obs_dict:
                state_parts.append(obs_dict[agent_id])
            else:
                state_parts.append(np.zeros(self.dim_info[agent_id][0], dtype=np.float32))
        return np.concatenate(state_parts)

    def run_episode(self, evaluate=False):
        episode_reward = 0.0
        obs_dict, _ = self.env.reset()
        done_dict = {agent_id: False for agent_id in self.agent_ids}

        if self.reward_scaling:
            self.reward_scaling.reset()

        for episode_step in range(self.env_cfg["episode_limit"]):
            actions_dict, logprobs_dict = self.agent.choose_action(obs_dict, evaluate=evaluate)

            obs_n = self._pad_obs_to_max_dim(obs_dict)
            s = self._get_global_state(obs_dict)
            v_n = self.agent.get_value(s)

            next_obs_dict, rewards_dict, terminated_dict, truncated_dict, _ = self.env.step(actions_dict)
            step_reward = sum(rewards_dict.values())
            episode_reward += step_reward

            for agent_id in terminated_dict:
                if terminated_dict[agent_id] or truncated_dict[agent_id]:
                    done_dict[agent_id] = True

            done = all(done_dict.values()) or len(self.env.agents) == 0
            done_n = np.array([done] * len(self.agent_ids), dtype=np.float32)

            if not evaluate:
                a_n = np.zeros((len(self.agent_ids), self.action_dim), dtype=np.float32)
                r_n = np.zeros(len(self.agent_ids), dtype=np.float32)
                a_logprob_n = np.zeros(len(self.agent_ids), dtype=np.float32)

                for i, agent_id in enumerate(self.agent_ids):
                    if agent_id in actions_dict:
                        a_n[i] = actions_dict[agent_id]
                    if agent_id in rewards_dict:
                        r_n[i] = rewards_dict[agent_id]
                    if logprobs_dict is not None and agent_id in logprobs_dict:
                        a_logprob_n[i] = logprobs_dict[agent_id]

                if self.reward_norm:
                    r_n = self.reward_norm(r_n)
                elif self.reward_scaling:
                    r_n = self.reward_scaling(r_n)

                self.replay_buffer.store_transition(
                    episode_step,
                    obs_n,
                    s,
                    v_n,
                    a_n,
                    a_logprob_n,
                    r_n,
                    done_n,
                )

            obs_dict = next_obs_dict
            if done:
                break

        if not evaluate:
            obs_n = self._pad_obs_to_max_dim(obs_dict)
            s = self._get_global_state(obs_dict)
            v_n = self.agent.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return episode_reward, episode_step + 1

    def train(self):
        episodes = self.train_cfg["episodes"]
        for episode in range(episodes):
            reward, steps = self.run_episode(evaluate=False)
            self.total_steps += steps
            self.writer.add_scalar("train/episode_reward", reward, episode + 1)
            if self.replay_buffer.episode_num == self.train_cfg["batch_size"]:
                self.agent.train(self.replay_buffer, self.total_steps)
                self.replay_buffer.reset_buffer()
            if (episode + 1) % 1 == 0:
                print(f"Episode {episode + 1}/{episodes}: reward={reward:.2f}, steps={steps}")

    def validate(self):
        validation_dir = os.path.join(self.run_dir, "validation")
        gif_dir = os.path.join(validation_dir, "gifs")
        os.makedirs(gif_dir, exist_ok=True)

        successes = 0
        steps_list = []
        metrics_rows = []

        for i in range(self.validation_cfg["num_episodes"]):
            seed = self.validation_cfg["seed_start"] + i
            env = make_env(self.env_cfg, self.validation_cfg["render_mode"])
            obs_dict, _ = env.reset(seed=seed)
            frames = []
            first_frame = env.render()
            if first_frame is not None:
                frames.append(first_frame)

            steps = 0
            for _ in range(self.validation_cfg["max_steps"]):
                actions_dict, _ = self.agent.choose_action(obs_dict, evaluate=True)
                obs_dict, _, terminated_dict, truncated_dict, _ = env.step(actions_dict)
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                steps += 1
                if all(terminated_dict.values()) or all(truncated_dict.values()) or len(env.agents) == 0:
                    break

            capture_success = get_capture_flag(env)
            successes += int(capture_success)
            steps_list.append(steps)

            gif_path = os.path.join(gif_dir, f"validation_episode_{i + 1}.gif")
            if frames:
                imageio.mimsave(gif_path, frames, fps=self.validation_cfg["gif_fps"])

            metrics_rows.append(
                {
                    "episode": i + 1,
                    "seed": seed,
                    "capture_success": capture_success,
                    "steps": steps,
                    "gif_path": gif_path,
                }
            )
            env.close()

        success_rate = successes / self.validation_cfg["num_episodes"]
        avg_steps = float(np.mean(steps_list)) if steps_list else 0.0

        metrics = {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "episodes": metrics_rows,
        }

        metrics_json = os.path.join(validation_dir, "metrics.json")
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        metrics_csv = os.path.join(validation_dir, "metrics.csv")
        with open(metrics_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["episode", "seed", "capture_success", "steps", "gif_path"],
            )
            writer.writeheader()
            writer.writerows(metrics_rows)

        self.writer.add_scalar("validation/success_rate", success_rate, self.total_steps)
        self.writer.add_scalar("validation/avg_steps", avg_steps, self.total_steps)
        self.writer.flush()

        print(f"Validation success rate: {success_rate:.2%}, avg steps: {avg_steps:.2f}")

    def save_checkpoint(self):
        checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        payload = {
            "actor_state_dict_by_agent": {
                agent_id: self.agent.agents[agent_id].actor.state_dict() for agent_id in self.agent_ids
            },
            "critic_state_dict_by_agent": {
                agent_id: self.agent.agents[agent_id].critic.state_dict() for agent_id in self.agent_ids
            },
        }
        checkpoint_path = os.path.join(checkpoints_dir, "happo_actor_critic.pth")
        torch.save(payload, checkpoint_path)


def main():
    config_path = os.path.join(CURRENT_DIR, "configs", "hunter_blocker_happo.yaml")
    config = load_config(config_path)

    run_name = config["experiment"]["name"]
    run_root = config["experiment"]["run_root"]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(CURRENT_DIR, run_root, f"{run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    runner = HunterBlockerRunner(config, run_dir)
    runner.train()
    runner.save_checkpoint()
    runner.validate()


if __name__ == "__main__":
    main()
