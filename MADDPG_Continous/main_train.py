from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
from envs import simple_tag_env, custom_agents_dynamics
from envs import hunter_blocker_env

from types import SimpleNamespace

from utils.config import load_config
from utils.runner import RUNNER

from agents.maddpg.MADDPG_agent import MADDPG
import torch
import os

import time
from datetime import timedelta
from utils.logger import TrainingLogger  # 添加导入

def get_env(env_name, config):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    ep_len = config.env.get("episode_length", 25)
    render_mode = config.env.get("render_mode", "None")
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len, continuous_actions=True)
    if env_name == 'simple_spread_v3':
        new_env = simple_spread_v3.parallel_env(max_cycles=ep_len, render_mode="rgb_array")
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(render_mode = render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    if env_name == 'simple_tag_env':
        new_env = simple_tag_env.parallel_env(render_mode = render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    if env_name == 'hunter_blocker_env':
        new_env = hunter_blocker_env.parallel_env(
            render_mode=render_mode,
            num_hunters=config.env.get("num_hunters", 2),
            num_blockers=config.env.get("num_blockers", 1),
            num_targets=config.env.get("num_targets", 1),
            num_obstacles=config.env.get("num_obstacles", 0),
            world_size=config.env.get("world_size", 2.5),
            capture_distance=config.env.get("capture_distance", 0.3),
            capture_steps=config.env.get("capture_steps", 5),
            hunter_speed=config.env.get("hunter_speed", 1.2),
            blocker_speed=config.env.get("blocker_speed", 1.0),
            target_speed=config.env.get("target_speed", 1.0),
            hunter_view_range=config.env.get("hunter_view_range", 0.6),
            blocker_view_range=config.env.get("blocker_view_range", 1.0),
            target_view_range=config.env.get("target_view_range", 0.8),
            max_cycles=ep_len,
            continuous_actions=True,
        )
    new_env.reset()
    _dim_info = {}
    action_bound = {}
    for agent_id in new_env.agents:
        print("agent_id:",agent_id)
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        action_bound[agent_id] = [] #[low action,  hign action]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0])
        action_bound[agent_id].append(new_env.action_space(agent_id).low)
        action_bound[agent_id].append(new_env.action_space(agent_id).high)
    print("_dim_info:",_dim_info)
    print("action_bound:",action_bound)
    return new_env, _dim_info, action_bound


def build_shared_policy_groups(dim_info):
    shared = {}
    for agent_id in dim_info.keys():
        if agent_id.startswith("hunter_"):
            shared[agent_id] = "hunter"
        elif agent_id.startswith("blocker_"):
            shared[agent_id] = "blocker"
        elif agent_id.startswith("target_"):
            shared[agent_id] = "target"
    return shared



if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
    #                         else 'cuda' if torch.cuda.is_available() else 'cpu')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "configs", "hunter_blocker.yaml")
    config = load_config(config_path)
    device = config.train.get("device", "cpu")
    print("Using device:", device)
    start_time = time.time() # 记录开始时间

    run_name = config.experiment.get("name", "maddpg_run")
    run_root = config.experiment.get("run_root", "runs")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(current_dir, run_root, f"{run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    chkpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")

    args = SimpleNamespace(
        env_name=config.env.get("env_name", "hunter_blocker_env"),
        render_mode=config.env.get("render_mode", "None"),
        episode_num=config.train.get("episode_num", 5),
        episode_length=config.env.get("episode_length", 500),
        learn_interval=config.train.get("learn_interval", 10),
        random_steps=config.train.get("random_steps", 500),
        tau=config.train.get("tau", 0.001),
        gamma=config.train.get("gamma", 0.99),
        buffer_capacity=config.train.get("buffer_capacity", int(1e6)),
        batch_size=config.train.get("batch_size", 128),
        actor_lr=config.train.get("actor_lr", 0.0002),
        critic_lr=config.train.get("critic_lr", 0.002),
        visdom=False,
        size_win=200,
    )
    # 创建环境
    print("Using Env's name", args.env_name)
    env, dim_info, action_bound = get_env(args.env_name, config)
    shared_groups = build_shared_policy_groups(dim_info)
    # 创建MA-DDPG智能体 dim_info: 字典，键为智能体名字 内容为二维数组 分别表示观测维度和动作维度 是观测不是状态 需要注意。
    agent = MADDPG(
        dim_info,
        args.buffer_capacity,
        args.batch_size,
        args.actor_lr,
        args.critic_lr,
        action_bound,
        _chkpt_dir=chkpt_dir,
        _device=device,
        shared_policy_groups=shared_groups,
    )
    # 创建运行对象
    runner = RUNNER(
        agent,
        env,
        args,
        device,
        mode='train',
        log_dir=log_dir,
        use_tensorboard=config.log.get("use_tensorboard", True),
    )
    # 开始训练
    runner.train()
    print("agent",agent)

    # 计算训练时间
    end_time = time.time()
    training_time = end_time - start_time
    # 转换为时分秒格式
    training_duration = str(timedelta(seconds=int(training_time)))
    print(f"\n===========训练完成!===========")
    print(f"训练设备: {device}")
    print(f"训练用时: {training_duration}")

    # 使用logger保存训练日志
    logger = TrainingLogger(log_dir=log_dir)
    current_time = logger.save_training_log(args, device, training_duration, runner)
    print(f"完成时间: {current_time}")

    print("--- saving trained models ---")
    agent.save_model()
    print("--- trained models saved ---")
    
