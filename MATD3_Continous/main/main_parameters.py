import argparse
import yaml
import os
import shutil

def load_yaml_config(config_path):
    """加载yaml配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Yaml配置文件不存在: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def merge_args(args, config):
    """
    合并参数：yaml配置覆盖argparse参数
    若需命令行参数优先级更高，可反转逻辑（仅用yaml填充未通过命令行指定的参数）
    """
    args_dict = vars(args)
    for key, value in config.items():
        if key in args_dict:
            # 类型校验（避免yaml参数类型与argparse不匹配）
            if isinstance(args_dict[key], type(value)) or value is None:
                setattr(args, key, value)
            else:
                raise TypeError(
                    f"参数{key}类型不匹配：argparse期望{type(args_dict[key])}，yaml提供{type(value)}"
                )
    return args

def backup_yaml_config(config_path, ckpt_dir):
    """备份yaml配置到ckpt目录"""
    os.makedirs(ckpt_dir, exist_ok=True)
    backup_path = os.path.join(ckpt_dir, "train_config_backup.yaml")
    with open(config_path, 'r', encoding='utf-8') as f_src, open(backup_path, 'w', encoding='utf-8') as f_dst:
        yaml.dump(yaml.safe_load(f_src), f_dst, indent=4, allow_unicode=True)
    print(f"Yaml配置已备份到: {backup_path}")

def main_parameters():
    parser = argparse.ArgumentParser("MATD3 legacy")
    
    parser.add_argument("--config", type=str, default=None, help="yaml配置文件路径（如config/train_config.yaml）")
    parser.add_argument("--seed", type=int, default=-1, help='随机种子 (使用-1表示不使用固定种子)')
    parser.add_argument("--dump_root", type=str, default="checkpoints", help="None | human | rgb_array")
    parser.add_argument("--use_variable_seeds", type=bool, default=False, help="使用可变随机种子")
    parser.add_argument("--env_name", type=str, default="simple_tag_v3", help="name of the env",   
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3', 'simple_tag_env']) 
    parser.add_argument("--render_mode", type=str, default="None", help="None | human | rgb_array")
    parser.add_argument("--episode_num", type=int, default=1500, help="训练轮数")
    parser.add_argument("--episode_length", type=int, default=100, help="每轮最大步数")
    parser.add_argument("--evaluate_interval", type=int, default=50, help="评估间隔")
    parser.add_argument("--evaluate_episode_num", type=int, default=100, help="评估轮数")
    parser.add_argument("--evaluate_episode_length", type=int, default=300, help="评估轮数")
    parser.add_argument('--learn_interval', type=int, default=10, help='学习间隔步数')
    parser.add_argument('--random_steps', type=int, default=200, help='初始随机探索步数')
    parser.add_argument('--tau', type=float, default=0.01, help='软更新参数')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='经验回放缓冲区容量')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--actor_lr', type=float, default=0.00001, help='Actor学习率')
    parser.add_argument('--critic_lr', type=float, default=0.0001, help='Critic学习率')
    parser.add_argument('--comm_lr', type=float, default=0.00001, help='Comm学习率')
    parser.add_argument('--message_dim', type=int, default=3, help='通信消息维度')
    parser.add_argument('--best_score', type=int, default= -20, help='最佳分数_初始值')
    parser.add_argument('--visdom', action="store_true", help="是否使用visdom可视化")
    parser.add_argument('--size_win', type=int, default=200, help="平滑窗口大小")
    parser.add_argument("--device", type=str, default='cpu', help="训练设备，默认自动选择cpu")

    # 解析命令行参数
    args = parser.parse_args()
    
    # 加载并合并yaml配置（若指定）
    if args.config is not None:
        yaml_config = load_yaml_config(args.config)
        args = merge_args(args, yaml_config)

    # 原有seed处理逻辑
    if args.seed == -1:
        args.seed = None
        
    return args

# 测试示例
if __name__ == "__main__":
    args = main_parameters()
    print(f"最终训练轮数: {args.episode_num}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练设备: {args.device}")