import numpy as np
import csv
import os
import threading
from datetime import datetime
import copy

import imageio  # 需要安装: pip install imageio

def tag_episode(episode, dir, message=''):
    with open(os.path.join(dir, 'save_log.txt'), 'w+') as f:
        f.write(f'Episode {episode}: \n')
        f.write(f'\t {message}\n')

class RUNNER:
    def __init__(self, agent, env, params, device):
        self.agent = agent
        self.env = env
        self.params = params
        # self.env_agents = {agent_id for agent_id in self.agent.agents.keys()} #  此处创建的是集合，顺序会出问题！ 三个小时花在这里了。。
        # print("self.env_agents:",self.env_agents)  # 此处打印顺序会乱

        self.env_agents = [agent_id  for agent_id in self.agent.agents.keys()]  #将键值 按顺序转换为列表self.env_agents = list(agent_id for agent_id in self.agent.agents.keys())
        self.done = {agent_id : False for agent_id in self.agent.agents.keys()} # 字典
        # print("self.env_agents:",self.env_agents)

        self.best_train_score = self.params.best_score
        self.best_eval_score = -10000
        self.best_capture_rate = -1
        self.best_capture_steps = self.params.evaluate_episode_length * 2

        # 添加奖励记录相关的属性
        self.episode_rewards = {}  # 存储每个智能体的详细奖励历史
        self.all_adversary_mean_rewards = []   #添加新的列表来存储每轮 追捕者 的平均奖励

        # 将 agent 的模型放到指定设备上
        for agent in self.agent.agents.values():
            agent.actor.to(device)
            agent.actor_target.to(device)
            agent.critic.to(device)
            agent.critic_target.to(device)

    def load_agent(self, load_dir=None):
        self.agent.load_model(load_dir)

    def train(self, exp_dir):
        self.exp_dir = exp_dir
        self.ckp_dir = os.path.join(exp_dir, 'ckp_models')
        self.best_train_score_dir = os.path.join(exp_dir, 'best_train_score_models')
        self.best_eval_score_dir = os.path.join(exp_dir, 'best_eval_score_models')
        self.best_capture_rate_dir = os.path.join(exp_dir, 'best_capture_rate_models')
        self.best_capture_steps_dir = os.path.join(exp_dir, 'best_capture_steps_models')

        step = 0
        # 记录每个智能体在每个episode的奖励
        self.episode_rewards = {agent_id: np.zeros(self.params.episode_num) for agent_id in self.env.agents}
        self.all_adversary_mean_rewards = []  # 追捕者平均奖励记录
        self.all_eval_scores = []
        self.all_capture_rates = []
        self.all_capture_steps = []

        # episode循环
        for episode in range(self.params.episode_num):
            print('='*20)
            print(f"Episode {episode}/{self.params.episode_num}")
            print('='*20)
            # print(f"This is episode {episode}")
            # 初始化环境 返回初始状态 为一个字典 键为智能体名字 即env.agents中的内容，内容为对应智能体的状态
            obs, _ = self.env.reset(self.params.seed)
            self.done = {agent_id : False for agent_id in self.env_agents}
            # 每个智能体当前episode的奖励
            agent_reward = {agent_id: 0 for agent_id in self.env_agents}

            # 每个智能体与环境进行交互
            while self.env.agents:  #  加入围捕判断
                step += 1  # 此处记录的是并行 的step，即统一执行后，step+1
                if step < self.params.random_steps:
                    action = {agent_id: self.env.action_space(agent_id).sample() for agent_id in self.env.agents}
                else:
                    action = self.agent.select_action(obs, explore=True, total_step=step, noise_type='gaussian') # 使用高斯噪声探索
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.done = {agent_id: bool(terminated[agent_id] or truncated[agent_id]) for agent_id in self.env_agents}
                self.agent.add(obs, action, reward, next_obs, self.done)

                # 计算当前episode每个智能体的奖励 每个step求和
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r

                if step >= self.params.random_steps and step % self.params.learn_interval == 0:
                    # 更新网络
                    self.agent.learn(self.params.batch_size, self.params.gamma) # 目标网络软更新放置在learn函数中，由policy_freq控制更新频率
                # 状态更新 - 创建深拷贝以避免引用问题
                #obs = next_obs
                obs = {k: v.copy() if hasattr(v, 'copy') else copy.deepcopy(v) for k, v in next_obs.items()}

            episode_adversary_rewards = [] # 每轮结束后的记录
            for agent_id, r in agent_reward.items():
                self.episode_rewards[agent_id][episode] = r
                if agent_id.startswith('adversary_'):
                    episode_adversary_rewards.append(r)
            adversary_mean = np.mean(episode_adversary_rewards)
            self.all_adversary_mean_rewards.append(adversary_mean)

            need_eval = episode % self.params.evaluate_interval == 0

            if adversary_mean > self.best_train_score:
                ss = f"New best score,{adversary_mean:>2f},>, {self.best_train_score:>2f}, saving models..."
                print(ss)
                self.agent.save_model(timestamp = False, save_dir=self.best_train_score_dir)  #存放在根目录
                self.best_train_score = adversary_mean
                tag_episode(episode, self.best_train_score_dir, ss)

                need_eval = True
                
            # 间隔一定episode，或得到最佳训练模型后，进行Evaluation
            eval_score = capture_rate = capture_steps = 0
            if need_eval:
                episode_dir = os.path.join(self.exp_dir, 'episodes', f"{episode:04d}")
                os.makedirs(episode_dir, exist_ok=True)
                res = self.evaluate()

                eval_score = res['avg_reward']
                capture_rate = res['capture_rate']
                capture_steps = res['capture_steps']

                if self.best_eval_score < eval_score:
                    ss = f"New best eval score,{eval_score:>2f},>, {self.best_eval_score:>2f}, saving models..."
                    print(ss)
                    self.best_eval_score = res['avg_reward']
                    self.agent.save_model(save_dir=self.best_eval_score_dir)
                    tag_episode(episode, self.best_eval_score_dir, ss)
                
                if self.best_capture_rate < capture_rate:
                    ss = f"New best capture rate,{capture_rate:>2f},>, {self.best_capture_rate:>2f}, saving models..."
                    print(ss)
                    self.best_capture_rate = capture_rate
                    self.agent.save_model(save_dir=self.best_capture_rate_dir)
                    tag_episode(episode, self.best_capture_rate_dir, ss)
                
                if capture_steps < self.best_capture_steps:
                    ss = f"New best capture steps,{capture_steps:>2f},<, {self.best_capture_steps:>2f}, saving models..."
                    print(ss)
                    self.best_capture_steps = capture_steps
                    self.agent.save_model(save_dir=self.best_capture_steps_dir)
                    tag_episode(episode, self.best_capture_steps_dir, ss)

            # # 打印进度
            # if (episode + 1) % 100 == 0:  # 每100轮打印一次
                message = f'Episode {episode}: '
                for agent_id, r in agent_reward.items():
                    message += f'{agent_id}: {r:>4f}; '
                message += f'train_score: {adversary_mean:>4f}, \
                            eval_score: {self.best_eval_score:>4f}, \
                            capture_rate: {self.best_capture_rate:>4f} \
                            capture_steps: {self.best_capture_steps:>4f}'
                print(message)

                print("Save checkpoints")
                self.agent.save_model(save_dir = self.ckp_dir)

            self.all_eval_scores.append(eval_score)
            self.all_capture_rates.append(capture_rate)
            self.all_capture_steps.append(capture_steps)

        # 奖励记录保存为csv
        self.save_rewards_to_csv(self.exp_dir)   

    def save_rewards_to_csv(self, chkpt_dir, prefix=''):
        """移植自runner.py的保存方法"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        filename = f"{prefix}rewards_{timestamp}.csv"
        
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # plot_dir = os.path.join(current_dir, '../plot/matd3_data')  # 调整保存路径
        # os.makedirs(chkpt_dir, exist_ok=True)
        
        with open(os.path.join(chkpt_dir, filename), 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Episode'] + list(self.episode_rewards.keys()) + ['Adversary_Mean'] + ['Eval_Adversary_Mean', 'Capture_Rate', 'Capture_Steps']
            writer.writerow(header)
            
            for ep in range(self.params.episode_num):
                row = [ep + 1]
                row += [self.episode_rewards[agent_id][ep] for agent_id in self.episode_rewards]
                row.append(self.all_adversary_mean_rewards[ep] if ep < len(self.all_adversary_mean_rewards) else 0)
                row.append(self.all_eval_scores[ep] if ep < len(self.all_eval_scores) else 0)
                row.append(self.all_capture_rates[ep] if ep < len(self.all_capture_rates) else 0)
                row.append(self.all_capture_steps[ep] if ep < len(self.all_capture_steps) else 0)
                
                writer.writerow(row)
        print(f"Data saved to {os.path.join(chkpt_dir, filename)}") 

#============================================================================================================
    def evaluate(self):
        """评估训练好的智能体"""
        print("evaluating...")
        # 添加胜率统计变量
        total_episodes = self.params.evaluate_episode_num
        successful_captures = 0
        
        total_steps = 0
        capture_steps = []

        total_rewards = 0.

        # 进行多次评估
        for episode in range(self.params.evaluate_episode_num):
            # 初始化环境
            if self.params.use_variable_seeds:
                obs, _ = self.env.reset(self.params.seed + episode)  # 使用不同的种子
            else:
                obs, _ = self.env.reset(self.params.seed)
            self.done = {agent_id: False for agent_id in self.env_agents}
            # 每个智能体当前episode的奖励
            agent_reward = {agent_id: 0 for agent_id in self.env_agents}

            # 记录当前episode的步数
            episode_step = 0
            # 每个智能体与环境进行交互
            while self.env.agents and not any(self.done.values()):
                episode_step += 1
                # 选择动作（评估模式，不添加噪声）
                action = self.agent.select_action(obs, explore=False)
                # 执行动作
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                self.done = {agent_id: bool(terminated[agent_id] or truncated[agent_id]) for agent_id in self.env_agents}
                
                self.captured = {agent_id: terminated[agent_id]  for agent_id in self.env_agents}
                captured_flag = any(self.captured.values()) # 捕获成功标志
                if captured_flag:
                    successful_captures += 1
                    capture_steps.append(episode_step)
                    
                # 记录奖励
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r
                # 更新观测
                obs = {k: v.copy() if hasattr(v, 'copy') else copy.deepcopy(v) for k, v in next_obs.items()}
                # 定期打印奖励信息
                # if episode_step % 10 == 0:  # print_interval
                #     print(f"评估 Episode {episode+1}, 步数: {episode_step}")
                #     for agent_id, r in agent_reward.items():
                #         print(f"{agent_id}: {r:.4f}", end="; ")
                    
                
                # 检查是否达到最大步数
                if episode_step >= self.params.evaluate_episode_length:
                    print(f"评估 Episode {episode+1}: 达到最大步数 {self.params.episode_length}")
                    break
            
            # 计算追捕者平均奖励
            episode_adversary_rewards = []
            for agent_id, r in agent_reward.items():
                if agent_id.startswith('adversary_'):
                    episode_adversary_rewards.append(r)
            adversary_mean = np.mean(episode_adversary_rewards) if episode_adversary_rewards else 0
            total_rewards += adversary_mean

            # # 打印每个评估episode的结果
            print(f"\n评估 Episode {episode+1} 完成, 总步数: {episode_step}")
            for agent_id, r in agent_reward.items():
                print(f"{agent_id}: {r:.4f}", end="; ")
            print(f"追捕者平均: {adversary_mean:.4f}")
            
            # 如果所有智能体都完成了，打印围捕成功
            if captured_flag:
                print(f"围捕成功！用时 {episode_step} 步")
            total_steps += episode_step
            print("-" * 50)  # 分隔线

        # 计算并打印胜率统计
        success_rate = successful_captures / total_episodes * 100
        avg_steps = total_steps / total_episodes
        if len(capture_steps) == 0:
            avg_capture_steps = 0
        else:
            avg_capture_steps = sum(capture_steps) / len(capture_steps)
        
        avg_reward = total_rewards / self.params.evaluate_episode_num

        print("\n评估完成")
        print("\n==== 评估统计 ====")
        print(f"总评估轮数: {total_episodes}")
        print(f"捕手平均Reward: {avg_reward:.2f}")
        print(f"成功围捕次数: {successful_captures}")
        print(f"围捕成功率: {success_rate:.2f}%")
        print(f"平均步数/轮: {avg_steps:.2f}")
        print(f"成功围捕平均步数: {avg_capture_steps:.2f}")
        print("=" * 20)

        return {
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'capture_rate': success_rate,
            'capture_counts': successful_captures,
            'capture_steps': avg_capture_steps
        }


class RecordingRunner(RUNNER):
    def evaluate(self, model_dir):
        if not os.path.exists(model_dir):
            print(f"Model in {model_dir} is not exist !!!!!!!!!!!!!!!")
            return
        
        self.agent.load_model(model_dir)

        # 记录每个episode的和奖励 用于平滑，显示平滑奖励函数
        self.reward_sum_record = []
        
        successful_captures = 0
        capture_steps = []

        total_rewards = 0.

        # episode循环
        for episode in range(self.params.evaluate_episode_num):
            step = 0  # 每回合step重置
            print(f"Evaluation episode {episode}/{self.params.evaluate_episode_num} 回合")
            # 初始化环境 返回初始状态
            obs, _ = self.env.reset(seed=self.params.seed)  # 重置环境，开始新回合
            self.done = {agent_id: False for agent_id in self.env_agents}
            # 每个智能体当前episode的奖励
            agent_reward = {agent_id: 0 for agent_id in self.env.agents}
            
            frames = []  # 用于存储渲染帧

            # 捕获初始帧
            frame = self.env.render()
            if frame is not None:
                frames.append(frame)
            
            # 每个智能体与环境进行交互
            ss = 'Normal end'
            # breakpoint()
            while self.env.agents and not any(self.done.values()):
                # breakpoint()
                step += 1
                # 使用训练好的智能体选择动作
                action = self.agent.select_action(obs)
                # 执行动作 获得下一状态 奖励 终止情况
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # 捕获当前帧
                frame = self.env.render()
                if frame is not None:
                    frames.append(frame)
                
                self.done = {agent_id: bool(terminated[agent_id] or truncated[agent_id]) for agent_id in self.env_agents}
                
                self.captured = {agent_id: terminated[agent_id]  for agent_id in self.env_agents}
                captured_flag = any(self.captured.values()) # 捕获成功标志
                if captured_flag:
                    successful_captures += 1
                    capture_steps.append(step)

                    ss = f'Capture successfully in {step:>4f} step'
                
                # 累积每个智能体的奖励
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r
                obs = next_obs
                # if step % 10 == 0:
                #     print(f"Step {step}, action: {action}, reward: {reward}, done: {self.done}")

                if step >= self.params.evaluate_episode_length:
                    print(f"评估 Episode {episode}: 达到最大步数 {self.params.episode_length}")
                    ss = f"Hit max steps: {self.params.episode_length}"
                    break
            
            # breakpoint()

            sum_reward = sum(agent_reward.values())
            self.reward_sum_record.append(sum_reward)
            print(f"回合 {episode + 1} 总奖励: {sum_reward}")

            tag_episode(episode, model_dir, ss)

            # 保存为GIF
            gif_path = os.path.join(model_dir, f'matd3_episode_{episode:04d}.gif')
            print(f"正在保存GIF到: {gif_path}")
            imageio.mimsave(gif_path, frames, fps=10)
        
