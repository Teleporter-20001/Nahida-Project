import os

import numpy as np
import pygame
import sys
import torch
import random

from tqdm import tqdm

from app.common.ProcessDrawer import ProcessDrawer
from app.common.Settings import Settings
from app.env.RewardSet import RewardSet
from app.env.TouhouEnv import TouhouEnv
from app.Agent.DQNTrain import DQNTrainer

class GameTrainer:
    """Game loop for training with DQN"""

    def __init__(self, env: TouhouEnv, trainer: DQNTrainer):
        self.env = env
        self.trainer = trainer  # DQNTrainer

    def run_training_episode(self, max_steps: int = 100000, render: bool = True, epsilon=0.1) -> RewardSet:
        """Run one episode with training (DQN)"""
        state = self.env.reset()
        total_reward_set = RewardSet()

        for step in range(max_steps):
            # pygame 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    sys.exit()
                try:
                    self.env.handle_event(event)
                except AttributeError:
                    pass

            # --- 选择动作 (epsilon-greedy) ---
            # action = None
            # if random.random() < epsilon:
            #     # action_idx = random.randrange(self.trainer.brain.net.num_actions)
            #     action = random.choice(list(self.env.action_space))
            # else:
                # obs_tensor = self.trainer.brain.net._obs_to_tensor(state.observation)
                # with torch.no_grad():
                #     q_values = self.trainer.policy_net(obs_tensor)
                #     action_idx = q_values.argmax(dim=1).item()
                #     action = list(self.env.action_space)[action_idx]
            with torch.no_grad():
                action = self.trainer.select_action(state, epsilon)

            # --- 环境交互 ---
            next_state, reward, done, info = self.env.step(action)
            total_reward_set += info['reward_details']

            # --- 存经验 ---
            # 将 action 转为索引再保存，方便后续用作 long tensor 索引
            from app.Agent.DataStructure import Action as _ActionEnum

            if isinstance(action, _ActionEnum):
                action_idx = list(_ActionEnum).index(action)
            else:
                # 优先尝试直接转换为 int（支持 numpy.int64 等），否则在 action_space 中查找
                try:
                    action_idx = int(action)
                except Exception:
                    try:
                        action_idx = list(self.env.action_space).index(action)
                    except ValueError:
                        raise ValueError(f"Cannot convert action to index: {action} (type={type(action)})")

            self.trainer.buffer.push(state.observation, action_idx, reward, next_state.observation, done)

            # --- 更新网络 ---
            self.trainer.optimize_model()

            state = next_state

            if render:
                self.env.render()

            if done:
                break

        return total_reward_set

    def train(self, episodes: int):
        # print(f'current dir: {os.getcwd()}')
        process_drawer = ProcessDrawer(
            title='Rewards',
            labels=['Total', 'survive', 'edge', 'hit', 'kill', 'behit', 'avoid'],
        )
        settings_server = Settings()
        begin_episode = Settings.begin_episode
        # ----- normal params -----

        # ----- this set of param is for teaching mode -----

        repeat_period = Settings.repeat_period


        with open(os.path.join('Agent', 'models', 'analysis', 'reward_record.csv'), 'w') as reward_file:

            reward_file.write("episode,epsilon,reward_total,survive,edge,hit,kill,behit,avoid\n")


            for episode in range(episodes):

                # epsilon = max(Settings.epsilon_end, Settings.epsilon_begin * Settings.epsilon_decay ** (episode % Settings.repeat_period))
                # epsilon = Settings.epsilon_begin - (Settings.epsilon_begin - Settings.epsilon_end) * ((episode % repeat_period) / repeat_period)
                epsilon = Settings.epsilon_end + 0.5 * (Settings.epsilon_begin - Settings.epsilon_end) * (1 + np.cos(np.pi * (episode % repeat_period) / repeat_period))
                assert 0 <= epsilon <= 1, f'invalid epsilon: {epsilon}'
                rewards = self.run_training_episode(render=Settings.render, epsilon=epsilon)

                print(f"Episode {episode} finished, epsilon={epsilon:.6f}, reward={rewards.value:.3f}")
                try:
                    reward_file.write(f"{episode},{epsilon},{rewards.value},{rewards.survive_reward},{rewards.edge_reward},{rewards.hit_reward},{rewards.kill_reward},{rewards.behit_reward},{rewards.avoid_reward}\n")
                    process_drawer.add_data(episode, rewards.value, rewards.survive_reward, rewards.edge_reward, rewards.hit_reward, rewards.kill_reward, rewards.behit_reward, rewards.avoid_reward)
                    process_drawer.update()
                except Exception as error:
                    print(f"Error writing reward to file: {error}")
                if episode % 50 == 0 and episode:
                    save_dir = os.path.join('Agent', 'models')
                    try:
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        save_path = os.path.join(save_dir, f'LinearNet_{episode + begin_episode}.pth')
                        torch.save(self.trainer.policy_net.state_dict(), save_path)
                        print(f'Model saved to {save_path}')
                    except Exception as e:
                        print(f"Error saving model: {e}")

    def optimize_network(self, steps: int=1000):
        """
        for offline training
        """
        begin_episode = Settings.begin_episode
        for step in tqdm(range(steps), desc="Optimizing network", unit="step"):
            self.trainer.optimize_model()
            if step % 10000 == 0 and step:
                save_dir = os.path.join('Agent', 'models')
                try:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = os.path.join(save_dir, f'LinearNet_{begin_episode}_offtrain_{step}.pth')
                    torch.save(self.trainer.policy_net.state_dict(), save_path)
                except Exception as e:
                    print(f"Error saving model: {e}")