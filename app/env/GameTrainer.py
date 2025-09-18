import os

import pygame
import sys
import torch
import random

from app.common.ProcessDrawer import ProcessDrawer
from app.env.TouhouEnv import TouhouEnv
from app.Agent.DQNTrain import DQNTrainer

class GameTrainer:
    """Game loop for training with DQN"""

    def __init__(self, env: TouhouEnv, trainer: DQNTrainer):
        self.env = env
        self.trainer = trainer  # DQNTrainer

    def run_training_episode(self, max_steps: int = 100000, render: bool = True, epsilon=0.1):
        """Run one episode with training (DQN)"""
        state = self.env.reset()
        total_reward = 0.0

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
            total_reward += reward

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

        return total_reward

    def train(self, episodes: int):
        # print(f'current dir: {os.getcwd()}')
        process_drawer = ProcessDrawer()
        begin_episode = 2350
        epsilon_begin = 1.0
        epsilon_end = 0.1
        epsilon_decay = 0.993
        epsilon = epsilon_begin

        with open(os.path.join('Agent', 'models', 'reward_record.csv'), 'w') as reward_file:

            reward_file.write("episode,epsilon,reward\n")


            for episode in range(episodes):

                epsilon = max(epsilon_end, epsilon_begin * epsilon_decay ** (episode % 300))
                reward = self.run_training_episode(render=True, epsilon=epsilon)

                print(f"Episode {episode} finished, epsilon={epsilon}, reward={reward}")
                try:
                    reward_file.write(f"{episode},{epsilon},{reward}\n")
                    process_drawer.add_data(episode, reward)
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