from app.Agent.BaseBrain import BaseBrain
from app.Agent.DQNTrain import DQNTrainer
from app.Agent.RandomBrain import RandomBrain
from app.Agent.SmartBrain1 import SmartBrain1
from app.common.Settings import Settings
from app.env.Game import Game
from app.env.GameTrainer import GameTrainer
from app.env.TouhouEnv import TouhouEnv

if __name__ == '__main__':
    play_mode: bool = False
    if play_mode:
        game: Game = Game()
        agent: BaseBrain = RandomBrain()
        once_reward = game.run_episode(agent, max_steps=100000, render=True)
        print(f'reward in one game: {once_reward}')
    else:
        env = TouhouEnv(Settings)
        trainer = DQNTrainer(SmartBrain1())
        game: GameTrainer = GameTrainer(env, trainer)
        game.train(30000)