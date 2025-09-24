from app.Agent.BaseBrain import BaseBrain
from app.Agent.DQNTrain import DQNTrainer
from app.Agent.MemoryBrain import MemoryBrain
from app.Agent.RandomBrain import RandomBrain
from app.Agent.SmartBrain1 import SmartBrain1
from app.common.Settings import Settings
from app.env.Game import Game
from app.env.GameTrainer import GameTrainer
from app.env.TouhouEnv import TouhouEnv

import cProfile
import pstats

if __name__ == '__main__':
    play_mode: bool = False
    if play_mode:
        settings = Settings()
        env = TouhouEnv(settings)
        game: Game = Game(env)
        agent: BaseBrain = MemoryBrain()
        once_reward = game.run_episode(agent, max_steps=100000, render=True)
        env._terminal = True
        env.close()
        print(f'reward in one game: {once_reward}')
    else:
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            settings: Settings = Settings()
            env = TouhouEnv(Settings)
            trainer = DQNTrainer(MemoryBrain(be_teached=Settings.teach_mode))
            game: GameTrainer = GameTrainer(env, trainer)
            # game.optimize_network(1200001)
            game.train(2)
            env.close()
        except KeyboardInterrupt:
            print('Training interrupted by user')
        except Exception as e:
            print(f'Training ended with error: {e}')

        print('-----------------------finish--------------------------')
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(50)