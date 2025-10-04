import cProfile
import pstats
from typing import cast

from app.Agent.Brains.BaseBrain import BaseBrain
from app.Agent.Brains.MemoryBrainV2 import MemoryBrainV2
from app.Agent.Brains.OptBrain import OptBrain
from app.Agent.Trainers.DRQNTrain import DRQNTrainer
from app.common.Settings import Settings
from app.env.Game import Game
from app.env.GameTrainer import GameTrainer
from app.env.TouhouEnv import TouhouEnv
from app.common.utils import printred, printgreen

if __name__ == '__main__':
    play_mode: bool = True
    if play_mode:
        # settings = Settings()
        env = TouhouEnv(Settings)
        game: Game = Game(env)
        agent: BaseBrain = OptBrain(memory_len=10, predict_len=30)
        once_reward = game.run_episode(agent, max_steps=100000, render=True)
        env._terminated = True
        env.close()
        del agent
        print(f'reward in one game: {once_reward}')
    else:
        performance_analyze = False
        profiler: cProfile.Profile = cast(cProfile.Profile, None)
        if performance_analyze:
            profiler = cProfile.Profile()
            profiler.enable()

        try:
            # settings: Settings = Settings()
            env = TouhouEnv(Settings)
            trainer = DRQNTrainer(MemoryBrainV2(be_teached=Settings.teach_mode), Settings.consider_bullets_num)
            game: GameTrainer = GameTrainer(env, trainer)
            # game.optimize_network(1200001)
            game.train(Settings.repeat_period + 1)
            env.close()
        except KeyboardInterrupt:
            printgreen('Training interrupted by user')
        except Exception as e:
            printred(f'Training ended with error: {e}')
            raise

        printgreen('-----------------------finish--------------------------')
        if performance_analyze:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            stats.print_stats(40)