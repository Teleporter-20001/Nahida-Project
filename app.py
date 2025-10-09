import cProfile
import pstats
from collections.abc import Callable
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


def normal_play():
    # global env, game
    env = TouhouEnv(Settings)
    game: Game = Game(env)
    agent: BaseBrain = OptBrain(memory_len=8, predict_len=15, action_predict_len=15)
    once_reward = game.run_episode(agent, max_steps=100000, render=True)
    env._terminated = True
    env.close()
    del agent
    print(f'reward in one game: {once_reward}')


def train_model():
    # global env, game
    env = TouhouEnv(Settings)
    trainer = DRQNTrainer(MemoryBrainV2(be_teached=Settings.teach_mode), Settings.consider_bullets_num)
    game: GameTrainer = GameTrainer(env, trainer)
    # game.optimize_network(1200001)
    game.train(Settings.repeat_period + 1)
    env.close()


def profile_run(function: Callable[[], None]):
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        function()
    except KeyboardInterrupt:
        printgreen('Process interrupted by user')
    except Exception as e:
        printred(f'Process ended with error: {e}')
        raise
    printgreen('-----------------------finish--------------------------')
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(50)


if __name__ == '__main__':
    play_mode: bool = True
    performance_analyze = False
    if play_mode:
        if performance_analyze:
            profile_run(normal_play)
        else:
            normal_play()
    else:
        if performance_analyze:
            profile_run(train_model)
        else:
            train_model()