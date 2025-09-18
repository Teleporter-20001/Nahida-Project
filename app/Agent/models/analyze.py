import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    raw_data = pd.read_csv('reward_record.csv')
    episodes = raw_data['episode'].values
    rewards = raw_data['reward'].values

    plt.plot(episodes, rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward over Episodes')
    plt.show()