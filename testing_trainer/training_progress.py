from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def training_progress_func(data_path):
    _dir = data_path
    algos = ['MF']
    all_rewards = []
    all_timestamps = []
    len_timestamps = []
    for i in range(len(algos)):
        pth = _dir.joinpath('progress.csv')
        df = pd.read_csv(pth)
        curr_reward = df['episode_reward_mean'][2:].values

        ts = df['timesteps_total'][2:].values
        all_rewards.append(curr_reward)
        all_timestamps.append(ts)
        for i in range(len(algos)):
            plt.plot(all_timestamps[i], all_rewards[i], label=algos[i])
        plt.legend()
        plt.ylabel('Episode Reward')
        plt.xlabel('Simulation Timesteps')
        plt.title('MF training curve')
        plt.savefig(_dir.joinpath('training_progress_timesteps.pdf'))
        plt.close()


        all_timestamps = []
        curr_timestamp = df['timestamp'][2:].values
        curr_relative_time = (curr_timestamp - curr_timestamp[0])/3600
        len_timestamps.append(len(curr_timestamp))
        all_rewards.append(curr_reward)
        all_timestamps.append(curr_relative_time)
        for i in range(len(algos)):
            plt.plot(all_timestamps[i], all_rewards[i], label=algos[i])
        plt.legend()
        plt.ylabel('Episode Reward')
        plt.xlabel('Time [hours]')
        plt.title('MF training curve')
        plt.savefig(_dir.joinpath('training_progress_time.pdf'))
        plt.close()


if __name__ == '__main__':
    _dir = Path('')
    training_progress_func(_dir)