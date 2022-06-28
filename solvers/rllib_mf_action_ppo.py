import os
import shutil
import time

import numpy as np
import ray
from ray.rllib.agents.ppo import ppo
from ray.tune import register_env

from utils import custom_log_creator

from testing_trainer.training_progress import training_progress_func

class RLLibSolver():
    """
    Approximate deterministic solutions using Rllib
    """

    def __init__(self, env_creator, **kwargs):
        super().__init__()
        self.env_creator = env_creator
        self.kwargs = kwargs

    def solve(self, env, **kwargs):
        ray.init(local_mode=True)
        # ray.init(local_mode=False, ignore_reinit_error=True)
        register_env("MFG-v0", self.env_creator)
        trainer = ppo.PPOTrainer(env="MFG-v0", logger_creator=custom_log_creator(self.kwargs.get('results_dir')),
                                 config={
                                     'framework': 'tf',
                                     "num_workers": 1,
                                     'multiagent': {
                                         'policies': {
                                             'default_policy': (None, env.observation_space, env.action_space, {})
                                         }
                                     }
                                 })
        logs = []
        avg_reward = []
        min_reward = []
        max_reward = []
        min_rew = None
        min_rew_thr = -0.1
        best_results_dir = self.kwargs.get('results_dir').joinpath('best_result')
        best_results_dir.mkdir(exist_ok=True)
        begin = time.time()
        a = True
        i = 0
        while a:
            i += 1
            print(i)
            log = trainer.train()
            time_elapsed = time.time() - begin
            print(f'step: {i}; time elapsed: {time_elapsed/60:.2f} mins')
            logs.append(log)

            if not np.isnan(log['episode_reward_mean']):
                if min_rew is None:
                    min_rew = log['episode_reward_mean']
                    trainer.save(checkpoint_dir=str(best_results_dir))
                elif min_rew - log['episode_reward_mean'] < min_rew_thr:
                    min_rew = log['episode_reward_mean']
                    shutil.rmtree(best_results_dir.joinpath(os.listdir(best_results_dir)[0]), ignore_errors=True)
                    trainer.save(checkpoint_dir=str(best_results_dir))


            print('mean reward', log['episode_reward_mean'])
            avg_reward.append(log['episode_reward_mean'])
            min_reward.append(log['episode_reward_min'])
            max_reward.append(log['episode_reward_max'])

            if i % 20 == 0:
                training_pth = best_results_dir.parent
                training_progress_func(training_pth)

            if i % 49 == 0 or i == env.outer_loop_iter-1:
                checkpoint_path = trainer.save(checkpoint_dir=str(self.kwargs.get('results_dir')))

        return avg_reward, min_reward, max_reward, trainer
