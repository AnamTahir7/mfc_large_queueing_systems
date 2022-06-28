import json
import os
import pickle5 as pickle
from pathlib import Path
import time
import shutil
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune import register_env

from envs_queues.queues_env import NQ_env
from testing_trainer.training_progress import training_progress_func

class Restore:
    @classmethod
    def restore_training_from_checkpoint(cls, checkpoint_path):

        checkpoint_path = Path(checkpoint_path)
        run_params_path = checkpoint_path.parent.parent.joinpath('run_params.json')
        with run_params_path.open('r') as jf:
            run_params = json.load(jf)
        run_params['config'] = 'mf'
        run_params.pop('trained_policy', None)

        with checkpoint_path.parent.parent.joinpath('params.pkl').open('rb') as pf:
            config = pickle.load(pf)

        _chkpnt_file = str(checkpoint_path.joinpath(checkpoint_path.stem.replace('_', '-')))
        config['env_config'] = run_params
        config['num_workers'] = 8

        ray.init(local_mode=True)
        register_env("RESTORE", lambda x: cls.create_env(x))
        agent = ppo.PPOTrainer(config, env='RESTORE')

        agent.restore(_chkpnt_file)

        logs = []
        avg_reward = []
        min_reward = []
        max_reward = []
        min_rew = None
        min_rew_thr = -0.1
        after_retore_results_dir = checkpoint_path.parent.parent.joinpath('Restored')
        after_retore_results_dir.mkdir(exist_ok=True)
        best_results_dir = after_retore_results_dir.joinpath('best_result')
        best_results_dir.mkdir(exist_ok=True)
        begin = time.time()
        a = True
        i = 0
        while a:
            i += 1
            print(i)
            log = agent.train()  # TODO compatibility?
            time_elapsed = time.time() - begin
            print(f'step: {i}; time elapsed: {time_elapsed/60:.2f} mins')
            logs.append(log)

            if not np.isnan(log['episode_reward_mean']):
                if min_rew is None:
                    min_rew = log['episode_reward_mean']
                    agent.save(checkpoint_dir=str(best_results_dir))
                elif min_rew - log['episode_reward_mean'] < min_rew_thr:
                    min_rew = log['episode_reward_mean']
                    shutil.rmtree(best_results_dir.joinpath(os.listdir(best_results_dir)[0]), ignore_errors=True)
                    agent.save(checkpoint_dir=str(best_results_dir))


            print('mean reward', log['episode_reward_mean'])
            avg_reward.append(log['episode_reward_mean'])
            min_reward.append(log['episode_reward_min'])
            max_reward.append(log['episode_reward_max'])

            if i % 20 == 0:
                training_pth = best_results_dir.parent.parent
                training_progress_func(training_pth)

            if i == 1 or i % 49 == 0:
                checkpoint_path = agent.save(checkpoint_dir=after_retore_results_dir)

    @staticmethod
    def create_env(params):
        env = NQ_env(**params)
        return env


if __name__ == '__main__':

    output_dir = ''
    Restore.restore_training_from_checkpoint(output_dir)
