import os
import shutil
import time

import numpy as np
import ray
from ray.rllib.agents.ppo import ppo

from envs_queues.ma_env_wrapper import make_multi_agent
from utils import custom_log_creator


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
        ma_env_cls = make_multi_agent(self.env_creator)
        ma_env = ma_env_cls({"num_agents": 1})
        ma_env.reset()
        obs_space = ma_env.observation_space
        act_space = ma_env.action_space

        policies = {"shared_policy": (None, obs_space[0], act_space, {})}
        trainer = ppo.PPOTrainer(
            env=ma_env_cls,
            logger_creator=custom_log_creator(self.kwargs.get('results_dir')),
            config={
                'framework': 'tf',
                "num_workers": 1,
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (lambda agent_id: "shared_policy"),
                },
            },
        )

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
            if i % 49 == 0 or i == 499:
                checkpoint_path = trainer.save(checkpoint_dir=str(self.kwargs.get('results_dir')))

        return avg_reward, min_reward, max_reward, trainer
