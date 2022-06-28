from pathlib import Path
from utils import create_file_path, save_params_to_file

import numpy as np

from utils import save_to_file


class DSED:
    def __init__(self, env_creator, run_params, **kwargs):
        super().__init__()
        self.env_creator = env_creator
        self.kwargs = kwargs
        self.results_dir = self.kwargs.get('results_dir')
        self.run_params = run_params

    def get_time_steps(self, delta_t):
        max_time = 500
        time_steps = np.round(max_time/delta_t).astype(int)
        return time_steps

    def solve(self, env):
        delta_t = str(np.int(env.delta_t))
        output_dir = self.results_dir.joinpath(delta_t)
        output_dir.mkdir(exist_ok=True)
        file_path = 'N_{}'.format(env.number_agents)
        agent_results_dir = output_dir.joinpath(file_path)
        agent_results_dir.mkdir(exist_ok=True)
        save_params_to_file(self.run_params, agent_results_dir.joinpath('run_params.json'))
        queue_path = '{}'.format(env.number_queues)
        agent_results_dir = agent_results_dir.joinpath(queue_path)
        agent_results_dir.mkdir(exist_ok=True)
        episode_timesteps = self.get_time_steps(env.delta_t)
        cum_reward_per_ep = np.zeros(episode_timesteps)
        mc = 100
        for j in range(mc):
            print(j)
            all_rewards = []
            curr_obs = env.reset()
            for i in range(episode_timesteps):
                curr_buffer_fillings = self.reshape_obs(curr_obs[1], env)
                action_all_agents = self.get_action(curr_buffer_fillings, env)
                curr_obs, joint_reward, _, _ = env.step(action_all_agents)
                all_rewards.append(joint_reward)  # per agent reward

            cum_reward_per_ep += np.cumsum(all_rewards)
            curr_output_json = {"no_agents": int(env.number_agents),
                                'number_queues': int(env.number_queues),
                                "reward": all_rewards
                                }
            save_to_file(curr_output_json, agent_results_dir, j)
        cum_reward_per_ep = cum_reward_per_ep/mc
        return cum_reward_per_ep

    def reshape_obs(self, obs, env):
        return np.reshape(obs, (env.number_agents, env.d))

    def get_action(self, curr_buffer_fillings, env):
        actions = np.zeros([env.number_agents, env.d])
        if env.number_diff_server_speeds == 2:      #SED
            xx = curr_buffer_fillings.copy()
            xx = xx.astype(int)
            mask_lt_2 = xx[:, 2] < 2

            mask_eq_01 = xx[:, 0] == xx[:, 1]
            mask_neq_01 = ~mask_eq_01

            _mask_lt2_eq01 = mask_lt_2 & mask_eq_01
            actions[_mask_lt2_eq01, np.random.choice(2, sum(_mask_lt2_eq01))] = 1

            _mask_lt2_neq01 = mask_lt_2 & mask_neq_01
            _min_index = np.argmin(xx[_mask_lt2_neq01, :2], axis=1)
            actions[_mask_lt2_neq01, _min_index] = 1

            mask_eq_2 = xx[:, 2] == 2      #[fast,slow]
            _mask_eq2_eq01 = mask_eq_2 & mask_eq_01
            actions[_mask_eq2_eq01, np.random.randint(env.d)] = 1

            _mask_eq2_neq01 = mask_eq_2 & mask_neq_01
            _min_index = np.argmin(xx[_mask_eq2_neq01, :2]/[max(env.alpha), min(env.alpha)], axis=1)  # first fats second slow
            actions[_mask_eq2_neq01, _min_index] = 1

            mask_eq_3 = xx[:, 2] == 3       #[slow,fast]
            _mask_eq2_eq01 = mask_eq_3 & mask_eq_01

            actions[_mask_eq2_eq01, np.random.randint(env.d)] = 1
            _mask_eq2_neq01 = mask_eq_3 & mask_neq_01
            _min_index = np.argmin(xx[_mask_eq2_neq01, :2]/[min(env.alpha), max(env.alpha)], axis=1)  #first slow second fast
            actions[_mask_eq2_neq01, _min_index] = 1

        else:       #JSQ
            mask = curr_buffer_fillings[:,0] == curr_buffer_fillings[:,1]
            if any(mask):
                rnd = np.random.choice(env.d, len(actions[mask]))
                actions[mask, rnd] = 1
                min_idx = np.argmin(curr_buffer_fillings[~mask],axis=1)
                actions[~mask, min_idx] = 1
            else:
                row_idx = np.arange(len(actions))
                min_idx = np.argmin(curr_buffer_fillings, axis=1)  # no equal so take min for all
                actions[row_idx, min_idx] = 1

        return actions


