from pathlib import Path
from utils import create_file_path, save_params_to_file

import numpy as np

from utils import save_to_file


class JIQ:
    def __init__(self, env_creator, run_params, **kwargs):
        super().__init__()
        self.env_creator = env_creator
        self.kwargs = kwargs
        self.results_dir = self.kwargs.get('results_dir')
        self.run_params = run_params
        self.number_agents = self.run_params['no_agents']
        self.number_queues = self.run_params['number_queues']
        self.i_queue = [[] for _ in range(self.number_agents)]   # list of queue at each scheduler
        self.d = self.run_params['d']

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
                curr_queue_access = self.reshape_obs(curr_obs[1], env.number_agents, env.d, env.number_diff_server_speeds)
                curr_queue_states = np.array(curr_obs[0])
                self.allocate_i_queues(curr_queue_states, i)
                action_all_agents = self.get_action(curr_queue_access, env)
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

    def reshape_obs(self, obs, number_agents, d, number_diff_server_speeds):
        if number_diff_server_speeds == 2:
            return np.reshape(obs, (number_agents, d+1))[:,:2]
        else:
            return np.reshape(obs, (number_agents, d))

    def allocate_i_queues(self, curr_queue_states, j):
        self.prev_idle_queues = [[] for _ in range(self.number_agents)]
        # allocate from the previous idle queues to current agents
        if j == 0:  # random allocation
            for i in range(self.number_agents):
                self.i_queue[i].append(np.random.choice(self.number_queues))
        else:
            for i in range(self.number_agents):
                if self.prev_idle_queues[i]:
                    self.i_queue[i].append(self.prev_idle_queues[i][0])

        # allocate idle queues randomly to agents based on current state to be used at the next time step
        idle_servers = np.where(curr_queue_states ==0)[0]
        for i in range(len(idle_servers)):
            random_scheduler = np.random.choice(self.number_agents)
            self.prev_idle_queues[random_scheduler].append(i)

    def get_action(self, curr_queue_access, env):
        # for non-empty I-queue pop FIFO form there and for others do random
        actions = np.zeros([env.number_agents])
        for i in range(self.number_agents):
            if self.i_queue[i]:
                idle_queue = self.i_queue[i].pop(0)
                actions[i] = idle_queue
            else:
                rand_queue = np.random.choice(curr_queue_access[i])
                actions[i] = rand_queue

        return actions


