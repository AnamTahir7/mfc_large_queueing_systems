import copy
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from utils import save_to_file, create_file_path, save_params_to_file


class JSQ:
    def __init__(self, buffer_size=5, number_queues=2, service_rates_all_queues=np.array([3, 3], dtype=np.int64),
                 drop_penalty=1, delta_t=np.float(0.5), global_arr_rate=np.array([5, 3], dtype=int),
                 testing_time_step=10, no_agents=2, d=2):

        # We need to distinguish between states and observations now

        self.drop_penalty = drop_penalty
        self.no_queues = number_queues
        self.no_agents = no_agents
        self.buffer_size = buffer_size
        self.total_size = self.buffer_size + 1  # +1 for empty queue state
        self.action_space = np.arange(self.no_queues)
        self.curr_obs = np.zeros(self.no_queues)
        self.curr_state = np.zeros(self.no_queues)
        self.delta_t = delta_t
        self.global_arr_rate = global_arr_rate
        self.discount = 0.95
        self.service_rates_all_queues = service_rates_all_queues
        self.episode_return = 0
        self.episode_timesteps = testing_time_step
        self.reward = 0
        self.discrete_state_space_n = self.total_size ** self.no_queues
        self.queue_allocated_each_agent = []
        self.agent_state = []
        ######
        self.d = d
        self.total_agents_state_space = [item for item in product(np.arange(self.total_size), repeat=self.d)]
        self.total_agents_state_space = [list(e) for e in self.total_agents_state_space if e]
        self.eps = None

    def reset(self):
        self.curr_state = np.zeros(self.no_queues)
        self.queue_allocated_each_agent, _ = self.sample_queue_access_each_agent()

    def sample_queue_access_each_agent(self):
        tot_combs_queues = range(len(self.total_agents_state_space))
        sampled_queue_allocated_each_agent = []
        agents_state = np.random.choice(tot_combs_queues, size=self.no_agents)  # uniformly at random
        for i in range(self.no_agents):
            sampled_queue_allocated_each_agent.append(self.total_agents_state_space[agents_state[i]])
        return sampled_queue_allocated_each_agent, agents_state

    def simulate_next_state(self, action, arr_each_agent, non_zero_arr_agents_idx):
        sample_tot_dept_each_queue = st.poisson.rvs(self.service_rates_all_queues *
                                                    self.delta_t)

        curr_state = copy.copy(self.curr_state)
        # remove departures
        next_state = np.maximum(np.zeros(self.no_queues), curr_state - sample_tot_dept_each_queue)
        for i in range(len(non_zero_arr_agents_idx)):
            agent_idx = non_zero_arr_agents_idx[i]
            arrs_curr_agent = arr_each_agent[agent_idx]
            action_curr_agent = action[i]
            next_state[action_curr_agent] += arrs_curr_agent
        x = next_state - self.buffer_size
        total_dropped_jobs_all_queues = np.sum(x[x > 0])
        if total_dropped_jobs_all_queues > 0:
            next_state = np.clip(next_state, 0, self.buffer_size)
        self.curr_state = next_state
        joint_reward = self.get_reward(next_state, total_dropped_jobs_all_queues)

        self.queue_allocated_each_agent, self.agent_state = self.sample_queue_access_each_agent()
        joint_obs = next_state  # fully observable right now

        return joint_reward, joint_obs, next_state

    def find_action(self, non_zero_arr_agents_idx):
        action = np.zeros(len(non_zero_arr_agents_idx))
        curr_state = self.curr_state
        for i in range(len(non_zero_arr_agents_idx)):  # do only for agents with arrivals
            curr_agent_idx = non_zero_arr_agents_idx[i]
            queue_alloc_curr_agent = self.queue_allocated_each_agent[curr_agent_idx]
            if len(queue_alloc_curr_agent) == 1:  # access to only one queue
                action[i] = queue_alloc_curr_agent[0]
            else:  # get curr state for allocated queues to this agent
                state_alloc_queues = curr_state[queue_alloc_curr_agent]
                if len(set(state_alloc_queues)) == 1:  # choose randomly
                    action[i] = np.random.choice(queue_alloc_curr_agent)
                else:  # choose shorter queue
                    shorter_queue_idx = np.where(state_alloc_queues == min(state_alloc_queues))[0]
                    if len(set(shorter_queue_idx)) == 1:
                        action[i] = queue_alloc_curr_agent[shorter_queue_idx[0]]
                    else:  # choose randomly
                        random_chosen_idx = np.random.choice(shorter_queue_idx)
                        action[i] = queue_alloc_curr_agent[random_chosen_idx]

        return action.astype(np.int)

    def calculate_arrivals_for_each_agent(self):
        sample_tot_arrs = st.poisson.rvs(self.global_arr_rate * self.delta_t)
        arr_each_agent = []
        if sample_tot_arrs > 0:
            sample_agents_for_arrival = np.random.choice(np.asarray(self.no_agents), size=sample_tot_arrs).tolist()
            for i in range(self.no_agents):
                arr_each_agent.append(sample_agents_for_arrival.count(i))
        non_zero_arr_agents_idx = np.nonzero(arr_each_agent)[0]

        return arr_each_agent, non_zero_arr_agents_idx

    def single_run_test(self):
        self.reset()
        all_rewards = []
        for i in range(self.episode_timesteps):
            arr_each_agent, non_zero_arr_agents_idx = self.calculate_arrivals_for_each_agent()
            action = self.find_action(non_zero_arr_agents_idx)
            joint_reward, joint_obs, next_state = self.simulate_next_state(action, arr_each_agent,
                                                                           non_zero_arr_agents_idx)
            all_rewards.append(joint_reward)
        return all_rewards

    def test(self):

        results_dir = create_file_path('jsq')

        run_params = {
            'trained_policy': 'jsq',
            'buffer_size': self.buffer_size,
            'number_queues': self.no_queues,
            'service_rates_all_queues': self.service_rates_all_queues,
            'drop_penalty': self.drop_penalty,
            'delta_t': self.delta_t,
            'global_arr_rate': self.global_arr_rate,
            'testing_time_step': self.episode_timesteps,
        }
        save_params_to_file(run_params, results_dir.joinpath('run_params.json'))
        no_mc = 500
        agent_nos_test = np.array([1, 2, 4, 6, 8, 10, 15, 20, 50, 70, 90, 100, 200])
        for no_agents_test_curr in agent_nos_test:
            print(no_agents_test_curr)

            self.no_agents = no_agents_test_curr
            sum_ep_rewards = np.zeros(self.episode_timesteps)
            cum_reward_per_ep = np.zeros(self.episode_timesteps)

            # output directory
            file_path = '{}'.format(no_agents_test_curr)
            agent_results_dir = results_dir.joinpath(file_path)
            agent_results_dir.mkdir(exist_ok=True)
            for i in range(no_mc):
                ep_rewards = self.single_run_test()
                cum_reward = np.cumsum(ep_rewards)
                sum_ep_rewards += ep_rewards
                cum_reward_per_ep += cum_reward
                curr_output_json = {"no_agents": int(self.no_agents),
                                    "reward": ep_rewards
                                    }
                # saving to file
                save_to_file(curr_output_json, agent_results_dir, i)

            # Take average of cum_reward over number of mc sims
            avg_cum_ep_reward = cum_reward_per_ep / no_mc
            plt.plot(np.arange(len(avg_cum_ep_reward)), avg_cum_ep_reward, 'g', label='Avg Cumulative reward')
            plt.title('{} agents'.format(no_agents_test_curr))
            plt.legend()
            plt.savefig(agent_results_dir.joinpath('{}agents_avg_cum_reward_testing.pdf'.format(no_agents_test_curr)))
            plt.close()

    def get_reward(self, curr_state, jobs_dropped):
        '''
        penalty for dropped jobs and current buffer filling status
        :return: joint reward
        '''
        reward = 0
        buffer_penalty = 0
        drop_penalty = self.drop_penalty * jobs_dropped
        reward -= drop_penalty + buffer_penalty
        return reward
