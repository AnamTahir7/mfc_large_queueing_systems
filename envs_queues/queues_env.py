from itertools import product
from collections import Counter
from itertools import product
import math

import gym
import numpy as np
import numpy_indexed as npi
import pandas as pd
import scipy as sp
import scipy.stats as st
import torch
import torch.nn.functional as f
from gym import spaces
from gym.spaces import Box
from numpy.random import default_rng
from ray.rllib.utils.spaces.simplex import Simplex


class NQ_env(gym.Env):
    def __init__(self, buffer_size=5, number_queues=2, service_rates_each_queues=np.array([1]),
                 service_rates_all_queues=np.array([3, 3], dtype=int), drop_penalty=1, delta_t=np.float(0.5),
                 global_arr_rate=np.array([5.0, 3.0], dtype=np.float32), no_agents=2, config='na_mq',
                 testing_time_step=500, d=2, outer_loop_iter=500, distr_srv=[0.5, 0.5], arr_rate_as_obs=True,
                 trained_policy='MF'):
        self.trained_policy = trained_policy
        self.arr_rate_as_obs = arr_rate_as_obs
        self.B_cap = 6
        self.eps = None
        self.number_agents = no_agents
        self.tot_agents = no_agents
        self.delta_t = delta_t
        self.distr_srv = np.array(distr_srv)
        self.service_rates_each_queues = service_rates_each_queues
        self.service_rates_all_queues = service_rates_all_queues    # homogenous servers
        self.alpha = self.service_rates_each_queues
        self.drop_penalty = drop_penalty
        self.number_queues = number_queues
        self.buffer_size = buffer_size
        self.total_size = self.buffer_size + 1  # +1 for empty queue state
        self.half_tot_size = self.total_size // 2
        self.config = config
        self.device = 'cpu'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.discount = 0.95
        self.done = False
        self.infos = {}
        self.episode_timesteps = testing_time_step
        self.outer_loop_iter = outer_loop_iter
        self.t = 0
        self.reward = 0
        ############
        self.d = d
        self.total_agents_state_space = list(product(np.arange(self.total_size), repeat=self.d))
        self.total_agents_state_space = [list(e) for e in self.total_agents_state_space if e]
        t = self.total_agents_state_space
        xx = np.vstack([t])
        self.total_agent_state_space_wo_server_speed = xx
        yy = np.zeros(xx.shape[0], dtype=int)
        action_states = list(product(np.arange(self.B_cap), repeat=self.d))
        self.action_state_space = [list(e) for e in action_states if e]
        tt = np.vstack([xx.T, yy]).T.tolist()
        self.total_agents_state_space = tt      #[[0,0,0],[0,1,0],[0,2,0],...,[5,5,0],[0,0,1],[0,1,1]...[5,5,3]]
        self.global_arr_rate = global_arr_rate
        if self.config == 'na_mq':
            self.action_space = Simplex(shape=(self.d,), dtype=np.float32)
            self.observation_space = spaces.Tuple((Box(0, 1, shape=(self.total_size,), dtype=np.float32),
                                                   Box(0, self.buffer_size, shape=(self.number_agents * self.d,),
                                                       dtype=int),
                                                   Box(0, np.inf, shape=(1,), dtype=np.float32)))
        elif self.config == 'mf':
            if self.arr_rate_as_obs:
                self.action_space = Box(0, 1, shape=(len(self.total_agents_state_space) * self.d,),
                                            dtype=np.float32)   # for ray==1.8.0
                self.observation_space = spaces.Tuple((Box(0, 1, shape=(self.total_size,),dtype=np.float32),
                                                           Box(0, np.inf, shape=(1,), dtype=np.float32)))
            else:
                self.action_space = Box(0, 1, shape=(len(self.total_agents_state_space) * self.d,),
                                        dtype=np.float32)   # for ray==1.8.0
                self.observation_space = Box(0, 1, shape=(self.total_size,), dtype=np.float32)

        else:
            raise NotImplementedError

    def get_agent_state_dist(self, nu, total_agents_state_space): #nu=[[0,0],[1,0],[2,0],...,[5,0],[0,1],[2,1],..,[5,1]]
        mu = np.zeros(len(total_agents_state_space))
        for i in range(len(mu)):
            idx = list(total_agents_state_space[i][0:self.d])
            mu[i] = np.prod(nu[idx])
        return mu

    def get_queue_filling_emp_distr(self):
        nu = np.zeros(self.total_size)
        cnt = Counter(np.array(self.curr_state.numpy(), dtype=int))
        for i in range(len(nu)):
            nu[i] = cnt[i]
        nu = nu / self.number_queues
        return nu

    def reset(self):
        '''
        :return: empty queues - state 0
                and reset time
        '''
        curr_obs = []

        if self.config == 'mf':
            nu = np.zeros(self.total_size)            #[[0,0],[1,0],[2,0],...,[5,0],[0,1],[2,1],..,[5,1]]
            nu[0] = 1    # initially all queues empty
            if self.arr_rate_as_obs:
                self.nu = nu
                self.mu = self.get_agent_state_dist(self.nu, self.total_agents_state_space)
                self.curr_state = self.nu
                self.arr_rate, self.eps = self.get_mmpp_arr_rate(self.eps, self.global_arr_rate)
                curr_obs = self.nu, np.array([self.arr_rate])
            else:
                self.nu = nu
                self.mu = self.get_agent_state_dist(self.nu, self.total_agents_state_space)
                self.curr_state = self.nu
                self.arr_rate, self.eps = self.get_mmpp_arr_rate(self.eps, self.global_arr_rate)
                curr_obs = self.nu

        elif self.config == 'na_mq':
            self.curr_state = torch.zeros(self.number_queues).to(self.device)   # state of all queues, initially empty
            self.arr_rate, self.eps = self.get_mmpp_arr_rate(self.eps, self.global_arr_rate)
            nu = self.get_queue_filling_emp_distr()
            self.queue_allocated_each_agent, queue_alloc_state = self.sample_d_queue_access_each_agent()
            if self.trained_policy == 'JIQ':
                self.agents_curr_state = np.array(queue_alloc_state).reshape(self.number_agents, self.d)
                curr_obs = self.curr_state, self.queue_allocated_each_agent
            else:
                self.agents_curr_state = np.array(queue_alloc_state).reshape(self.number_agents, self.d)
                curr_obs = nu, queue_alloc_state, np.array([self.arr_rate])

        self.done = False
        self.t = 0
        self.reward = 0
        self.infos = {}
        self.total_job_drops = []
        return curr_obs

    def sample_d_queue_access_each_agent(self):
        # out of all queues allocate d queues randomly to each agent
        if self.number_agents > 1 and self.number_queues > 1:
            rnd = np.random.choice(self.number_queues, size=[self.number_agents, self.d])
            mask = rnd[:, 0] == rnd[:, 1]
            rnd[mask, 1] = (rnd[mask, 1] + np.random.randint(self.number_queues / 2)) % self.number_queues
            agents_state = np.array(self.curr_state)[rnd].flatten().tolist()
            sampled_queue_allocated_each_agent = rnd
            agn = np.array(agents_state, dtype=int).reshape(self.number_agents, self.d)
            idx = npi.indices(np.array(self.total_agent_state_space_wo_server_speed), agn)
        else:
            sampled_queue_allocated_each_agent = [0,0]
            agents_state = [0,0]
            idx = 0
        self.idx = idx
        return sampled_queue_allocated_each_agent, agents_state

    def normalise_action(self, action_np):
        action_np = np.array(action_np)
        action_np = action_np.reshape([len(self.total_agents_state_space), self.d])
        action = torch.from_numpy(action_np).type(torch.float32)
        # squashing not needed for ray version == 1.8, uncomment for lower versions
        # # action = ((torch.tanh(action) + 1.0) / 2.0)     # for [-inf,inf] range with tanh squashing
        if self.config == 'na_mq':
            action = action.T.divide(torch.max(action.abs(),axis=1)[0]).T   # for [-inf,inf] range without tanh squashing
            action = (action + 1) / 2
        action += 1e-10
        action = f.normalize(action, p=1)
        return action

    def step(self, action_np):
        '''
        :param action (list): will come as joint action from all agents or in mf case a policy for each agent state
                                also caters for when both agents send to the same queue
        :return: observation (object): each agent's observation of the current environment
                                        (individual obs from each queue)
            reward (float) : amount of reward returned after previous joint action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        '''
        if self.config == 'mf':
            if len(action_np) == len(self.total_agents_state_space)*self.d:     # non dirichlet policy
                action = self.normalise_action(action_np)
            else:
                action = torch.from_numpy(action_np).type(torch.float32)        #dirichlet policy
            state_action_dist = self.get_mf_dist(action)
            joint_reward, joint_obs, next_state = self.simulate_state_action_dist(state_action_dist)

        elif self.config == 'na_mq':
            joint_reward, joint_obs, next_state = self.simulate(action_np)

        if self.t >= self.episode_timesteps - 1:
            self.infos["observation"] = joint_obs
            self.done = True
        self.curr_state = next_state
        self.t += 1

        self.arr_rate, self.eps = self.get_mmpp_arr_rate(self.eps, self.global_arr_rate)
        joint_obs_np = np.array(joint_obs.tolist())
        joint_reward_np = np.float64(joint_reward)

        if self.config == 'mf':
            self.nu = joint_obs
            self.mu = self.get_agent_state_dist(self.nu, self.total_agents_state_space)
            if self.arr_rate_as_obs:
                joint_obs_np = self.nu, np.array([self.arr_rate])
            else:
                joint_obs_np = self.nu

        elif self.config == 'na_mq':
            self.queue_allocated_each_agent, queue_alloc_state = self.sample_d_queue_access_each_agent()
            nu = self.get_queue_filling_emp_distr()
            self.agents_curr_state = np.array(queue_alloc_state).reshape(self.number_agents, self.d)
            if self.trained_policy == 'JIQ':
                joint_obs_np = self.curr_state, self.queue_allocated_each_agent
            else:
                joint_obs_np = nu, queue_alloc_state, np.array([self.arr_rate])
        return joint_obs_np, joint_reward_np, self.done, self.infos

    def sample_queue_for_each_agent(self, state_action_dist):
        action_np = []
        if self.config == 'mf':
            for i in range(self.number_agents):
                alloc_queue = self.curr_state[self.queue_allocated_each_agent[i]].tolist()
                state_idx = self.get_state_index(alloc_queue)
                action_probs_curr_agent = state_action_dist[state_idx].numpy()
                norm_action_probs_curr_agent = np.divide(action_probs_curr_agent, np.sum(action_probs_curr_agent,
                                                                                        dtype=np.float), dtype=np.float)
                action_curr_agent = np.random.choice(self.d, p=norm_action_probs_curr_agent)
                action_np.append(action_curr_agent)
        if self.config == 'na_mq':
            for i in range(self.number_agents):
                alloc_queue_filling = self.curr_state[self.queue_allocated_each_agent[i]].tolist()
                action_probs_curr_agent = state_action_dist[i].numpy()
                norm_action_probs_curr_agent = np.divide(action_probs_curr_agent, np.sum(action_probs_curr_agent,
                                                                                        dtype=np.float), dtype=np.float)
                action_curr_agent = np.random.choice(self.d, p=norm_action_probs_curr_agent)
                action_np.append(action_curr_agent)
        return action_np

    def get_agent_state_index(self, agent_index):
        return self.get_state_index(self.queue_allocated_each_agent[agent_index])

    def get_state_index(self, state):
        return self.total_agents_state_space.index(state)

    def get_state(self, state_index):
        return self.total_agents_state_space[state_index]

    def get_mf_dist(self, action):
        if self.config == 'na_mq':
            agent_state_dist = torch.zeros(len(self.total_agents_state_space))
            for i in range(self.number_agents):
                curr_queue_filling = self.curr_state[self.queue_allocated_each_agent[i]]
                idx = self.get_state_index(curr_queue_filling.tolist())
                agent_state_dist[idx] += 1
            agent_state_dist = agent_state_dist / self.number_agents
        else:
            agent_state_dist = torch.Tensor(self.mu)

        return agent_state_dist.unsqueeze(1) * action

    @staticmethod
    def get_mmpp_arr_rate(curr_eps, global_arr_rate):
        T = [[0.8, 0.2], [0.5, 0.5]]   # TODO get an accurate Transition matrix
        if curr_eps is None: # use ini distr
            ini_p = 0.5
            ini_eps = st.bernoulli(ini_p).rvs()
            if ini_eps == 0:
                arr_rate = global_arr_rate[0]
            else:
                arr_rate = global_arr_rate[1]
            curr_eps = ini_eps
        else:
            if curr_eps == 0:
                T_p = T[curr_eps]
                p = T_p[1]
                eps_t = st.bernoulli(p).rvs()
                if eps_t == 0:
                    arr_rate = global_arr_rate[0]
                else:
                    arr_rate = global_arr_rate[1]
            else:
                T_p = T[curr_eps]
                p = T_p[0]
                eps_t = st.bernoulli(p).rvs()
                if eps_t == 0:
                    arr_rate = global_arr_rate[0]
                else:
                    arr_rate = global_arr_rate[1]
            curr_eps = eps_t
        return arr_rate, curr_eps

    def get_new_arr_rate(self, state_action_dist):
        mmpp_arr_rate = self.arr_rate
        norm_state_action_dist = f.normalize(state_action_dist, p=1).numpy()
        new_arr_rates = np.zeros(self.total_size)
        one_vec_ident = np.identity(self.total_size)
        for x in range(len(new_arr_rates)):  # for each x in nu
            one_vector = one_vec_ident[x]
            sum_u_all_s = np.sum(norm_state_action_dist * one_vector[self.total_agent_state_space_wo_server_speed], axis=1)
            sum_s_u_all_s = np.sum(self.mu * sum_u_all_s)
            new_arr_rates[x] = mmpp_arr_rate * sum_s_u_all_s   #MMPP arr rate
        new_arr_rates = np.divide(new_arr_rates, self.nu, out=np.zeros_like(new_arr_rates), where=self.nu != 0)
        return new_arr_rates

    def generate_A_matrix(self, new_arr_rate_x, alpha, total_size):
        A = np.zeros([total_size,total_size])
        for x in range(total_size):
            if x == 0:  #only arrivals
                A[x, x+1] = new_arr_rate_x
            elif x == total_size-1:   # only dept
                A[x, x-1] = alpha
            else:
                A[x, x+1] = new_arr_rate_x
                A[x, x-1] = alpha
            A[x, x] -= sum(A[x])
        return A.transpose()

    def simulate_state_action_dist(self, state_action_dist):
        '''
        return reward, next state and obs together from this simulator by using the ode solution
        '''
        r = 0
        new_arr_rates = self.get_new_arr_rate(state_action_dist)
        P_0 = np.identity(self.total_size)  # for each x P is a vector of probs
        zz = np.zeros(self.total_size)
        zz = np.expand_dims(zz, axis=1)
        P_D_concat_0 = np.hstack((P_0, zz))
        topA = np.zeros([self.total_size, 1])
        botA = np.zeros([1, self.total_size - 1])
        nu = np.zeros(self.total_size)
        for x in range(self.total_size):
            A = self.generate_A_matrix(new_arr_rates[x], self.alpha, self.total_size)
            B = np.block([
                [A, topA],
                [botA, new_arr_rates[x] * np.ones([1, 1]), np.zeros([1, 1])]
                ])
            P_D_concat_0_x = P_D_concat_0[x]
            B_exp = sp.linalg.expm(B * self.delta_t)
            P_D_concat_x = B_exp @ P_D_concat_0_x
            nu += self.nu[x] * P_D_concat_x[:-1]
            r -= self.nu[x] * P_D_concat_x[-1]
            norm_nu = nu
        return r, norm_nu, norm_nu

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

    def simulate(self, action):
        '''
        return reward, next state and obs together from this simulator for N agent case
        :param action:
        :param curr_state:
        :return: total reward to all for now, can also be factored, next state and obs
        '''

        total_dropped_jobs_all_queues = 0
        arr_rate = self.arr_rate * sum(self.service_rates_all_queues)
        arr_rate_to_each_agent = arr_rate / self.number_agents
        tot_N_access_queue = np.zeros(self.number_queues)
        arr_rate_all_queues = np.zeros(self.number_queues)
        if self.trained_policy == 'JIQ':
            qu, cnt = np.unique(action, return_counts=True)
            if len(qu) < self.number_queues:
                cnt_all = np.zeros(self.number_queues)
                cnt_all[qu.astype(int)] = cnt
                cnt = cnt_all
            arr_rate_all_queues = cnt * arr_rate_to_each_agent

        else:
            if isinstance(action, list):
                if isinstance(action[0], np.ndarray):
                    af = np.array(action).flatten() * arr_rate_to_each_agent
                else:
                    af = torch.stack(action).flatten().detach().numpy() * arr_rate_to_each_agent
            else:
                af = np.array(action).flatten() * arr_rate_to_each_agent
            qa = self.queue_allocated_each_agent
            qf = qa.flatten()
            qu, cnt = np.unique(qf, return_counts=True)
            tot_N_access_queue[qu] = cnt
            xx = np.vstack([qf, af])
            xx= pd.DataFrame(xx.T, columns=['qf', 'af'])
            res = xx.groupby(by='qf').agg({'af': 'sum'})
            arr_rate_all_queues[res.index.astype(int)] = res.af
            assert(sum(tot_N_access_queue) == (self.number_agents * self.d))
        dept_rate_all_queues = self.service_rates_all_queues
        sum_exp_all_queues = arr_rate_all_queues + dept_rate_all_queues
        prob_arr_all_queues = arr_rate_all_queues / sum_exp_all_queues
        timer_all_queues = np.zeros(self.number_queues)
        curr_state_all_queues = self.curr_state.cpu().numpy()
        while np.any(timer_all_queues <= self.delta_t):     # Do for all qeueus at the same time
            when_event_occurs_each_queue = st.expon(scale=np.reciprocal(sum_exp_all_queues)).rvs()
            timer_all_queues += when_event_occurs_each_queue
            time_remaining_which_queues = np.where(timer_all_queues <= self.delta_t)[0]
            if len(time_remaining_which_queues) > 0:
                which_event_occured_in_each_queue = st.binom.rvs(1, prob_arr_all_queues[time_remaining_which_queues])
                if np.isscalar(which_event_occured_in_each_queue):
                    which_event_occured_in_each_queue = np.array(which_event_occured_in_each_queue)
                which_event_occured_in_each_queue[which_event_occured_in_each_queue == 0] = -1
                """ Add and remove packets depending on event """
                curr_state_all_queues[time_remaining_which_queues] = \
                    curr_state_all_queues[time_remaining_which_queues] + which_event_occured_in_each_queue
                """ Remove packets that were dropped """
                dropped_packets = np.maximum(np.zeros_like(curr_state_all_queues),
                                             curr_state_all_queues - self.buffer_size)
                total_dropped_jobs_all_queues += np.sum(dropped_packets)
                """ Cannot drop below or above minimum or maximum of buffer size """
                curr_state_all_queues = np.clip(curr_state_all_queues, 0, self.buffer_size)
        next_state = torch.from_numpy(curr_state_all_queues)
        joint_reward = self.get_reward(next_state, total_dropped_jobs_all_queues)  # rewrd based on next state and drops
        joint_obs = next_state  # fully observable right now
        return joint_reward, joint_obs, next_state