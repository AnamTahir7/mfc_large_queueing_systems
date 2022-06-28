import json
import os
import pickle5 as pickle
from argparse import ArgumentParser
from ast import literal_eval
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune import register_env

from envs_queues.queues_env import NQ_env
from utils import save_to_file


class EvaluatePolicy:
    def __init__(self, trainer, results_dir, mf_eval, **run_parameters):
        """

        :param trainer: preloaded trainer
        :param results_dir: a directory with parameters and checkpoints
        :param run_parameters: preloaded run parameters for the environment
        """
        self.params = run_parameters
        self.trained_policy = self.params.pop('trained_policy', 'MF')
        if mf_eval:
            self.config = 'mf'
        else:
            self.config = 'na_mq'
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.trainer = trainer
        self.equal_time = True


    @classmethod
    def from_checkpoint(cls, checkpoint_path, mf_eval, results_dir_suffix='eval'):

        chk = checkpoint_path.split('checkpoint_')[1]
        results_dir_suffix = results_dir_suffix+chk

        checkpoint_path = Path(checkpoint_path)
        run_params_path = checkpoint_path.parent.parent.joinpath('run_params.json')
        with run_params_path.open('r') as jf:
            run_params = json.load(jf)

        trained_policy = run_params.pop('trained_policy', None)
        if 'service_rates_all_queues' in run_params:
            run_params['service_rates_all_queues'] = np.array(run_params['service_rates_all_queues'])

        with checkpoint_path.parent.parent.joinpath('params.pkl').open('rb') as pf:
            config = pickle.load(pf)

        _chkpnt_file = str(checkpoint_path.joinpath(checkpoint_path.stem.replace('_', '-')))
        config['env_config'] = run_params
        config['num_workers'] = 1

        ray.init(local_mode=True)
        register_env("EVAL", lambda x: cls.create_env(x))
        agent = ppo.PPOTrainer(config, env='EVAL')

        agent.restore(_chkpnt_file)
        run_params['trained_policy'] = trained_policy
        return cls(agent, checkpoint_path.parent.joinpath(results_dir_suffix), mf_eval, **run_params)

    @staticmethod
    def create_env(params):
        env = NQ_env(**params)
        return env

    def extract_h_policy(self, action_trainer, env, no_agents_test):
        action_queue_access = action_trainer[env.idx]
        return action_queue_access

    def get_time_steps(self, delta_t):
        max_time = 500
        time_steps = np.round(max_time/delta_t).astype(int)
        return time_steps

    def single_run_test(self, env, no_agents_test, equal_time, mf_eval=False):
        all_rewards = []
        obs = env.reset()
        if equal_time:
            time_steps = self.get_time_steps(self.params.get('delta_t'))
        else:
            time_steps = self.params.get('testing_time_step', 500)
        for i in range(time_steps):
            if self.trained_policy == 'MF':
                if mf_eval:
                    action_trainer = self.trainer.compute_single_action(obs)  # for mf
                    action = env.normalise_action(action_trainer).numpy()
                else:
                    obs_mf = obs[0], obs[2]
                    action_trainer = self.trainer.compute_single_action(obs_mf)  # for mf
                    action_trainer = env.normalise_action(action_trainer)
                    action = self.extract_h_policy(action_trainer, env, no_agents_test)
            elif self.trained_policy == 'PS':
                action = []
                agent_states = obs[1]
                for j in range(no_agents_test):
                    obss = obs[0], np.array([agent_states[j]]), obs[2]
                    action_per_policy = self.trainer.compute_action(obss, policy_id='shared_policy')  # for ps
                    action.append(action_per_policy)
            elif self.trained_policy == 'NA':
                action = []
                agent_states = obs[1]
                for j in range(no_agents_test):
                    obss = obs[0], np.array([agent_states[j]]), obs[2]
                    action_per_policy = self.trainer.compute_action(obss,
                                                                    policy_id='policy_{0}'.format(j))  # for n agent
                    action.append(action_per_policy)
            else:
                raise NotImplementedError(f'Policy is not implemented for {self.params.get("trained_policy")}')

            obs, joint_reward_np, _, _ = env.step(action)
            all_rewards.append(joint_reward_np)

        cum_reward = np.cumsum(all_rewards)
        return cum_reward, all_rewards

    def run_test(self, equal_time, agent_nos_test, queue_nos_test, mf_eval):
        no_mc = 100
        no_agents_test_curr = agent_nos_test
        no_queues_test_curr = queue_nos_test
        print(no_agents_test_curr, no_queues_test_curr)
        if equal_time:
            sum_ep_rewards = np.zeros(self.get_time_steps(self.params.get('delta_t')))
            cum_reward_per_ep = np.zeros(self.get_time_steps(self.params.get('delta_t')))
        else:
            sum_ep_rewards = np.zeros(self.params.get('testing_time_step', 500))
            cum_reward_per_ep = np.zeros(self.params.get('testing_time_step', 500))

        current_params = deepcopy(self.params)
        current_params['config'] = self.config
        current_params['no_agents'] = no_agents_test_curr
        service_rates_each_queues = self.params.get('service_rates_each_queues')
        queues_distr = np.array(self.params.get('distr_srv')) * no_queues_test_curr
        if len(queues_distr) > 0:
            service_rates_all_queues = np.repeat(np.array(service_rates_each_queues), queues_distr.astype(int))
        else:
            service_rates_all_queues = np.repeat(np.array(service_rates_each_queues), no_queues_test_curr)
        current_params['number_queues'] = no_queues_test_curr
        current_params['service_rates_all_queues'] = service_rates_all_queues
        test_env = self.create_env(current_params)

        # output directory
        file_path = 'N_{}/{}'.format(no_agents_test_curr, no_queues_test_curr)
        agent_results_dir = self.results_dir.joinpath(file_path)
        agent_results_dir.mkdir(parents=True,exist_ok=True)
        # print(current_params)
        for i in range(no_mc):
            print(i)
            cum_reward, ep_rewards = self.single_run_test(test_env, no_agents_test_curr, equal_time, mf_eval)
            sum_ep_rewards += ep_rewards
            cum_reward_per_ep += cum_reward
            curr_output_json = {"no_agents": int(no_agents_test_curr),
                                "no_queues": int(no_queues_test_curr),
                                "global_arr_rate": current_params['global_arr_rate'],
                                "service_rates_all_queues": current_params['service_rates_all_queues'],
                                "delta_t": current_params['delta_t'],
                                "reward": ep_rewards
                                }
            # saving to file
            save_to_file(curr_output_json, agent_results_dir, i)

        # Take average of cum_reward over number of mc sims
        avg_cum_ep_reward = cum_reward_per_ep / no_mc

        plt.plot(np.arange(len(avg_cum_ep_reward)), avg_cum_ep_reward, 'g', label='Avg Cumulative reward')
        plt.title('{a} agents, {q} queues'.format(a=no_agents_test_curr, q=no_queues_test_curr))
        plt.legend()
        plt.savefig(agent_results_dir.joinpath('{}agents_avg_cum_reward_testing.pdf'.format(no_agents_test_curr)))
        plt.close()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('nr_agents', type=int)
    parser.add_argument('nr_queues', type=int)
    parser.add_argument('path', type=str)
    parser.add_argument('mf_eval', type=str)
    parser.add_argument('equal_time', type=str)
    args = parser.parse_args()
    equal_time = literal_eval(args.equal_time)      # if you want to run for equal time for all delta_t
    mf_eval = literal_eval(args.mf_eval)
    if equal_time:
        if mf_eval:
            results_dir_suffix = 'eval_eq_mf'
        else:
            results_dir_suffix = 'eval_eq_non_same_d'
    else:
        results_dir_suffix = 'eval_non_same_d'

    current_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.path
    ep = EvaluatePolicy.from_checkpoint(output_dir, mf_eval, results_dir_suffix=results_dir_suffix)
    ep.run_test(equal_time, agent_nos_test=args.nr_agents, queue_nos_test=args.nr_queues, mf_eval=mf_eval)
