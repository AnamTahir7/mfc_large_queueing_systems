import numpy as np

from utils import save_to_file


class RNDMF:
    def __init__(self, env_creator, **kwargs):
        super().__init__()
        self.env_creator = env_creator
        self.kwargs = kwargs
        self.results_dir = self.kwargs.get('results_dir')

    def solve(self, env):
        file_path = '{}'.format(env.number_agents)
        agent_results_dir = self.results_dir.joinpath(file_path)
        agent_results_dir.mkdir(exist_ok=True)
        cum_reward_per_ep = np.zeros(env.episode_timesteps)
        actions = self.get_action(env)
        mc = 100
        for j in range(mc):
            print(j)
            all_rewards = []
            curr_obs = env.reset()
            for i in range(env.episode_timesteps):
                action_all_agents = actions
                curr_obs, joint_reward, _, _ = env.step(action_all_agents)
                all_rewards.append(joint_reward)  # per agent reward

            cum_reward_per_ep += np.cumsum(all_rewards)
            curr_output_json = {"no_agents": int(env.number_agents),
                                'number_queues': int(env.number_queues),
                                "reward": all_rewards
                                }
            save_to_file(curr_output_json, agent_results_dir, j)
        cum_reward_per_ep = cum_reward_per_ep / mc
        return cum_reward_per_ep, [], [], []

    def get_action(self, env):
        action = np.ones(len(env.total_agents_state_space) * env.d)
        action = action * (1 / env.d)
        action = np.reshape(action, (len(env.total_agents_state_space), env.d))
        return action
