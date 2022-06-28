import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from gym import spaces
from gym.spaces import Box


def make_multi_agent(env_name_or_creator):
    """Convenience wrapper for any sigle-agent env to be converted into MA.
    Agent IDs are int numbers starting from 0 (first agent).
    Args:
        env_name_or_creator (Union[str, Callable[]]: String specifier or
            env_maker function.
    Returns:
        Type[MultiAgentEnv]: New MultiAgentEnv class to be used as env.
            The constructor takes a config dict with `num_agents` key
            (default=1). The reset of the config dict will be passed on to the
            underlying single-agent env's constructor.
    Examples:
         # >>> # By gym string:
         # >>> ma_cartpole_cls = make_multi_agent("CartPole-v0")
         # >>> # Create a 2 agent multi-agent cartpole.
         # >>> ma_cartpole = ma_cartpole_cls({"num_agents": 2})
         # >>> obs = ma_cartpole.reset()
         # >>> print(obs)
         # ... {0: [...], 1: [...]}
         # >>> # By env-maker callable:
         # >>> ma_stateless_cartpole_cls = make_multi_agent(
         # ...    lambda config: StatelessCartPole(config))
         # >>> # Create a 2 agent multi-agent stateless cartpole.
         # >>> ma_stateless_cartpole = ma_stateless_cartpole_cls(
         # ...    {"num_agents": 2})
    """

    class MultiEnv(MultiAgentEnv):
        def __init__(self, config):
            self.env = env_name_or_creator(config)
            obs = self.env.observation_space
            self.observation_space = {i: spaces.Tuple((obs[0], Box(obs[1].low[0], obs[1].high[0], shape=(1,),
                                                                   dtype=np.int64), obs[2])) for i in range(self.env.number_agents)}

            self.action_space = self.env.action_space

        def reset(self):
            obs_state = self.env.reset()
            agents_state = obs_state[1]
            return {i: (obs_state[0], np.array([agents_state[i]]), np.array(obs_state[2])) for i in range(self.env.number_agents)}

        def step(self, action_dict):
            obs, rew, dones, info = {}, {}, {}, {}
            action_list = []
            for i, action in action_dict.items():
                action_list.append(action)
            joint_obs_np, joint_reward_np, done, _ = self.env.step(action_list)
            agents_state = joint_obs_np[1]
            dones["__all__"] = done
            rew = {i: joint_reward_np for i in range(self.env.number_agents)}
            obs = {i: (joint_obs_np[0], np.array([agents_state[i]]), np.array(joint_obs_np[2])) for i in range(self.env.number_agents)}
            return obs, rew, dones, info

    return MultiEnv
