import itertools

import numpy as np


class ExactSolver:
    def __init__(self, **kwself):
        super().__init__(**kwself)
        self.kwself = kwself

    def solve(self, env):
        Vs = []
        Qs = []
        curr_V = [0 for _ in range(env.discrete_state_space_n)]

        for t in range(env.episode_timesteps).__reversed__():
            Q_t = []
            for x in range(env.discrete_state_space_n):
                Q_tx = []
                for u in list(itertools.product(*[range(x) for x in env.action_space.nvec[0]])):
                    transition_probs, expected_reward = env.discrete_state_transition_probs_and_reward(t, x, u)
                    Q_tx.append(expected_reward + np.vdot(curr_V, transition_probs))
                Q_t.append(Q_tx)
            curr_V = [np.max(Q_t[x]) for x in range(len(curr_V))]

            Vs.append(curr_V)
            Qs.append(Q_t)

        Vs.reverse()
        Qs.reverse()
