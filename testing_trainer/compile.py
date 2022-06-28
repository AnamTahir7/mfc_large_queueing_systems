"""This file contains all the functions required to retrieve stored data in json format and average them"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import utils


class Compile:
    def __init__(self, path, expected_iter=100):

        if type(path) is list:
            _files = []
            for i in range(len(path)):
                _path = Path(path[i])
                _filepaths = _path.glob('**/*.json')
                _files.extend([f for f in _filepaths if not (str(f.stem).startswith('._') or 'average' in str(f)
                                                             or 'eps' in str(f) or 'comp_only_one_data' in str(f))])
            self._filepaths = _files
        else:
            self.path = Path(path)
            _filepaths = self.path.glob('**/*.json')
            self._filepaths = [f for f in _filepaths if not (str(f.stem).startswith('._') or 'average' in str(f)
                                                             or 'eps' in str(f))]
        self.expected_iter = expected_iter
        print("Total files found: {}, Files expected: {}".format(len(self._filepaths), self.expected_iter))
        self.tot_files = len(self._filepaths)

    @staticmethod
    def load(path: Path):
        # print(path)
        with path.open("r") as jf:
            return json.load(jf)

    def get_time_steps(self, delta_t):
        max_time = 500
        time_steps = np.round(max_time/delta_t).astype(int)
        return time_steps

    def compile(self, delta_t):
        mc = len(self._filepaths)
        accum_data = []
        individual_mc_cum_data = []
        no_events = self.get_time_steps(delta_t)
        all_cum_data = np.zeros(no_events)
        for path in tqdm(self._filepaths[:mc], desc='Compiling'):
            if path.stem[0] == 'd':
                _data = self.load(path)
                # cum_data = np.cumsum(_data['reward'])
                cum_data = np.cumsum(_data['reward'][0:no_events])    # for close NM
                # accum_data.extend(_data['reward'])
                accum_data.extend(_data['reward'][0:no_events])       # for close NM
                individual_mc_cum_data.append(cum_data[-1])
                all_cum_data += cum_data
        all_cum_data = all_cum_data / self.tot_files
        # get std
        std_cumsum = np.std(individual_mc_cum_data)
        # get confidence interval
        ci = (std_cumsum / np.sqrt(len(individual_mc_cum_data))) * 2
        #get sum
        sum_cumsum = np.sum(individual_mc_cum_data, axis=0)
        # get avg
        avg_cumsum = sum_cumsum / len(individual_mc_cum_data)
        return std_cumsum, avg_cumsum, ci, all_cum_data
        # return self, accum_data

if __name__ == '__main__':
    utils.figure_configuration_ieee_standard()
    ### for all delta_ts in one plot
    colors = ['#2ca02c','#1f77b4','#ff7f0e']
    delta_all = [1,2,3,4,5,6,7,8,9,10]
    config = [[10000,100],[40000,200], [160000,400], [360000, 600], [640000,800], [1000000, 1000]]
    curr_dir = ''
    output_dir = ''
    algos = [ 'MF','DJSQ','RND']
    for no_agents, no_queues in config:
        final_cum_reward_per_nm = []
        final_cum_reward_per_nm_positive = []
        std_cum_reward_per_nm = []
        ci_cum_reward_per_nm = []
        for algo in algos:
            final_cum_reward_per_algo = []
            final_cum_reward_per_algo_positive = []
            std_cum_reward_per_algo = []
            ci_cum_reward_per_algo = []
            for delta_t in delta_all:
                file_name = curr_dir + '/delta_t_' + str(delta_t) + '/' + algo + '/N_' + str(no_agents) + '/' + str(no_queues)
                avg_compiler = Compile(file_name, expected_iter=100)
                std_cumsum, cum_rew_last, ci, _ = avg_compiler.compile(delta_t)
                std_cumsum = std_cumsum / no_queues
                cum_rew_last = cum_rew_last / no_queues
                ci = ci / no_queues
                final_cum_reward_per_algo.append(cum_rew_last)
                final_cum_reward_per_algo_positive.append(cum_rew_last * -1)
                std_cum_reward_per_algo.append(std_cumsum)
                ci_cum_reward_per_algo.append(ci)
            final_cum_reward_per_nm.append(final_cum_reward_per_algo)
            final_cum_reward_per_nm_positive.append(final_cum_reward_per_algo_positive)
            std_cum_reward_per_nm.append(std_cum_reward_per_algo)
            ci_cum_reward_per_nm.append(ci_cum_reward_per_algo)
        colors = ['r','g','b', 'orange']

        for l in range(len(algos)):
            if algos[l] == 'DJSQ':
                label = 'JSQ(2)'
            elif algos[l] == 'MF':
                label = 'MF-NM'
            else:
                label = algos[l]
            plt.errorbar(np.arange(len(delta_all)), final_cum_reward_per_nm_positive[l], ci_cum_reward_per_nm[l],
                         capsize=5, label=label, color=colors[l])

        plt.xticks(ticks=np.arange(len(delta_all)), labels=delta_all)
        plt.ylabel('Total packets dropped')
        plt.xlabel(r'$\Delta t$')
        plt.title('M={}'.format( no_queues))
        plt.legend(loc='lower right')
        plt.savefig(output_dir + '/' + str(no_agents) + '_' + str(no_queues) + '_new8.pdf')
        plt.close()


    ### for individual delta ts
    delta_t_all = [1,2,3,4,5,6,7,8,9,10]
    for delta_t in delta_t_all:
        print(delta_t)
        final_last_cum_reward = []
        final_std_cum_reward = []
        final_ci_cum_reward = []
        final_last_cum_reward_positive = []
        final_cum_reward = []
        final_cum_reward_positive = []
        std_cum_reward = []
        ci_cum_reward = []

        # To compare for different N and M
        algos = ['MF', 'DJSQ', 'RND']
        plot_dir = Path('')
        all_dr = [
            ''
        ]
        config = [[10000, 100], [40000, 200], [160000, 400], [360000, 600], [640000, 800], [1000000, 1000]]
        fig_name = str(plot_dir) + '/'
        M = [100, 200, 400, 600, 800, 1000]
        j = 0
        for curr_algo in algos:
            print(curr_algo)
            final_cum_reward = []
            final_cum_reward_positive = []
            final_cum_reward_error = []
            std_cum_reward = []
            ci_cum_reward = []
            curr_path = all_dr[j]
            j += 1
            for n,m in config:
                print(n,m)
                path = curr_path
                avg_compiler = Compile(path, expected_iter=100)
                std_cumsum, cum_rew_last, ci, _ = avg_compiler.compile(delta_t)
                std_cumsum = std_cumsum / m
                cum_rew_last = cum_rew_last / m
                ci = ci / m
                final_cum_reward.append(cum_rew_last)
                pos_cum_reward = cum_rew_last * -1
                final_cum_reward_positive.append(pos_cum_reward)
                std_cum_reward.append(std_cumsum)
                ci_cum_reward.append(ci)

            final_last_cum_reward.append(final_cum_reward)
            final_last_cum_reward_positive.append(final_cum_reward_positive)
            final_std_cum_reward.append(std_cum_reward)
            final_ci_cum_reward.append(ci_cum_reward)
        colors = ['r','g','b', 'orange']
        for l in range(0,4):
            if algos[l] == 'DJSQ':
                label = 'JSQ(2)'
            elif algos[l] == 'MF':
                label = 'MF-NM'
            else:
                label = algos[l]
            plt.errorbar(np.array(M), final_last_cum_reward_positive[l], final_ci_cum_reward[l],
                         capsize=5, label=label, linewidth=2.5)
        labels = M
        plt.xticks(labels, labels)
        plt.ylabel('Average packet drops')
        plt.xlabel('M configuration')
        plt.title(r'$\Delta t={}$'.format(delta_t))
        plt.legend(loc='upper right')
        plt.savefig(fig_name + 'delta' + str(delta_t) + '1.pdf')
        plt.close()
