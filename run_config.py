import time
from utils import create_file_path, save_params_to_file

from envs_queues.queues_env import NQ_env
from solvers import rnd, d_jsq, d_sed, centralised_d_jsq_mf, centralised_rnd_mf,\
    rllib_mf_action_ppo,rllib_nagent_action_ppo, rllib_nagent_action_ppo_ps, jiq


def run_training(trained_policy, buffer_size, number_queues, service_rates_each_queues, service_rates_all_queues,
                 drop_penalty, delta_t, global_arr_rate, number_agents, testing_time_step, d, outer_loop_iter,
                 distr_srv, arr_rate_as_obs):

    if trained_policy != 'MF':
        results_dir = create_file_path(trained_policy)
    else:
        results_dir = create_file_path(trained_policy + f'{number_agents}'+'_'+f'{number_queues}''_'+f'{delta_t}')
    run_params = {
        'trained_policy': trained_policy,
        'buffer_size': buffer_size,
        'number_queues': number_queues,
        'service_rates_each_queues': service_rates_each_queues,
        'drop_penalty': drop_penalty,
        'delta_t': delta_t,
        'global_arr_rate': global_arr_rate,
        'no_agents': number_agents,
        'testing_time_step': testing_time_step,
        'd': d,
        'outer_loop_iter': outer_loop_iter,
        'distr_srv': distr_srv,
        'arr_rate_as_obs': arr_rate_as_obs
    }
    if trained_policy == 'MF':
        save_params_to_file(run_params, results_dir.joinpath('run_params.json'))      # use if you want to save datewise

    if trained_policy in ('MF','MFDJSQ', 'MFRND'):
        config = 'mf'
    elif trained_policy in ('PS', 'NA', 'DJSQ', 'RND', 'DSED', 'JIQ'):
        config = 'na_mq'
    else:
        raise NotImplementedError

    env = NQ_env(buffer_size, number_queues, service_rates_each_queues, service_rates_all_queues,
                 drop_penalty, delta_t, global_arr_rate, number_agents,
                 config, testing_time_step, d, outer_loop_iter, distr_srv, arr_rate_as_obs, trained_policy)

    def env_creator(env_config):
        return NQ_env(buffer_size, number_queues, service_rates_each_queues,
                      service_rates_all_queues, drop_penalty,
                      delta_t, global_arr_rate, number_agents, config,
                      testing_time_step, d, outer_loop_iter, distr_srv, arr_rate_as_obs, trained_policy)

    if trained_policy == 'NA':
        print("Current configuration:", config)
        print("N agent PPO solver")
        solver = rllib_nagent_action_ppo.RLLibSolver(env_creator, results_dir=results_dir)
        begin = time.time()
        avg_reward, min_reward, max_reward, trainer = solver.solve(env)
        print(time.time() - begin)

    elif trained_policy == 'PS':
        print("Current configuration:", config)
        print("N agent PPO solver with parameter sharing")
        solver = rllib_nagent_action_ppo_ps.RLLibSolver(env_creator, results_dir=results_dir)
        begin = time.time()
        avg_reward, min_reward, max_reward, trainer = solver.solve(env)
        print(time.time() - begin)

    elif trained_policy == 'MF':
        print("Current configuration:", config)
        print("MF PPO solver")
        # ray.init(local_mode=True)
        solver = rllib_mf_action_ppo.RLLibSolver(env_creator, results_dir=results_dir)
        begin = time.time()
        avg_reward, min_reward, max_reward, trainer = solver.solve(env)
        print(time.time() - begin)

    elif trained_policy == 'MFDJSQ':
        print("Current configuration:", config)
        print("MF-DJSQ PPO solver")
        solver = centralised_d_jsq_mf.DJSQMF(env_creator, results_dir=results_dir)
        begin = time.time()
        avg_reward, min_reward, max_reward, trainer = solver.solve(env)
        print(time.time() - begin)

    elif trained_policy == 'MFRND':
        print("Current configuration:", config)
        print("MF-RND PPO solver")
        solver = centralised_rnd_mf.RNDMF(env_creator, results_dir=results_dir)
        begin = time.time()
        avg_reward, min_reward, max_reward, trainer = solver.solve(env)
        print(time.time() - begin)

    elif trained_policy == 'DJSQ':
        print("Current configuration:", config)
        print("DJSQ solver")
        solver = d_jsq.DJSQ(env_creator, run_params, results_dir=results_dir)
        begin = time.time()
        avg_reward = solver.solve(env)
        print(time.time() - begin)

    elif trained_policy == 'DSED':
        print("Current configuration:", config)
        print("DSED solver")
        solver = d_sed.DSED(env_creator, run_params, results_dir=results_dir)
        begin = time.time()
        avg_reward = solver.solve(env)
        print(time.time() - begin)

    elif trained_policy == 'RND':
        print("Current configuration:", config)
        print("RND solver")
        solver = rnd.RND(env_creator, run_params, results_dir=results_dir)
        begin = time.time()
        avg_reward = solver.solve(env)
        print(time.time() - begin)

    elif trained_policy == 'JIQ':
        print("Current configuration:", config)
        print("JIQ solver")
        solver = jiq.JIQ(env_creator, run_params, results_dir=results_dir)
        begin = time.time()
        avg_reward = solver.solve(env)
        print(time.time() - begin)

    else:
        raise NotImplementedError