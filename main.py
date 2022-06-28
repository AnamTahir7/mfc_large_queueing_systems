import numpy as np
import os
from run_config import run_training

from argparse import ArgumentParser


def run(solver, number_agents, number_queues, delta_t):
    testing_time_step = 500
    outer_loop_iter = 1
    job_drop_penalty = 1
    d = 2
    distr_srv = []
    service_rates_each_queues = 1
    service_rates_all_queues = np.full([number_queues], service_rates_each_queues)
    global_arr_rate = np.array([0.9*service_rates_each_queues, 0.6 * service_rates_each_queues], dtype=np.float32)

    buffer_size = 5     # finite capacity queue size
    arr_rate_as_obs = True
    run_training(solver, buffer_size, number_queues, service_rates_each_queues, service_rates_all_queues,
                 job_drop_penalty, delta_t, global_arr_rate, number_agents, testing_time_step, d,
                 outer_loop_iter, distr_srv, arr_rate_as_obs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('solver', help='name of the solver', choices=['MF', 'RND', 'DJSQ', 'MFDJSQ', 'MFRND'])
    parser.add_argument('nr_agents', type=int)
    parser.add_argument('nr_queues', type=int)
    parser.add_argument('delta_t', type=float)
    parser.add_argument('--gpu', help='0 or 1 or leave it for all GPUS', type=int)

    args = parser.parse_args()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.gpu is not None and cvd is not None:
        if args.gpu < 0 or args.gpu > 1:
            raise TypeError(f'this {args.gpu}')

        os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'
    run(solver=args.solver, number_agents=args.nr_agents, number_queues=args.nr_queues, delta_t=args.delta_t)

    # Reset the original value of the CUDA Device
    if cvd is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cvd
    # DJSQ 20000 200 1