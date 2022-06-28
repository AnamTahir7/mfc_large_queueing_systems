import subprocess

if __name__ == '__main__':
    child_processes = []
    delta_t_all = [1,2,3,4,5,6,7,8,9,10]
    solver_all = ['DJSQ', 'RND', 'DSED']
    nm_config = [[160000,400], [360000, 600], [640000,800], [1000000, 1000]]
    import multiprocessing
    num_cores = multiprocessing.cpu_count()

    diff_srv_speeds_all = [1]
    inf_queue = [0]
    for solver in solver_all:
        for no_agents, no_queues in nm_config:
            for delta_t in delta_t_all:
                for diff_srv_speeds in diff_srv_speeds_all:
                    for inf_q in inf_queue:
                        p = subprocess.Popen(['python', './main.py', solver, str(no_agents), str(no_queues),
                                              str(delta_t), str(diff_srv_speeds),  str(inf_q)])
                        child_processes.append(p)
                        if len(child_processes) > num_cores-1:
                            for p in child_processes:
                                p.wait()
                            child_processes = []

    for p in child_processes:
        p.wait()
