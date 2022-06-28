import subprocess

# use enviornment py38

if __name__ == '__main__':
    child_processes = []

    nm_config = [[10000, 100], [40000, 200], [160000, 400], [360000, 600], [640000, 800], [1000000, 1000]]
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    mf_eval = 'False'
    equal_time = 'True'    # if you want to run for equal time for all delta_t
    all_dir = [
        ''
        ]
    for no_agents, no_queues in nm_config:
        for curr_dir in all_dir:
            p = subprocess.Popen(['python', '-m', 'testing_trainer.evaluate_policy', str(no_agents), str(no_queues),
                                  curr_dir, mf_eval, equal_time])
            child_processes.append(p)
            if len(child_processes) > num_cores-1:
                for p in child_processes:
                    p.wait()
                child_processes = []

    for p in child_processes:
        p.wait()
