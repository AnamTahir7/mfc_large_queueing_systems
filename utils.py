""" A place to store all utility functions """
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from ray.tune.logger import UnifiedLogger

def create_directory(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def get_datetime():
    """
    Returns current data and time as e.g.: '2019-4-17_21_40_56'
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_short_datetime():
    """
    Returns current data and time as e.g.: '0417_214056'
    """
    return datetime.now().strftime("%m%d_%H%M%S")


def ndarray_to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, np.int_):
        return int(x)
    return x


def recursive_conversion(d, func=ndarray_to_list):
    if isinstance(d, dict):
        return {k: recursive_conversion(v) for k, v in d.items()}
    return func(d)


def create_file_path(config):
    if config.startswith('MF'):
        script_path = Path(__file__).absolute().parent
        results_dir = script_path.joinpath('Results')
        results_dir.mkdir(exist_ok=True)
        run_time = get_short_datetime()
        file_name = '{config}-{t}'.format(config=config, t=run_time)
        output_dir = results_dir.joinpath(file_name)
        create_directory(output_dir)
    else:
        script_path = Path(__file__).absolute().parent
        results_dir = script_path.joinpath('Results')
        results_dir.mkdir(exist_ok=True)
        run_time = get_short_datetime()
        file_name = '{config}'.format(config=config)
        output_dir = results_dir.joinpath(file_name)
        create_directory(output_dir)
    return output_dir

def save_to_file(curr_output_json, output_dir, i):
    curr_output_json = recursive_conversion(curr_output_json)
    with open(os.path.join(output_dir, 'data_{:05}.json'.format(i)), 'w') as json_file:
        json.dump(curr_output_json, json_file, indent=4)


def save_params_to_file(params: dict, path: Path):
    _params = recursive_conversion(params)
    with path.open('w') as rf:
        json.dump(_params, rf, indent=4)

def custom_log_creator(results_dir):
    def logger_creator(config):
        return UnifiedLogger(config, str(results_dir), loggers=None)

    return logger_creator
