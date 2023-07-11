from enum import Enum

import gym
import numpy as np
import pathos.multiprocessing as pmp
import torch.multiprocessing as mp
import multiprocess.context as context

from typing import List, Tuple, Callable, Dict, TypeVar
from os.path import exists

from fabricatio_rl import FabricatioRL
from fabricatio_rl.interface_templates import SchedulingUserInputs


T = TypeVar('T')


class NoDaemonProcess(context.Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class NoDaemonPool(pmp.Pool):
    # noinspection PyMethodMayBeStatic
    def Process(self, *args, **kwds):
        return NoDaemonProcess(*args, **kwds)


def parallelize_heterogeneously(fns: List[Callable],
                                args: List[Tuple]):
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    n_threads = len(fns)
    assert n_threads == len(args)
    returns = []
    # pool = NoDaemonPool(n_threads)  # we need pthos mp for this
    pool = NoDaemonPool(n_threads)
    workers = [
        pool.apply_async(fns[i], args=args[i])
        for i in range(n_threads)]
    for w in workers:
        fn_return = w.get()
        returns.append(fn_return)
    pool.close()
    pool.join()
    return returns


def parallelize_homogeneously(fn: Callable, args: Tuple, n_threads: int):
    returns = []
    pool = NoDaemonPool(n_threads)
    workers = [
        pool.apply_async(fn, args=args)
        for _ in range(n_threads)]
    for w in workers:
        fn_return = w.get()
        returns.append(fn_return)
    pool.close()
    pool.join()
    return returns


def partition_benchmark_paths(instance_fnames, benchmark_directory):
    n_threads = int(mp.cpu_count() / 2)
    partitions = [([],) for _ in range(n_threads)]
    i = 0
    for f_name in sorted(instance_fnames):
        partitions[i % n_threads][0].append(f'{benchmark_directory}/{f_name}')
        i += 1
    return partitions, n_threads


def partition_list(full_list, n_threads):
    partitions = [[] for _ in range(n_threads)]
    i = 0
    for item in full_list:
        part_idx = i % n_threads
        partitions[part_idx].append(item)
        i += 1
    return partitions, n_threads


def import_tf(gpu=True):
    import os
    import logging
    import warnings
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # force computeation on CPU
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(3)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"     # computation on GPU 1
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(3)
        gpu_available = tf.test.is_gpu_available()
        print('\nCUDA capable GPU to be used for training found: {0}\n'
              ''.format(gpu_available))
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            # dynamically manage GPU memory
            tf.config.experimental.set_memory_growth(device, True)
    logging.getLogger('tensorflow').disabled = True
    return tf


def store_seeds(seeds, f_name):
    f = open(f_name, 'w')
    i = 1
    seed_str = ''
    lines = []
    for seed in seeds:
        if i != 0 and i % 20 != 0:
            seed_str += f'{seed} '
        else:
            seed_str += f'{seed}\n'
            lines.append(seed_str)
            seed_str = ''
        i += 1
    if i - 1 % 20 != 0:
        lines.append(seed_str)
    f.writelines(lines)
    f.close()


def read_seeds(filepath):
    if not exists(filepath):
        return np.array([])
    seed_file = open(filepath, 'r')
    all_seeds = []
    for line in seed_file:
        nums = [int(n) for n in line.split()]
        all_seeds += nums
    seed_file.close()
    return list(set(all_seeds))


class SchedulingSetup(Enum):
    JOBSHOP = 'jm10(100)'
    FLEXIBLEJOBSHOP = 'fjc10(100)'


def get_benchmark_env(paths: List[str],
                      env_name: str = 'fabrikatio-v0',
                      online: Tuple[int, int] = None,
                      logfile_path: str = '',
                      seed: int = 999,
                      precedence='Jm',
                      machine_capa=None,
                      inter_arrival_time='balanced') -> FabricatioRL:
    sis: List[SchedulingUserInputs] = []
    for path in paths:
        sis.append(
            SchedulingUserInputs(
                path=path,
                n_jobs=online[1],
                operation_types=precedence,
                operation_precedence=precedence,
                machine_capabilities=machine_capa,
                inter_arrival_time=inter_arrival_time
            ))
    env_args = dict(
        scheduling_inputs=sis,
        logfile_path=logfile_path,
        seeds=[seed]
    )
    return make_env(env_name, env_args)


def make_env(env_name: str, env_args: Dict[str, T]):
    if env_name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[env_name]
    gym.register(
        id=env_name,
        entry_point='fabricatio_rl:FabricatioRL',
        kwargs=env_args
    )
    environment: FabricatioRL = gym.make(env_name)
    return environment
