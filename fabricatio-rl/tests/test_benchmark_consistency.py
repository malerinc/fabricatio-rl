import numpy as np

from time import time
from os import listdir
from pathlib import Path

from tests.t_helpers import init_env

# from testfixtures import TempDirectory

BENCHMARK_DIR = Path(__file__).resolve().parent / 'jssp_test_instances'
SOLUTION_DIR = Path(__file__).resolve().parent / 'jssp_test_solutions'


def test_jssp_simulation_benchmark_consistency():
    logpath = ''
    env_seed = np.random.randint(low=1, high=1000, size=1).tolist()
    f_names = []
    if BENCHMARK_DIR != '':  # TODO: if its a valid dir...
        for file_name in listdir(BENCHMARK_DIR):
            sln_base_name, sln_extension = file_name.split(".")
            instance_path = f'{BENCHMARK_DIR}/{file_name}'
            sln_path = f'{SOLUTION_DIR}/{sln_base_name}_sln.{sln_extension}'
            env = init_env(env_seed, instance_path, sln_path, logpath=logpath)
            t_start = time()
            state_repr, done, curr_makespan = env.reset(), False, 0
            env.set_core_seq_autoplay(False)
            n_steps = 0
            print(f"Testing results on {file_name}")
            while not done:
                state_repr, curr_makespan, done, _ = env.step(0)
                n_steps += 1
            t = time() - t_start
            print(f"Testing results: {curr_makespan}")
            lower_bound = int(((file_name.split('__')[-1]).split('_'))[0][2:])
            assert lower_bound == int(curr_makespan)
            env.close()
