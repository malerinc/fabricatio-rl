from time import time

from fabricatio_controls.comparison_utils import get_benchmark_env
from os import listdir
from fabricatio_controls.heuristics import (
    SPT, LPT, LOR, MOR, SRPT, LRPT, LTPO, MTPO, EDD)
from fabricatio_controls.heuristics import (
    SimulationSearch, SimulationSearchControl)


def test_sim_search_autoplay():
    basedir = '../benchmarks/jssp_test_instances'
    for f_name in listdir(basedir)[:2]:
        opt_result = int(f_name.split('__')[-1][:-4].split('_')[0][2:])
        env = get_benchmark_env(
            paths=[f'{basedir}/{f_name}'],
            logfile_path='')
        control = SimulationSearchControl(SimulationSearch(
            None, seq_optimizers=[SPT(), LPT(), LOR(), MOR(), SRPT(),
                                  LRPT(), LTPO(), MTPO(), EDD()],
            tra_optimizers=None, criterium='makespan', p_completion=1))
        ts = time()
        end_state = control.play_game(env)
        print(f'Instance {f_name} runtime with SimSearch autoplay, env skips: '
              f'{time() - ts}')
        ts = time()
        control.autoplay = False
        end_state_alt1 = control.play_game(env)
        print(f'Instance {f_name} runtime w/o autoplay, env skips: : '
              f'{time() - ts}')
        ts = time()
        # control.autoplay = True
        # control.optimizer.n_threads = 2
        # end_state_alt = control.play_game(env)
        # print(f'Instance {f_name} runtime autoplay 2 threads: '
        #       f'{time() - ts}\n\n')
        env.set_core_rou_autoplay(False)
        env.set_core_seq_autoplay(False)
        ts = time()
        end_state_alt2 = control.play_game(env)
        print(f'Instance {f_name} runtime with autoplay, no env skips: '
              f'{time() - ts}')
        control.autoplay = True
        ts = time()
        end_state_alt3 = control.play_game(env)
        print(f'Instance {f_name} runtime with autoplay, no env skips: '
              f'{time() - ts}')
        ts = time()
        assert end_state_alt1.system_time == end_state.system_time
        assert end_state_alt2.system_time == end_state_alt1.system_time
        assert end_state_alt3.system_time == end_state_alt2.system_time
        assert end_state.matrices.op_duration.sum() == 0
        assert end_state_alt1.matrices.op_duration.sum() == 0
        assert end_state_alt2.matrices.op_duration.sum() == 0
        assert end_state_alt3.matrices.op_duration.sum() == 0


# TODO: test that when env skips all, SimSearch need not skip anything


if __name__ == '__main__':  # do not forget this guard for parallel procs!!!
    test_sim_search_autoplay()
