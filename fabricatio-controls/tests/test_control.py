from fabricatio_controls.comparison_utils import get_benchmark_env
from os import listdir
from fabricatio_controls.cp import FixedCPPlanSequencing, CPControl


def test_cp_sequencing():
    basedir = '../benchmarks/jssp_test_instances'
    for f_name in listdir(basedir):
        opt_result = int(f_name.split('__')[-1][:-4].split('_')[0][2:])
        env = get_benchmark_env([f'{basedir}/{f_name}'])
        control = CPControl(FixedCPPlanSequencing(100))
        end_state = control.play_game(env)
        print(f'Instance {f_name} CP results: \n{end_state.system_time}')
        assert int(end_state.system_time) == opt_result


# test_cp_sequencing()
