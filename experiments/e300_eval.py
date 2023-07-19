from commons import ControlLoader
from commons import load_fjc_env_fixed_params, load_jm_env_fixed_params
from e222_model_selection_fc_feaH import init_stable_baselies_control_d2
from fabricatio_controls.comparison_utils import Evaluator
from fabricatio_controls.comparison_utils import read_seeds
from fabricatio_rl.env_utils import create_folders

if __name__ == '__main__':
    loader = ControlLoader(
        rl_model_names=[
            'dqn__h6A_fea10S_mR__FJc14_100_v4_4',
        ],
        models_path='./2_models/',
        control_initializer=init_stable_baselies_control_d2)
    eval_seeds = read_seeds('./1_data/seeds/3_seeds_eval.log')
    Evaluator(
        env_loader=load_jm_env_fixed_params,
        # control_loader=loader.get_cp_baseline,
        control_loader=loader.get_dqnrf_experiment_controls,
        # control_loader=loader.get_sim_search_baseline,
        # control_loader=loader.get_mcts_baseline,
        # control_loader=loader.get_power_mcts_baseline,
        # control_loader=loader.get_power_sim_search_baseline,
        test_seeds=eval_seeds,
        log_runs=False,
        results_path=create_folders(
            f'./1_data/results/2_jm_10x14(100)_mcts2_tt_s6000.csv'),
        n_threads=4  # 40
    ).compare_controls()

    loader = ControlLoader(
        rl_model_names=[
            'dqn__h6A_fea10S_mR__FJc14_100_v4_4',
        ],
        models_path='./2_models/',
        control_initializer=init_stable_baselies_control_d2)
    eval_seeds = read_seeds('./1_data/seeds/3_seeds_eval.log')
    Evaluator(
        env_loader=load_fjc_env_fixed_params,
        # control_loader=loader.get_cp_baseline,
        # control_loader=loader.get_dqnrf_experiment_controls,
        # control_loader=loader.get_sim_search_baseline,
        # control_loader=loader.get_power_sim_search_baseline,
        # control_loader=loader.get_mcts_baseline,
        control_loader=loader.get_power_mcts_baseline,
        test_seeds=eval_seeds,
        log_runs=False,
        results_path=create_folders(
            f'./1_data/results/1_fjc_10x14(100)_mcts2__s6000.csv'),
        n_threads=4  # 48  # max 9 for dqn games ;)
    ).compare_controls()
