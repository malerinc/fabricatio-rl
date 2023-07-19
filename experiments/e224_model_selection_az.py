from fabricatio_controls.comparison_utils import Evaluator, read_seeds
from fabricatio_controls.comparison_utils import import_tf
from fabricatio_controls.heuristics import LQT
from fabricatio_controls.heuristics import SPT, LPT, LOR, MOR, LRPT, LTPO
from fabricatio_controls.rl.az import AZControl, RLSetupArgs, AZAgentArgs
from fabricatio_controls.rl import MakespanReward
from fabricatio_rl.env_utils import create_folders
from commons import load_fjc_env_fixed_params, load_jm_env_fixed_params
from commons import CriticalReviewTransformer, FeaturesWithTimeTransformer


ret_trnsformer = CriticalReviewTransformer(
            FeaturesWithTimeTransformer([
                'estimated_flow_time_std',  # 1
                'duration_std',  # 2
                'throughput_time_j_rel_std',  # 3
                'job_op_max_rel_completion_rate_std',  # 4
                'job_work_max_rel_completion_rate_std',  # 5
                'legal_action_len_stream_std',  # 6
                'duration_ave',  # 7
                'job_work_completion_rate_std',  # 8
                'throughput_time_j_rel_avg',  # 9
                'estimated_flow_time_ave'  # 10
            ]), MakespanReward())
optimizers = [SPT(), LPT(), LOR(), MOR(), LRPT(), LTPO(), LQT()]


if __name__ == '__main__':
    # # TRAINING SCRIPT
    seeds_path = '../1_data/seeds'
    seedfile = f"{seeds_path}/2_seeds_training.log"

    training_seeds = read_seeds(seedfile)
    training_envs = load_fjc_env_fixed_params(training_seeds)

    azsa = RLSetupArgs(
        AZAgentArgs(
            memlen=10000,
            learning_rate=0.0001,
            tb_logdir=create_folders('./1_data/tensorboard/az_lr-4'),
            res_blocks=3, res_filters=64, itermax=2, c_puct=2.5, temperature=1,
            stack_size=3
        ),
        baseln_hs="", baseln_hr="", reward_delay=1,
        models_base_dir=create_folders("./2_models/az_lr-4/")
    )
    az = AZControl(azsa, training_envs, ret_trnsformer, optimizers, 'az')
    trained_model_path = az.train_agent_parallel(10000, 20)


    def get_az_control():
        return [AZControl(
            azsa=RLSetupArgs(AZAgentArgs(
                filepath='./2_models/az_lr-4/dqn_selfplay9080',
                itermax=5, c_puct=2.5, stack_size=3)),
            state_adapter=ret_trnsformer,
            optimizers=optimizers,
            name='AZ')]

    # EVALUATION SCRIPT
    import_tf(gpu=False)
    eval_seeds = read_seeds('./1_data/seeds/3_seeds_eval.log')
    Evaluator(
        # env_loader=load_fjc_env,
        env_loader=load_jm_env_fixed_params,
        control_loader=get_az_control,
        test_seeds=eval_seeds,
        log_runs=False,
        results_path=create_folders(
            f'../1_data/results/2_jm_10x14(100)_AZlr-4__s6000.csv'),
        n_threads=40  # max 5 for AZ games ;)
    ).compare_controls()
