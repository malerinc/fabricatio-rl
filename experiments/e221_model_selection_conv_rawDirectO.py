from collections import defaultdict
from os import getcwd
from os.path import exists
from typing import Union, List

import numpy as np
from stable_baselines3 import DQN

from fabricatio_controls.comparison_utils import read_seeds, store_seeds
from fabricatio_controls.heuristics import LQT
from fabricatio_controls.rl import FCDQNPolicy, MlpExtractor
from fabricatio_controls.rl import JmFullFlatState
from fabricatio_controls.rl import MakespanReward
from fabricatio_controls.rl import StableBaselinesRLControl
from commons import ControlLoader
from commons import (load_fjc_env_fixed_params, linear_schedule,
                     log_stable_baselines_run, get_model_run_name,
                     CriticalReviewTransformer)


def init_stable_baselies_control_d1(seeds: Union[int, List[int]] = -1,
                                    control_name='dqn', training=False):
    learning_rate = 0.00001
    kwargs = dict(
        initial_env=None,
        training_envs=load_fjc_env_fixed_params(seeds),
        validation_envs=load_fjc_env_fixed_params(seeds),
        optimizers=[LQT()],
        return_transformer=CriticalReviewTransformer(
            JmFullFlatState(), MakespanReward()),
        model_class=DQN,
        model_policy=FCDQNPolicy,
        model_parameters=dict(
            tensorboard_log=f"./1_data/tensorboard/{rl_control_name}",
            policy_kwargs=dict(
                features_extractor_class=MlpExtractor,
                features_extractor_kwargs=dict(features_dim=256),
                net_arch=[512, 256]
            ),
            learning_rate=linear_schedule(learning_rate),  # 0.0001
            buffer_size=100000,  # 1500
            learning_starts=2000,  # 0 # 9000 # 100
            batch_size=128,
            tau=0.8,  # 1
            gamma=0.99,
            train_freq=1000,  # (1, "episode")
            gradient_steps=1,  # 2
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            optimize_memory_usage=False,
            target_update_interval=2000,  # 500
            exploration_fraction=0.33,
            exploration_initial_eps=0.5,  # 1
            exploration_final_eps=0.01,    # 0.05
            max_grad_norm=10,
            verbose=1,
            device='cuda:1'
        ),
        model_path=f'./2_models/{control_name}',
        eval_freq=4001,  # 500
        eval_n_episodes=21,  # 3
        eval_deterministic=True,  # set it to true!
        learn_total_timesteps=int(1e6),  # 2e5
        control_name=control_name,
        retrain=False,
        log_freq=10  # in episodes!
    )
    control = StableBaselinesRLControl(**kwargs)
    if training:
        kwargs['model_parameters']['learning_rate'] = learning_rate
        log_stable_baselines_run(kwargs, './1_data/sb_runs')
    return control


def get_training_seeds(hcs):
    hw_seeds = defaultdict(set)
    while len(hw_seeds.keys()) < 10:
        seed = np.random.randint(0, 10000, size=1, dtype='int32')[0]
        env = load_fjc_env_fixed_params(seed)
        hr = []
        for c in hcs:
            end_state = c.play_game(env)
            hr.append(end_state.system_time)
        winner = np.argmax(hr)
        hw_seeds[winner].add(seed)
        print(hw_seeds.keys())
    return hw_seeds


if __name__ == "__main__":
    rl_control_name = get_model_run_name(
        'dqn__directOA_fullS_mR__FJc14_100', run_logdir='./1_data/sb_runs')
    seeds_path = './1_data/seeds'
    print(getcwd())

    seedfile = f"{seeds_path}/2_seeds_training.log"
    if exists(seedfile):
        training_seeds = read_seeds(seedfile)
    else:
        seed_contenders = get_training_seeds(
            ControlLoader([], '', None).get_heuristic_controls())
        training_seeds = []
        for w, s in seed_contenders.items():
            training_seeds.append(next(iter(s)))
        print(training_seeds)
        store_seeds(training_seeds, seedfile)
    dqn__directOA_fullS_euR__FJc14_100 = init_stable_baselies_control_d1(
        training_seeds, rl_control_name, training=True)
    dqn__directOA_fullS_euR__FJc14_100.learn()

    # dqn__h10A_fea10S_smR__FJc14_100 = init_stable_baselies_control_d3(
    #     training_seeds, rl_control_names[0], tensorboard_path, models_path,
    #     optimizers=[SPT(),  LPT(), LOR(), MOR(), SRPT(), LRPT(), LTPO(),
    #                 MTPO(),
    #                 EDD(), LUDM(), LQT()],
    #     transformer=JmFlatFeatureNormedMakespanReward(),
    #     training=True)
    # dqn__h10A_fea10S_smR__FJc14_100.learn()
    # # eval_seeds = np.random.randint(1, 10000, 500)
    # eval_seeds = read_seeds('./1_data/seeds/3_seeds_eval.log')
    # Evaluator(
    #     env_loader=load_fjc_env,
    #     # control_loader=cl.get_cp_baseline,
    #     control_loader=cl.get_dqnrf_experiment_controls,
    #     test_seeds=eval_seeds,
    #     log_runs=False,
    #     results_path=create_folders(
    #         f'{evaluation_path}/{experiment_name}_s100_v3.csv'),
    #     n_threads=2
    # ).compare_controls()
    # </editor-fold>
