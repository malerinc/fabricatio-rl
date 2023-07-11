from collections import defaultdict
from os import getcwd
from os.path import exists
from typing import Union, List

import numpy as np
from stable_baselines3 import DQN
from torch import nn

from fabricatio_controls.comparison_utils import read_seeds, store_seeds
from fabricatio_controls.heuristics import LQT
from fabricatio_controls.heuristics import SPT, LPT, LOR, MOR, LRPT, LTPO
from fabricatio_controls.rl import FCDQNPolicy, MlpExtractor
from fabricatio_controls.rl import MakespanReward
from fabricatio_controls.rl import StableBaselinesRLControl
from commons import ControlLoader, FeaturesWithTimeTransformer
from commons import (load_fjc_env_fixed_params, linear_schedule,
                     log_stable_baselines_run, get_model_run_name,
                     CriticalReviewTransformer)


def init_stable_baselies_control_d2(seeds: Union[int, List[int]] = -1,
                                    control_name='dqn', training=False):
    learning_rate = 0.00001
    kwargs = dict(
        initial_env=None,
        training_envs=load_fjc_env_fixed_params(seeds),
        validation_envs=load_fjc_env_fixed_params([
            8485, 2799, 5082, 203, 6433, 8889, 5704, 6900, 9711, 5150, 8585,
            2793, 5081, 202, 6432, 8888, 5703, 6909, 9710, 5159, 56890]),
        optimizers=[
            SPT(), LPT(), LOR(), MOR(), LRPT(), LTPO(), LQT()
        ],
        return_transformer=CriticalReviewTransformer(
            FeaturesWithTimeTransformer([
                'estimated_flow_time_std',               # 1
                'duration_std',                          # 2
                'throughput_time_j_rel_std',             # 3
                'job_op_max_rel_completion_rate_std',    # 4
                'job_work_max_rel_completion_rate_std',  # 5
                'legal_action_len_stream_std',           # 6
                'duration_ave',                          # 7
                'job_work_completion_rate_std',          # 8
                'throughput_time_j_rel_avg',             # 9
                'estimated_flow_time_ave'                # 10
            ]), MakespanReward()),
        # 'wip_to_arrival_time_ratio' 'makespan_lb_ub_ratio'
        # 'tardiness_rate' 'estimated_flow_time_std'
        model_class=DQN,
        model_policy=FCDQNPolicy,
        model_parameters=dict(
            tensorboard_log=f"./1_data/tensorboard/{control_name}",
            policy_kwargs=dict(
                features_extractor_class=MlpExtractor,
                features_extractor_kwargs=dict(features_dim=128),
                net_arch=[], activation_fn=nn.ReLU
            ),
            learning_rate=linear_schedule(learning_rate),  # 0.0001
            buffer_size=100000,  # 1500
            learning_starts=2000,  # 0 # 9000 # 100
            batch_size=2000,  # size of the sample from the replay buffer ;)
            tau=0.8,  # 1
            gamma=0.99,  # discount factor
            train_freq=1000,   # (1, "episode")  # default: steps
            gradient_steps=-1,  # as many gradient steps as the buffer size
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            optimize_memory_usage=False,
            target_update_interval=2000,  # 500
            exploration_fraction=0.33,
            exploration_initial_eps=0.5,  # 1
            exploration_final_eps=0.01,    # 0.05
            max_grad_norm=1000,
            verbose=0,
            device='cuda:0'
        ),
        model_path=f'./2_models/{control_name}',
        eval_freq=4001,  # 500
        eval_n_episodes=21,  # 3
        eval_deterministic=True,  # set it to true!
        learn_total_timesteps=int(2e6),  # 2e5
        control_name=control_name,
        retrain=False,
        log_freq=10  # in episodes!
    )
    control = StableBaselinesRLControl(**kwargs)
    if training:
        # this function changes the parameter dict, so make sure the model has
        # already been initialized!        # just for logging ;)
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
    print("I'm starting")
    rl_control_name = get_model_run_name(
        'dqn__h6A_fea10S_mR__FJc14_100_v4', run_logdir='./1_data/sb_runs')
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
    dqn__h6A_fea10S_mR__FJc14_100_v4 = init_stable_baselies_control_d2(
        training_seeds, rl_control_name, training=True)
    dqn__h6A_fea10S_mR__FJc14_100_v4.learn()

    # </editor-fold>
