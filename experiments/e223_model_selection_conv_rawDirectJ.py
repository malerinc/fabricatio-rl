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
from fabricatio_rl.core_state import State
from fabricatio_rl.interface_templates import Optimizer
from commons import ControlLoader
from commons import (load_fjc_env_fixed_params, linear_schedule,
                     log_stable_baselines_run, get_model_run_name,
                     CriticalReviewTransformer)


class MostValuableJob(Optimizer):
    """
    Preferrentially picks the first operation from the highest priority job as
    an action. If the high priority job is not found, the an illegal action (-1)
    will be returned.
    """
    def __init__(self, j_idx):
        super().__init__('sequencing')
        self.j_idx = j_idx

    def get_action(self, state: 'State') -> int:
        n_j_operations = state.params.max_n_operations
        wait_flag = n_j_operations * state.params.max_jobs_visible
        if state.legal_actions[-1] == wait_flag:
            action_arr = np.array(state.legal_actions[:-1])
        else:
            action_arr = np.array(state.legal_actions)
        j_idxs = action_arr // n_j_operations
        prefferred_ops = np.where(j_idxs == self.j_idx)[0]
        if prefferred_ops.any():
            return state.legal_actions[prefferred_ops[0]]
        else:
            return -1


def init_stable_baselies_control_d3(seeds: Union[int, List[int]] = -1,
                                    control_name='dqn', training=False):
    # noinspection PyTypeChecker
    learning_rate = 0.001
    kwargs = dict(
        initial_env=None,
        training_envs=load_fjc_env_fixed_params(seeds),
        validation_envs=load_fjc_env_fixed_params(seeds),
        optimizers=[MostValuableJob(i) for i in range(14)] + [LQT()],
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
            buffer_size=10000,  # 1500
            learning_starts=2000,  # 0 # 9000 # 100
            batch_size=2000,
            tau=0.8,  # 1
            gamma=0.99,
            train_freq=1000,  # (1, "episode")
            gradient_steps=-1,  # 2
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            optimize_memory_usage=False,
            target_update_interval=2000,  # 500
            exploration_fraction=0.33,
            exploration_initial_eps=0.5,  # 1
            exploration_final_eps=0.01,    # 0.05
            verbose=0,
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
        'dqn__directJA_partialRawS_mR__FJc14_100',
        run_logdir='./1_data/sb_runs')
    seeds_path = '../1_data/seeds'
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
    dqn__directJA_partialRawS_euR__FJc14_100 = init_stable_baselies_control_d3(
        training_seeds, rl_control_name, training=True)
    dqn__directJA_partialRawS_euR__FJc14_100.learn()
