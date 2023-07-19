from copy import deepcopy
from time import time
from typing import Union, List

import numpy as np
import pandas as pd

from fabricatio_controls.comparison_utils import ControlResult, SchedulingSetup
from fabricatio_controls.comparison_utils import parallelize_heterogeneously
from fabricatio_controls.comparison_utils import partition_list
from fabricatio_controls.comparison_utils import read_seeds, store_seeds
from fabricatio_controls.heuristics import LQT
from fabricatio_controls.heuristics import SPT, LPT, LTPO, MOR, LOR
from fabricatio_controls.heuristics import SRPT, LRPT, MTPO, EDD, LUDM
from fabricatio_controls.heuristics import SequencingHeuristic
from fabricatio_controls.rl import FeatureTransformer
from fabricatio_controls.rl import (UtilScaledReward,
                                    UtilIntReward,
                                    UtilDeviationReward,
                                    MakespanNormedReward,
                                    MakespanReward, UtilDiffReward,
                                    UtilExpReward, BufferLenReward)
from fabricatio_rl.core_state import State
from fabricatio_rl.env_utils import create_folders
from fabricatio_rl.interface import FabricatioRL
from fabricatio_rl.interface_templates import ReturnTransformer
from commons import load_fjc_env, load_jm_env


class StateInformationCollector:
    def __init__(self, feature_list: Union[List[str], None] = None):
        self.feature_collector = FeatureTransformer(feature_list)
        if not feature_list:
            feature_names = self.feature_collector.get_feature_names()
            self.feature_collector.feature_list = feature_names
        self.names = []

    def transform_state(self, state: State) -> np.ndarray:
        wip = state.job_view_to_global_idx
        features = self.feature_collector.transform_state(state)
        # CAUTION: the matrices in the lazy feature evaluator are only available
        # after __get_state_base_nfo was called in the transform_state function
        # of the transform_state function
        D = self.feature_collector.lfi.bfi.O_D.flatten()
        T = self.feature_collector.lfi.bfi.O_T.flatten()
        L = state.matrices.op_location[wip].flatten()
        S = state.matrices.op_status[wip].flatten()
        if not self.names:
            self.names += [f'op_d_{i}' for i in range(D.shape[0])]
            self.names += [f'op_t_{i}' for i in range(T.shape[0])]
            self.names += [f'op_l_{i}' for i in range(L.shape[0])]
            self.names += [f'op_l_{i}' for i in range(S.shape[0])]
            self.names += self.feature_collector.feature_list + ['m_nr']
        all_values = np.concatenate([D, T, L, S, features,
                                     np.array([state.current_machine_nr])])
        # print(features)
        return all_values


class StateRewardCollector:
    def __init__(self):
        self.urd = UtilDiffReward()
        self.uri = UtilIntReward()
        self.ure = UtilExpReward()
        self.urs = UtilScaledReward()
        self.udi = UtilDeviationReward()
        self.mr = MakespanReward()
        self.mrn = MakespanNormedReward()
        self.blr = BufferLenReward()
        self.names = [
            'r_util_ave_diff_continuous', 'r_util_ave_diff_discrete',
            'r_util_exp', 'r_util_timescaled',
            'r_util_std_diff_discrete',
            'r_makespan_continuous', 'r_makespan_normed', 'r_buff_len'
        ]

    def transform_reward(self, state: State, illegal=False):
        reward_list = [
            self.urd.transform_reward(state, illegal),
            self.uri.transform_reward(state, illegal),
            self.ure.transform_reward(state, illegal),
            self.urs.transform_reward(state, illegal),
            self.udi.transform_reward(state, illegal),
            self.mr.transform_reward(state, illegal),
            self.mrn.transform_reward(state, illegal),
            self.blr.transform_reward(state, illegal)
        ]
        return reward_list

    def get_dummy_data(self):
        """
        Returns a list of nan for every reward name to mark the initial reward
        after a reset.

        :return: A dummy reward vector consisting of np.nan.
        """
        return [np.nan] * len(self.names)


class DataCollectionTransformer(ReturnTransformer):
    def __init__(self):
        self.state_transformer = StateInformationCollector()
        self.reward_transformer = StateRewardCollector()

    def reset(self):
        self.__init__()

    def transform_state(self, state):
        return self.state_transformer.transform_state(state)

    def transform_reward(self, state: State, illegal=False, environment=None):
        return self.reward_transformer.transform_reward(state, illegal)


class DataReport:
    def compile_data(self, *args):
        raise NotImplementedError


class ReusltDataReport(DataReport):
    def __init__(self, env: FabricatioRL):
        self.env = env
        self.h_seq_optimizers = [SPT(), LPT(), LOR(), MOR(), SRPT(), LRPT(),
                                 LTPO(), MTPO(), EDD(), LUDM()]
        self.h_rou_optimizers = [LQT()]
        # track endscores of all heuristics
        self.names = []
        self.values = []
        self.makespan_winner_idx = None
        # track end cumulative rewards of *ALL* heuristics
        self.cumulative_rewards_now = None

    def reset(self):
        self.names = []
        self.values = []
        self.makespan_winner_idx = None
        # track end cumulative rewards of *ALL* heuristics
        self.cumulative_rewards_now = None

    def compile_data(self, current_rewards,
                     reward_transformer) -> (np.ndarray, int):
        result_values, result_names, cum_r_names = [], [], []
        scores = []
        self.update_current_rewards(current_rewards)
        # test = self.play_measuring_game()
        for h_optimizer in self.h_seq_optimizers:
            # end_state = control.play_game(deterministic_copy)
            end_state, cum_r_v = self.play_reward_collection_game(
                h_optimizer, reward_transformer)
            ctrl_name = (h_optimizer.__class__.__name__ + '_'
                         + self.h_rou_optimizers[0].__class__.__name__)
            metric_names, metric_values = ControlResult(
                ctrl_name, end_state, '').to_metric_list()
            if not self.names:
                result_names += metric_names
                result_names += [f'{ctrl_name}_cum_' + r_name
                                 for r_name in reward_transformer.names]
            result_values += metric_values
            result_values += cum_r_v
            scores.append(end_state.system_time)
        if not self.names:
            self.names = result_names
        self.values = result_values
        self.makespan_winner_idx = int(np.argmin(scores))

    def update_current_rewards(self, current_rewards):
        # get object for action selection and reward transformation;
        # update current cumulative rewards
        if np.isfinite(current_rewards[0]):
            self.cumulative_rewards_now += np.array(current_rewards)
        else:
            self.cumulative_rewards_now = np.zeros(len(current_rewards))

    def play_reward_collection_game(self, h_optimizer: SequencingHeuristic,
                                    reward_transformer):
        rewards_future = np.zeros(self.cumulative_rewards_now.shape)
        rt = deepcopy(reward_transformer)
        # copy and configure env
        env_c = deepcopy(self.env)
        # TODO: pass routing opt as argument, and use it on the loop while
        #  differentiating between sequencing and routing modes ;)
        env_c.set_optimizers(self.h_rou_optimizers)
        env_c.set_transformer(None)
        env_c.set_core_rou_autoplay(True)
        env_c.set_core_seq_autoplay(True)
        env_c.make_deterministic()
        # run game
        done, state, n_steps = False, env_c.core.state, 0
        while not done:
            action = h_optimizer.get_action(state)
            _, _, done, _ = env_c.step(action)
            rewards = np.array(rt.transform_reward(env_c.core.state))
            rewards_future += rewards
        rewards_future = self.cumulative_rewards_now + rewards_future
        return env_c.core.state, rewards_future.tolist()

    def play_measuring_game(self):
        env_c = deepcopy(self.env)
        env_c.set_transformer(None)
        done, state = False, env_c.core.state
        waiting_jobs = []
        n_steps = 0
        while not done:
            state, _, done, _ = env_c.step(0)
            n_steps += 1
            if n_steps % env_c.core.state.params.max_jobs_visible == 0:
                waiting_jobs += [[len(state.system_job_queue),
                                  state.trackers.n_jobs_in_window]]
        arr = np.array(waiting_jobs)
        return arr

    def get_winning_direct_action(self):
        optimizer = self.h_seq_optimizers[self.makespan_winner_idx]
        return optimizer.get_action(self.env.core.state)


class ActionDataReport(DataReport):
    def __init__(self, train_env):
        self.env = train_env
        self.names = [
            't_action', 'direct_action', 'indirect_action', 'n_steps',
            'n_jobs_done', 'benchmark_name'
        ]
        self.values = None

    def compile_data(self, action_direct, h_idx, env_nr):
        self.values = [
            self.env.core.state.system_time,
            action_direct,
            h_idx,
            self.env.core.state.n_sequencing_steps,
            self.env.core.state.trackers.n_completed_jobs,
            env_nr
        ]
        # print(self.env.core.state.n_sequencing_steps)


class GlobalDataReport(DataReport):
    def __init__(self, scheduling_setup, dirpath):
        self.records = []
        self.names = None
        self.scheduling_setup = scheduling_setup
        self.basepath = dirpath

    def compile_data(self, state_data: np.ndarray, reward_data: List[float],
                     adr: ActionDataReport, rdr: ReusltDataReport):
        state_data = state_data.tolist()
        # noinspection PyUnresolvedReferences
        self.records.append(state_data + reward_data
                            + adr.values + rdr.values)

    def save_run_results(self, env_seed: int, train_env_name: str,
                         adr: ActionDataReport, rdr: ReusltDataReport,
                         dct: DataCollectionTransformer):
        if self.names is None:
            state_data_names = dct.state_transformer.names
            reward_data_names = dct.reward_transformer.names
            action_data_names = adr.names
            result_data_names = rdr.names
            self.names = (state_data_names + reward_data_names
                          + action_data_names + result_data_names)
        df = pd.DataFrame(self.records, columns=self.names)
        path = (f'{self.basepath}/'
                f'{self.scheduling_setup}/{train_env_name.split("/")[-1]}_'
                f'seeded{env_seed}__data.csv')
        create_folders(path)
        df.to_csv(path)
        # store_seed(env_seed, f'./1_data/seeds/1_analysis_seeds_'
        #                      f'{self.scheduling_setup.value}.txt')
        self.__purge_records()

    def __purge_records(self):
        self.records = []


def init_env(scheduling_setup, env_seed, bmdir):
    # 1. init env with 10x10(100) benchmarks with data collection
    if scheduling_setup == SchedulingSetup.JOBSHOP:
        train_env = load_jm_env(env_seed,
                                basedir=bmdir,
                                n_jobs_initial=14,
                                inter_arrival_time='over')
    elif scheduling_setup == SchedulingSetup.FLEXIBLEJOBSHOP:
        train_env = load_fjc_env(env_seed,
                                 basedir=bmdir,
                                 n_jobs_initial=14,
                                 inter_arrival_time='balanced')
    else:
        raise TypeError
    # 2. add data collection transformer and heuristic optimizers
    optimizers = [SPT(), LPT(), LOR(), MOR(), SRPT(),
                  LRPT(), LTPO(), MTPO(), EDD(), LUDM(), LQT()]
    dct = DataCollectionTransformer()
    train_env.set_optimizers(optimizers)
    train_env.set_transformer(dct)
    train_env.set_core_seq_autoplay(True)
    train_env.set_core_rou_autoplay(True)
    return train_env, optimizers, dct


def assert_h_upper_bound(env, h_result_vec):
    assert env.core.state.scheduling_mode == 0
    wip = env.core.state.job_view_to_global_idx
    upper_bound = env.core.state.matrices.op_duration[wip].sum()
    h_start_rel_makespan = h_result_vec[0] - env.core.state.system_time
    try:
        assert upper_bound >= h_start_rel_makespan
    except AssertionError:
        print(h_start_rel_makespan, upper_bound)
        raise AssertionError


def collect_data(scheduling_setup, env_seed, dirpath, bmdir):
    # 1. setup env with heuristic optimizers and DataCollectionTransformer
    train_env, optimizers, dct = init_env(scheduling_setup, env_seed, bmdir)
    adr, rdr = ActionDataReport(train_env), ReusltDataReport(train_env)
    last_env_name = train_env.parameters.scheduling_inputs.name
    train_env_name = ''
    gdr = GlobalDataReport(scheduling_setup, dirpath)
    #   2. run main loop
    env_nr = 0
    print(f"Started data colection. Breaking loop on encountering "
          f"{last_env_name} again.")
    while last_env_name != train_env_name:
        # 2.1 prep container for results
        data_state = train_env.reset()
        train_env_name = train_env.parameters.name
        print(f'Collecting data for seed {env_seed} on '
              f'the benchmark {train_env_name} ...')
        data_reward = dct.reward_transformer.get_dummy_data()
        state, done = train_env.core.state, False
        # 2.2 run main loop and at every step:
        n_steps = 0
        while not done:
            # 3.3 run heuristic controls
            rdr.compile_data(data_reward, dct.reward_transformer)
            action_direct = rdr.get_winning_direct_action()
            indirect_action = rdr.makespan_winner_idx
            # 3.4 concatenate rewards, state, h results and direct action
            adr.compile_data(action_direct, indirect_action, env_nr)
            gdr.compile_data(data_state, data_reward, adr, rdr)
            # assert_h_upper_bound(train_env, data_result)
            # 2.5 step with best_h_index
            data_state, data_reward, done, _ = train_env.step(indirect_action)
            # print(n_steps)
            n_steps += 1
            print(n_steps)
        print(f'Simulation seeded {env_seed} on '
              f'the benchmark {train_env_name} ended after {n_steps}.')
        env_nr += 1
        #   3. save the resulting dataframe
        gdr.save_run_results(env_seed, train_env_name, adr, rdr, dct)
        rdr.reset()


def collect_nseed_data(setup_setup, seed_list, dirpath, bmdir):
    for s in seed_list:
        collect_data(setup_setup, s, dirpath, bmdir)


def get_new_seeds(n_seeds, setup_str):
    f_name = f'./1_data/seeds/1_analysis_seeds_{setup_str}.txt'
    prev_seeds = set(read_seeds(f_name))
    other_seeds = []
    if len(prev_seeds) >= n_seeds:
        return list(prev_seeds)[:n_seeds]
    while len(other_seeds) + len(prev_seeds) < n_seeds:
        seed = np.random.randint(low=30000, high=40000, size=1,
                                 dtype='int32')[0]
        if seed in prev_seeds:
            continue
        else:
            other_seeds.append(seed)
    all_seeds = np.array(list(prev_seeds) + other_seeds, dtype='int32')
    store_seeds(all_seeds, f_name)
    return [s for s in all_seeds]


if __name__ == "__main__":
    n_threads = 3
    n_seeds = 1752
    setup = SchedulingSetup.FLEXIBLEJOBSHOP
    new_seeds = get_new_seeds(n_seeds, setup.value)
    basepath = f'./1_data/setup'
    benchmark_dir = '../benchmarks/fjssp_all'
    seed_partitions, _ = partition_list(np.array(new_seeds), n_threads)
    t0 = time()
    parallelize_heterogeneously(
        [collect_nseed_data] * n_threads,
        [(setup, sp, basepath, benchmark_dir) for sp in seed_partitions])
    print(f"Ran {n_threads} episodes in parallel in {time() - t0} seconds.")

    n_threads = 3
    n_seeds = 1468
    setup = SchedulingSetup.JOBSHOP
    new_seeds = get_new_seeds(n_seeds, setup.value)
    basepath = f'./1_data/setup'
    benchmark_dir = '../../benchmarks/jssp_all'
    seed_partitions, _ = partition_list(np.array(new_seeds), n_threads)
    t0 = time()
    parallelize_heterogeneously(
        [collect_nseed_data] * n_threads,
        [(setup, sp, basepath, benchmark_dir) for sp in seed_partitions])
    print(f"Ran {n_threads} episodes in parallel in {time() - t0} seconds.")
