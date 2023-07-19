import datetime
from os import listdir
from os.path import exists
from typing import Union, List, Callable, TypeVar

import jsons
import numpy as np
from numpy import int32, int64

from fabricatio_controls.comparison_utils import make_env
from fabricatio_controls.cp import CPControl, ReplanningCPSequencing
from fabricatio_controls.heuristics import HeuristicControl
from fabricatio_controls.heuristics import (LPT, LOR, MOR, SRPT, LRPT,
                                            LTPO, MTPO, EDD, LUDM, SPT)
from fabricatio_controls.heuristics import LQT, LQO
from fabricatio_controls.heuristics import (SimulationSearchControl,
                                            SimulationSearch)
from fabricatio_controls.mcts import MCTSControl, MCTS
import fabricatio_controls.rl.transformer_state_representations as ts
from fabricatio_controls.rl import FeatureTransformer
from fabricatio_rl import FabricatioRL
from fabricatio_rl.core_state import State
from fabricatio_rl.env_utils import create_folders
from fabricatio_rl.interface_templates import (SchedulingUserInputs,
                                               ReturnTransformer)

T = TypeVar('T')


def log_stable_baselines_run(kwargs, run_logdir):
    kwargs['time'] = str(datetime.datetime.now())
    kwargs['optimizers'] = [
        o.__class__.__name__ for o in kwargs['optimizers']]
    state_transformer = kwargs['return_transformer'].state_transformer
    state_transformer_name = state_transformer.__class__.__name__
    if state_transformer_name == "FeatureTransformer":
        kwargs['input_features'] = state_transformer.feature_list
    kwargs['state_transformer'] = state_transformer_name
    kwargs['reward_transformer'] = kwargs[
        'return_transformer'].reward_transformer.__class__.__name__
    del kwargs['return_transformer']
    for k in ['training_envs', 'validation_envs']:
        kwargs[k] = list(set([
            si.path for si in kwargs[k].scheduling_inputs])) + [
                        kwargs[k].seeds]
    logpath = f'{run_logdir}/{kwargs["control_name"]}.log'
    print(f"saving parameter file tp {logpath}")
    with open(logpath, 'w') as f:
        f.write(jsons.dumps(kwargs, indent=4))


def get_model_run_name(model_type_name, run_logdir='1_data/sb_runs'):
    if not exists(run_logdir):
        create_folders(run_logdir)
        experiment_nr = 1
    else:
        experiment_nr = len([ex_log for ex_log in listdir(run_logdir)
                             if ex_log.startswith(model_type_name)]) + 1
    name = f"{model_type_name}_{experiment_nr}"
    return name


def load_fjc_env_fixed_params(
        seed: Union[int, List[int], np.ndarray]) -> FabricatioRL:
    if type(seed) not in [list, np.ndarray]:
        assert type(seed) in [int, np.int32]
        seeds = [seed]
    else:
        seeds = seed
    env_args = dict(
        scheduling_inputs=[SchedulingUserInputs(
            path="../benchmarks/fjssp_all/10_10-0.0__10-5.0__orb10v.fjs",
            n_jobs=100,
            max_jobs_visible=14,
            n_jobs_initial=14,
            operation_types='Jm',
            operation_precedence='Jm',
            machine_capabilities=True,
            inter_arrival_time='balanced')],
        logfile_path="",
        seeds=seeds
    )
    return make_env('fabricatio-v2', env_args)


def load_fjc_env(seed: Union[int, List[int]],
                 n_jobs_initial=10, n_jobs_total=100,
                 inter_arrival_time='balanced',
                 basedir='../benchmarks/fjssp_all'):
    if type(seed) != list:
        assert type(seed) in [int, int32]
        seeds = [seed]
    else:
        seeds = seed
    env_args = dict(
        scheduling_inputs=[SchedulingUserInputs(
            path=f"{basedir}/10_10-0.0__10-5.0__orb10v.fjs",
            n_jobs_initial=n_jobs_initial,
            n_jobs=n_jobs_total,
            max_jobs_visible=n_jobs_initial,
            operation_types='Jm',
            operation_precedence='Jm',
            machine_capabilities=True,
            inter_arrival_time=inter_arrival_time)],
        logfile_path="",
        seeds=seeds
    )
    return make_env('fabricatio-v2', env_args)


def load_jm_env(seed: Union[int, List[int]],
                n_jobs_initial=10, n_jobs_total=100,
                inter_arrival_time='balanced',
                basedir='../benchmarks/fjssp_all'):
    if type(seed) != list:
        assert type(seed) in [int, int32, int64]
        seeds = [seed]
    else:
        seeds = seed
    env_args = dict(
        scheduling_inputs=[SchedulingUserInputs(
            path=f"{basedir}/10_10 __orb10__lb944_ub944.csv",
            n_jobs_initial=n_jobs_initial,
            max_jobs_visible=n_jobs_initial,
            n_jobs=n_jobs_total,
            operation_types='Jm',
            operation_precedence='Jm',
            machine_capabilities=None,
            inter_arrival_time=inter_arrival_time)],
        logfile_path="",
        seeds=seeds
    )
    return make_env('fabricatio-v2', env_args)


def load_jm_env_fixed_params(seed: Union[int, List[int]]):
    if type(seed) != list:
        assert type(seed) in [int, np.int32, np.int64]
        seeds = [seed]
    else:
        seeds = seed
    env_args = dict(
        scheduling_inputs=[SchedulingUserInputs(
            path=f"../benchmarks/jssp_all/10_10 __orb10__lb944_ub944.csv",
            n_jobs=100,
            n_jobs_initial=14,
            max_jobs_visible=14,
            operation_types='Jm',
            operation_precedence='Jm',
            machine_capabilities=None,
            inter_arrival_time='balanced')],
        logfile_path="",
        seeds=seeds
    )
    return make_env('fabricatio-v2', env_args)


class ControlLoader:
    def __init__(self, rl_model_names: List[str],
                 models_path: str,
                 control_initializer: Union[Callable, None] = None):
        self.model_names = rl_model_names
        self.control_initializer = control_initializer
        self.models_path = models_path
        self.seq_heuristics = [
            LPT(), LOR(), MOR(), SRPT(), LRPT(), LTPO(),
            MTPO(), EDD(), LUDM(), SPT()
        ]
        self.rou_heuristics = [LQT()]

    @staticmethod
    def get_cp_baseline():
        return [
            CPControl(ReplanningCPSequencing(10, n_ops_per_job=3), 'CP3'),
        ]

    def get_baselines(self):
        # noinspection PyTypeChecker
        return (ControlLoader.get_cp_baseline() +
                self.get_heuristic_controls() +
                self.get_sim_search_baseline())

    def get_sim_search_baseline(self):
        return [
            SimulationSearchControl(SimulationSearch(
                None, seq_optimizers=self.seq_heuristics,
                tra_optimizers=self.rou_heuristics,
                criterium='makespan', p_completion=0.6))
        ]

    def get_power_sim_search_baseline(self):
        return [
            SimulationSearchControl(SimulationSearch(
                None, seq_optimizers=self.seq_heuristics,
                tra_optimizers=[LQT(), LQO()],
                criterium='makespan', p_completion=0.6), n_steps=10)
        ]

    def get_mcts_baseline(self):
        return [
            MCTSControl(optimizer=MCTS(
                env=None, itermax=8,
                heuristic=None, criterium='makespan',
                model=None, normed=False, uctk=2
            ), transformer=None, heuristics=self.seq_heuristics,
                autoplay=True, name='')
        ]

    def get_power_mcts_baseline(self):
        return [
            MCTSControl(optimizer=MCTS(
                env=None, itermax=8,
                heuristic=None, criterium='makespan',
                model=None, normed=False, uctk=2
            ), transformer=None,
                heuristics=self.seq_heuristics + [LQT(), LQO()],
                autoplay=True, name='')
        ]

    def get_heuristic_controls(self):
        h_controls = []
        for seq_h in self.seq_heuristics:
            for rou_h in self.rou_heuristics:
                h_controls += [HeuristicControl(seq_h, rou_h, 1.0)]
        return h_controls

    # noinspection DuplicatedCode
    def get_dqnrf_experiment_controls(self):
        controls = []  # ControlLoader.get_heuristic_controls()
        for i in range(len(self.model_names)):
            dqn_control = self.control_initializer(training=False)
            dqn_control.load(f'{self.models_path}/{self.model_names[i]}')
            controls += [dqn_control]
        controls += self.get_heuristic_controls()
        # controls += self.get_sim_search_baseline()
        # controls += [
        #     # CPControl(ReplanningCPSequencing(10, n_ops_per_job=3), 'CP3'),
        #     RandomControl(nowait=False),
        #     RandomControl(),
        # ]
        return controls


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func


class CriticalReviewTransformer(ReturnTransformer):
    def __init__(self, state_transformer, reward_transformer):
        self.reward_transformer = reward_transformer
        self.state_transformer = state_transformer

    def transform_state(self, state: State):
        return self.state_transformer.transform_state(state)

    def transform_reward(self, state: State, illegal=False, environment=None):
        return self.reward_transformer.transform_reward(state, illegal)

    def reset(self):
        self.reward_transformer.__init__()
        if isinstance(self.state_transformer, ts.FeatureTransformer):
            self.state_transformer.__init__(
                self.state_transformer.feature_list)
        else:
            self.state_transformer.__init__()


class FeaturesWithTimeTransformer(FeatureTransformer):
    def __init__(self, feature_list: List[str]):
        super().__init__(feature_list)

    def transform_state(self, state: State) -> np.ndarray:
        features = super().transform_state(state)
        return np.concatenate([features, np.array([state.system_time])])
