from collections import namedtuple
from copy import deepcopy
from time import time
from typing import List, Union, Dict, Callable, TYPE_CHECKING

import numpy as np
import pandas as pd

from fabricatio_controls.heuristics import HeuristicControl
from fabricatio_controls.heuristics import LQT
from fabricatio_controls.heuristics import (SPT, LPT, LOR, MOR, SRPT, LRPT,
                                            LTPO, MTPO, EDD, LUDM)
from fabricatio_controls.comparison_utils import (
    partition_list, parallelize_heterogeneously)

if TYPE_CHECKING:
    from fabricatio_controls import Control
    from fabricatio_rl import FabricatioRL
    from fabricatio_rl.core_state import State


Scalers = namedtuple('Scalers', ['makespan', 'tardiness', 'throughput_op',
                                 'throughput_j', 'flow_time', 'utilization'])


class Evaluator:
    def __init__(self,
                 env_loader: Callable[[int], 'FabricatioRL'],
                 control_loader: Callable[[], List['Control']],
                 test_seeds,
                 results_path='', log_runs=False,
                 n_threads: int = 3):
        self.load_controls = control_loader
        self.load_env = env_loader
        self.test_seeds = test_seeds
        self.results_path = results_path
        self.log_runs = log_runs
        self.log_resuts = True if results_path else False
        self.n_threads = n_threads

    def compare_controls(self):
        # todo: logging if parameter present
        results = ControlResults(verbose=False)
        n_ex = 0
        # for seed in self.test_seeds:
        # for debugging: single thread no return version ;)
        # Evaluator.compare_single_seed(
        #     n_ex, self.test_seeds, self.load_env, self.load_controls,
        #     self.log_runs)
        partitions, n_threads = partition_list(self.test_seeds, self.n_threads)
        fns = [Evaluator.compare_single_seed] * self.n_threads
        args = [(
            n_ex, seed_partition,
            self.load_env,
            self.load_controls,
            self.log_runs) for seed_partition in partitions]
        returns = parallelize_heterogeneously(fns, args)
        for res in returns:
            results += res
        if self.log_resuts:
            df_results = pd.DataFrame(results.get_records())
            df_results.to_csv(self.results_path)

    @staticmethod
    def compare_single_seed(experiment_num, seeds,
                            load_env: Callable[[int], 'FabricatioRL'],
                            load_controls: Callable[[], List['Control']],
                            logging=False):
        results = ControlResults(verbose=False)
        control_list = load_controls()
        for seed in seeds:
            test_env = load_env(seed)
            last_env_name = test_env.parameters.scheduling_inputs.name
            test_env_name = ''
            while last_env_name != test_env_name:
                initial_abservations = test_env.reset()
                test_env_name = test_env.parameters.name
                print(f'Experiment {experiment_num}; '
                      f'Instance: {test_env_name}; '
                      f'Seed: {seed}')
                h_score, h_name = Evaluator.get_best_heuristic(
                    test_env, initial_abservations)
                for control in control_list:
                    if logging:
                        test_env.core.logger.logdir = (
                            f'../visualization_d3js/sim_data/{control.name}')
                        test_env.core.logger.on = True
                    t_s = time()
                    final_state = control.play_game(deepcopy(test_env),
                                                    initial_abservations)
                    dt = time() - t_s
                    test_env.core.logger.on = False
                    c_score = final_state.system_time
                    winner = control.name if c_score <= h_score \
                        else h_name
                    print(f'{control.name} vs '
                          f'Virtual Best Heuristic Selector (VBHS):')
                    print(f"\tFinished in {dt} seconds")
                    print(f"\tWinner: {winner}. VBS to control ratio:"
                          f" {c_score / h_score}")
                    results.add(control.name, final_state,
                                f'{seed}__{test_env_name}', time() - t_s, -1)
                    experiment_num += 1
        return results

    @staticmethod
    def get_best_heuristic(test_env, initial_abservations):
        hs = [SPT(), LPT(), LOR(), MOR(), SRPT(), LRPT(),
              LTPO(), MTPO(), EDD(), LUDM()]
        winner_name = None
        best_score = np.inf
        for heuristic in hs:
            hc = HeuristicControl(heuristic, LQT(), 1.0)
            fin_state = hc.play_game(deepcopy(test_env), initial_abservations)
            if fin_state.system_time < best_score:
                best_score = fin_state.system_time
                winner_name = hc.name
        return best_score, winner_name


class ControlResult:
    """
    Control results container. The object is used to store experiment results
    in an OO fashion.

    The object's constructor uses extracts information about
        1.) makespan
        2.) tardiness
        3.) operation throughput
        4.) job throughput
        5.) flow time and
        6.) utilization
    from an environment state and stores it in aptly named fields.

    Additionally the instance name, which should ideally contain the seed used
    in the stochastic case, the number of decisions needed until the passed
    state  was reached, the name of the control used to run the simulation and
    the computation time required to reach the state are tracked.

    The metric_names is an internal list meant for easy filtering of metrics
    from other fields. The contained strings should exactly match the
    corresponding fiel names.
    """
    def __init__(self, c_name: str, final_state: Union['State', float],
                 instance: str, duration=None, n_decisions=None):
        """
        Result container constructor.

        :param c_name: The name of the employed control method.
        :param final_state: The last state after a simulation run to read the
            results from or the float representing a benchmark makespan result.
        :param duration: The time required to control the instance from
            start to finish.
        :param n_decisions: The number of decisions taken.
        :param instance: The description of the instance; either a seed or a
            benchmark name.
        """
        if type(final_state) == float:
            self.makespan = final_state
            self.tardiness = None
            self.throughput_op = None
            self.throughput_j = None
            self.flow_time = None
            self.utilization = None
        else:
            self.makespan = final_state.system_time
            self.tardiness = final_state.trackers.tardiness.mean()
            self.throughput_op = final_state.trackers.operation_throughput
            self.throughput_j = final_state.trackers.job_throughput
            self.flow_time = final_state.trackers.flow_time.mean()
            self.utilization = final_state.trackers.utilization_times.mean()
        self.duration = duration
        self.n_decisions = n_decisions
        self.instance = instance
        self.control_name = c_name
        self.metric_names = ['makespan', 'tardiness',
                             'throughput_op', 'throughput_j',
                             'flow_time', 'utilization']

    def to_record_dict(self) -> Dict[str, float]:
        """
        Creates a dictionary representation of the object which is compatible
        with pandas dataframe record representation.

        :return: The dict for a pandas dataframe record.
        """
        field_dict = vars(self)
        del field_dict['metric_names']
        return field_dict

    def to_metric_list(self) -> (List[str], List[float]):
        """
        Creates a list of values corresponding to the scheduling metric field
        and return it together with the list of corresponding metric names. The
        employed control name is prepended to the metric field names.

        :return: A tuple of two lists containing the metric fiel values and the
            corresponding field names prefixed with the control name.
        """
        result_names = []
        result_values = []
        field_dict = vars(self)
        for k in self.metric_names:
            result_names.append(f'{self.control_name}_{k}')
            result_values.append(field_dict[k])
        return result_names, result_values

    def scale_value(self, scalers: Scalers):
        self.makespan *= scalers.makespan
        if self.tardiness is not None:
            self.tardiness *= scalers.tardiness
            self.throughput_op *= scalers.throughput_op
            self.throughput_j *= scalers.throughput_j
            self.flow_time *= scalers.flow_time
            self.utilization *= scalers.utilization

    def __str__(self):
        return str(self.to_record_dict())


class ControlResults:
    def __init__(self, verbose=False):
        self.experiment_dict = {}
        self.verbose = verbose

    def add(self, c_name: str, final_state: 'State', instance: str,
            duration=None, n_decisions=None):
        res = ControlResult(c_name, final_state, instance,
                            duration, n_decisions)
        if res.instance in self.experiment_dict:
            self.experiment_dict[res.instance].append(res)
        else:
            self.experiment_dict[res.instance] = [res]
        if self.verbose:
            print(str(res))

    def normalize(self, min_is_best=True):
        for ex in self.experiment_dict.keys():
            min_makespan = min_tardiness = min_flow_time = np.infty
            max_throughput_op = max_throughput_j = max_utilization = -np.infty
            for r in self.experiment_dict[ex]:
                if r.makespan < min_makespan:
                    min_makespan = r.makespan
                if r.tardiness is not None and r.tardiness < min_tardiness:
                    min_tardiness = r.tardiness
                if r.throughput_op is not None and \
                        r.throughput_op > max_throughput_op:
                    max_throughput_op = r.throughput_op
                if r.throughput_j is not None and \
                        r.throughput_j > max_throughput_j:
                    max_throughput_j = r.throughput_j
                if r.flow_time is not None and r.flow_time < min_flow_time:
                    min_flow_time = r.flow_time
                if r.utilization is not None and \
                        r.utilization > max_utilization:
                    max_utilization = r.utilization
            scalers = Scalers(1 / min_makespan, 1 / min_tardiness,
                              1 / max_throughput_op, 1 / max_throughput_j,
                              1 / min_flow_time, 1 / max_utilization)
            for x in self.experiment_dict[ex]:
                x.scale_value(scalers)

    def get_records(self):
        recs = []
        for ex in self.experiment_dict.keys():
            recs += [x.to_record_dict() for x in self.experiment_dict[ex]]
        return recs

    def __add__(self, other):
        for k in other.experiment_dict.keys():
            if k in self.experiment_dict:
                self.experiment_dict[k] += other.experiment_dict[k]
            else:
                self.experiment_dict[k] = other.experiment_dict[k]
        return self

    def print_makespan_cmp(self):
        for k in self.experiment_dict.keys():
            res_string = f'{k}:\n'
            for r in self.experiment_dict[k]:
                res_string += f'\t{r.control_name}: {r.makespan}\n'
            print(res_string)
