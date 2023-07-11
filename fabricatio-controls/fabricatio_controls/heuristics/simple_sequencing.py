from fabricatio_rl.interface_templates import Optimizer

from collections import namedtuple
import numpy as np
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from fabricatio_rl.core_state import State

Operation = namedtuple('Operation', ['index', 'duration',
                                     'remaining_ops_no',
                                     'remaining_processing_time',
                                     'time_per_remaining_ops',
                                     'due_date'])


Op = namedtuple('Op', ['index', 'downstream_load'])


class SequencingHeuristic(Optimizer):
    def get_action(self, state: 'State') -> int:
        pass

    def __init__(self):
        self._j_idxs: Union[np.ndarray, None] = None
        self._op_idxs: Union[np.ndarray, None] = None
        super().__init__('sequencing')

    def _get_operations(self, state: 'State', cmp_prop='duration',
                        minimum=True, key='duration'):
        assert state.scheduling_mode == 0
        legal_actions = state.legal_actions
        op_durations = state.matrices.op_duration
        n_jobs = state.params.n_jobs
        n_j_operations = state.params.max_n_operations
        ops = []
        wait_flag = n_j_operations * state.params.max_jobs_visible
        wip = state.job_view_to_global_idx
        # vectorized alternative
        try:
            if state.legal_actions[-1] == wait_flag:
                action_arr = np.array(state.legal_actions[:-1])
            else:
                action_arr = np.array(state.legal_actions)
        except IndexError:
            print('herehere!')
        j_idxs = action_arr // n_j_operations
        j_idxs_glob = np.take(wip, j_idxs)
        op_idxs = action_arr % n_j_operations
        self._j_idxs = j_idxs_glob
        self._op_idxs = op_idxs


class LUDM(SequencingHeuristic):
    """
    LeastUsedDownstreamMachines
    """
    def __init__(self):
        super().__init__()

    def get_action(self, state: 'State') -> int:
        return LUDM.get_min_load_op(state)

    @staticmethod
    def get_min_load_op(state: 'State') -> int:
        n_j_operations = state.params.max_n_operations
        wait_flag = n_j_operations * state.params.max_jobs_visible
        wip = state.job_view_to_global_idx
        action = wait_flag
        val_min = np.infty
        for action_nr in state.legal_actions:
            if action_nr == wait_flag:
                continue
            j_idx = action_nr // n_j_operations
            j_idx_glob = wip[j_idx]
            op_idx = action_nr % n_j_operations
            downstream_machine_load_sum = 0
            for t in state.matrices.op_type[j_idx_glob, op_idx:]:
                # do not consider "empty" operations resulting from vnops
                if t != 0:
                    downstream_machine_load_sum += LUDM.get_machine_load(
                        state, t)
            if downstream_machine_load_sum < val_min:
                val_min = downstream_machine_load_sum
                action = action_nr
        return action

    @staticmethod
    def get_machine_load(state: 'State', operation_type: int):
        machines = state.matrices.machine_capab_dt[operation_type]
        m_avg_buffer_loads = state.trackers.buffer_times[
            np.array(list(machines)) - 1].mean()
        m_remaining_times = state.trackers.remaining_processing_times[
            np.array(list(machines)) - 1].mean()
        return m_avg_buffer_loads + m_remaining_times


class SPT(SequencingHeuristic):
    """
        Shortest Processing Time heuristic.
    """
    def __init__(self):
        super().__init__()

    def get_action(self, state: 'State') -> int:
        self._get_operations(state)
        # criteria
        i = np.argmin(state.matrices.op_duration[self._j_idxs, self._op_idxs])
        return state.legal_actions[i]


class LPT(SequencingHeuristic):
    """
        Longest Processing Time heuristic.
    """
    def __init__(self):
        super().__init__()

    def get_action(self, state: 'State') -> int:
        self._get_operations(state)
        i = np.argmax(state.matrices.op_duration[self._j_idxs, self._op_idxs])
        return state.legal_actions[i]


class LOR(SequencingHeuristic):
    """
            Least Operations Remaining
    """
    def __init__(self):
        super().__init__()

    def get_action(self, state: 'State') -> int:
        self._get_operations(state)
        i = np.argmin(state.trackers.job_n_remaining_ops[self._j_idxs])
        return state.legal_actions[i]


class MOR(SequencingHeuristic):
    """
        Most Operations Remaining
    """
    def __init__(self):
        super().__init__()

    def get_action(self, state: 'State') -> int:
        self._get_operations(state)
        i = np.argmax(state.trackers.job_n_remaining_ops[self._j_idxs])
        return state.legal_actions[i]


class SRPT(SequencingHeuristic):
    """
        Shortest Remaining Processing Time heuristic.
    """
    def __init__(self):
        super().__init__()

    def get_action(self, state: 'State') -> int:
        self._get_operations(state)
        i = np.argmin(state.trackers.job_remaining_time[self._j_idxs])
        return state.legal_actions[i]


class LRPT(SequencingHeuristic):
    """
        Longest Remaining Processing Time heuristic.
    """
    def __init__(self):
        super().__init__()

    def get_action(self, state: 'State') -> int:
        self._get_operations(state)
        i = np.argmax(state.trackers.job_remaining_time[self._j_idxs])
        return state.legal_actions[i]


class LTPO(SequencingHeuristic):
    """
        Least Time per Remaining Operations.
    """
    def __init__(self):
        super().__init__()

    def get_action(self, state: 'State') -> int:
        self._get_operations(state)
        remaining_ops = state.trackers.job_n_remaining_ops[self._j_idxs]
        remaining_t = state.trackers.job_remaining_time[self._j_idxs]
        remaining_time_per_op = remaining_ops / remaining_t
        i = np.argmin(remaining_time_per_op)
        return state.legal_actions[i]


class MTPO(SequencingHeuristic):
    """
        Most Time per Remaining Operations.
    """
    def __init__(self):
        super().__init__()

    def get_action(self, state: 'State') -> int:
        self._get_operations(state)
        remaining_ops = state.trackers.job_n_remaining_ops[self._j_idxs]
        remaining_t = state.trackers.job_remaining_time[self._j_idxs]
        remaining_time_per_op = remaining_ops / remaining_t
        i = np.argmax(remaining_time_per_op)
        return state.legal_actions[i]


class EDD(SequencingHeuristic):
    """
    Earliest Due Date.
    """

    def __init__(self):
        super().__init__()

    def get_action(self, state: 'State') -> int:
        self._get_operations(state)
        i = np.argmin(state.matrices.deadlines[self._j_idxs])
        return state.legal_actions[i]
