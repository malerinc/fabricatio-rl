import numpy as np

from copy import deepcopy

from fabricatio_rl.core_state import State

from fabricatio_controls.heuristics import HeuristicControl
from fabricatio_controls.heuristics import SPT, LPT, LOR, MOR
from fabricatio_controls.heuristics import SRPT, LRPT
from fabricatio_controls.heuristics import LTPO, MTPO, EDD


class BufferLenReward:
    """
    Based on [24]
    """
    def transform_reward(self, state: State, illegal=False, environment=None):
        buffer_lengths_total = state.trackers.buffer_lengths.sum()
        if illegal:
            return -state.params.max_jobs_visible
        return -buffer_lengths_total


class UtilDiffReward:
    """
    Based on R1 [35].
    """
    def __init__(self):
        self.utl_prev = 0
        self.n_steps = 0
        # self.norm = 0

    def transform_reward(self, state: State, illegal=False, environment=None):
        if self.n_steps == 0:
            self.n_steps += 1
            return 0
        else:
            utl_times = state.trackers.utilization_times
            t = state.system_time
            utl_curr = utl_times.mean() / t if t != 0 else 0
            reward = utl_curr - self.utl_prev
            self.utl_prev = utl_curr
        if illegal:
            return -1
        return reward


class UtilScaledReward:
    """
    NEW REWARD ;)
    """
    def __init__(self):
        self.t_prev = 0
        self.n_steps = 0
        # self.norm = 0

    def transform_reward(self, state: State, illegal=False, environment=None):
        utl_times = state.trackers.utilization_times
        t = state.system_time
        if self.n_steps == 0:
            # assert all(utl_times == 0)
            self.n_steps += 1
            # self.norm = state.trackers.initial_durations.sum()
            return 0
        else:
            utl_curr = utl_times.mean()
            dt = t - self.t_prev
            # reward = (utl_curr - self.utl_prev) / dt if dt != 0 else 0
            reward = utl_curr * dt / (t**2) if t != 0 else 0
            self.t_prev = t
        # assert -1 <= utl_diff <= 1
        if illegal:
            return -1
        return reward


class UtilExpReward:
    """
    From 38. Without weight!
    """
    def __init__(self, weight=1):
        self.weight = weight

    def transform_reward(self, state: State, illegal=False, environment=None):
        utl_times = state.trackers.utilization_times
        t = state.system_time
        utl_curr = utl_times.mean() / t if t != 0 else 0
        reward = np.exp(utl_curr / 1.5) - 1
        if illegal:
            return -1
        return reward * self.weight


class UtilIntReward:
    """
    Based on [1].
    """
    def __init__(self):
        self.utl_prev = 0
        self.n_steps = 0

    def transform_reward(self, state: State, illegal=False, environment=None):
        if illegal:
            return -2
        if self.n_steps == 0:
            self.n_steps += 1
            return 0
        else:
            utl_times = state.trackers.utilization_times
            t = state.system_time
            utl_curr = utl_times.mean() / t if t != 0 else 0
            if self.utl_prev < utl_curr:
                reward = 1
            elif self.utl_prev * 0.95 < utl_curr:
                reward = 0
            else:
                reward = -1
            self.utl_prev = utl_curr
            return reward


class MakespanReward:
    """
    From [35]
    """
    def __init__(self):
        self.time_prev = 0

    def transform_reward(self, state: State, illegal=False, environment=None):
        if illegal:
            r = min(-20000, (100 - state.n_sequencing_steps) * -1000)
        else:
            time_current = state.system_time
            td = (time_current - self.time_prev)
            self.time_prev = time_current
            r = -td
        try:
            assert r <= 0
        except AssertionError:
            print('herehere!')
        return r


class MakespanNormedReward:
    """
    Based on [35]
    """
    def __init__(self):
        self.time_prev = -1

    def transform_reward(self, state: State, illegal=False, environment=None):
        if self.time_prev == -1:
            self.time_prev = 0
            return 0
        time_current = state.system_time
        if illegal:
            return -1
        if time_current != self.time_prev:
            wip = state.job_view_to_global_idx
            makespan_norm = state.trackers.initial_durations[wip].sum()
            normed_makespan = ((time_current - self.time_prev)
                               / makespan_norm)
            self.time_prev = time_current
            return 1 - normed_makespan
        else:
            return 0


# <editor-fold desc="Currently not Considered">
class SimulativeMakespanReward:
    """
    Inspired by AZ.
    """
    def __init__(self):
        self.h_controls = [
            HeuristicControl(SPT()),
            HeuristicControl(LPT()),
            HeuristicControl(LOR()),
            HeuristicControl(MOR()),
            HeuristicControl(SRPT()),
            HeuristicControl(LRPT()),
            HeuristicControl(LTPO()),
            HeuristicControl(MTPO()),
            HeuristicControl(EDD())]

    def transform_reward(self, state: State, illegal=False, environment=None):
        if illegal:
            return -1
        results = []
        deterministic_copy = deepcopy(environment)
        deterministic_copy.make_deterministic()
        if not state.legal_actions:  # this was the end state
            return 0
        for h in self.h_controls:
            results.append(h.play_game(deterministic_copy).system_time)
        wip = state.job_view_to_global_idx
        m = environment.core.parameters.dims.n_machines
        c_max_h = min(results) - state.system_time
        c_max_lower = state.matrices.op_duration[wip].sum() / m
        reward = c_max_lower / c_max_h
        return reward


class UtilDeviationReward:
    def __init__(self):
        self.utl_prev = None
        self.n_steps = 0
        self.t_prev = None
        self.utl_dt_prev = None

    def transform_reward(self, state: State, illegal=False, environment=None):
        if illegal:
            return -2
        if self.n_steps == 0:
            self.utl_prev = state.trackers.utilization_times
            self.t_prev = 0
            self.utl_dt_prev = np.zeros(
                state.trackers.utilization_times.shape[0])
            self.n_steps += 1
            return 0
        utl_curr = state.trackers.utilization_times
        utl_diff = utl_curr - self.utl_prev
        t = state.system_time
        dt = t - self.t_prev
        if dt == 0:
            return 0
        else:
            self.t_prev = t
            utl_dt = utl_diff / dt
            if utl_dt.std() <= self.utl_dt_prev.std():
                self.utl_prev = deepcopy(utl_curr)
                self.utl_dt_prev = utl_dt
                return 1
            else:
                self.utl_prev = utl_curr
                self.utl_dt_prev = utl_dt
                return -1
# </editor-fold desc="Currently not Considered">
