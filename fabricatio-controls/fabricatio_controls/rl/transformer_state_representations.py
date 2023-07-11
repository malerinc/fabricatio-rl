import numpy as np

from scipy.spatial import distance
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.stats import kendalltau

from fabricatio_rl.core_state import State, Trackers
from fabricatio_controls.heuristics import (SPT, LPT, LOR, MOR, SRPT, LRPT,
                                            LTPO, MTPO, EDD, LUDM)

from typing import List, Union


class StateTransformer:
    @staticmethod
    def transform_state(state: State):
        raise NotImplementedError


class JmFlatFullState(StateTransformer):
    @staticmethod
    def transform_state(state):
        wip = state.job_view_to_global_idx
        O_D = state.matrices.op_duration[wip]
        O_T = state.matrices.op_type[wip]
        L = state.matrices.op_location[wip]
        S = state.matrices.op_status[wip]
        machine_nr = state.current_machine_nr
        current_j_nr = state.current_job
        return np.concatenate([
            O_D.flatten(), O_T.flatten(), L.flatten(), S.flatten(),
            np.array([machine_nr, current_j_nr])])


class JmFlatFullStateNormalized(StateTransformer):
    @staticmethod
    def transform_state(state):
        wip = state.job_view_to_global_idx
        O_D = state.matrices.op_duration[wip].astype('float32')
        scaler = O_D.max()
        O_D *= 1.0 / scaler if scaler != 0 else 0
        O_T = state.matrices.op_type[wip]
        L = state.matrices.op_location[wip]
        S = state.matrices.op_status[wip]
        machine_nr = state.current_machine_nr
        current_j_nr = state.current_job
        simulation_mode = state.scheduling_mode
        return np.concatenate([
            O_D.flatten(), O_T.flatten(), L.flatten(), S.flatten(),
            np.array([machine_nr, current_j_nr])])


class JmMainMatrices:
    @staticmethod
    def transform_state(state):
        wip = state.job_view_to_global_idx
        O_D = state.matrices.op_duration[wip]
        O_T = state.matrices.op_type[wip]
        L = state.matrices.op_location[wip]
        return np.stack([O_D, O_T, L], axis=0)


class JmFullFlatState:
    @staticmethod
    def transform_state(state):
        wip = state.job_view_to_global_idx
        O_D = state.matrices.op_duration[wip].flatten()
        O_T = state.matrices.op_type[wip].flatten()
        L = state.matrices.op_location[wip].flatten()
        S = state.matrices.op_status[wip].flatten()
        m = state.current_machine_nr
        t = state.system_time
        return np.concatenate([O_D, O_T, L, S, np.array([m, t])],
                              axis=0).astype('float32')


class JmPartialBoxState:
    @staticmethod
    def transform_state(state):
        wip = state.job_view_to_global_idx
        O_D = state.matrices.op_duration[wip]
        O_T = state.matrices.op_type[wip]
        L = state.matrices.op_location[wip]
        S = state.matrices.op_status[wip]
        machine_nr = state.current_machine_nr
        current_j_nr = state.current_job
        simulation_mode = state.scheduling_mode
        n_jobs, n_j_operations = (state.params.max_jobs_visible,
                                  state.params.max_n_operations)
        legal_actions = np.zeros(O_D.shape)
        ops = []
        wait_flag = n_j_operations * n_jobs
        # TODO: vectorize!
        for action_nr in state.legal_actions:
            if action_nr == wait_flag:
                continue
            j_idx = action_nr // n_j_operations
            op_idx = action_nr % n_j_operations
            legal_actions[(j_idx, op_idx)] = 1
        return np.stack([O_D, O_T, legal_actions], axis=0)


class JmFullBoxState:
    @staticmethod
    def transform_state(state):
        wip = state.job_view_to_global_idx
        O_D = state.matrices.op_duration[wip]
        O_T = state.matrices.op_type[wip]
        L = state.matrices.op_location[wip]
        S = state.matrices.op_status[wip]
        machine_nr = state.current_machine_nr
        current_j_nr = state.current_job
        simulation_mode = state.scheduling_mode
        n_jobs, n_j_operations = (state.params.max_jobs_visible,
                                  state.params.max_n_operations)
        legal_actions = np.zeros(O_D.shape)
        ops = []
        wait_flag = n_j_operations * n_jobs
        for action_nr in state.legal_actions:
            if action_nr == wait_flag:
                continue
            j_idx = action_nr // n_j_operations
            op_idx = action_nr % n_j_operations
            legal_actions[(j_idx, op_idx)] = 1
        return np.stack([O_D, O_T, L, S, legal_actions], axis=0)


class BaseFeatureInfo:
    """
    Helper class storing selected state information for easier feature
    calculation.
    """
    def __init__(self):
        self.O_D, self.O_T, self.wip, self.job_queue = (None for _ in range(4))
        self.track: Union[Trackers, None] = None
        self.legal_actions = None
        self.__wip_size = -1
        self.__j_n_o: Union[np.ndarray, None] = None
        self.__n_o: int = -1
        self.m = -1
        self.__legal_action_lengths = []
        self.__type_bin_counts: Union[np.ndarray, None] = None
        self.__duration_bin_counts: Union[np.ndarray, None] = None
        self.__j_start_time: Union[np.ndarray, None] = None
        self.__j_visible_time: Union[np.ndarray, None] = None
        self.__wip_start_time: Union[int, None] = None
        self.__wip_rel_sys_t: int = 0
        self.__decision_skips: Union[int, None] = None
        self.__decisions_total: Union[int, None] = None
        self.__state: Union[State, None] = None

    @property
    def n_o(self):
        """
        The total number of operations over all WIP jobs as counted at
        scheduling problem definition time (original, unprocessed jobs).

        :return: The n_o property.
        """
        return self.__n_o

    @property
    def j_n_o(self):
        """
        A numpy array containing the original number of operations for each job
        in WIP.

        :return: The j_n_o attribute.
        """
        return self.__j_n_o

    @property
    def wip_size(self):
        """
        The number of WIP job slots (WIP window size/maximum number of visible
        jobs).

        :return: The wip_size attribute.
        """
        return self.__wip_size

    @property
    def legal_action_lengths(self):
        """
        Contains the stream of legal action sizes from the beginning of the
        scheduling process as a list.

        :return: The legal_action_lengths attribute.
        """
        return self.__legal_action_lengths

    @property
    def type_bin_counts(self):
        """
        A numpy array of size n_types containing the type bin counts. The bins
        represent the distribution of the WIP operation types. Zeros in the wip
        operation type matrix are not counted.

        This information is maintained within the Tracker object of the
        simulation state. See: Trackers.type_counts

        :return: The type_counts attribute.
        """
        return self.__type_bin_counts

    @property
    def duration_bin_counts(self):
        """
        A numpy array of size n_types containing the duration bin counts. The
        bins are created by distributing the operations from the current state
        over n_types + 1 consecutive, equally sized intervals starting at 0 and
        ending at the current WIP operation duration matrix maximum with respect
        to their durations. Zeros are ignored from the count.

        This information is maintained by the Tracker object of the simulation
        state. See: Trackers.duration_counts

        :return: The duration_bin_counts attribute.
        """
        return self.__duration_bin_counts

    @property
    def j_start_time(self):
        """
        An numpy array of size equal to the WIP size containing the points in
        time when the respective jobs were first processed (first job operation
        start time).

        :return: The j_start_time attribute.
        """
        return self.__j_start_time

    @property
    def j_visible_time(self):
        """

        :return:
        """
        return self.__j_visible_time

    @property
    def wip_start_time(self):
        """
        The point in time relative to the beginning of the scheduling process
        when the last job change was made to the wip.

        :return: The wip_start_time attribute.
        """
        return self.__wip_start_time

    @property
    def wip_rel_sys_t(self):
        """
        The amount of time elapsed since the last job was added to the work in
        progress window. This is essentially the system_time - wip_start_time.

        :return: The wip_rel_sys_t attribute.
        """
        return self.__wip_rel_sys_t

    @property
    def decision_skips(self):
        """
        The number of skipped decisions. Decisions are skipped when there is a
        single possible sequencing/routing decision to be made. If decision
        skips are turned of within the simulation, this property will always be
        0.

        :return: The decision_skips attribute.
        """
        return self.__decision_skips

    @property
    def decisions_total(self):
        return self.__decisions_total

    @property
    def state(self):
        return self.__state

    @staticmethod
    def remove_nan(a):
        a[np.isnan(a) | ~np.isfinite(a)] = 0
        return a

    @staticmethod
    def std(arr, avg):
        n = arr.shape[0]
        val = np.sqrt(np.sum((arr - avg) ** 2) / n)
        # assert np.isfinite(val)
        return val

    # noinspection PyArgumentList
    def get_state_base_nfo(self, state: State):
        """
        Reassigns the common feature calculation parameters based on the current
        state.

        :param state: The current simulation state.
        :return: None
        """
        self.wip = state.job_view_to_global_idx
        self.O_D = state.matrices.op_duration[self.wip]
        # assert (self.O_D >= 0).all()
        self.O_T = state.matrices.op_type[self.wip]
        self.__wip_size = state.params.max_jobs_visible
        self.__j_n_o = state.params.n_operations[self.wip]
        self.__n_o = self.j_n_o.sum()
        self.m = state.params.n_machines
        self.track = state.trackers
        self.job_queue = state.system_job_queue
        self.__type_bin_counts = state.trackers.type_counts[1:]  # ignore zeros
        self.__duration_bin_counts = state.trackers.duration_counts[1:]
        self.__j_start_time = self.track.job_start_times[self.wip]
        self.__j_visible_time = self.track.job_visible_dates[self.wip]
        self.__wip_start_time = self.j_start_time.min()
        self.__wip_rel_sys_t = state.system_time - self.wip_start_time
        self.__decision_skips = state.trackers.n_decision_skips
        self.__decisions_total = state.trackers.n_total_decisions
        self.legal_actions = state.legal_actions
        self.__legal_action_lengths.append(len(state.legal_actions))
        self.__state = state


class LazyFeatureInfo:
    def __init__(self, bfi: BaseFeatureInfo):
        self.estimated_flow_time: Union[np.ndarray, None] = None
        self.taus: Union[np.ndarray, None] = None
        self.bfi = bfi
        self.job_clustering_vecs = None
        self.O_D_max = None
        self.O_D_min = None
        self.O_T_max = None
        self.O_T_min = None
        self.work_remaining_total: Union[int, None] = None
        self.makespan_lb: Union[int, None] = None
        self.makespan_ub: Union[int, None] = None
        self.remaining_ops: Union[np.ndarray, None] = None
        self.remaining_ops_wip_avg: Union[int, None] = None
        self.ops_completed_max: Union[int, None] = None
        self.remaining_ops_sum: Union[int, None] = None
        self.remaining_times_wip_avg = None
        self.load_ratios: Union[np.ndarray, None] = None
        self.buffer_load_ratios_avg = None
        self.utl_avg: Union[int, None] = None
        self.throughput_time_j_abs_std = None
        self.dt_arrival: Union[np.ndarray, None] = None
        self.throughput_time_j_abs_avg = None
        self.throughput_time_j_abs = None
        self.throughput_time_j_rel_std = None
        self.throughput_time_j_rel_avg = None
        self.throughput_time_j_rel = None
        self.j_latest_time: Union[np.ndarray, None] = None
        self.ops_completed: Union[np.ndarray, None] = None
        self.job_op_completion_rate: Union[float, None] = None
        self.work_throughput = None
        self.work_original_jobs: Union[np.ndarray, None] = None
        self.work_done: Union[np.ndarray, None] = None
        self.work_remaining = None
        self.work_done_wip_sum: Union[float, None] = None
        self.work_done_max: Union[float, None] = None
        self.work_original_sum: Union[float, None] = None
        self.job_op_completion_rate = None
        self.d_hamming: Union[np.ndarray, None] = None
        self.d_hamming_max: Union[int, None] = None
        self.d_euclidean: Union[np.ndarray, None] = None
        self.d_euclidean_max: Union[int, None] = None

    def add_ops_completed(self):
        self.ops_completed = self.bfi.track.job_n_completed_ops[self.bfi.wip]

    def add_job_op_completion_rate(self):
        """
        Adds the array of completed operations (units) to original operations
        for feature construction.

        :return: None.
        """
        if self.ops_completed is None:
            self.add_ops_completed()
        self.job_op_completion_rate = self.ops_completed / self.bfi.j_n_o

    def add_work_done(self):
        self.work_done = self.bfi.track.job_completed_time[self.bfi.wip]

    def add_work_done_wip_sum(self):
        if self.work_done is None:
            self.add_work_done()
        self.work_done_wip_sum = self.work_done.sum()

    def add_work_done_max(self):
        if self.work_done is None:
            self.add_work_done()
        # noinspection PyArgumentList
        self.work_done_max = self.work_done.max()

    def add_work_original_jobs(self):
        """
        Adds the distribution of original work over jobs for feature
        computation.

        :return: None.
        """
        self.work_original_jobs = self.bfi.track.work_original_jobs[
            self.bfi.wip]

    def add_work_original_sum(self):
        """
        Adds the total original work volume (processing times) for the current
        WIP to the feature construction information.

        :return: None.
        """
        if self.work_original_jobs is None:
            self.add_work_original_jobs()
        self.work_original_sum = self.work_original_jobs.sum()

    def add_throughput_time_j_rel(self):
        """
        Adds the ratio of work done to timedelta since the beginning of job
        processing for all jobs.

        :return: None.
        """
        if self.j_latest_time is None:
            self.add_j_latest_time()
        if self.work_done is None:
            self.add_work_done()
        if self.dt_arrival is None:
            self.add_dt_arrival()
        self.throughput_time_j_rel = np.divide(
            self.work_done, self.dt_arrival, out=np.zeros_like(self.work_done),
            where=self.dt_arrival != 0)

    def add_throughput_time_j_rel_avg(self):
        """
        Adds the mean of the distribution of work done relative to the timedelta
        since the first processing of the job over all jobs.

        :return: None.
        """
        if self.throughput_time_j_rel is None:
            self.add_throughput_time_j_rel()
        self.throughput_time_j_rel_avg = self.throughput_time_j_rel.mean()

    def add_throughput_time_j_rel_std(self):
        """
        Adds the standard deviation of the distribution  work done relative to
        the timedelta since the first processing of the job over all jobs.

        :return: None.
        """
        if self.throughput_time_j_rel is None:
            self.add_throughput_time_j_rel()
        self.throughput_time_j_rel_std = self.throughput_time_j_rel.std()

    def add_throughput_time_j_abs(self):
        if self.dt_arrival is None:
            self.add_dt_arrival()
        if self.work_done is None:
            self.add_work_done()
        if self.j_latest_time is None:
            self.add_j_latest_time()
        # noinspection PyArgumentList
        self.throughput_time_j_abs = np.divide(
            self.work_done, self.dt_arrival.max(),
            out=np.zeros_like(self.work_done),
            where=self.j_latest_time != 0)

    def add_throughput_time_j_abs_avg(self):
        if self.throughput_time_j_abs is None:
            self.add_throughput_time_j_abs()
        self.throughput_time_j_abs_avg = self.throughput_time_j_abs.mean()

    def add_dt_arrival(self):
        if self.j_latest_time is None:
            self.add_j_latest_time()
        self.dt_arrival = self.j_latest_time - self.bfi.j_visible_time

    def add_j_latest_time(self):
        self.j_latest_time = self.bfi.track.job_last_processed_time[
            self.bfi.wip]

    def add_throughput_time_j_abs_std(self):
        if self.throughput_time_j_abs is None:
            self.add_throughput_time_j_abs()
        if self.throughput_time_j_abs_avg is None:
            self.add_throughput_time_j_abs_avg()
        self.throughput_time_j_abs_std = self.throughput_time_j_abs.std()

    def add_utl_avg(self):
        self.utl_avg = self.bfi.track.utilization_times.mean()

    def add_buffer_load_ratios_avg(self):
        if self.load_ratios is None:
            self.add_load_ratios()
        self.buffer_load_ratios_avg = self.load_ratios.mean()

    def add_load_ratios(self):
        """
        Adds the buffer load to maximum buffer load ratios for every machine
        to the feature construction information base.

        :return: None
        """
        max_load = self.bfi.track.buffer_times.max()
        if max_load == 0:
            # zero buffer times all over
            self.load_ratios = self.bfi.track.buffer_times
        else:
            self.load_ratios = self.bfi.track.buffer_times / max_load

    def add_remaining_times_wip_avg(self):
        if self.work_remaining is None:
            self.add_work_remaining()
        self.remaining_times_wip_avg = self.work_remaining.mean()

    def add_work_remaining(self):
        self.work_remaining = self.bfi.track.job_remaining_time[self.bfi.wip]

    def add_remaining_ops_wip_avg(self):
        """
        Adds the remaining operations for every job as well as the mean thereof
        for lazy feature construction.

        :return: None.
        """
        if self.remaining_ops is None:
            self.remaining_ops = self.bfi.track.job_n_remaining_ops[
                self.bfi.wip]
        self.remaining_ops_wip_avg = self.remaining_ops.mean()

    def add_remaining_ops(self):
        self.remaining_ops = self.bfi.track.job_n_remaining_ops[self.bfi.wip]

    def add_makespan_lb(self):
        if self.work_remaining_total is None:
            self.add_work_remaining_total()
        self.makespan_lb = (self.bfi.wip_rel_sys_t
                            + self.work_remaining_total / self.bfi.m)

    def add_work_remaining_total(self):
        if self.work_remaining is None:
            self.add_work_remaining()
        self.work_remaining_total = self.work_remaining.sum()

    def add_makespan_ub(self):
        if self.work_remaining_total is None:
            self.add_work_remaining_total()
        self.makespan_ub = self.bfi.wip_rel_sys_t + self.work_remaining_total

    def add_remaining_ops_total(self):
        if self.remaining_ops is None:
            self.add_remaining_ops()
        self.remaining_ops_sum = self.remaining_ops.sum()

    def add_type_matrix_minimum(self):
        self.O_T_min = self.bfi.O_T.min()

    def add_type_matrix_maximum(self):
        self.O_T_max = self.bfi.O_T.max()

    def add_duration_matrix_minimum(self):
        self.O_D_min = self.bfi.O_D.min()

    def add_duration_matrix_maximum(self):
        self.O_D_max = self.bfi.O_D.max()

    def add_job_clustering_vec(self):
        if self.O_T_max is None:
            self.add_type_matrix_maximum()
        if self.O_T_min is None:
            self.add_type_matrix_minimum()
        if self.O_D_max is None:
            self.add_duration_matrix_maximum()
        if self.O_D_min is None:
            self.add_duration_matrix_minimum()
        # normalize types
        if self.O_T_max == 0:
            O_T_normed = self.bfi.O_T
        elif self.O_T_max == self.O_T_min:
            O_T_normed = np.ones(self.bfi.O_T.shape)
        else:
            O_T_norm_factor = self.O_T_max - self.O_T_min
            O_T_normed = (self.bfi.O_T - self.O_T_min) / O_T_norm_factor
        # normalize durations
        if self.O_D_max == 0:
            O_D_normed = self.bfi.O_D
        elif self.O_D_max == self.O_D_min:
            O_D_normed = np.ones(self.bfi.O_D.shape)
        else:
            O_D_norm_factor = self.O_D_max - self.O_D_min
            O_D_normed = (self.bfi.O_D - self.O_D_min) / O_D_norm_factor
        # concatenate matrices
        self.job_clustering_vecs = np.concatenate(
            (O_T_normed, O_D_normed), axis=1)

    def add_taus(self, taus: np.ndarray):
        """
        Saves the Kendal tau values measuring the difference between the
        identity permutation and the operation type sequence in different jobs
        as feature construction info.

        :param taus: The Kendal tau values for individual jobs.
        :return: None.
        """
        self.taus = taus

    def add_max_completed_ops(self):
        """
        Adds the maximum over the remaining operations distribution over WIP job
        to the feature construction information.

        :return: None.
        """
        if self.ops_completed is None:
            self.add_ops_completed()
        # noinspection PyArgumentList
        self.ops_completed_max = self.ops_completed.max()

    def add_d_hamming(self):
        ds_nominal = distance.cdist(self.bfi.O_T,
                                    self.bfi.O_T, 'hamming').flatten()
        self.d_hamming = ds_nominal[ds_nominal != 0]

    def add_d_hamming_max(self):
        if self.d_hamming is None:
            self.add_d_hamming()
        if self.d_hamming.size == 0:
            self.d_hamming_max = 0
        else:
            # noinspection PyArgumentList
            self.d_hamming_max = self.d_hamming.max()

    def add_d_euclidean(self):
        ds_numeric = distance.cdist(self.bfi.O_D,
                                    self.bfi.O_D, 'euclidean').flatten()
        self.d_euclidean = ds_numeric[ds_numeric != 0]

    def add_d_euclidean_max(self):
        if self.d_euclidean is None:
            self.add_d_euclidean()
        if self.d_euclidean.size == 0:
            self.d_euclidean_max = 0
        else:
            # noinspection PyArgumentList
            self.d_euclidean_max = self.d_euclidean.max()

    def add_estimated_flow_time(self):
        if self.dt_arrival is None:
            self.add_dt_arrival()
        if self.utl_avg is None:
            self.add_utl_avg()
        if self.work_original_sum is None:
            self.add_work_original_sum()
        if self.work_remaining is None:
            self.add_work_remaining()
        flow_time = (self.dt_arrival
                     + self.work_remaining  # is never < 0
                     * (1 / self.utl_avg)) if self.utl_avg != 0 else 0
        # noinspection PyArgumentList
        norm = self.work_original_sum
        eft = (flow_time / norm) if norm != 0 else np.zeros(
            self.work_remaining.shape[0])
        self.estimated_flow_time = eft
        try:
            assert eft.all() >= 0
        except AssertionError:
            print('herehere!')


class FeatureTransformer(StateTransformer):
    # TODO: normalize all features to fit betweext -1 and 1
    # <editor-fold desc="INITIALIZATION FUNCTIONS">
    def __init__(self, feature_list: List[str]):
        bfi = BaseFeatureInfo()
        self.__lfi = LazyFeatureInfo(bfi)
        self.__feature_list = feature_list

    @property
    def feature_list(self):
        return self.__feature_list

    @feature_list.setter
    def feature_list(self, new_list):
        self.__feature_list = new_list

    @property
    def lfi(self):
        return self.__lfi

    def __get_state_base_nfo(self, state: State):
        """
        Updates the lazy feature data by re-instantiating the LazyFeatureInfo
        attribute. Since some of the information in the BaseFeatureInfo (lfi)
        instance of the LazyFeatureInformation attribute is persistent (e.g.
        legal_action_lengths), a reference to this object must be stored,
        followed by an object update and the reinitialization of lfi with the
        updated object.

        :param state: The simulation state for updates.
        :return: None.
        """
        bfi_temp = self.__lfi.bfi
        bfi_temp.get_state_base_nfo(state)
        self.__lfi = LazyFeatureInfo(bfi_temp)
    # </editor-fold desc="INITIALIZATION FUNCTIONS">

    # <editor-fold desc="[JOB-CENTRIC] COMPLETION RATE (TIME/OPS) FEATURES">
    def f_op_completion_rate(self):
        """
        Calculates the ratio between the sum of completed operations and total
        operations in WIP.

        :return: The WIP operation completion rate.
        """
        if self.__lfi.ops_completed is None:
            self.__lfi.add_ops_completed()
        return self.__lfi.ops_completed.sum() / self.__lfi.bfi.n_o

    def f_job_op_completion_rate_std(self):
        """
        Calculates the standard deviation over the distribution of job
        completion ratios (number of finished job operations over total job
        operations).

        :return: The standard deviation of the job completion distribution.
        """
        if self.__lfi.job_op_completion_rate is None:
            self.__lfi.add_job_op_completion_rate()
        return self.__lfi.job_op_completion_rate.std()

    def f_job_op_completion_rate_ave(self):
        """
        See f_job_op_completion_rate_std.

        :return: The average job completion with respect to the initial number
            of operations.
        """
        if self.__lfi.job_op_completion_rate is None:
            self.__lfi.add_job_op_completion_rate()
        return self.__lfi.job_op_completion_rate.mean()

    def f_job_op_max_rel_completion_rate_avg(self):
        """
        Calculates the average of the completed operation distribution over all
        jobs. The distribution quantities are normalized with the current
        maximum number of completed operations over jobs.

        :return: The normalized average of the remaining WIP operations over
            every job.
        """
        if self.__lfi.ops_completed is None:
            self.__lfi.add_ops_completed()
        if self.__lfi.ops_completed_max is None:
            self.__lfi.add_max_completed_ops()
        if self.__lfi.ops_completed_max == 0:
            return 0
        return (self.__lfi.ops_completed /
                self.__lfi.ops_completed_max).mean()

    def f_job_op_max_rel_completion_rate_std(self):
        """
        See f_job_op_max_rel_completion_rate_avg.

        :return: The standard deviation of ratio distribution of the completed
            job operations to the maximum current completed operations over
            jobs.
        """
        if self.__lfi.ops_completed is None:
            self.__lfi.add_ops_completed()
        if self.__lfi.ops_completed_max is None:
            self.__lfi.add_max_completed_ops()
        if self.__lfi.ops_completed_max == 0:
            return 0
        return (self.__lfi.ops_completed /
                self.__lfi.ops_completed_max).std()

    def f_work_completion_rate(self):
        """
        The ratio between the sum of completed operation durations and the total
        operation durations before job processing had started for the current
        WIP.

        :return: The completed work ratio.
        """
        if self.__lfi.work_done_wip_sum is None:
            self.__lfi.add_work_done_wip_sum()
        if self.__lfi.work_original_sum is None:
            self.__lfi.add_work_original_sum()
        return self.__lfi.work_done_wip_sum / self.__lfi.work_original_sum

    def f_job_work_completion_rate_avg(self):
        """
        Calculates the average of the ratio of completed work (in terms of
        operation processing time) to initial work over all jobs in the current
        WIP jobs.

        :return: The average remaining work ratio for WIP jobs.
        """
        if self.__lfi.work_done is None:
            self.__lfi.add_work_done()
        if self.__lfi.work_original_jobs is None:
            self.__lfi.add_work_original_jobs()
        work_original: np.ndarray = self.__lfi.work_original_jobs
        return (self.__lfi.work_done / work_original).mean()

    def f_job_work_completion_rate_std(self):
        """
        Calculates the standard deviation for the distribution of the ratio
        of remaining work (in terms of operation processing time) to initial
        work over all jobs in the current WIP jobs.

        :return: The standard deviation of remaining work ratio over all WIP
            jobs.
        """
        if self.__lfi.work_done is None:
            self.__lfi.add_work_done()
        if self.__lfi.work_original_jobs is None:
            self.__lfi.add_work_original_jobs()
        work_original: np.ndarray = self.__lfi.work_original_jobs
        return (self.__lfi.work_done / work_original).std()

    def f_job_work_max_rel_completion_rate_avg(self):
        """
        Calculates the average of completed work (in terms of
        operation processing time) over all jobs. the value is normalized using
        the current maximum work over all job in WIP.

        :return: The average completed work ratio for WIP jobs.
        """
        if self.__lfi.work_done is None:
            self.__lfi.add_work_done()
        if self.__lfi.work_done_max is None:
            self.__lfi.add_work_done_max()
        work_max: float = self.__lfi.work_done_max
        if work_max == 0:
            return 0
        return self.__lfi.work_done.mean() / work_max

    def f_job_work_max_rel_completion_rate_std(self):
        """
        Calculates the average of the ratio of completed work (in terms of
        operation processing time) to initial work over all jobs in the current
        WIP jobs.

        :return: The average remaining work ratio for WIP jobs.
        """
        if self.__lfi.work_done is None:
            self.__lfi.add_work_done()
        if self.__lfi.work_done_max is None:
            self.__lfi.add_work_done_max()
        # noinspection PyTypeChecker
        work_max: np.ndarray = self.__lfi.work_done_max
        if work_max == 0:
            return 0
        return self.__lfi.work_done.std() / work_max

    def df_remaining_time_ratio(self):
        """
        DEPRECATED FEATURE (df) since this correlates perfectly with the
        f_work_completion_rate.

        :return: None.
        """
        pass
        # if self.__lfi.completed_time_ratio is None:
        #     self.__lfi.add_completed_time_ratio()
        # return 1 - self.__lfi.completed_time_ratio
    # </editor-fold desc="[JOB-CENTRIC] COMPLETION RATE (TIME/OPS) FEATURES">

    # <editor-fold desc="[JOB-CENTRIC] THROUGHPUT FEATURES">
    def f_throughput_time_j_rel_avg(self):
        # TODO: arrival_relative_job_throughput_rate
        if self.__lfi.throughput_time_j_rel_avg is None:
            self.__lfi.add_throughput_time_j_rel_avg()
        return self.__lfi.throughput_time_j_rel_avg

    def f_throughput_time_j_rel_std(self):
        if self.__lfi.throughput_time_j_rel_std is None:
            self.__lfi.add_throughput_time_j_rel_std()
        return self.__lfi.throughput_time_j_rel_std

    def f_throughput_time_j_abs_avg(self):
        if self.__lfi.throughput_time_j_abs_avg is None:
            self.__lfi.add_throughput_time_j_abs_avg()
        return self.__lfi.throughput_time_j_abs_avg

    def f_throughput_time_j_abs_std(self):
        if self.__lfi.throughput_time_j_abs_std is None:
            self.__lfi.add_throughput_time_j_abs_std()
        return self.__lfi.throughput_time_j_abs_std
    # </editor-fold>

    # <editor-fold desc="[MACHINE-CENTRIC] WORKLOAD FEATURES">
    def f_utl_avg(self):
        """
        Returns the average utilization over all machines since the beginning of
        the scheduling process. First the work done is averaged over machines,
        then the indicator is normalized to [0, 1] by dividing by the current
        system time.

        :return: The average machine utilization since the simulation start.
        """
        if self.__lfi.utl_avg is None:
            self.__lfi.add_utl_avg()
        norm = self.__lfi.bfi.state.system_time
        return self.__lfi.utl_avg / norm if norm != 0 else 0

    def f_utl_std(self):
        """
        Returns the utilization standard over all machines since the beginning
        of the scheduling process. First the standard deviation of work done
        done by machines is computed, then the indicator is normalized to [0, 1]
        by dividing by the current system time.

        :return: The average machine utilization since the simulation start.
        """
        if self.__lfi.utl_avg is None:
            self.__lfi.add_utl_avg()
        norm = self.__lfi.bfi.state.system_time
        return ((self.__lfi.bfi.track.utilization_times.std() / norm)
                if norm != 0 else 0)

    def f_utl_current(self):
        """
        Returns the current ratio of active machines in the system. Thos is
        different from the f_utl_ave and f_utl_std in that it captures a
        snapshot utilization at the current time as opposed to the evolution
        of the utilization distribution over time.

        :return: The current utilization ratio.
        """
        active_machines = self.__lfi.bfi.state.machines.machine_active
        total_active_machines = active_machines.astype('int').sum()
        return total_active_machines / self.__lfi.bfi.m

    def f_buffer_load_ratios_avg(self):
        """
        Calculates the average of the normalized buffer load distribution over
        all machine buffers.

        Individual machine buffer loads are calculated by summing
        up the processing times of all the buffered operation. The maximum
        load is used as a normalization factor.

        :return: The average normalized buffered processing time of the system.
        """
        if self.__lfi.buffer_load_ratios_avg is None:
            self.__lfi.add_buffer_load_ratios_avg()
        return self.__lfi.buffer_load_ratios_avg

    def f_buffer_load_ratios_std(self):
        """
        Calculates the standard deviation of the normalized buffer load
        distribution over all machine buffers.

        Individual machine buffer loads are calculated by summing up the
        processing times of all the buffered operation. The maximum
        load is used as a normalization factor.

        :return: The standard deviation of the normalized buffered processing
            time of the system.
        """
        if self.__lfi.load_ratios is None:
            self.__lfi.add_load_ratios()
        return self.__lfi.load_ratios.std()
    # </editor-fold desc="[MACHINE-CENTRIC] WORKLOAD FEATURES">

    # <editor-fold desc="[GOAL-CENTRIC] GOAL ESTIMATE FEATURES">
    def f_estimated_flow_time_ave(self):
        """
        The average estimated flow time over all jobs. The estimation is based
        on the elapsed times since job releases, to which we add the remaining
        work scaled by the inverse historic utilization average.

        The value is normalized using the original WIP durations.
        :return:
        """
        if self.__lfi.estimated_flow_time is None:
            self.__lfi.add_estimated_flow_time()
        return self.__lfi.estimated_flow_time.mean()

    def f_estimated_flow_time_std(self):
        """
        The standard deviation of the estimated flow time over all jobs.
        The estimation is based on the elapsed times since job releases,
        to which we add the remaining work scaled by the inverse historic
        utilization average.

        The value is normalized using the original WIP durations.
        :return:
        """
        if self.__lfi.estimated_flow_time is None:
            self.__lfi.add_estimated_flow_time()
        return self.__lfi.estimated_flow_time.std()

    def f_makespan_lb_ub_ratio(self):
        if self.__lfi.makespan_lb is None:
            self.__lfi.add_makespan_lb()
        if self.__lfi.makespan_ub is None:
            self.__lfi.add_makespan_ub()
        return self.__lfi.makespan_lb / self.__lfi.makespan_ub

    def f_estimated_tardiness_rate(self):
        """
        Calculates the ratio of estimated total tardy operations and total
        remaining operations. Operations are estimated to be tardy if they would
        finish after the containing job due date. It is assumed that future
        operations can be executed consecutively.

        See luo2020_djssp.

        TODO: Optimize computation by maintaining cumsums in the
            Trackers object.
        :return:
        """
        if self.__lfi.remaining_ops_sum is None:
            self.__lfi.add_remaining_ops_total()
        t = self.__lfi.bfi.state.system_time
        ecd = self.__lfi.bfi.O_D.cumsum(axis=1)  # estimated completion duration
        ect = ecd.astype('float')   # estimated completion time
        ect = (ect + t) * self.__lfi.bfi.O_D.astype('bool')  # trailing zeros!
        wip = self.__lfi.bfi.wip
        deadlines = self.__lfi.bfi.state.matrices.deadlines[wip]
        n_tardy_ops_future = np.apply_along_axis(
            lambda x: x > deadlines, axis=0, arr=ect
        ).sum()
        try:
            assert n_tardy_ops_future <= self.__lfi.remaining_ops_sum
        except AssertionError:
            print(self.__lfi.bfi.state.params.seed)
            print(self.__lfi.bfi.state.params.name)
        if self.__lfi.remaining_ops_sum == 0:
            return 0
        return n_tardy_ops_future / self.__lfi.remaining_ops_sum

    def f_tardiness_rate(self):
        """
        Calculates the ratio of of tardy and remaining operations. Tardy
        operations are defined as operations being contained by jobs whose due
        dates have elapsed.

        See luo2020_djssp.

        TODO: Optimize computation by maintaining cumsums in the
            Trackers object.
        :return:
        """
        if self.__lfi.remaining_ops_sum is None:
            self.__lfi.add_remaining_ops_total()
        t = self.__lfi.bfi.state.system_time
        estimated_completion_time = self.__lfi.bfi.O_D.cumsum(axis=1) + t
        wip = self.__lfi.bfi.wip
        elapsed_deadlines = self.__lfi.bfi.state.matrices.deadlines[wip] < t
        n_tardy_ops_current = (
                elapsed_deadlines * self.__lfi.remaining_ops
        ).sum()
        if self.__lfi.remaining_ops_sum == 0:
            return 0
        return n_tardy_ops_current / self.__lfi.remaining_ops_sum
    # </editor-fold">

    # <editor-fold desc="JOB ARRIVAL FEATURES">
    def f_wip_to_arrival_ratio(self):
        jobs_in_wip = self.__lfi.bfi.track.n_jobs_in_window
        return ((jobs_in_wip + len(self.__lfi.bfi.job_queue))
                / self.__lfi.bfi.wip_size)

    def f_wip_to_arrival_time_ratio(self):
        """
        Calculates the ratio of work in WIP to the work in both the WIP and the
        jobs that are already released but not yet in WIP.

        :return: None.
        """
        if self.__lfi.work_remaining_total is None:
            self.__lfi.add_work_remaining_total()
        wip_external_work = self.__lfi.bfi.track.job_remaining_time[
                    self.__lfi.bfi.job_queue].sum()
        wip_work = self.__lfi.work_remaining_total
        total_work = wip_external_work + wip_work
        if total_work == 0:
            return 0
        return wip_work / total_work

    def f_wip_rel_sys_t(self):
        """
        Calculates the ratio between the time elapsed since the processing of
        the current WIP began and the total work over all the original WIP jobs.

        :return: The WIP relative system time to original WIP work ratio.
        """
        if self.__lfi.work_original_sum is None:
            self.__lfi.add_work_original_sum()
        return self.__lfi.bfi.wip_rel_sys_t / self.__lfi.work_original_sum
    # </editor-fold desc="JOB ARRIVAL FEATURES">

    # <editor-fold desc="STOCHASTICITY FEATURES">
    def f_type_entropy(self):
        """
        Calculates the entropy over the type matrix of the current WIP.

        :return: The current type matraix entropy.
        """
        # Note that type_probabilities and duration_probabilities could be
        # features in their own right, though their size varies with the
        # instance
        if self.__lfi.remaining_ops_sum is None:
            self.__lfi.add_remaining_ops_total()
        if self.__lfi.remaining_ops_sum == 0:
            return 0
        type_probabilities = (self.__lfi.bfi.type_bin_counts /
                              self.__lfi.remaining_ops_sum)
        type_probabilities = type_probabilities[type_probabilities != 0]
        type_entropy = -np.sum(type_probabilities * np.log2(type_probabilities))
        # TODO: move to tests!
        # try:
        #     assert (self.__lfi.remaining_ops_total
        #             == self.__lfi.bfi.type_bin_counts.sum())
        #     assert (self.__lfi.bfi.type_bin_counts >= 0).all()
        #     assert np.isfinite(self.__lfi.bfi.type_bin_counts).all()
        # except AssertionError:
        #     print("Check Everything!")
        return type_entropy

    def f_duration_entropy(self):
        """
        Calculates the WIP operation duration matrix entropy.

        :return: The current operation duration matrix entropy.
        """
        if self.__lfi.remaining_ops_sum is None:
            self.__lfi.add_remaining_ops_total()
        if self.__lfi.remaining_ops_sum == 0:
            return 0
        duration_probabilities = (self.__lfi.bfi.duration_bin_counts
                                  / self.__lfi.remaining_ops_sum)
        duration_probabilities = duration_probabilities[
            duration_probabilities != 0]
        duration_entropy = -np.sum(
            duration_probabilities * np.log2(duration_probabilities))
        #  TODO: move to tests!
        # try:
        #     bin_sum = self.__lfi.bfi.duration_bin_counts.sum()
        #     assert (self.__lfi.remaining_ops_total == bin_sum)
        #     assert (self.__lfi.bfi.duration_bin_counts >= 0).all()
        #     assert np.isfinite(self.__lfi.bfi.duration_bin_counts).all()
        # except AssertionError:
        #     print("Check Everything2!")
        return duration_entropy

    def __get_taus(self) -> np.ndarray:
        """
        For two input lists $\tau_1$, $\tau_2$, the Kendall Tau distance counts
        the number of discordant pairs, i.e.
        $$
        K(t1, t2) = |{(i, j): i < j and ((t1(i) < t2(j) and t1(i) > t2(j)) or
                                        (t1(i) > t2(j) and t1(i) < t2(j))}|
         See https://en.wikipedia.org/wiki/Kendall_tau_distance for an example.

        The metric can be normalized with the total number of pairs possible
        index pairs (n choose 2) n * (n - 1) / 2.

        For permutations (or rankings), i.e. lists with the same numbers without
        repetitions, Kendal Tau, normalized or not, defines a distance metric,
        i.e. the three properties identity, symmetry and triangle inequality
        hold. For sequences containing repetitions, Kendall Tau is still well
        defined, albeit not a metric anymore.

        Kendal Tau can be calculated by counting the number of bubble-sort swaps
        required to transform one sequence into the other. An implementation
        building upon merge-sort has an asymptotic runtime of O(n log(n)).
        For python, there is a scipy implementation of the Kendal coefficient
        normalized between -1 and 1. See the documentation at
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html

        This function uses the scipy implementation and abuse the coefficient
        somewhat (because our operation type sequences are not necessarily
        rankings) to calculate the Kendall tau coefficient between the job
        operation type sequences in WIP and the identity permutation
        1, 2, ..., max_n_job_ops. If the job operation types are all zero,
        kendalltau returns a nan. We handle this by returning a suggestive 0
        correlation in such cases.

        :return: The Kendall Tau coefficients between the operation type
            sequence and the reference identity permutation vector for every
            job.
        """
        op_type_matrix = self.__lfi.bfi.O_T
        n = op_type_matrix.shape[0]
        taus = np.zeros(n)
        reference = np.arange(op_type_matrix.shape[1]) + 1
        for j in range(n):
            kt = kendalltau(reference, op_type_matrix[j], nan_policy='omit')[0]
            if not np.isfinite(kt):
                taus[j] = 0
            else:
                taus[j] = kt
        return taus

    def f_kendalltau_ave(self):
        """
        Calculate the mean the Kendall tau coefficient between the job operation
        type sequences in WIP and the identity permutation
        (1, 2, ..., max_n_job_ops).

        :return: The average tau coefficient.
        """
        if self.__lfi.taus is None:
            self.__lfi.add_taus(self.__get_taus())
        return self.__lfi.taus.mean()

    def f_kendalltau_std(self):
        """
        Calculate the standard deviation the Kendall tau coefficient between the
        job operation type sequences in WIP and the identity permutation
        (1, 2, ..., max_n_job_ops).

        :return: The average tau coefficient.
        """
        if self.__lfi.taus is None:
            self.__lfi.add_taus(self.__get_taus())
        return self.__lfi.taus.std()

    def f_type_hamming_mean(self):
        """
        Calculates the mean hamming distance between all job operation type
        sequences.

        :return: The average hamming distance between the job type sequences.
        """
        if self.__lfi.d_hamming is None:
            self.__lfi.add_d_hamming()
        if self.__lfi.d_hamming_max is None:
            self.__lfi.add_d_hamming_max()
        if self.__lfi.d_hamming_max == 0:
            return 0
        return self.__lfi.d_hamming.mean() / self.__lfi.d_hamming_max

    def f_type_hamming_std(self):
        """
        Calculates the standard deviation of the hamming distances between all
        job operation type sequences.

        :return: The standard deviation of the hamming distances between the job
            type sequences.
        """
        if self.__lfi.d_hamming is None:
            self.__lfi.add_d_hamming()
        if self.__lfi.d_hamming_max is None:
            self.__lfi.add_d_hamming_max()
        if self.__lfi.d_hamming_max == 0:
            return 0
        return self.__lfi.d_hamming.std() / self.__lfi.d_hamming_max

    def f_duration_distance_mean(self):

        """
        Calculates the mean distance between all job duration vectors.

        :return: The mean distance between job duration vectors normalized by
            the maximum distance.
        """
        if self.__lfi.d_hamming is None:
            self.__lfi.add_d_hamming()
        if self.__lfi.d_hamming_max is None:
            self.__lfi.add_d_hamming_max()
        if self.__lfi.d_hamming_max == 0:
            return 0
        return self.__lfi.d_hamming.mean() / self.__lfi.d_hamming_max

    def f_duration_distance_std(self):
        """
        Calculates the standard deviation of the distribution of distances
        between all job duration vectors.

        :return: The standard deviation of the distribution of distances between
            job duration vectors.
        """
        if self.__lfi.d_euclidean is None:
            self.__lfi.add_d_euclidean()
        if self.__lfi.d_euclidean_max is None:
            self.__lfi.add_d_euclidean_max()
        if self.__lfi.d_euclidean_max == 0:
            return 0
        return self.__lfi.d_euclidean.mean() / self.__lfi.d_euclidean_max

    def ff_permutation_distance_ave(self):
        """
        Future feature (ff). Some editing distance between the different jobs.
        """
        # TODO: consider following additional "permutation distances":
        #   kendal tau, cycle rearrangement distance, l1 distance
        #  see this post:
        #  https://math.stackexchange.com/questions/2492954/distance-between-two-permutations
    # </editor-fold desc="STOCHASTICITY FEATURES">

    # <editor-fold desc="DECISION FEATURES">
    def f_decision_skip_ratio(self):
        n_decisions_skipped = self.__lfi.bfi.decision_skips
        n_decisions_total = self.__lfi.bfi.decisions_total
        return (n_decisions_skipped / n_decisions_total
                if n_decisions_total != 0 else 0)

    def f_legal_action_job_ratio(self):
        """
        Calculates the ratio between the currently legal direct actions with
        respect to the maximum possible legal actions for most cases, namely
        the number of wip jobs.

        Note that in rare cases where the scheduling problem contains both a
        reoccurring operations within individual jobs (recrc, vnops) and partial
        order precedence constraints, the maximum number of legal operations
        may exeed the number of wip jobs.
        :return:
        """
        # TODO: wip jobs, not total jobs!
        return len(self.__lfi.bfi.legal_actions) / self.__lfi.bfi.wip_size

    def f_heuristic_agreement_entropy(self):
        """
        Calculates and return the entropy for the operation indices yielded by
        the 10 benchmark heuristics. The

        :return: The heuristic decision overlap entropy.
        """
        hs = [SPT(), LPT(), LOR(), MOR(), SRPT(),
              LRPT(), LTPO(), MTPO(), EDD(), LUDM()]
        h_agreement = np.zeros(len(hs))
        if (self.__lfi.bfi.legal_actions and
                self.__lfi.bfi.state.scheduling_mode == 0):
            encoding = 0
            # def f(h): return h.get_action(self.lfi.state)
            # vf = np.vectorize(f, otypes=[np.int32])
            # vf(hs)
            # TODO: adapt to fit with routing!!!
            for i in range(h_agreement.shape[0]):
                h_agreement[i] = hs[i].get_action(self.__lfi.bfi.state)
            _, decision_counts = np.unique(h_agreement, return_counts=True)
            decision_probabilities = decision_counts / len(hs)
            return -np.sum(decision_probabilities
                           * np.log2(decision_probabilities))
        else:  # this is the end state which needs no action ;)
            return 0

    def f_legal_action_len_stream_ave(self):
        la_array = (np.array(self.__lfi.bfi.legal_action_lengths)
                    / self.__lfi.bfi.wip_size)
        return la_array.mean()

    def f_legal_action_len_stream_std(self):
        la_array = (np.array(self.__lfi.bfi.legal_action_lengths)
                    / self.__lfi.bfi.wip_size)
        return la_array.std()
    # </editor-fold desc="DECISION FEATURES">

    # <editor-fold desc="MATRIX DESCRIPTION FEATURES">
    def f_type_ave(self) -> float:
        """
        Computes the min-max scaled type matrix mean. Note that this is not
        expected to be very impactful since it interprets an underlying
        categorical variable as nummeric.

        :return: The normalized type mean.
        """
        if self.__lfi.O_T_max is None:
            self.__lfi.add_type_matrix_maximum()
        if self.__lfi.O_T_min is None:
            self.__lfi.add_type_matrix_minimum()
        t_max = self.__lfi.O_T_max
        t_min = self.__lfi.O_T_min
        if t_max == 0:
            return 0.0
        elif t_max == t_min:
            return 1.0
        else:
            normalized_avg = ((self.__lfi.bfi.O_T.mean() - t_min)
                              / (t_max - t_min))
            return normalized_avg

    def f_type_std(self) -> float:
        """
        Computes the min-max scaled type matrix standard deviation.
        Note that this is not expected to be very impactful since it interprets
        an underlying categorical variable as nummeric.

        :return: The normalized type standard deviation.
        """
        if self.__lfi.O_T_max is None:
            self.__lfi.add_type_matrix_maximum()
        if self.__lfi.O_T_min is None:
            self.__lfi.add_type_matrix_minimum()
        t_max = self.__lfi.O_T_max
        t_min = self.__lfi.O_T_min
        if t_max == 0:
            return 0.0
        elif t_max == t_min:
            return 1.0
        else:
            normalized_std = ((self.__lfi.bfi.O_T.std() - t_min) /
                              (t_max - t_min))
            return normalized_std

    def f_duration_ave(self) -> float:
        """
        Computes the min-max scaled duration matrix mean.

        :return: The normalized duration mean.
        """
        if self.__lfi.O_D_max is None:
            self.__lfi.add_duration_matrix_maximum()
        if self.__lfi.O_D_min is None:
            self.__lfi.add_duration_matrix_minimum()

        d_max = self.__lfi.O_D_max
        d_min = self.__lfi.O_D_min
        if d_max == 0:
            return 0.0
        elif d_max == d_min:
            return 1.0
        else:
            normalized_avg = ((self.__lfi.bfi.O_D.mean() - d_min)
                              / (d_max - d_min))
            return normalized_avg

    def f_duration_std(self) -> float:
        """
        Computes the min-max scaled duration matrix standard deviation.

        :return: The normalized duration standard deviation.
        """
        if self.__lfi.O_D_max is None:
            self.__lfi.add_duration_matrix_maximum()
        if self.__lfi.O_D_min is None:
            self.__lfi.add_duration_matrix_minimum()
        d_max = self.__lfi.O_D_max
        d_min = self.__lfi.O_D_min
        if d_max == 0:
            return 0.0
        elif d_max == d_min:
            return 1.0
        else:
            normalized_std = ((self.__lfi.bfi.O_D.std() - d_min)
                              / (d_max - d_min))
            return normalized_std
    # </editor-fold desc="MATRIX DESCRIPTION FEATURES">

    # <editor-fold desc="CLUSTERING FEATURES">
    def __silhouette_over_kmeans(self, k):
        """
        Calculates the silhouette score over a KMeans generated clustering for
        samples made up of concatenated WIP job types and durations. This
        function makes sure the k parameter passed to KMeans is never smaller
        than two or smaller than the number of unique job vectors in WIP.

        :param k: The number of clusters to pass to KMeans.
        :return: The silhouette score of the KMeans clustering for the current
            WIP jobs.
        """
        if self.__lfi.job_clustering_vecs is None:
            self.__lfi.add_job_clustering_vec()
        n_unique = np.unique(self.__lfi.job_clustering_vecs, axis=0).shape[0]
        if n_unique < 2:
            return 0
        elif n_unique < k:
            k = n_unique
        model = KMeans(n_clusters=k).fit(self.__lfi.job_clustering_vecs)
        centers = model.predict(self.__lfi.job_clustering_vecs)
        return silhouette_score(self.__lfi.job_clustering_vecs, centers)

    def f_silhouette_kmeans_min_k(self):
        """
        Calculates the silhouette score for the KMeans clustered WIP jobs
        assuming there are only two clusters present in the data. The
        samples are generated by concatenating operation types and durations.

        :return: The silhouette score of the WIP job KMeans clustering with k=2.
        """
        return self.__silhouette_over_kmeans(2)

    def f_silhouette_kmeans_max_k(self):
        """
        Calculates the silhouette score for the KMeans clustered WIP jobs
        assuming there number of clusters present in the data is equal to the
        number of samples - 1. The samples are generated by concatenating
        operation types and durations.

        :return: The silhouette score of the WIP job KMeans clustering with k=2.
        """
        return self.__silhouette_over_kmeans(self.__lfi.bfi.wip_size - 1)

    def f_silhouette_kmeans_mid_k(self):
        """
        Calculates the silhouette score for the KMeans clustered WIP jobs
        assuming there number of clusters present in the data is equal to the
        half the number of samples. The samples are generated by concatenating
        operation types and durations.

        :return: The silhouette score of the WIP job KMeans clustering with k=2.
        """
        return self.__silhouette_over_kmeans(int(self.__lfi.bfi.wip_size / 2))
    # </editor-fold desc="CLUSTERING FEATURES">

    # <editor-fold desc="FEATURE ACCUMULATION FUNCTIONS">
    def __add_clustering_features(self, container):
        for fea_name in [
            'silhouette_kmeans_min_k',
            'silhouette_kmeans_mid_k',
            'silhouette_kmeans_max_k'
        ]:
            container.append(getattr(self, f'f_{fea_name}')())

    def __add_global_matrix_features(self, container):
        for fea_name in [
            'type_ave',
            'type_std',
            'duration_ave',
            'duration_std'
        ]:
            container.append(getattr(self, f'f_{fea_name}')())

    def __add_throughput_features(self, container: List):
        for fea_name in [
            # THROUGHPUT OP FEATURES
            'op_completion_rate',
            # THROUGHPUT TIME FEATURES
            'wip_job_completion_rate_std',
            'wip_job_completion_rate_ave',
            'completed_time_ratio',
            'work_throughput',
            'throughput_time_j_rel_avg',
            'throughput_time_j_rel_std',
            'throughput_time_j_abs_avg',
            'throughput_time_j_abs_std'
        ]:
            container.append(getattr(self, f'f_{fea_name}')())

    def __add_workload_features(self, container: List):
        for fea_name in [
            # MACHINE FEATURES
            'utl_avg',
            'utl_std',
            'buffer_load_ratios_avg',
            'buffer_load_ratios_std',
            # WORKLOAD FEATURES
            'remaining_time_ratio',
            'remaining_times_wip_avg',
            'remaining_times_wip_std',
            'remaining_ops_wip_avg',
            'remaining_ops_wip_std'
        ]:
            container.append(getattr(self, f'f_{fea_name}')())

    def __add_bounds_and_arrival_ratio(self, container):
        for fea_name in [
            'makespan_lb_ub_ratio',
            'wip_to_arrival_ratio',
            'wip_to_arrival_time_ratio',
            'sys_t_rel'
        ]:
            container.append(getattr(self, f'f_{fea_name}')())

    def __add_stochasticity_metrics(self, container: List):
        for fea_name in [
            'type_entropy',
            'duration_entropy',
        ]:
            container.append(getattr(self, f'f_{fea_name}')())

    def __add_decision_features(self, container):
        for fea_name in [
            'decision_skip_ratio',
            'legal_action_job_ratio',
            'heuristic_agreement_entropy',
            'legal_action_len_stream_ave',
            'legal_action_len_stream_std'
        ]:
            container.append(getattr(self, f'f_{fea_name}')())

    def accumulate_all_features(self, state: State) -> List:
        feature_list = []
        self.__add_throughput_features(feature_list)
        self.__add_workload_features(feature_list)
        self.__add_bounds_and_arrival_ratio(feature_list)
        self.__add_stochasticity_metrics(feature_list)
        self.__add_decision_features(feature_list)
        self.__add_global_matrix_features(feature_list)
        self.__add_clustering_features(feature_list)
        return feature_list
    # </editor-fold desc="FEATURE ACCUMULATION FUNCTIONS">

    # <editor-fold desc="LOCAL MACHINE FEATURES">
    def f_buffer_length_ratio(self):
        """
        Calculates and returns the ratio of the operations buffered at the
        current machine relative to the total remaining operations.

        :return: The buffer length ratio.
        """
        m = self.__lfi.bfi.state.current_machine_nr
        if self.__lfi.remaining_ops_sum is None:
            self.__lfi.add_remaining_ops_total()
        norm = self.__lfi.remaining_ops_sum
        if norm == 0:
            return 0
        else:
            return self.__lfi.bfi.state.trackers.buffer_lengths[m - 1] / norm

    def f_buffer_time_ratio(self):
        """
        Calculates and returns the ratio of the total estimated processing time
        buffered at the current machine relative to the total remaining
        operation durations.

        :return: The buffered time (work) ratio.
        """
        m = self.__lfi.bfi.state.current_machine_nr
        if self.__lfi.work_remaining_total is None:
            self.__lfi.add_work_remaining_total()
        if self.__lfi.work_remaining_total == 0:
            # all the buffer should have a length of 0
            return self.__lfi.bfi.state.trackers.buffer_lengths[m - 1]
        return (self.__lfi.bfi.state.trackers.buffer_lengths[m - 1]
                / self.__lfi.work_remaining_total)
    # </editor-fold">

    def get_feature_names(self):
        available_features = []
        for obj_attr in dir(self):
            if obj_attr.startswith('f_'):
                available_features.append(obj_attr[2:])
        return available_features

    def transform_state(self, state: State) -> np.ndarray:
        self.__get_state_base_nfo(state)
        feature_vector = []
        if self.__feature_list:
            for fea_name in self.__feature_list:
                feature_vector.append(getattr(self, f'f_{fea_name}')())
        else:
            feature_vector = self.accumulate_all_features(state)
        # assert not np.isnan(ret).any()
        # assert np.isfinite(ret).all()
        return np.array(feature_vector)


class JmMCTSFeatureState(StateTransformer):
    def __init__(self):
        super().__init__()
        # self.ur = UtilScaledReward()
        self.feature_extractor = FeatureTransformer(feature_list=[
            # 'n_steps',  # 1
            'legal_action_len_stream_ave',  # 2
            'legal_action_len_stream_std',  # 3
            'utl_avg',  # 4
            'utl_std',  # 5
            'op_completion_rate',  # 6
            'remaining_times_wip_std',  # 7
            'wip_to_arrival_time_ratio',  # 8
            'type_entropy',  # 9
            'sys_t_rel',  # 10
            'wip_to_arrival_ratio',  # 11
            'type_std',  # 12
            'throughput_time_j_abs_std',  # 13
            'duration_entropy',  # 14
            'remaining_times_wip_avg',  # 15
            'buffer_time_ratio',  # 16
            # 'n_jobs_done',  # 17
            'throughput_time_j_rel_std',  # 18
            'makespan_lb_ub_ratio'  # 19
        ])

    def transform_state(self, state: State):
        feature_vector = self.feature_extractor.transform_state(state)
        return feature_vector
