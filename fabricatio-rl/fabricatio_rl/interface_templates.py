from collections import deque
from os import getcwd
from typing import Union, List, Dict, Callable, Tuple, TypeVar, TYPE_CHECKING

import numpy as np

from fabricatio_rl.env_utils import faster_deepcopy, GraphUtils


if TYPE_CHECKING:
    from fabricatio_rl.core_state import State
    # indicates generic types
    T = TypeVar('T')
    SamplingFunction2D = Callable[[Tuple[int, int]], np.ndarray]
    SamplingFunction1D = Callable[[int], np.ndarray]
    PrecedenceDictionary = Dict[Tuple[int], List[int]]


class Optimizer:
    def __init__(self, target_mode):
        assert target_mode in ['sequencing', 'transport', 'universal']
        self.target_mode = target_mode

    def get_action(self, state: 'State') -> int:
        raise NotImplementedError

    def update(self, action: int, state: 'State'):
        """
        Method to be called after the core action has been executed. To update
        any particular optimizer attributes that depend on the action taken.
        An example of this is the case of the SimulationSearch optimizer which
        maintains a copy of the simulation that needs to be rolled forward
        concomitantly with the embedding simulation.

        :param state:
        :param action: The direct action used within the current step of the
            simulation.
        :return: None
        """
        pass


class ReturnTransformer:
    def transform_state(self, state: 'State') -> np.ndarray:
        """
        Transforms the environment state into the agent state. Note that a
        numpy array has to be returned for compatibility with the environment
        interface state shape inference.

        :param state: The state as defined by the environment Core.
        :return: The state representation as a numpy array.
        """
        raise NotImplementedError

    def transform_reward(self, state: 'State', illegal=False, environment=None):
        return state.system_time

    def reset(self):
        pass


class FJSSPInstanceInfo(object):
    def __init__(self):
        self.t_next = 1  # type counter
        self.T_dict = {}
        self.n_jobs = -1
        self.n_machines = -1
        self.op_types = np.array([[]])  # dummy
        self.op_durations = np.array([[]])  # dummy
        self.n_ops: Union[np.array, None] = []  # dummy
        self.machine_capabilities = {}  # dummy
        self.machine_speeds = np.array([])  # dummy
        self.As = deque([])  # operation durations on machines
        self.bs = deque([])  # duration total
        self.speed_constraint_ids = set({})

    def initialize_machine_speeds(self):
        assert self.n_machines != -1
        self.machine_speeds = np.zeros(self.n_machines, dtype='uint16')

    def add_speed_restrictions(self, machine_indexes, operation_durations):
        duration_vec = np.zeros(self.n_machines)
        duration_vec[machine_indexes] = operation_durations
        d_nonzero = operation_durations[operation_durations != 0]
        if d_nonzero.shape[0] == 0:
            return
        ref = d_nonzero.min()
        a = duration_vec  # / ref
        b = ref * len(d_nonzero)
        # c = (tuple(a), b)
        # c_identifier = hash(c)
        # if c_identifier in self.speed_constraint_ids:
        #     return
        # self.speed_constraint_ids.add(c_identifier)
        self.As.append(a)
        self.bs.append(b)
        # for comb in combinations(range(machine_indexes.shape[0]), 2):
        #     # print(n_machines, comb)
        #     speed_rel = np.zeros((self.n_machines,))
        #     # print(speed_rel[machine_indexes[list(comb)]])
        #     ratios = operation_durations[list(comb)]
        #     ratios[-1] = -ratios[-1]
        #     speed_rel[machine_indexes[list(comb)]] = ratios
        #     self.speed_constraint_coeffs.append(speed_rel)

    def get_best_speed_solution(self):
        # TODO: assert speeds are non-negative and non-zero!
        As = np.array(self.As)
        bs = np.array(self.bs)
        Abs = np.hstack([As, bs.reshape(As.shape[0], -1)])
        r_A, r_Ab = np.linalg.matrix_rank(Abs), np.linalg.matrix_rank(As)
        if r_Ab > r_A:
            print("Underdetermined System! "
                  "The machine speeds are just approximations.")
        elif r_Ab < r_A:
            print("Overdetermined system. Filling in the zeros...")
        speeds, residuals, duration_contrib_matrix_rank, _ = np.linalg.lstsq(
            As, bs, rcond=None)
        print(speeds, residuals)
        return speeds
        # S = np.array(self.speed_constraint_coeffs)
        # TODO: assert rank(S) <= S.shape[1], i.e. system is at most well
        #    defined. if a system were overdetermined, that would mean that the
        #    machine speeds are unrelated, which violates the definition of a
        #    FJc (aka fjsp aka fjssp)); use np.linalg.matrix_rank (
        #    https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html)
        # q, r = np.linalg.qr(S, mode='complete')
        # S_reduced_l = []
        # max_rank = min(r.shape)
        # for i in range(max_rank):
        #     if r[i][i] != 0:
        #         S_reduced_l.append(S[i])
        # S_reduced = np.array(S_reduced_l)
        # S_sqared = np.dot(S_reduced.T, S_reduced)
        # eig_vals, eig_vecs = np.linalg.eig(S_sqared)
        # speeds = eig_vecs[:, np.argmin(eig_vals)].T
        # speed_unit_scaler = 1 / (speeds[speeds != 0].min())
        # speeds_scaled = (speeds * speed_unit_scaler).astype('uint8')
        # speeds_scaled[speeds_scaled == 0] = 1
        # return speeds_scaled

    def initialize_frame_information(self, segs):
        self.n_jobs, self.n_machines = int(segs[0]), int(segs[1])
        self.n_ops = np.zeros(self.n_jobs, dtype='uint16')


class SchedulingUserInputs:
    def __init__(self, n_jobs=20, n_machines=20, n_tooling_lvls=0, n_types=20,
                 min_n_operations=20, max_n_operations=20, n_operations=None,
                 max_n_failures=0, n_jobs_initial=-1, max_jobs_visible=20,
                 operation_precedence: Union[
                     np.ndarray, List['PrecedenceDictionary'], str] = 'Jm',
                 operation_types: Union[
                     np.ndarray, 'SamplingFunction2D', str] = 'Jm',
                 operation_durations: Union[
                     np.ndarray, 'SamplingFunction2D', str
                 ] = 'default_sampling',
                 operation_tool_sets: Union[
                     np.ndarray, 'SamplingFunction2D', str, None] = None,
                 job_pool: Union[np.ndarray, None] = None,
                 inter_arrival_time: Union[
                     np.ndarray, 'SamplingFunction1D', str] = 'under',
                 due_dates: Union[
                     np.ndarray, float, str] = 'default_sampling',
                 perturbation_processing_time: Union[
                     np.ndarray, 'SamplingFunction2D', float, str, None] = None,
                 perturbation_due_date='',
                 machine_speeds=None,
                 machine_distances=None,
                 machine_buffer_capacities=None,
                 machine_capabilities=None,
                 tool_switch_times=None,
                 machine_failures=None,
                 name='', path='', fixed_instance=False):
        # DIMENSIONS
        self.__n_jobs = n_jobs  # n
        self.__n_machines = n_machines  # m
        self.__n_tooling_lvls = n_tooling_lvls  # l
        self.__n_types = n_types  # t
        self.__min_n_operations = min_n_operations
        self.__max_n_operations = max_n_operations  # o
        self.__n_operations = n_operations
        self.__max_n_failures = max_n_failures  # f
        self.__n_jobs_initial = n_jobs_initial  # jobs with arrival time 0
        self.__max_jobs_visible = max_jobs_visible  # entries in {1 .. n}
        # MATRIX INFORMATION
        self.__operation_precedence = operation_precedence  # 'Jm', 'Fm', 'POm'
        self.__operation_types = operation_types  # 'Jm', 'Fm'
        self.__operation_durations = operation_durations
        self.__operation_tool_sets = operation_tool_sets
        self.__job_pool = job_pool
        self.__inter_arrival_time = inter_arrival_time  # or 'over'/'balanced'
        self.__due_dates = due_dates  # TODO: loose, tight, balanced :)
        self.__perturbation_processing_time = perturbation_processing_time
        self.__perturbation_due_date = perturbation_due_date  # TODO!
        # MACHINE INFORMATION
        self.__machine_speeds = machine_speeds
        self.__machine_distances = machine_distances
        self.__machine_buffer_capacities = machine_buffer_capacities
        self.__machine_capabilities = machine_capabilities
        self.__type_indexed_capabilities = False
        self.__tool_switch_times = tool_switch_times
        self.__machine_failures = machine_failures
        # INSTANCE INFO
        self.__path = path
        self.__name = name
        if self.__path != '':
            self.__set_pool_from_file()
            self.__set_instance_name(self.path)
        # USE EXACT INSTANCE
        self.fixed_instance = fixed_instance

    def __deepcopy__(self, memodict):
        return faster_deepcopy(self, memodict)

    def __init(self, scheduling_input: Dict[str, 'T']):
        """
        DEPRECATED!

        Initializes the user input container from the scheduling_input
        dictionary passed to the simulation. The scheduling_input keys need to
        perfectly match the SchedulingUserInput class attributes. To make sure
        that no illegal state can be reached, this method first checks the
        scheduling_input keys against the object properties. Should a key value
        be an undefined SchedulingUserInput property, an AttributeError with a
        message listing the legal properties is raised.

        :param scheduling_input: The dictionary of user inputs
        :return: None.
        """
        for key in scheduling_input:
            mangled_key = f'_{self.__class__.__name__}__{key}'
            if mangled_key in self.__dict__:
                setattr(self, mangled_key, scheduling_input[key])
            else:
                fields_string = ', '.join(self.__dict__.keys())
                raise AttributeError(
                    f"The scheduling_inputs dictionary does not "
                    f"support the {key} key. Accepted keys are:"
                    f" {fields_string}")

    def __set_pool_from_file(self):
        if (self.operation_types == 'Jm'
            # TODO: use custom JSSP information container (analogous nfo ;)
            and self.operation_precedence == 'Jm'
                and self.machine_capabilities is None):  # problem: Jm
            nfo = SchedulingUserInputs.extract_jssp_benchmark_info(self.path)
            n_jobs, n_machines, job_pool = nfo
            self.__job_pool = np.array(job_pool)
            if self.__n_jobs_initial == -1:
                self.__n_jobs = n_jobs
                self.__n_jobs_initial = n_jobs
                self.__max_jobs_visible = n_jobs
            self.__n_types = n_machines
            self.__n_machines = n_machines
            self.__n_operations = np.repeat(n_machines, n_jobs)
            self.__max_n_operations = n_machines
            self.__min_n_operations = n_machines
        elif (self.operation_types == 'Jm'
              and self.operation_precedence == 'Jm'
              and self.machine_capabilities is not None):  # problem: FJc
            ret = SchedulingUserInputs.extract_fjssp_benchmark_info(self.path)
            nfo, jobs = ret
            if self.__n_jobs_initial == -1:
                self.__n_jobs = nfo.n_jobs
                self.__n_jobs_initial = nfo.n_jobs
                self.__max_jobs_visible = nfo.n_jobs
            self.__n_types = len(nfo.T_dict)
            self.__n_machines = nfo.n_machines
            self.__n_operations = nfo.n_ops
            # noinspection PyArgumentList
            self.__max_n_operations = self.__n_operations.max()
            # noinspection PyArgumentList
            self.__min_n_operations = self.__n_operations.min()
            self.__machine_speeds = nfo.get_best_speed_solution()
            self.__machine_capabilities = nfo.machine_capabilities
            self.__type_indexed_capabilities = True
            job_pool = []
            job_template_vec = np.zeros(self.__max_n_operations, dtype='uint16')
            for i in range(len(jobs)):
                j_precedence = GraphUtils.graph_chain_precedence(
                    list(range(nfo.n_ops[i])))
                j_precedence[(0,)] = [0]
                job_op_sequence = job_template_vec.copy()
                job_op_sequence[:nfo.n_ops[i]] = jobs[i][0]
                job_op_durations = job_template_vec.copy()
                job_op_durations[:nfo.n_ops[i]] = jobs[i][1]
                job_pool.append([job_op_sequence, job_op_durations,
                                 job_template_vec, j_precedence])
            self.__job_pool = np.array(job_pool)
        # TODO!!!
        # elif (self.operation_types == 'Fm'
        #       and self.operation_precedence == 'Jm'
        #       and self.machine_capabilities is None):   # problem: Fm
        #     pass
        # elif (self.operation_types == 'Fm'
        #       and self.operation_precedence == 'Jm'
        #       and self.machine_capabilities is not None):  # problem: FFc
        #     pass
        # elif (self.operation_types == 'Jm'
        #       and self.operation_precedence == 'Om'
        #       and self.machine_capabilities is None):  # problem: Om
        #     pass
        # elif (self.operation_types == 'Jm'
        #       and self.operation_precedence == 'POm'
        #       and self.machine_capabilities is None):  # problem: POm
        #     pass
        # elif (self.operation_types == 'Jm'
        #       and self.operation_precedence == 'FPOc'
        #       and self.machine_capabilities is None):  # problem: FPOc
        #     pass
        # else:
        #     pass
        #     # raise error ;)

    def __set_instance_name(self, f_name: str):
        while f_name.startswith('.') or f_name.startswith('/'):
            f_name = f_name[1:]
        self.__name = f_name[:-4]

    # <editor-fold desc="JSSP Benchmark Loaders">
    @staticmethod
    def extract_jssp_benchmark_info(benchmark_f_name: str) -> \
            (int, int, np.ndarray, np.ndarray):
        """
        Reads JSSP file and extracts the duration and type matrices and
        transfers them to a format accepted by FabrikatioRL.

        The JSSP instances are expected to be in the standard specification (see
        http://jobshop.jjvh.nl/explanation.php), meaning that rows in the file
        correspond to jobs. The types of the machine to be visited alternate
        with the duration specification, with even row entries i corresponding
        to the machine types to be visited and the immediate odd entries i + 1
        to the associated operation durations. The type numbering starts at 0.
        The first row in the file contains the total number of jobs and
        operations separated by a whitespace.

        FabrikatioRL indexes types starting at 1, so the read type matrix will
        be changed correspondingly. Note that some benchmarking instances (e.g.
        orb07) contain 0 durations and 0 types. So as not to confuse Fabrikatio,
        the types corersponding to 0 durations are zeroed out.

        :param benchmark_f_name: The path to the JSSP benchmark file.
        :return: The tuple of containing the number of machines, the number of
            jobs, as well as the Fabricatio format operation type and duration
            matrices.
        """
        print(getcwd())
        instance_file = open(benchmark_f_name)
        n_jobs, n_machine_groups = map(
            int, instance_file.readline().strip().split(' '))
        i = 0
        job_pool = []
        for line in instance_file:
            line_elements = line.strip().split('  ')
            j_sequence = np.zeros(n_machine_groups, dtype='int16')
            j_durations = np.zeros(n_machine_groups, dtype='int16')
            if bool(line_elements):
                for j in range(len(line_elements)):
                    if j % 2 == 0:
                        j_sequence[j // 2] = int(line_elements[j]) + 1
                    else:
                        j_durations[j // 2] = int(line_elements[j])
                        if line_elements[j] == 0:
                            j_sequence[j // 2] = 0
                j_tooling_times = np.zeros(n_machine_groups, dtype='int16')
                j_precedence = GraphUtils.graph_chain_precedence(
                    list(range(n_machine_groups)))
                j_precedence[(0,)] = [0]
                job_pool.append([j_sequence, j_durations,
                                 j_tooling_times, j_precedence])
                i += 1
        instance_file.close()
        return n_jobs, n_machine_groups, job_pool
    # </editor-fold desc="Benchmark Loaders">

    # <editor-fold desc="FJSSP Benchmark Loaders">
    @staticmethod
    def process_job_information(nfo: FJSSPInstanceInfo, segs: List[str],
                                j_nr: int):
        op_idx = 0
        idx_next_op = 1
        job_op_types, job_op_durations = [], []
        while idx_next_op < len(segs):
            # for this operation
            n_tgt_machines = int(segs[idx_next_op])
            op_durations = []
            op_machines = []
            for i in range(n_tgt_machines):
                tgt_m = int(segs[idx_next_op + 2 * i + 1])
                tgt_m_duration = int(segs[idx_next_op + 2 * i + 2])
                if tgt_m_duration != 0:
                    # make sure there is no zero duration in the future
                    op_durations.append(tgt_m_duration)
                    op_machines.append(tgt_m)
            if len(op_machines) != 0:
                nfo.n_ops[j_nr] += 1
                # add op type and update entries in T matrix
                if tuple(op_machines) in nfo.T_dict:
                    job_op_types.append(nfo.T_dict[tuple(op_machines)])
                else:
                    nfo.T_dict[tuple(op_machines)] = nfo.t_next
                    job_op_types.append(nfo.t_next)
                    nfo.machine_capabilities[nfo.t_next] = op_machines
                    nfo.t_next += 1
                # add speed restrictions
                # if n_tgt_machines > 1:
                nfo.add_speed_restrictions(
                    np.array(op_machines) - 1, np.array(op_durations))
                # add op base duration
                job_op_durations.append(min(op_durations))
            # update loop vars
            idx_next_op = idx_next_op + n_tgt_machines * 2 + 1
            op_idx += 1
        return job_op_types, job_op_durations

    @staticmethod
    def extract_fjssp_benchmark_info(path):
        f = open(path)
        nfo = FJSSPInstanceInfo()
        jobs = []
        j_nr = 0
        # get max n_ops and init matrices
        for line in f:
            segs = line.strip().split()
            if len(segs) == 0:
                continue
            if len(segs) == 3:
                nfo.initialize_frame_information(segs)
                nfo.initialize_machine_speeds()
                continue
            jt, jd = SchedulingUserInputs.process_job_information(
                nfo, segs, j_nr)
            j_nr += 1
            jobs.append([jt, jd])
        f.close()
        return nfo, jobs
    # </editor-fold desc="FJSSP Benchmark Loaders">

    # <editor-fold desc="Dimension Getters">
    @property
    def n_operations(self):
        """
        The number of operations per job defined by the user. This can be either
        a numpy array of integers of size n (1), where n is the total number of
        jobs in the scheduling problem, the 'random_sampling' string (2) or
        None (3).

        In case of (1) the array will be used as is. In case of (2) the number
        of operations per job will be sampled uniformly from {min_n_operations,
        ...,  max_n_operations}. In case of (3) all job operations will be the
        same. Note that in this case min_n_operations has to be equal to
        max_n_operations.

        :return: The user parameter specifying the number of operations per job.
        """
        return self.__n_operations

    @property
    def n_jobs(self):
        """
        An integer marking the total number of jobs for the given scheduling
        problem.

        :return: The total number of jobs defined by the user.
        """
        return self.__n_jobs

    @property
    def n_machines(self):
        """
        An integer marking the number of machines in the scheduling problem.

        :return: The user defined number of machines
        """
        return self.__n_machines

    @property
    def n_tooling_lvls(self):
        """
        An integer specifying the total number of distinct tool sets present.

        :return: The number of tool sets defined by the user.
        """
        return self.__n_tooling_lvls

    @property
    def n_types(self):
        """
        The total number of distinct operations types.

        :return: The user defined number of operation types.
        """
        return self.__n_types

    @property
    def min_n_operations(self):
        """
        The minimum allowed operations per job.

        :return: The user defined minimum allowed operations per job.
        """
        return self.__min_n_operations

    @property
    def max_n_operations(self):
        """
        The maximum allowed operations per job.

        :return: The user defined maximum allowed operations per job.
        """
        return self.__max_n_operations

    @property
    def max_jobs_visible(self):
        """
        The maximum number of jobs visible to the scheduling agents at any time
        during the simulation; Corresponds to the work in progress (WIP) window
        size.

        :return: The WIP size specified by the user.
        """
        return self.__max_jobs_visible

    @property
    def max_n_failures(self):
        """
        The maximum number of machine failures to simulate.

        :return: The user specified number of machine failures.
        """
        return self.__max_n_failures

    @property
    def n_jobs_initial(self):
        """
        The initial number of visible jobs (initial jobs in WIP).

        :return: The number of jobs before any new arrivals specified by the
            user.
        """
        return self.__n_jobs_initial

    # </editor-fold desc="DIMENSION GETTERS">

    # <editor-fold desc="Job Matrices Information Getters">
    @property
    def job_pool(self):
        """
        A numpy matrix of dimension k x 3 x max_n_operations jobs to sample
        from. The jobs need to respect the scheduling problem dimension
        properties (e.g. n_operations).

        Each job entry in the pool matrix consists of three arrays of equal size
        corresponding to operation types, operation durations and operation
        tooling levels.

        @see: The constructor of the JobMatrices class.

        :return: The job pool specified by the user.
        """
        return self.__job_pool

    @property
    def operation_types(self):
        """
        A matrix of dimension n_jobs x max_n_operations representing the
        operation types (1), a sampling function taking a 2d-shape as a
        parameter and returning a numpy array (2), or one of the
        'default_sampling' (3), 'Jm' (4), or 'Fm' (5)  strings.

        In cases (2) to (5), the operation type matrix will be sampled randomly.
        In the (2) case, the passed function will be used with a shape of
        (n_jobs, max_n_operations).
        In the case of (3), the number of operations for each job (n_oj) will be
        sampled uniformly at random from {min_n_operations, ...,
        max_n_operations}. Then n_oj operation types will be sampled uniformly
        at random from {1 ... n_types}.

        In case of (4) a jssp style operation types, i.e. permutations of
        {1, ..., n_types} will be sampled for each job. Note that in this
        case max_n_operations == min_n_operations and n_types
        == max_n_operations needs to hold, else an error will be thrown.

        In the case of (5) fssp style operation types will be sampled, i.e.
        a random permutation of {1, ..., n_types}  will be sampled and
        replicated for every job.

        @See: __set_op_types in JobMatrices.

        :return: The operation types user specification.
        """
        return self.__operation_types

    @property
    def operation_durations(self):
        """
        The user specification for operation durations. The attribute can be
        either a function taking a 2d-shape as an argument and returning a
        numpy array of corresponding dimensions (1), the numpy array directly
        (2) or the 'default_sampling' string (3).

        Cases (1) and (2) will be handled analogously to operation_types. Case
        (3) will induce the generation of type dependent bi-modal gaussian mixes
        from which the duration will be subsequently sampled for each operation.

        @See: __set_op_duration in class JobMatrices.

        :return: The user specification for the operation duration matrix.
        """
        return self.__operation_durations

    @property
    def operation_tool_sets(self):
        """
        The user specification for the operation tool sets. Can be either a
        numpy array (1), a sampling function (2), the 'random_sampling' string
        (3) or None (4).

        Cases (1) to (3) are handled analogously to the corresponding operation
        duration cases. Case (4) leads to operation tool sets being set to 0 for
        all operations, marking the absence of setup times.

        @see: __set_op_tool_sets in the JobMatrices class.

        :return: The user specification for operation tool sets.
        """
        return self.__operation_tool_sets

    @property
    def operation_precedence(self):
        """
        The operation precedence specification, which can take one of five
        forms, namely 'Jm' (1), 'Om' (2), 'POm'(3), a three dimensional numpy
        array (i.e. an array of adjacency matrices, one for each job) (4) or a
        list of adjacency dictionaries (one for each job) (5).

        In cases (1) and (2) the graphs representing precedence constraints for
        each job will correspond to lists and disconnected nodes respectively.
        In case (3) a random directed acyclical graph (DAG) will be generated
        for each job.

        In cases (4) and (5) the custom precedence constrints specified will be
        used.

        @see: __set_op_precedence in the JobMatrices class.

        :return: The operation precedence user specification.
        """
        return self.__operation_precedence

    @property
    def due_dates(self):
        """
        The due date specification for jobs. Can be either a numpy array of
        dimension n_jobs (1), the 'default_sampling' string (2) or a float (3).

        In case (1) the custom due dates passed by the user will be used. In
        case (2) the due dates will be computed by summing the operation
        durations in each job and multiplying the results by a safaty factor of
        1.5 and adding these values to the job release dates. In case (4) the
        due date computation is analogous to (3), but the safety factor is
        passed by the user.

        @see: set_due_dates in the JobMatrices class.

        :return: User specification for the job due dates.
        """
        return self.__due_dates

    @property
    def inter_arrival_time(self):
        """
        The job inter-arrival time specification given by the user. Accepted
        values are a numpy array of dimension n_jobs (1), a sampling function
        taking a 2d-shape (or simply an integer) as an argument and returning a
        numpy array of corresponding size (2) or one of the strings 'balanced'
        (3), 'over' (4) or 'under' (5).

        In case (3), a deterministic version of the simulation will be used on a
        single full WIP to infer the average utilization time (Utl_ave), the
        individual job flow times (F) and the average number of operations in
        the WIP window (wor) over the distinct scheduling steps. A truncated
        normal distribution is then constructed using the distribution defined
        by F / (Utl_ave * wor) for parametrization purposes. Finally,
        inter-arrival times are sampled from the truncnormal distribution and
        the cumulative sum is computed to generate the release-dates.

        This user specification leads to the subsequent definition of release
        dates by computing the stepwise cumulative sum of the inter-arrival
        times.

        @See: __set_release_dates in the JobMatrices class.
        @See: finalize_stochastic_input in the Input class.
        :return:
        """
        return self.__inter_arrival_time

    @property
    def perturbation_processing_time(self):
        """
        The user defined operation duration noise. Can be either a sampling
        function (1), numpy array of shape corresponding to the operation
        duration matrix shape (2), a float between 0.1 and 2 (3), the
        'random_sampling' string (3) or None.

        TODO: Detail cases!

        @see: __set_op_perturbation in the JobMatrices class.

        :return: The user specification for operation duration noise.
        """
        return self.__perturbation_processing_time

    # perturbation_processing_time: T,
    # </editor-fold desc="Job Matrices Information Getters">

    # <editor-fold desc="Machine Matrices Information Getters">
    @property
    def machine_speeds(self):
        return self.__machine_speeds

    @property
    def machine_distances(self):
        return self.__machine_distances

    @property
    def machine_buffer_capacities(self):
        return self.__machine_buffer_capacities

    @property
    def machine_capabilities(self):
        return self.__machine_capabilities

    @property
    def type_indexed_capabilities(self):
        return self.__type_indexed_capabilities

    @property
    def tool_switch_times(self):
        return self.__tool_switch_times

    @property
    def machine_failures(self):
        """

        :return:
        """
        return self.__machine_failures

    # </editor-fold desc="Machine Matrices Information Getters">

    # <editor-fold desc="Instance Information Getters">
    @property
    def name(self):
        return self.__name

    @property
    def path(self):
        return self.__path
    # </editor-fold desc="Instance Information Getters">
