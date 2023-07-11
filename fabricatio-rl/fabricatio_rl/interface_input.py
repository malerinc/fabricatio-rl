from collections import deque
from copy import deepcopy, copy
from typing import (TypeVar, TYPE_CHECKING,
                    Callable, Union, List, Set, Tuple, Dict)

import numpy as np
import scipy.stats as stats

from fabricatio_rl.env_utils import (UndefinedInputType,
                                                   faster_deepcopy,
                                                   GraphUtils,
                                                   decouple_view)
from fabricatio_rl.interface_rng import FabricatioRNG

# type-hinting finalize_stochastic inputs would cause a cyclical dependency;
# to circumvent that, we load the module only for the benefit of typechecking
# tools; see the following post:
# https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports/43041058#43041058
from fabricatio_rl.interface_templates import SchedulingUserInputs

if TYPE_CHECKING:
    from fabricatio_rl.core import Core
    from fabricatio_rl.core_state import State
    # indicates generic types
    T = TypeVar('T')
    SamplingFunction2D = Callable[[Tuple[int, int]], np.ndarray]
    PrecedenceDictionary = Dict[Tuple[int], List[int]]
    # TODO: make the dictionary values consistent (i.e. sets ;)
    CapabilityDict = Dict[int, Union[Set[int], List[int]]]


class SchedulingDimensions:
    """
    Initializes and stores scheduling problem dimensions. This read-only object
    is stored within the Input object and is invariant between resets. A
    SchedulingDimensions object is used throughout the simulation initialization
    to set the appropriate dimensions for containers, as well a during the
    simulation execution for quick information access, offset computation etc.
    """

    # noinspection PyProtectedMember
    def __init__(self, si: SchedulingUserInputs,
                 rng: np.random.Generator, seed):
        """
        The SchedulingDimensions constructor.

        :param si: The scheduling inputs set by the user.
        :param rng: A numpy random number generator.
        """
        self.__n_jobs = si.n_jobs
        self.__n_machines = si.n_machines
        self.__n_tooling_lvls = si.n_tooling_lvls
        self.__n_types = si.n_types
        # TODO: initialize this with the rest of the matrices?
        if type(si.n_operations) == np.ndarray:
            self.__n_operations = si.n_operations
        elif si.n_operations == 'default_sampling':
            assert si.min_n_operations < si.max_n_operations
            self.__n_operations = rng.integers(
                low=si.min_n_operations, high=si.max_n_operations,
                size=si.n_jobs)
        elif si.min_n_operations == si.max_n_operations:
            self.__n_operations = np.repeat(si.max_n_operations, si.n_jobs)
        else:
            raise UndefinedInputType(
                type(si.n_operations), " n_operations parameter.")
        self.__n_operations.flags.writeable = False
        self.__min_n_operations = si.min_n_operations
        self.__max_n_operations = si.max_n_operations
        self.__max_n_failures = si.max_n_failures
        self.__max_jobs_visible = si.max_jobs_visible
        self.__n_jobs_initial = si.n_jobs_initial
        # so that we can get seed and instance name from the state later on;
        # for debugging purposes
        self.__name = si.name
        self.__seed = seed

    def __deepcopy__(self, memo: dict):
        """
        Creates a deep copy of the SchedulingDimensions object.

        :param memo: The object copy memory.
        :return: A deep copy of the current object.
        """
        # TODO: check if shallow copy could suffice here (most likely it will)
        return copy(self)

    # <editor-fold desc="Getters">
    @property
    def name(self):
        return self.__name

    @property
    def seed(self):
        return self.__seed

    @property
    def n_jobs(self):
        """
        An integer marking the total number of jobs in the scheduling problem.

        :return: The n_jobs instance attribute.
        """
        return self.__n_jobs

    @property
    def n_machines(self):
        """
        The total number of machines in the system.

        :return: The n_machines attribute.
        """
        return self.__n_machines

    @property
    def n_tooling_lvls(self):
        """
        The total number of distinct tool sets present.

        :return: The n_tooling_levels attribute.
        """
        return self.__n_tooling_lvls

    @property
    def n_types(self):
        """
        The total number of distinct operations types.

        :return: The n_types attribute.
        """
        return self.__n_types

    @property
    def n_operations(self):
        """
        An numpy array containing the number of operations in each job.

        :return: The n_operations attribute.
        """
        return self.__n_operations

    @property
    def min_n_operations(self):
        """
        The minimum over the n_operations array.

        :return: The min_n_operations attribute.
        """
        return self.__min_n_operations

    @property
    def max_n_operations(self):
        """
        The maximum over the n_operations array.

        :return: The max_n_operations attribute.
        """
        return self.__max_n_operations

    @property
    def max_n_failures(self):
        """
        The maximum number of machine failures to simulate.

        :return: The max_n_failures attribute.
        """
        return self.__max_n_failures

    @property
    def max_jobs_visible(self):
        """
        The maximum number of jobs visible to the scheduling agents at any time
        during the simulation; Corresponds to the work in progress (WIP) window
        size.

        :return: The max_jobs_visible attribute.
        """
        return self.__max_jobs_visible

    @property
    def n_jobs_initial(self):
        """
        The initial number of visible jobs (initial jobs in WIP).

        :return: The n_jobs_initial attribute.
        """
        return self.__max_jobs_visible
    # </editor-fold>

    def finalize_n_operations(self, job_idxs):
        """
        Used when jobs are sampled from a job pool if the n_operations array
        was read from a file. In this case the property is expanded using the
        view indices of the job pool passed as parameters. After the property
        reassignment, the array is made readonly.

        :param job_idxs: The job pool indices.
        :return: None.
        """
        self.__n_operations = self.__n_operations[job_idxs]
        self.__n_operations.flags.writeable = False


class JobMatrices:
    @staticmethod
    def __test_row_independence(array: Union[np.ndarray, List]):
        addresses = set({})
        for row in array:
            address = id(row)
            assert address not in addresses
            addresses.add(address)

    def __init__(self, dims: SchedulingDimensions, si: SchedulingUserInputs,
                 rng: np.random.Generator):  # perturbation_due_date: T
        self.__rng = rng
        self.simulation_needed = False
        if si.job_pool is not None:
            if not si.fixed_instance:
                job_idxs = rng.choice(
                    len(si.job_pool), dims.n_jobs, replace=True)
            else:
                job_idxs = np.arange(dims.n_jobs)
            jobs = si.job_pool[job_idxs]
            # todo: pool as an object
            # todo: check and report pool consistency (i.e. dims vs matrix dims)
            self.__op_type = np.stack(jobs[:, 0])
            # JobMatrices.__test_row_independence(self.__op_type)
            self.__op_duration = np.stack(jobs[:, 1])
            # self.__test_row_independence(self.__op_duration)
            self.__op_tool_set = np.stack(jobs[:, 2])
            op_precedences = decouple_view(jobs[:, 3])
            dims.finalize_n_operations(job_idxs)
            # self.__test_row_independence(op_precedences)
            als, ams = JobMatrices.__set_op_precedence_from_spec(
                dims, op_precedences, from_pool=True)
            self.__op_precedence_l, self.__op_precedence_m = als, ams
        else:
            self.__op_type = self.__set_op_types(dims, si.operation_types)
            self.__op_duration = self.__set_op_duration(
                dims, self.__op_type, si.operation_durations)
            self.__op_tool_set = self.__set_op_tool_sets(
                dims, self.__op_type, si.operation_tool_sets)
            als, ams = self.__set_op_precedence(dims, si.operation_precedence)
            self.__op_precedence_l, self.__op_precedence_m = als, ams
        self.__op_perturbations = self.__set_op_perturbation(
            dims, si.perturbation_processing_time)
        self.__job_release_times = self.__set_release_dates(
            dims, si.inter_arrival_time, self.__op_duration)
        self.__job_due_dates = self.set_due_dates(
            dims, si.due_dates, self.__job_release_times,
            self.__op_duration)
        # todo!
        # self.__job_due_date_perturbation = JobMatrices.__set_due_date_noise(
        #     perturbation_due_date)

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)

    # <editor-fold desc="Constructor Helpers">
    def __set_op_types(
            self, dims: SchedulingDimensions,
            op_types: Union[np.ndarray, 'SamplingFunction2D', str]
    ) -> np.ndarray:
        """
        TODO: docstrings!

        :param dims:
        :param op_types:
        :return:
        """
        if type(op_types) == np.ndarray:
            return np.array(op_types).astype('int16')
        else:
            n, o_max = dims.n_jobs, dims.max_n_operations
            if callable(op_types):
                set_op_types = JobMatrices.__sample(op_types, (n, o_max))
            elif op_types == 'default_sampling':  # op_types == '':
                set_op_types = self.__rng.choice(
                    range(1, dims.n_types + 1), (n, o_max), replace=True)
            elif op_types == 'Jm':
                assert dims.n_types == dims.max_n_operations
                assert dims.max_n_operations == dims.min_n_operations
                set_op_types = np.zeros((n, o_max))
                for j in range(n):
                    set_op_types[j, :] = self.__rng.permutation(
                        dims.n_types) + 1
            elif op_types == 'Fm':
                assert dims.n_types == dims.max_n_operations
                assert dims.max_n_operations == dims.min_n_operations
                set_op_types = np.zeros((n, o_max))
                job_structure = self.__rng.permutation(dims.n_types) + 1
                for j in range(n):
                    set_op_types[j, :] = job_structure.copy()
            else:
                raise UndefinedInputType(
                    type(op_types),
                    " operation_types parameter. Accepted inputs are"
                    "the 'deafault_sampling' string, a sampling function "
                    "taking a shape as a parameter and returning a numpy array "
                    "of corresponding size, the string 'Pm' or 'Fm'.")
            # delete types that are too many
            for j in range(n):
                o_j = dims.n_operations[j]
                set_op_types[j, o_j:] = np.zeros(o_max - o_j)
            return set_op_types.astype('int16')

    def __set_op_duration(
            self, dims: SchedulingDimensions, op_types: np.ndarray,
            op_durations: Union[np.ndarray, 'SamplingFunction2D', str]
    ) -> np.ndarray:
        """
        TODO: docstrings!

        :param dims:
        :param op_types:
        :param op_durations:
        :return:
        """
        if type(op_durations) == np.ndarray:
            return np.array(op_durations).astype('int16')
        else:
            n, o_max, n_ty = dims.n_jobs, dims.max_n_operations, dims.n_types
            if callable(op_durations):
                set_op_duration = JobMatrices.__sample(
                    op_durations, (n, o_max))
            elif op_durations == 'default_sampling':  # op_durataions == ''
                # type conditional duration dist
                tcdd = JobMatrices.__create_tcdd(n_ty, self.__rng)
                tcdd[0] = np.zeros(2)  # type 0 means no operation, hence d = 0
                tcdd_samples = deque([])
                for ty in range(0, n_ty + 1):
                    tcdd_sample = self.__rng.choice(tcdd[ty], n)
                    tcdd_samples.append(tcdd_sample.astype('int16'))
                tcdd_samples_t = np.array(tcdd_samples).transpose()
                set_op_duration = np.zeros((n, o_max))
                for j in range(n):
                    type_seq = op_types[j, :]
                    set_op_duration[j, :] = tcdd_samples_t[j, :][type_seq]
            else:
                raise UndefinedInputType(type(op_durations),
                                         " operation_types parameter.")
            return set_op_duration

    def __set_op_tool_sets(
            self, dims: SchedulingDimensions, op_types: np.ndarray,
            op_tools: Union[np.ndarray, 'SamplingFunction2D', str, None]
    ) -> np.ndarray:
        """
        TODO: docstring!
        :param dims:
        :param op_types:
        :param op_tools:
        :return:
        """
        if type(op_tools) == np.ndarray:
            return np.array(op_tools).astype('int16')
        else:
            n, o_max = dims.n_jobs, dims.max_n_operations
            n_ty, n_tl = dims.n_types, dims.n_tooling_lvls
            if op_tools is None:
                return np.zeros((n, o_max)).astype('int16')
            elif callable(op_tools):
                op_tools_sample = JobMatrices.__sample(
                    op_tools, (n, o_max)).astype('int16')
            elif op_tools == 'default_sampling':  # op_tools == ''
                # type conditional tool set dist
                tctsd = JobMatrices.__create_tctsd(n_ty, n_tl, self.__rng)
                tctsd[0] = np.zeros(2)  # type 0 means no operation, hence d = 0
                tctsd_samples = deque([])
                for ty in range(0, n_ty + 1):
                    tctsd_sample = self.__rng.choice(tctsd[ty], n).astype(
                        'int16')
                    tctsd_samples.append(tctsd_sample)
                tctsd_samples_t = np.array(tctsd_samples).transpose()
                op_tools_sample = np.zeros((n, o_max)).astype('int16')
                for j in range(n):
                    type_seq = op_types[j, :]
                    op_tools_sample[j, :] = tctsd_samples_t[j, :][type_seq]
            else:
                raise UndefinedInputType(
                    type(op_tools), " operation_tool_sets parameter. "
                                    "Accepted parameter values are None, "
                                    "'default_sampling' or a sampling function")
            return op_tools_sample.astype('int16')

    @staticmethod
    def __set_op_precedence_from_spec(
            dims: SchedulingDimensions,
            # TODO: more explicit typing
            precedence_specification: Union['PrecedenceDictionary', np.ndarray],
            from_pool=False
    ) -> (list, np.ndarray):
        """
        Transforms a user defined precedence constraint specification (either a
        list of adjacency dictionaries or a tensor of dimensions
        n_jobs x max_n_ops x max_n_ops) into its complement representation and
        returns both.

        :param dims: The problem dimensions object.
        :param precedence_specification:
            List of adjacency dicts or prec. tensor.
        :return: List of adjacency dicts and precedence tensor.
        """
        n, o = dims.n_jobs, dims.max_n_operations
        if (type(precedence_specification) == np.ndarray
                and len(precedence_specification.shape) == 3):
            ams, als = precedence_specification, []
            for j in range(dims.n_jobs):
                al = GraphUtils.graph_matrix_to_adjacency_list(
                    ams[j, :], o, j)
                als.append(al)
        elif ((type(precedence_specification) == list)
              or (type(precedence_specification) == np.ndarray
                  and len(precedence_specification.shape) == 1)):
            ams, als = np.zeros((n, o, o)), precedence_specification
            for j in range(dims.n_jobs):
                if from_pool:
                    try:
                        if j != 0:
                            als[j][(j,)] = als[j][(0,)]
                            del als[j][(0,)]
                    except KeyError:
                        print("herehere")
                am = GraphUtils.graph_adjacency_list_to_matrix(
                    als[j], o, (j,))
                ams[j, :] = am
        else:
            raise UndefinedInputType(
                type(precedence_specification),
                " operation_precedence parameter. Accepted inputs are "
                "the strings 'Om', 'Jm', 'POm' for no order, total order and "
                "partial order respectively, a tensor of graphs or a list of "
                "precedence graphs as a adjacency dictionaries.")
        return als, ams

    def __set_op_precedence(
            self, dims: SchedulingDimensions,
            op_precedence: Union[np.ndarray, List['PrecedenceDictionary'], str]
    ) -> (np.ndarray, 'PrecedenceDictionary'):
        """
        Computes the list of adjacency dicts and precedence matrix tensor for
        the init function.

        :param dims: Problem dimension object.
        :param op_precedence: Precedence spec; either 'Jm', 'Om', 'POm', ndarray
            or list of dicts.
        :return: The precedence constraints as a multidimensional matrix and the
            equivalent list of adjacency dictionaries.
        """
        n, o_max, o_js = dims.n_jobs, dims.max_n_operations, dims.n_operations
        als = []
        prec_matrix = np.zeros((n, o_max, o_max))
        if op_precedence == 'Jm':
            for j in range(n):
                als = GraphUtils.get_job_chain_precedence_graphs(n, o_js)
                for i in range(o_js[j] - 1):
                    prec_matrix[j][i][i + 1] = 1
        elif op_precedence == 'POm':
            for j in range(n):
                al, am = GraphUtils.get_random_precedence_relation(
                    o_js[j], o_max, self.__rng)
                al[(j,)] = al.pop(-1)
                als.append(al)
                prec_matrix[j, :, :] = am
        elif op_precedence == 'Om':
            for j in range(n):
                al = {(j,): list(range(o_js[j]))}
                als.append(al)
                # prec matrix stays 0
        else:  # matrix was user specified: either ndarray or list of adj. dicts
            als, prec_matrix = JobMatrices.__set_op_precedence_from_spec(
                dims, op_precedence)  # raises an exception if input doesn't fit
        return als, prec_matrix

    def __set_op_perturbation(
            self, dims: SchedulingDimensions,
            op_perturbation: Union[np.ndarray,
                                   Callable[
                                       [Tuple[int, int]], np.ndarray], float,
                                   str, None]
    ) -> np.ndarray:
        """
        Creates an array of floats in greater than 0 to scale the operation
        durations with. Accepted inputs are '' in case a truncated normal
        distribution will be used, a scaler between 0 and one in which case
        a uniform distribution of ten values between -scaler and +scaler will
        be used, or a custom sampling function.

        :param dims: The Scheduling problem dimensions.
        :param op_perturbation: '', scaler or callable defining.
        :return: A scaler mask for operation durations.
        """
        n, o, n_ops = dims.n_jobs, dims.max_n_operations, dims.n_operations
        if op_perturbation == 'default_sampling':
            mu, sigma, lo, hi = 1, 1, 0.1, 2
            dist = stats.truncnorm(
                (lo - mu) / sigma, (hi - mu) / sigma, loc=mu, scale=sigma)
            m_op_perturbations = dist.rvs((n, o), random_state=self.__rng)
        elif callable(op_perturbation):
            m_op_perturbations = JobMatrices.__sample(op_perturbation, (n, o))
        elif type(op_perturbation) == float:
            assert 1 > op_perturbation >= 0
            p_range_step = (2 * op_perturbation) / 10
            p_range = np.arange(-op_perturbation, op_perturbation, p_range_step)
            m_op_perturbations = self.__rng.choice(1 + p_range, (n, o))
        elif type(op_perturbation) == np.ndarray:
            assert (op_perturbation > 0).all()
            m_op_perturbations = op_perturbation
        elif op_perturbation is None:
            m_op_perturbations = np.ones((n, o))
        else:
            raise UndefinedInputType(
                type(op_perturbation),
                " operation perturbation parameter (delta). Accepted inputs are"
                "the empty string, a float between 0 (inclusively) and one "
                "(exclusively), a sampling function taking a shape as a "
                "parameter and returning a numpy array of corresponding size or"
                "an ndarray of positive entries.")
        for j in range(n):
            m_op_perturbations[j, n_ops[j]:] = np.zeros(o - n_ops[j])
        return m_op_perturbations

    def __set_release_dates(self, dims: SchedulingDimensions,
                            time_inter_release: 'T', op_durations: np.ndarray):
        """
        DEPRECATED COMMENT!

        Sets the relese date for jobs past the n_jobs_initial (which have a
        release date of 0) by summing inter-arrival times as defined per
        inter_arrival_time argument. If the empty string is passed in the latter
        the method defaults to sampling inter-arrival times from a truncated
        normal distribution informed by the operation processing time
        distribution.

        If a sampling function is passed, it will be used for
        inter-arrival times. Alternatively one can pass the vector of
        inter-arrival times directly.

        @see: finalize_stochastic_inputs in the Input class

        :param dims: The scheduling problem dimensions.
        :param time_inter_release: The inter-arrival time sampling function,
            vector or the empty string.
        :param op_durations: The operation duration matrix.
        :return: The vector of release times for jobs.
        """
        n, n_0, tir = dims.n_jobs, dims.n_jobs_initial, time_inter_release
        job_arrivals = np.zeros(n, dtype='uint32')
        if n == n_0:
            return job_arrivals
        elif time_inter_release in ['balanced', 'over', 'under']:
            max_duration = op_durations.sum()
            job_arrivals = np.zeros(n)
            job_arrivals[n_0:] = np.repeat(max_duration, n - n_0)
            self.simulation_needed = True
            return job_arrivals
        elif callable(time_inter_release):
            job_arrivals[n_0:] = JobMatrices.__sample(tir, n - n_0)
            return job_arrivals
        elif type(time_inter_release) == np.ndarray:
            # forcibly set job arrival with idx < n_0 to 0 to avoid future
            # unpleasentness
            job_arrivals[n_0:] = time_inter_release[n_0:]
            return job_arrivals
        else:
            raise UndefinedInputType(
                type(time_inter_release),
                "release dates parameter. Accepted inputs are"
                "the empty string, a sampling function taking a shape and "
                "returning a corresponding ndarray or an ndarray with due "
                "dates.")

    @staticmethod
    def set_due_dates(dims: SchedulingDimensions,
                      due_dates: Union[np.ndarray, str, float],
                      release_dates: np.ndarray,
                      op_durations: np.ndarray) -> np.ndarray:
        """
        Sets the due dates for jobs by adding scaling (default 1.5) the job
        duration lower  bound with the passed due_dates parameter and adding it
        to the release dates. Alternatively, the due dates vector can be passed
        directly as an ndarray.

        :param dims: The scheduling problem dimensions.
        :param due_dates: The due date vector, scaler or the empty string.
        :param release_dates: The job release dates.
        :param op_durations: The job operation duration matrix.
        :return: The due dates vector.
        """
        n, n_0 = dims.n_jobs, dims.n_jobs_initial
        if due_dates == 'default_sampling':
            job_durations = op_durations.sum(axis=1)
            vec_due_dates = release_dates + 1.5 * job_durations
        elif type(due_dates) == float:
            job_durations = op_durations.sum(axis=1)
            vec_due_dates = release_dates + due_dates * job_durations
        elif type(due_dates) == np.ndarray:
            assert len(due_dates.shape) == 1 and due_dates.shape[0] == n
            assert due_dates >= 0
            vec_due_dates = due_dates.astype('uint16')
        else:
            raise UndefinedInputType(
                type(due_dates),
                "due dates parameter. Accepted inputs are"
                "the empty string, a float > 0 or a numpy array of positive "
                "integers of length equal to the number of jobs.")
        return vec_due_dates

    # @staticmethod
    # def __set_due_date_noise(dims: SchedulingDimensions,
    #                          perturbation_due_date: T) -> np.ndarray:
    #     # TODO!
    #     return np.ones(dims.n_jobs)
    # </editor-fold>

    # <editor-fold desc="Sampling Functions">
    @staticmethod
    def __sample(sampling_function: Callable[[Union[int, tuple]], np.ndarray],
                 size: Union[tuple, int]):
        """
        Samples a np array of shape defined by size using the sampling function
        passed.

        :param sampling_function: A function taking the sample shape as a
            parameter.
        :param size: The shape of the distribution sample.
        :return: The sample of the requested shape from the specified
        distribution.
        """
        return sampling_function(size)

    @staticmethod
    def __create_tcdd(n_types: int, rng: np.random.Generator) -> dict:
        """
        Creates a operation type conditional bi-modal processing time
        distribution. To 1000 samples are drawn from two distinct lognormal
        distributions with a type dependent mean distribution parameter prior
        concatenation.

        :param n_types: The number of operation types.
        :param rng: The numpy random number generator.
        :return: A dictionary with types as keys and lists of 2000 distribution
            points.
        """
        conditional_dist = {}
        for i in range(1, n_types + 1):
            step = (i / n_types)
            bimodal_sample = np.concatenate(
                [rng.lognormal(np.log(50 + 50 * step), 0.2, 1000),
                 rng.lognormal(np.log(150 - 50 * step), 0.08, 1000)])
            operation_times = np.ceil(bimodal_sample).astype('uint16')
            conditional_dist[i] = rng.choice(operation_times, 2000)
        return conditional_dist

    @staticmethod
    def __create_tctsd(n_types: int, total_tools: int,
                       rng: np.random.Generator) -> dict:
        """
        Creates a type dependent tool set distribution. The number of n_t of
        tool sets associated with each type are first drawn. Then n_t tool types
        are drawn uniformly at random from {1 .. total_tools} for each
        type t.

        :param n_types: Number of operation types.
        :param total_tools: Number of tools in the system.
        :param rng: The numpy random number generator.
        :return: Dictionary of type dependent tool set distributions.
        """
        assert total_tools >= 2
        conditional_dist = {}
        tool_range = range(1, total_tools + 1)
        for i in range(1, n_types + 1):
            n_type_tools = rng.integers(1, total_tools // 2 + 1)
            conditional_dist[i] = rng.choice(tool_range, n_type_tools)
        return conditional_dist

    @staticmethod
    def __sample_symmetric_matrix(sampling_function, size, diag):
        """
        Samples a full matrix, as specified by size from the distribution
        provided. The lower diagonal matrix is then eliminated, with or without
        the matrix diagonal itself. The upper diagonal matrix is then transposed
        and added to its original. If the diagonal elements were kept, these
        are devided by 2 to recreate the original samples prior to the function
        return.

        :param sampling_function: A list of strings where the first is
            the distribution name followed by the distribution parameters.
        :param size: The shape of the distribution sample.
        :param diag: True if the symmetric matrix should contain diagonal
            elements.
        :return: The symmetric matrix with entries sampled from the requested
            distribution.
        """
        assert size[0] == size[1]
        upper_diagonal_matrix = np.triu(
            JobMatrices.__sample(sampling_function, size), 1 - diag)
        lower_diagonal_matrix = upper_diagonal_matrix.transpose()
        symmetric_m = upper_diagonal_matrix + lower_diagonal_matrix
        if diag:
            np.fill_diagonal(
                symmetric_m, symmetric_m.diagonal() / 2)
        return symmetric_m

    # </editor-fold>

    # <editor-fold desc="Getters">
    @property
    def operation_precedence_m(self):
        return self.__op_precedence_m

    @property
    def operation_precedence_l(self):
        return self.__op_precedence_l

    @property
    def operation_types(self):
        return self.__op_type

    @property
    def operation_durations(self):
        return self.__op_duration

    @property
    def operation_tool_sets(self):
        return self.__op_tool_set

    @property
    def operation_perturbations(self):
        return self.__op_perturbations

    @property
    def job_arrivals(self):
        return self.__job_release_times

    @property
    def job_due_dates(self):
        return self.__job_due_dates

    # @property
    # def job_due_date_perturbations(self):
    #     return self.__job_due_date_perturbation
    # </editor-fold>

    # <editor-fold desc="Setters">
    @job_arrivals.setter
    def job_arrivals(self, value):
        assert self.simulation_needed
        self.__job_release_times = value

    @job_due_dates.setter
    def job_due_dates(self, value):
        assert self.simulation_needed
        self.__job_due_dates = value
    # </editor-fold desc="Setters">


class MachineMatrices:
    def __init__(self, dims: SchedulingDimensions, si: SchedulingUserInputs,
                 job_matrices: JobMatrices, rng: np.random.Generator):
        self.__rng = rng
        self.__machine_distances = self.__set_machine_distances(
            dims, si.machine_distances, job_matrices.operation_durations)
        self.__tool_switch_times = self.__set_tool_switch_times(
            dims, si.tool_switch_times, job_matrices.operation_durations)
        self.__machine_speeds = self.__set_machine_speeds(
            dims, si.machine_speeds)
        self.__machine_buffer_capa = MachineMatrices.__set_machine_buffer_capa(
            dims, si.machine_buffer_capacities)
        cdm, cdt, cm = self.__set_machine_capabilities(
            dims, si.machine_capabilities, si.type_indexed_capabilities)
        self.__machine_capabilities_dm = cdm
        self.__machine_capabilities_dt = cdt
        self.__machine_capabilities_m = cm
        # stochastic influences
        self.__machine_failures = self.__set_machine_failures(
            dims, si.machine_failures, job_matrices.operation_durations)

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)

    # <editor-fold desc="Constructor Helpers">
    def __set_machine_distances(self, dims: SchedulingDimensions,
                                machine_distances: 'T',
                                op_durations: np.ndarray) -> np.ndarray:
        """
        Constructs a machine distance matrix as specified by the user. The
        machine_distances parameter can be either the 'default_sampling' string
        a sampling function, a float, the matrix directly as an ndarray or None.

        If the machine_distance parameter is the empty string, a random
        symmetric matrix with a 0 diagonal is constructed by sampling from a
        truncated normal distribution approximating that of the operation
        durations. A float can be used to adapt the transport time distribution
        by scaling the truncnormal distribution bounds and mean. The sampling
        function will be used to create the entries of symmetric matrix instead.
        If machine_distances is none, the internal transport time matrx will be
        initialized to zeros.

        The distance matrix will contain an additional row modelling the
        production source.

        :param dims: Scheduling dimensions.
        :param machine_distances: '', float, callable or ndarray.
        :param op_durations: The operation durations.
        :return: The distance matrix mapping transport times between machines or
            machines and source.
        """
        m = dims.n_machines
        if machine_distances is None:
            transport_times = np.zeros((m + 1, m + 1))
        elif machine_distances == 'default_sampling':
            dist_f = self.__get_truncnormal_op_duration_approx(
                op_durations)
            transport_times = MachineMatrices.__sample_symmetric_matrix(
                dist_f, (m + 1, m + 1), False).astype('uint16')
        elif type(machine_distances) == float:
            dist_f = self.__get_truncnormal_op_duration_approx(
                machine_distances)
            transport_times = MachineMatrices.__sample_symmetric_matrix(
                dist_f, (m + 1, m + 1), False).astype('uint16')
        elif callable(machine_distances):
            transport_times = MachineMatrices.__sample_symmetric_matrix(
                machine_distances, (m + 1, m + 1), False).astype('uint16')
        elif type(machine_distances) == np.ndarray:
            transport_times = machine_distances
        else:
            raise UndefinedInputType(
                type(machine_distances),
                "machine distance parameter. Accepted inputs are None, "
                "the 'default_sampling' string, a sampling function taking a "
                "shape and returning a corresponding np.ndarray, a foat to "
                "scale the transport time distribution relative to the "
                "operation duration distribution or an ndarray with pre-set "
                "machine distances as positive integers or 0.")
        return transport_times

    def __set_tool_switch_times(self, dims: SchedulingDimensions,
                                tool_switch_times: 'T',
                                op_durations: np.ndarray) -> np.ndarray:
        """
        Analogous to "__set_machine_distances"
        """
        tl_lvls = dims.n_tooling_lvls
        if tool_switch_times == 'default_sampling':
            dist_f = self.__get_truncnormal_op_duration_approx(
                op_durations, 0.1)
            tooling_times = MachineMatrices.__sample_symmetric_matrix(
                dist_f, (1 + tl_lvls, 1 + tl_lvls), False).astype('uint16')
        elif tool_switch_times is None:
            tooling_times = np.zeros((1 + tl_lvls, 1 + tl_lvls))
        elif type(tool_switch_times) == float:
            dist_f = self.__get_truncnormal_op_duration_approx(
                op_durations, tool_switch_times)
            tooling_times = MachineMatrices.__sample_symmetric_matrix(
                dist_f, (tl_lvls + 1, tl_lvls + 1), False).astype('uint16')
        elif callable(tool_switch_times):
            tooling_times = MachineMatrices.__sample_symmetric_matrix(
                tool_switch_times, (tl_lvls + 1, tl_lvls + 1), False).astype(
                'uint16')
        elif type(tool_switch_times) == np.ndarray:
            tooling_times = tool_switch_times
        else:
            raise UndefinedInputType(
                type(tool_switch_times),
                "tool switch time parameter. Accepted inputs are"
                "the empty string, a sampling function taking a shape and "
                "returning a corresponding np.ndarray, a foat to scale the "
                "tooling time distribution relative to the operation duration"
                "distribution or an ndarray with pre-set "
                "toling as positive integers or 0.")
        return tooling_times

    def __set_machine_speeds(self, dims: SchedulingDimensions,
                             machine_speeds: 'T') -> np.ndarray:
        """
        Samples machine speeds uniformly at random from {0.5, 0.1 .. 1.5}
        if the machine_speeds parameter is the 'default_sampling' string,
         initializes all machine speeds if None is passed, or
        returns the machine speeds directly if an ndarray of corresponding
        length was passed.

        :param dims: The scheduling problem dimensions.
        :param machine_speeds: Either '' or an ndarray with machine speed
            scalars between (0, infinity)
        :return: The vector of machine speeds.
        """
        m = dims.n_machines
        if machine_speeds is None:
            return np.ones(m)
        elif (type(machine_speeds) == str and
              machine_speeds == 'default_sampling'):
            return self.__rng.choice(np.arange(0.5, 1.5, 0.1), m)
        elif type(machine_speeds) == np.ndarray:
            assert len(machine_speeds.shape) == 1
            assert machine_speeds.shape[0] == m
            return machine_speeds
        else:
            raise UndefinedInputType(
                type(machine_speeds),
                " machine_speeds parameter. Accepted inputs are"
                "the 'default_sampling' string, None or an ndarray with"
                "pre-set machine speeds as positive floats.")

    @staticmethod
    def __set_machine_buffer_capa(dims: SchedulingDimensions,
                                  machine_buffer_capacities: 'T'):
        """
        Sets the buffer capacities for machines to infinity if the empty string
        is passed in machine_buffer_capacities. If said parameter is an
        ndarray of positive integers, the latter is returned.

        :param dims: The scheduling input dimensions.
        :param machine_buffer_capacities: The empty string or an ndarray of
            positive integers.
        :return: The buffr capacities as an ndarray indexed by machine numbers.
        """
        m = dims.n_machines
        if machine_buffer_capacities is None:
            return np.repeat(np.inf, m)
        elif type(machine_buffer_capacities) == np.ndarray:
            assert len(machine_buffer_capacities.shape) == 1
            assert machine_buffer_capacities.shape[0] == m
            assert np.issubdtype(
                type(machine_buffer_capacities.shape[0]), np.integer)
            return machine_buffer_capacities
        else:
            raise UndefinedInputType(
                type(machine_buffer_capacities),
                "machine_buffer_capacities parameter. Accepted inputs are"
                "None or an ndarray with pre-set "
                "buffer capacities as positive integers.")

    @staticmethod
    def __invert_capab_dict(capabilities: 'CapabilityDict') -> 'CapabilityDict':
        """
        Flips the key-value scheme of the machine capability dictionary from
        machines -> list of types to type -> list of machines and vice-versa.

        :param capabilities: The machine or type indexed capability
            dictionary.
        :return: The complementary capability dictionary.
        """
        t_idexed_m_capa = {}
        for m_i in capabilities.keys():
            for t_i in capabilities[m_i]:
                if t_i in t_idexed_m_capa:
                    t_idexed_m_capa[t_i].add(m_i)
                else:
                    t_idexed_m_capa[t_i] = {m_i}
        return t_idexed_m_capa

    @staticmethod
    def __to_capab_matrix(dims: SchedulingDimensions,
                          capab_dict: dict) -> np.ndarray:
        capab_matrix = np.zeros((dims.n_machines, dims.n_types), dtype=bool)
        for m_i in capab_dict.keys():
            for t_i in capab_dict[m_i]:
                capab_matrix[m_i - 1][t_i - 1] = 1
        return capab_matrix

    @staticmethod
    def __to_capab_dict(capab_matrix: np.ndarray) -> dict:
        capab_dict = {}
        for m_i in range(capab_matrix.shape[0]):
            for t_i in range(capab_matrix.shape[1]):
                if capab_matrix[m_i][t_i] == 1:
                    if m_i in capab_dict:
                        # types and machines are idexed starting at 1
                        capab_dict[m_i + 1].append(t_i + 1)
                    else:
                        capab_dict[m_i + 1] = [t_i + 1]
        return capab_dict

    def __set_machine_capabilities(
            self, dims: SchedulingDimensions, machine_capabilities: 'T',
            type_indexed_capabilities=False
    ) -> (dict, np.ndarray):
        """
        Defines machine type capabilities as a dictionary with types as keys
        and a collection of compatible machines as values, as well as a matrix
        of boolean with rows corresponding to machines and column corresponding
        to types. Machine-type compatibility is signaled by the value 1.

        The dictionary is required to run the simulation fast while the matrix
        encodes the capability information for the agent.

        Depending on the machine_capability parameter, the dictionary and
        corresponding matrix will be generated randomly (enpty string), the
        matrix will get converted to a dict (parameter type == np.ndarray) or
        the dict to a matrix (parameter type == dict).

        Note that the dict input should be presented in the form
        {m_id_1: []}
        # TODO: finish comments!

        :param dims:
        :param machine_capabilities:
        :return:
        """
        m, t = dims.n_machines, dims.n_types
        if machine_capabilities == 'default_sampling':
            capab_matrix = np.zeros((m, t), dtype=bool)
            capab_dict_m = {}
            encountered_types = set({})
            for i in range(1, m + 1):  # m indices start at 1
                n_capab = self.__rng.integers(1, t + 1)
                m_capab = self.__rng.choice(
                    np.arange(t) + 1, n_capab, replace=False)
                capab_dict_m[i] = list(m_capab)
                capab_matrix[i - 1, [x - 1 for x in capab_dict_m[i]]] = np.ones(
                    len(capab_dict_m[i]))
                encountered_types |= set(m_capab)
            missing_types = set(range(1, t + 1)) - encountered_types
            # make sure each type occurs at least once!!!
            for mty in missing_types:
                rand_m = self.__rng.integers(1, m + 1)
                capab_dict_m[rand_m].append(mty)
                capab_matrix[rand_m - 1, mty - 1] = 1
            # convert to type indexed
            capab_dict = MachineMatrices.__invert_capab_dict(capab_dict_m)
            return capab_dict_m, capab_dict, capab_matrix
        elif machine_capabilities is None:
            assert dims.n_types == dims.n_machines
            capab_dict_m = {m: [m] for m in range(1, dims.n_types + 1)}
            capab_dict = MachineMatrices.__invert_capab_dict(capab_dict_m)
            capab_matrix = MachineMatrices.__to_capab_matrix(
                dims, capab_dict_m)
            return capab_dict_m, capab_dict, capab_matrix
        elif type(machine_capabilities) == np.ndarray:
            capab_dict_m = MachineMatrices.__to_capab_dict(machine_capabilities)
            capab_dict = MachineMatrices.__invert_capab_dict(capab_dict_m)
            return capab_dict_m, capab_dict, machine_capabilities
        elif type(machine_capabilities) == dict:
            if not type_indexed_capabilities:
                capab_dict_t = MachineMatrices.__invert_capab_dict(
                    machine_capabilities)
                capab_dict_m = machine_capabilities
            else:
                capab_dict_m = MachineMatrices.__invert_capab_dict(
                    machine_capabilities)
                capab_dict_t = machine_capabilities
            capab_matrix = MachineMatrices.__to_capab_matrix(
                dims, capab_dict_m)
            return capab_dict_m, capab_dict_t, capab_matrix
        else:
            raise UndefinedInputType(
                type(machine_capabilities),
                "machine_capabilities parameter. Accepted inputs are"
                "the empty string, a boolean numpy matrix mapping machines to "
                "compatible types or a dictionary indexed by machine numbers "
                "with the compatible type lists as values.")

    def __set_machine_failures(
            self, dims: SchedulingDimensions, failure_times: 'T',
            op_durations: np.ndarray) -> (np.ndarray, list):
        machine_fails = {}
        if failure_times == 'default_sampling':
            job_lentghs = op_durations.sum(axis=1)
            j_min, j_max = int(job_lentghs.min()), int(job_lentghs.max())
            mtbf = job_lentghs.sum() / 3  # mean time between failure
            exp_sample = self.__rng.exponential(mtbf, 1000)
            for m in range(1, dims.n_machines + 1):
                # 0.5 chance mach cannot fail, 0.5 it fails at most three times
                if self.__rng.choice([0, 1]) == 1:
                    # flip reliability dist and sample 3 points
                    fails = np.cumsum(self.__rng.choice(
                        1 - exp_sample + exp_sample.max(initial=-1e5), 3))
                    repair_times = self.__rng.choice(
                        range(j_min, j_max, int(j_max - j_min / 10)), 3)
                    machine_fails[m] = list(zip(fails, repair_times))
            return machine_fails
        elif type(failure_times) == dict:
            return failure_times
        elif failure_times is None:
            return machine_fails
        else:
            raise UndefinedInputType(
                type(failure_times),
                "machine_failures parameter. Accepted inputs are"
                "the empty string or, dictionary indexed by machine ids with"
                "lists of (failure time, repair duration) tuples as values or "
                "None.")

    # </editor-fold>

    # <editor-fold desc="Sampling Functions">
    def __get_truncnormal_op_duration_approx(
            self, op_durations: np.ndarray, scaler=1.0) -> Callable:
        """
        Constructs a truncnormal sampling function with parameters derived from
        the operation duration distribution.

        :param op_durations: The operation duration matrix.
        :param scaler: A scaler to describe the times in the distribution
            approximation relative to the original, e.g. 0.1 -- mean time == 10%
            of the processing time mean.
        :return: A sampling function taking a shape tuple as a parameter and
            returning a corresponding ndarray sampled from the calculated
            distribution .
        """
        ops = op_durations.flatten()
        ops = ops[np.nonzero(ops)]
        lo, hi = scaler * ops.min(initial=1e5), scaler * ops.max(initial=-1e5)
        mu, sigma = scaler * ops.mean(), ops.std()
        dist = stats.truncnorm(
            (lo - mu) / sigma, (hi - mu) / sigma, loc=mu, scale=sigma)
        return lambda x: dist.rvs(x, random_state=self.__rng)

    @staticmethod
    def __sample_symmetric_matrix(sampling_function, size, diag):
        """
        Samples a full matrix, as specified by size from the distribution
        provided. The lower diagonal matrix is then eliminated, with or without
        the matrix diagonal itself. The upper diagonal matrix is then transposed
        and added to its original. If the diagonal elements were kept, these
        are devided by 2 to recreate the original samples prior to the function
        return.

        :param sampling_function: A function taking a shape as an argument and
            returning a corresponding ndarray.
        :param size: The shape of the distribution sample.
        :param diag: True if the symmetric matrix should contain diagonal
            elements.
        :return: The symmetric matrix with entries sampled from the requested
            distribution.
        """
        assert size[0] == size[1]
        upper_diagonal_matrix = np.triu(sampling_function(size), 1 - diag)
        lower_diagonal_matrix = upper_diagonal_matrix.transpose()
        symmetric_m = upper_diagonal_matrix + lower_diagonal_matrix
        if diag:
            np.fill_diagonal(
                symmetric_m, symmetric_m.diagonal() / 2)
        return symmetric_m

    # </editor-fold>

    # <editor-fold desc="Getters">
    @property
    def machine_speeds(self):
        return self.__machine_speeds

    @property
    def machine_distances(self):
        return self.__machine_distances

    @property
    def machine_buffer_capa(self):
        return self.__machine_buffer_capa

    @property
    def machine_capabilities_dm(self):
        return self.__machine_capabilities_dm

    @property
    def machine_capabilities_dt(self):
        return self.__machine_capabilities_dt

    @property
    def machine_capabilities_m(self):
        return self.__machine_capabilities_m

    @property
    def machine_failures(self):
        return self.__machine_failures

    @property
    def tool_switch_times(self):
        return self.__tool_switch_times
    # </editor-fold>


class Input:
    """
    Object containing all the simulation parameters.
    All random sampling, if required, is executed within this class.
    """

    def __init__(self, si: SchedulingUserInputs,
                 sim_rng: FabricatioRNG, seed=-1,
                 logfile_path=''):
        # saved inputs for re-sampling
        self.__scheduling_inputs = si
        # sampling with seeded RNG
        self.__rng = sim_rng.reinitialize_rng(seed=seed).rng
        self.__dims = SchedulingDimensions(si, self.__rng, seed)
        self.__matrices_j = JobMatrices(self.__dims, si, self.__rng)
        self.__matrices_m = MachineMatrices(self.__dims, si, self.__matrices_j,
                                            self.__rng)
        # logging
        self.__logfile_path = logfile_path
        self.name = si.name
        self.seed = seed

    def __deepcopy__(self, memo):
        """
        Return a deepcopy of the input object.
        TODO: when initializing the state, deepcopy all mutable objects; in
            doing so, we could return a shallow copy here instead of a deep one

        :param memo: The object copy memory for recursive calls.
        :return: The deep copy of the Input object.
        """
        return faster_deepcopy(self, memo)

    # <editor-fold desc="Second-Stage Initialization Methods">
    @staticmethod
    def get_least_busy_machine(s: 'State'):
        tgt_machines = s.legal_actions
        buffer_lengths = s.trackers.buffer_lengths[
            np.array(tgt_machines) - 1]
        m = np.argmin(buffer_lengths)
        return tgt_machines[m]if len(m.shape) == 0 else m[0]

    def run_full_wip_simulation(self, wip_size: int,
                                sim_core: 'Core') -> ('State', np.array):
        """
        TODO: Explain!!!
        :param wip_size:
        :param sim_core:
        :return:
        """
        s, done = sim_core.state, False
        n_ops_total = wip_size * self.dims.max_n_operations
        st = None
        remaining_op_ratios = deque([])
        while not done:
            if s.in_postbuffer_routing_mode():
                action = Input.get_least_busy_machine(s)
            else:
                legal_actions = sim_core.state.legal_actions
                action = legal_actions[0]
            assert action != n_ops_total
            s, done = sim_core.step(action)
            st = s.trackers
            remaining_op_ratios.append(st.job_n_remaining_ops[:wip_size].sum())
        wip_op_ave = np.array(remaining_op_ratios).mean() / n_ops_total
        utilization_times = st.utilization_times / s.system_time
        return st.flow_time, utilization_times, wip_op_ave

    def ammend_release_dates(self, u, f, wip_op_ave, n0):
        """
        TODO: explain!!!
        :param u:
        :param f:
        :param wip_op_ave:
        :param n0:
        :return:
        """
        n, m = (self.dims.n_jobs, self.dims.n_machines)
        job_arrivals = None
        si = self.scheduling_inputs
        if si.inter_arrival_time == 'balanced':
            utl_agg = u.mean()
        elif si.inter_arrival_time == 'under':
            utl_agg = u.min()
        elif si.inter_arrival_time == 'over':
            utl_agg = u.max()
        else:
            raise AttributeError(f"Illegal scheduling input combination "
                                 f"involving the {si.inter_arrival_time} value "
                                 f"for the 'inter_arrival_time' key.")
        # inter-arrival parametrization distribution ;)
        ia_param_dist: np.ndarray = (f[:n0] * 0.9 * wip_op_ave) / (m * utl_agg)
        try:
            assert (ia_param_dist > 0).all()
        except AssertionError:
            print("herehere")

        job_arrivals = np.zeros(n)
        lo, hi = ia_param_dist.min(), ia_param_dist.max()
        mu, sigma = ia_param_dist.mean(), ia_param_dist.std()
        dist = stats.truncnorm(
            (lo - mu) / sigma, (hi - mu) / sigma,
            loc=mu, scale=sigma)
        t_interarrival_samples = dist.rvs(n - n0, random_state=self.__rng)
        t_arrivals = np.cumsum(t_interarrival_samples)
        job_arrivals[n0:] = t_arrivals
        self.__matrices_j.job_arrivals = job_arrivals
        return job_arrivals

    def ammend_due_dates(self, release_dates: np.ndarray):
        self.matrices_j.job_due_dates = self.matrices_j.set_due_dates(
            self.dims, self.__scheduling_inputs.due_dates, release_dates,
            self.matrices_j.operation_durations)

    def finalize_stochastic_inputs(self, core: 'Core'):
        """
        TODO: Explain!!!
        :param core:
        :return:
        """
        if not self.matrices_j.simulation_needed:
            return core
        # else:
        # 1. copy the simulation core
        core_c = deepcopy(core)
        # 2. make simulation deterministic
        core_c.make_deterministic()
        # 3. extract release date parametrization (utililzation etc.)
        n0 = self.dims.max_jobs_visible
        f, u, wip_op_ave = self.run_full_wip_simulation(n0, core_c)
        # 4. ammend release dates, due dates etc
        rd = self.ammend_release_dates(u, f, wip_op_ave, n0)
        self.ammend_due_dates(rd)
        self.matrices_j.simulation_needed = False
        # 5. reurn self
        return self

    # </editor-fold desc="Second-Stage Initialization Methods">

    # <editor-fold desc="Getters">
    @property
    def scheduling_inputs(self):
        return self.__scheduling_inputs

    @property
    def dims(self):
        return self.__dims

    @property
    def matrices_j(self):
        return self.__matrices_j

    @property
    def matrices_m(self):
        return self.__matrices_m

    @property
    def logfile_path(self):
        return self.__logfile_path
    # </editor-fold>
