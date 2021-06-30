import numpy as np
import math
import bisect
import scipy.stats as stats
from typing import TypeVar, Callable
from gym_fabrikatioRL.envs.env_utils import UndefinedInputType
from copy import deepcopy

# indicates generic types
T = TypeVar('T')


class SchedulingDimensions:
    """
    Initializes and stores scheduling problem dimensions.
    """
    def __init__(self, n_jobs, n_machines, n_tooling_lvls, n_types,
                 n_operations, min_n_operations, max_n_operations,
                 max_n_failures, max_jobs_visible, n_jobs_initial):
        self.__n_jobs = n_jobs
        self.__n_machines = n_machines
        self.__n_tooling_lvls = n_tooling_lvls
        self.__n_types = n_types
        if type(n_operations) == np.ndarray:
            self.__n_operations = n_operations
        elif n_operations == 'default_sampling':
            assert min_n_operations < max_n_operations
            self.__n_operations = np.random.randint(
                min_n_operations, max_n_operations, n_jobs)
        elif min_n_operations == max_n_operations:
            self.__n_operations = np.repeat(max_n_operations, n_jobs)
        else:
            raise UndefinedInputType(
                type(n_operations), " n_operations parameter.")
        self.__min_n_operations = min_n_operations
        self.__max_n_operations = max_n_operations
        self.__max_n_failures = max_n_failures
        self.__max_jobs_visible = max_jobs_visible
        self.__n_jobs_initial = n_jobs_initial

    # <editor-fold desc="Getters">
    @property
    def n_jobs(self):
        return self.__n_jobs

    @property
    def n_machines(self):
        return self.__n_machines

    @property
    def n_tooling_lvls(self):
        return self.__n_tooling_lvls

    @property
    def n_types(self):
        return self.__n_types

    @property
    def n_operations(self):
        return self.__n_operations

    @property
    def min_n_operations(self):
        return self.__min_n_operations

    @property
    def max_n_operations(self):
        return self.__max_n_operations

    @property
    def max_n_failures(self):
        return self.__max_n_failures

    @property
    def max_jobs_visible(self):
        return self.__max_jobs_visible

    @property
    def n_jobs_initial(self):
        return self.__max_jobs_visible
    # </editor-fold>


class GraphUtils:
    """
    Methods for precedence graph generation and and transformation.
    """
    # <editor-fold desc="Transformation Functions">
    @staticmethod
    def graph_adjacency_list_to_matrix(graph_adjacency_list: dict,
                                       max_n_ops: int,
                                       current_job: T = -1) -> np.ndarray:
        """
        Converts an adjacency list representation of the precedence constraints
        graph to a matrix representation. The graph root given by current_job
        parameter is ignored.

        :param current_job: Representation of the job root; this node's
            children represent the first eligible operations in a job.
        :param graph_adjacency_list: The adjacency list representation of the
            precedence constraints partial order graph.
        :param max_n_ops: The size of the matrix; needed for consitency with the
            other jobs.
        :return: The matrix representation of the precedence constraints graph.
        """
        graph_matrix = np.zeros((max_n_ops, max_n_ops))
        for node in graph_adjacency_list.keys():
            if node == current_job:  # ignore job root
                continue
            for neighbor in graph_adjacency_list[node]:
                graph_matrix[node][neighbor] = 1
        return graph_matrix

    @staticmethod
    def graph_matrix_to_adjacency_list(matrix: np.ndarray, n_ops: int,
                                       current_job: int) -> dict:
        """
        Converts a n_operations x n_operations matrix containing the
        an adjacency into a corresponding adjacency dictionary representation.

        :param matrix: A precedence graph matrix.
        :param n_ops: List of the number of operations in each job.
        :param current_job: Current job number for the root label.
        :return: The adjacency dictionary representation of the matrix.
        """
        ingress_counts, job_adjacency = {}, {}
        for node_out in range(n_ops):
            for node_in in range(n_ops):
                edge = matrix[node_out][node_in]
                if edge == 0:
                    continue
                if node_out not in job_adjacency:
                    job_adjacency[node_out] = {node_in}
                else:
                    job_adjacency[node_out].add(node_in)
                if node_in not in ingress_counts:
                    ingress_counts[node_in] = 1
                else:
                    ingress_counts[node_in] += 1
        job_adjacency[(current_job,)] = (set(range(n_ops)) -
                                         set(ingress_counts.keys()))
        return job_adjacency
    # </editor-fold>

    # <editor-fold desc="POm Precedence Generation">
    @staticmethod
    def get_random_precedence_relation(n_ops: int,
                                       n_ops_max: int) -> (dict, np.ndarray):
        """
        DEPRECATED COMMENT
        Creates random hasse diagrams representing the operation precedence
        relation. It work by iteratively sampling random integers smaller than
        10e6 and adding their divisors (found in O(sqrt(n))) to a list until the
        latter's length is between n_ops_min and n_ops_max. Then the adjacency
        list representation corresponding to the 'divides' relation is computed
        and transitively reduced.

        The nodes (divisors) are renamed while building the adjacency lists to
        sequential integers, wile maintaining the original relation.

        It is ensured that 1 (renamed to 0) is always part of the relation,
        such that the job root dummy can easily be inserted.

        :param n_ops: The minimum number of operations.
        :param n_ops_max: The maximum number of job operations.
        :return: The Hasse diagram of the the job operation precedence
            constraints in its matrix and adjacency list representation.
        """
        divisors = set([])
        while len(divisors) < n_ops + 1:
            new_int = np.random.randint(1000000)
            divisors |= set(GraphUtils.__get_divisors(new_int))
            if len(divisors) > n_ops + 1:
                while len(divisors) > n_ops + 1:
                    divisors.pop()
                break
        if 1 not in divisors:
            divisors.pop()
            divisors.add(1)
        graph = GraphUtils.__get_adjacency(divisors)
        al_hasse = GraphUtils.__transitive_reduction(graph)
        am = GraphUtils.graph_adjacency_list_to_matrix(
            al_hasse, n_ops_max, -1)  # -1 is the generic job root node
        return al_hasse, am

    @staticmethod
    def __get_divisors(n):
        """
        Finds the divisors of a number. O(sqrt(n))
        Source: https://github.com/tnaftali/hasse-diagram-processing-py
        """
        divisors = []
        limit = int(str(math.sqrt(n)).split('.')[0])
        for i in range(1, limit + 1):
            if n % i == 0:
                bisect.insort(divisors, i)
                if i != (n / i):
                    bisect.insort(divisors, n / i)
        return divisors

    @staticmethod
    def __get_adjacency(divisors: set):
        """
        Constructs Adjacency list repesentation for the division relation;
        Renames the nodes sequentially; O(n^2).
        """
        latest_node_nr = -1
        sequential_names = {}
        graph = {}
        for i in divisors:
            if i not in sequential_names:
                sequential_names[i] = latest_node_nr
                latest_node_nr += 1
            neighbors = set([])
            for j in divisors:
                if j not in sequential_names:
                    sequential_names[j] = latest_node_nr
                    latest_node_nr += 1
                if j % i == 0 and i != j:
                    neighbors.add(sequential_names[j])
            graph[sequential_names[i]] = neighbors
        return graph

    @staticmethod
    def __transitive_closure(graph, node, closure, visited):
        """
        Adds all nodes reacheable from the node parameter in the graph
        parameter to the closure parameter.
        """
        if node in visited:
            return
        visited |= {node}
        closure |= graph[node]  # O(1)
        for neighbor in graph[node]:  # O(|V| + |E|)
            GraphUtils.__transitive_closure(graph, neighbor, closure, visited)

    @staticmethod
    def __transitive_reduction(graph):
        """
        Computes the transitive reduction by eliminating direct node
        neighbors who are present in the union of all the direct neighbor
        transitive clauses. O(N)
        """
        reduced_graph = {}
        for node in graph.keys():
            neighbor_closure, visited = set({}), set({})
            good_neighbors = set({})
            for neighbor in graph[node]:
                GraphUtils.__transitive_closure(
                    graph, neighbor, neighbor_closure, visited)
            for neighbor in graph[node]:
                if neighbor not in neighbor_closure:
                    good_neighbors.add(neighbor)
            reduced_graph[node] = good_neighbors
        return reduced_graph
    # </editor-fold>

    # <editor-fold desc="Jm/Fm Precedence Generation">
    @staticmethod
    def get_job_chain_precedence_graphs(n_jobs: int, n_ops: np.ndarray) -> list:
        """
        Creates a list of dictionaries containing adjacency list representations
        of linear precedence constraints, one for every job. Every job graph has
        a the following tuple as a root node: (j_index,).

        Example n_jobs == 2, n_ops == [3, 5]:
            [{(0,): [0], 0: [1], 1: [2]},
             {(1,): [0], 0: [1], 1: [2], 2: [3], 3: [4]}]

        :param n_jobs: Number of jobs for which to construct the chain
            precedence graphs.
        :param n_ops: List containing the number of operations in every job.
        :return: List of dictionaries representing the operation precedence
            (chain) graphs.
        """
        graphs = []
        for i in range(n_jobs):
            graph_dict = GraphUtils.__graph_chain_precedence(
                list(range(n_ops[i])))
            graph_dict[(i,)] = [0]  # dummy element for job root
            graphs.append(graph_dict)
        return graphs

    @staticmethod
    def __graph_chain_precedence(operations_range: list) -> dict:
        adjacency_dict = {}
        start_node = operations_range[0]
        for node in operations_range[1:]:
            adjacency_dict[start_node] = [node]
            start_node = node
        return adjacency_dict
    # </editor-fold>


class JobMatrices:
    def __init__(self, dims: SchedulingDimensions,
                 job_pool: np.ndarray, op_types: T, op_durations: T,
                 op_tool_sets: T, op_precedence: T, due_dates: T,
                 time_inter_release: T,
                 perturbation_processing_time: T):  # perturbation_due_date: T
        if job_pool is not None:
            job_idxs = np.random.choice(
                len(job_pool), dims.n_jobs, replace=True)
            jobs = job_pool[job_idxs]
            # todo: pool as an object
            # todo: check and report pool consistency (i.e. dims vs matrix dims)
            self.__op_type = jobs[:, 0]
            self.__op_duration = jobs[:, 1]
            self.__op_tool_set = jobs[:, 2]
            als, ams = JobMatrices.__set_op_precedence_from_spec(
                dims, jobs[:, 3])
            self.__op_precedence_l, self.__op_precedence_m = als, ams
        else:
            self.__op_type = JobMatrices.__set_op_types(dims, op_types)
            self.__op_duration = JobMatrices.__set_op_duration(
                dims, op_durations, self.__op_type)
            self.__op_tool_set = JobMatrices.__set_op_tool_sets(
                dims, op_tool_sets, self.__op_type)
            als, ams = JobMatrices.__set_op_precedence(dims, op_precedence)
            self.__op_precedence_l, self.__op_precedence_m = als, ams
        self.__op_perturbations = JobMatrices.__set_op_perturbation(
            dims, perturbation_processing_time)
        self.__job_release_times = JobMatrices.__set_release_dates(
            dims, time_inter_release, self.operation_durations)
        self.__job_due_dates = JobMatrices.__set_due_dates(
            dims, due_dates, self.__job_release_times, self.operation_durations)
        # todo!
        # self.__job_due_date_perturbation = JobMatrices.__set_due_date_noise(
        #     perturbation_due_date)

    # <editor-fold desc="Constructor Helpers">
    @staticmethod
    def __set_op_types(dims: SchedulingDimensions, op_types: T):
        if type(op_types) == np.ndarray:
            return np.array(op_types).astype('int16')
        else:
            n, o_max = dims.n_jobs, dims.max_n_operations
            if callable(op_types):
                set_op_types = JobMatrices.__sample(op_types, (n, o_max))
            elif op_types == 'default_sampling':  # op_types == '':
                set_op_types = np.random.choice(
                    range(1, dims.n_types + 1), (n, o_max), replace=True)
            elif op_types == 'Jm':
                assert dims.n_types == dims.max_n_operations
                assert dims.max_n_operations == dims.min_n_operations
                set_op_types = np.zeros((n, o_max))
                for j in range(n):
                    set_op_types[j, :] = np.random.permutation(dims.n_types) + 1
            elif op_types == 'Fm':
                assert dims.n_types == dims.max_n_operations
                assert dims.max_n_operations == dims.min_n_operations
                set_op_types = np.zeros((n, o_max))
                job_structure = np.random.permutation(dims.n_types) + 1
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

    @staticmethod
    def __set_op_duration(dims: SchedulingDimensions, op_durations: T,
                          op_types: np.ndarray):
        if type(op_durations) == np.ndarray:
            return np.array(op_durations).astype('int16')
        else:
            n, o_max, n_ty = dims.n_jobs, dims.max_n_operations, dims.n_types
            if callable(op_durations):
                set_op_duration = JobMatrices.__sample(
                    op_durations, (n, o_max))
            elif op_durations == 'default_sampling':  # op_durataions == ''
                # type conditional duration dist
                tcdd = JobMatrices.__create_tcdd(n_ty)
                tcdd[0] = np.zeros(2)  # type 0 means no operation, hence d = 0
                tcdd_samples = []
                for ty in range(0, n_ty + 1):
                    tcdd_sample = np.random.choice(tcdd[ty], n)
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

    @staticmethod
    def __set_op_tool_sets(dims: SchedulingDimensions, op_tools: T,
                           op_types: np.ndarray):
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
                tctsd = JobMatrices.__create_tctsd(n_ty, n_tl)
                tctsd[0] = np.zeros(2)  # type 0 means no operation, hence d = 0
                tctsd_samples = []
                for ty in range(0, n_ty + 1):
                    tctsd_sample = np.random.choice(tctsd[ty], n).astype(
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
            precedence_specification: T) -> (list, np.ndarray):
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
        if type(precedence_specification) == np.ndarray:
            ams, als = precedence_specification, []
            for j in range(dims.n_jobs):
                al = GraphUtils.graph_matrix_to_adjacency_list(
                    ams[j, :], o, j)
                als.append(al)
        elif type(precedence_specification) == list:
            ams, als = np.zeros(n, o, o), precedence_specification
            for j in range(dims.n_jobs):
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

    @staticmethod
    def __set_op_precedence(dims: SchedulingDimensions,
                            op_precedence: T) -> (np.ndarray, list):
        """
        Computes the list of adjacency dicts and precedence matrix tensor for
        the init function.

        :param dims: Problem dimension object.
        :param op_precedence: Precedence spec; either 'Jm', 'Om', 'POm', ndarray
            or list of dicts.
        :return: The precedence constraints as a multidimensional matrix and the
            equivalen list of adjacency dictionaries.
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
                    o_js[j], o_max)
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

    @staticmethod
    def __set_op_perturbation(dims: SchedulingDimensions,
                              op_perturbation: T) -> np.ndarray:
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
            m_op_perturbations = dist.rvs((n, o))
        elif callable(op_perturbation):
            m_op_perturbations = JobMatrices.__sample(op_perturbation, (n, o))
        elif type(op_perturbation) == float:
            assert 1 > op_perturbation >= 0
            p_range_step = (2 * op_perturbation) / 10
            p_range = np.arange(-op_perturbation, op_perturbation, p_range_step)
            m_op_perturbations = np.random.choice(1 + p_range, (n, o))
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

    @staticmethod
    def __set_release_dates(dims: SchedulingDimensions,
                            time_inter_release: T, op_durations: np.ndarray):
        """
        Sets the relese date for jobs past the n_jobs_initial (which have a
        release date of 0) by summing inter-arrival times as defined per
        time_inter_release argument. If the empty string is passed in the latter
        the method defaults to sampling inter-arrival times from a truncated
        normal distribution informed by the operation processing time
        distribution.

        If a sampling function is passed, it will be used for
        inter-arrival times. Alternatively one can pass the vector of
        inter-arrival times directly.

        :param dims: The scheduling problem dimensions.
        :param time_inter_release: The inter-arrival time sampling function,
            vector or the empty string.
        :param op_durations: The operation duration matrix.
        :return: The vector of release times for jobs.
        """
        n, n_0, tir = dims.n_jobs, dims.n_jobs_initial, time_inter_release
        m = dims.n_machines * 0.4
        job_arrivals = np.zeros(n, dtype='uint16')
        if n == n_0:
            return job_arrivals
        elif time_inter_release == 'default_sampling':
            ds = op_durations
            lo, hi = ds.min(initial=np.inf) / m, ds.max(initial=-10) / m
            mu, sigma = ds.mean() / m, ds.std()
            dist = stats.truncnorm(
                (lo - mu) / sigma, (hi - mu) / sigma,
                loc=mu, scale=sigma)
            job_arrivals[n_0:] = np.cumsum(dist.rvs(n - n_0))
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
    def __set_due_dates(dims: SchedulingDimensions, due_dates: T,
                        release_dates: np.ndarray,
                        op_durations: np.ndarray) -> np.ndarray:
        """
        Sets the due dates for jobs by adding scaling (default 1.5) the job
        duration lower  bound with the passed due_dates parameter and addint it
        to the release dates. Alternatively, the due dates vector can be passed
        directly as an ndarray.

        :param dims: The scheduling problem dimensions.
        :param due_dates: The due date vector, scaler or the empty string.
        :param release_dates: The job release dates.
        :param op_durations: The job operation duration matrix.
        :return: The due dates vector.
        """
        n, n_0 = dims.n_jobs, dims.n_jobs_initial
        vec_due_dates = np.zeros(n, dtype='uint16')
        if due_dates == 'default_sampling':
            vec_due_dates[:n_0] = np.cumsum(
                1.5 * op_durations[:n_0].sum(axis=1))
            vec_due_dates[n_0:] = (release_dates[n_0:] +
                                   1.5 * op_durations[n_0:].sum(axis=1))
        elif type(due_dates) == float:
            vec_due_dates[:n_0] = np.cumsum(
                due_dates * op_durations[:n_0].sum(axis=1))
            vec_due_dates[n_0:] = (release_dates[n_0:] +
                                   due_dates * op_durations[n_0:].sum(axis=1))
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
    def __sample(sampling_function, size):
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
    def __create_tcdd(n_types: int) -> dict:
        """
        Creates a operation type conditional bi-modal processing time
        distribution. To 1000 samples are drawn from two distinct lognormal
        distributions with a type dependent mean distribution parameter prior
        concatenation.

        :param n_types: The number of operation types.
        :return: A dictionary with types as keys and lists of 2000 distribution
            points.
        """
        conditional_dist = {}
        for i in range(1, n_types + 1):
            step = (i/n_types)
            bimodal_sample = np.concatenate(
                [np.random.lognormal(np.log(50 + 50 * step), 0.2, 1000),
                 np.random.lognormal(np.log(150 - 50 * step), 0.08, 1000)])
            operation_times = np.ceil(bimodal_sample).astype('uint16')
            conditional_dist[i] = np.random.choice(operation_times, 2000)
        return conditional_dist

    @staticmethod
    def __create_tctsd(n_types: int, total_tools: int) -> dict:
        """
        Creates a type dependent tool set distribution. The number of n_t of
        tool sets associated with each type are first drawn. Then n_t tool types
        are drawn uniformly at random from the ran {1 .. total_tools} for each
        type t.

        :param n_types: Number of operation types.
        :param total_tools: Number of tools in the system.
        :return: Dictionary of type dependent tool set distributions.
        """
        assert total_tools >= 2
        conditional_dist = {}
        tool_range = range(1, total_tools + 1)
        for i in range(1, n_types + 1):
            n_type_tools = np.random.randint(1, total_tools // 2 + 1)
            conditional_dist[i] = np.random.choice(tool_range, n_type_tools)
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


class MachineMatrices:
    def __init__(self, dims: T, machine_speeds: T, machine_distances: T,
                 machine_buffer_capacities: T, machine_capabilities: T,
                 machine_failures: T, tool_switch_times: T,
                 job_matrices: JobMatrices):
        self.__machine_distances = MachineMatrices.__set_machine_distances(
            dims, machine_distances, job_matrices.operation_durations)
        self.__tool_switch_times = MachineMatrices.__set_tool_switch_times(
            dims, tool_switch_times, job_matrices.operation_durations)
        self.__machine_speeds = MachineMatrices.__set_machine_speeds(
            dims, machine_speeds)
        self.__machine_buffer_capa = MachineMatrices.__set_machine_buffer_capa(
            dims, machine_buffer_capacities)
        cdm, cdt, cm = (MachineMatrices.__set_machine_capabilities(
            dims, machine_capabilities))
        self.__machine_capabilities_dm = cdm
        self.__machine_capabilities_dt = cdt
        self.__machine_capabilities_m = cm
        # stochastic influences
        self.__machine_failures = MachineMatrices.__set_machine_failures(
            dims, machine_failures, job_matrices.operation_durations)

    # <editor-fold desc="Constructor Helpers">
    @staticmethod
    def __set_machine_distances(dims: SchedulingDimensions,
                                machine_distances: T,
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
            dist_f = MachineMatrices.__get_truncnormal_op_duration_approx(
                op_durations)
            transport_times = MachineMatrices.__sample_symmetric_matrix(
                dist_f, (m + 1, m + 1), False).astype('uint16')
        elif type(machine_distances) == float:
            dist_f = MachineMatrices.__get_truncnormal_op_duration_approx(
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
                "shape and returning a corresponding ndarray, a foat to scale "
                "the transport time distribution relative to the operation "
                "duration distribution or an ndarray with pre-set "
                "machine distances as positive integers or 0.")
        return transport_times

    @staticmethod
    def __set_tool_switch_times(dims: SchedulingDimensions,
                                tool_switch_times: T,
                                op_durations: np.ndarray) -> np.ndarray:
        """
        Analogous to "__set_machine_distances"
        """
        tl_lvls = dims.n_tooling_lvls
        if tool_switch_times == 'default_sampling':
            dist_f = MachineMatrices.__get_truncnormal_op_duration_approx(
                op_durations, 0.1)
            tooling_times = MachineMatrices.__sample_symmetric_matrix(
                dist_f, (1 + tl_lvls, 1 + tl_lvls), False).astype('uint16')
        elif tool_switch_times is None:
            tooling_times = np.zeros((1 + tl_lvls, 1 + tl_lvls))
        elif type(tool_switch_times) == float:
            dist_f = MachineMatrices.__get_truncnormal_op_duration_approx(
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
                "returning a corresponding ndarray, a foat to scale the "
                "tooling time distribution relative to the operation duration"
                "distribution or an ndarray with pre-set "
                "toling as positive integers or 0.")
        return tooling_times

    @staticmethod
    def __set_machine_speeds(dims: SchedulingDimensions,
                             machine_speeds: T) -> np.ndarray:
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
            return np.random.choice(np.arange(0.5, 1.5, 0.1), m)
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
                                  machine_buffer_capacities: T):
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
    def __to_type_indexed_capab(m_idxed_m_capa: dict) -> dict:
        t_idexed_m_capa = {}
        for m_i in m_idxed_m_capa.keys():
            for t_i in m_idxed_m_capa[m_i]:
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

    @staticmethod
    def __set_machine_capabilities(
            dims: SchedulingDimensions,
            machine_capabilities: T) -> (dict, np.ndarray):
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
                n_capab = np.random.randint(1, t + 1)
                m_capab = np.random.choice(
                    np.arange(t) + 1, n_capab, replace=False)
                capab_dict_m[i] = list(m_capab)
                capab_matrix[i - 1, [x - 1 for x in capab_dict_m[i]]] = np.ones(
                    len(capab_dict_m[i]))
                encountered_types |= set(m_capab)
            missing_types = set(range(1, t + 1)) - encountered_types
            # make sure each type occurs at least once!!!
            for mty in missing_types:
                rand_m = np.random.randint(1, m + 1)
                capab_dict_m[rand_m].append(mty)
                capab_matrix[rand_m - 1, mty - 1] = 1
            # convert to type indexed
            capab_dict = MachineMatrices.__to_type_indexed_capab(capab_dict_m)
            return capab_dict_m, capab_dict, capab_matrix
        elif machine_capabilities is None:
            assert dims.n_types == dims.n_machines
            capab_dict_m = {m: [m] for m in range(1, dims.n_types + 1)}
            capab_dict = MachineMatrices.__to_type_indexed_capab(capab_dict_m)
            capab_matrix = MachineMatrices.__to_capab_matrix(
                dims, capab_dict_m)
            return capab_dict_m, capab_dict, capab_matrix
        elif type(machine_capabilities) == np.ndarray:
            capab_dict_m = MachineMatrices.__to_capab_dict(machine_capabilities)
            capab_dict = MachineMatrices.__to_type_indexed_capab(capab_dict_m)
            return capab_dict_m, capab_dict, machine_capabilities
        elif type(machine_capabilities) == dict:
            capab_matrix = MachineMatrices.__to_capab_matrix(
                dims, machine_capabilities)
            capab_dict = MachineMatrices.__to_type_indexed_capab(
                machine_capabilities)
            return machine_capabilities, capab_dict, capab_matrix
        else:
            raise UndefinedInputType(
                type(machine_capabilities),
                "machine_capabilities parameter. Accepted inputs are"
                "the empty string, a boolean numpy matrix mapping machines to "
                "compatible types or a dictionary indexed by machine numbers "
                "with the compatible type lists as values.")

    @staticmethod
    def __set_machine_failures(dims: SchedulingDimensions,
                               failure_times: T,
                               op_durations: np.ndarray) -> (np.ndarray, list):
        machine_fails = {}
        if failure_times == 'default_sampling':
            job_lentghs = op_durations.sum(axis=1)
            j_min, j_max = int(job_lentghs.min()), int(job_lentghs.max())
            mtbf = job_lentghs.sum() / 3   # mean time between failure
            exp_sample = np.random.exponential(mtbf, 1000)  # reliability sample
            for m in range(1, dims.n_machines + 1):
                # 0.5 chance mach cannot fail, 0.5 it fails at most three times
                if np.random.choice([0, 1]) == 1:
                    # flip reliability dist and sample 3 points
                    fails = np.cumsum(np.random.choice(
                        1 - exp_sample + exp_sample.max(initial=-1e5), 3))
                    repair_times = np.random.choice(
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
    @staticmethod
    def __get_truncnormal_op_duration_approx(
            op_durations: np.ndarray, scaler=1.0) -> Callable:
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
        return dist.rvs

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
    Read-Only Object containing all the simulation parameters.
    All random sampling, if required, is executed within this class.
    """
    def __init__(self, scheduling_inputs, seed=-1, logfile_path=''):
        # saved inputs for re-sampling
        self.__scheduling_inputs = None
        self.__unpack_scheduling_input(scheduling_inputs)
        si = deepcopy(self.__scheduling_inputs)
        # start sampling with seeded RNG
        if seed != -1:
            np.random.seed(seed)
        self.__dims = SchedulingDimensions(
            si['n_jobs'], si['n_machines'], si['n_tooling_lvls'], si['n_types'],
            si['n_operations'], si['min_n_operations'], si['max_n_operations'],
            si['max_n_failures'], si['max_jobs_visible'], si['n_jobs_initial']
        )
        self.__matrices_j = JobMatrices(
            self.__dims,
            si['job_pool'], si['operation_types'], si['operation_durations'],
            si['operation_tool_sets'], si['operation_precedence'],
            si['time_job_due'], si['time_inter_release'],
            si['perturbation_processing_time'],
            # TODO! scheduling_inputs['perturbation_due_date'],
        )
        # machine matrices
        self.__matrices_m = MachineMatrices(
            self.__dims,
            si['machine_speeds'], si['machine_distances'],
            si['machine_buffer_capa'], si['machine_capabilities'],
            si['machine_failures'], si['tool_switch_times'],
            self.__matrices_j   # informs some of the default sampling
        )
        # logging
        self.__logfile_path = logfile_path
        self.seed = seed

    def __unpack_scheduling_input(self, user_inputs: dict):
        self.__scheduling_inputs = {
            'n_jobs': 20,                 # n
            'n_machines': 20,             # m
            'n_tooling_lvls': 0,          # l
            'n_types': 20,                # t
            'min_n_operations': 20,
            'max_n_operations': 20,       # o
            'n_operations': None,         # n vec of nr. ops per job or rnd
            'max_n_failures': 0,          # f
            'n_jobs_initial': 20,         # jobs with arrival time 0
            'max_jobs_visible': 20,       # entries in {1 .. n}
            'operation_precedence': 'Jm',
            'operation_types': 'Jm',
            'operation_durations': 'default_sampling',
            'operation_tool_sets': None,
            'machine_speeds': None,
            'machine_distances': None,
            'machine_buffer_capa': None,
            'machine_capabilities': None,
            'tool_switch_times': None,
            'time_inter_release': 'default_sampling',
            'time_job_due': 'default_sampling',
            'machine_failures': None,
            'perturbation_processing_time': None,
            'perturbation_due_date': '',
            'job_pool': None}
        for key in user_inputs:
            if key not in self.__scheduling_inputs:
                raise ValueError(f"The scheduling inputs dictionary does not "
                                 f"support the {key} key.")
            else:
                self.__scheduling_inputs[key] = user_inputs[key]

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
