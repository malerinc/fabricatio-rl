import bisect
import marshal
import math
import numpy as np

from inspect import isclass
from copy import deepcopy
from os import makedirs
from os.path import join, exists

from typing import TypeVar

T = TypeVar('T')


# <editor-fold desc="Exceptions">
class UndefinedSimulationModeError(Exception):
    def __init__(self):
        super().__init__('This simulation only accept modes '
                         '0 - scheduling decisions -, '
                         '1 - regular routing decisions - or '
                         '2 - routing decisions from behind a broken machine')


class MultipleParameterSetsError(Exception):
    def __init__(self):
        super().__init__('Cannot instantiate more than one simulation parameter'
                         ' sets! The class "EnvParameters" is a singleton."')


class InvalidHeuristicName(Exception):
    def __init__(self):
        super().__init__('Cannot instantiate ComparableAction: chosen heuristic'
                         ' name is unavailable.')


class InvalidStepNumber(Exception):
    def __init__(self):
        super().__init__(
            'The number of decisions exceeds the maximum allowed for this '
            'simulation run. Possible endless loop.')


class UndefinedInputType(Exception):
    def __init__(self, input_type, input_desc):
        super().__init__(f'The input type {input_type} is undefined for '
                         f'{input_desc}.')


class UndefinedLegalActionCall(Exception):
    def __init__(self, opt_conf, sim_mode):
        super().__init__(f'Cannot cal "get_legal_actions" in simulation mode '
                         f'{sim_mode} with an optimizer configuration of '
                         f'{opt_conf}')


class UndefinedOptimizerConfiguration(Exception):
    def __init__(self):
        err_string = f'The optimizer combination is not supported.'
        super().__init__(err_string)


class UndefinedOptimizerTargetMode(Exception):
    def __init__(self):
        err_string = (f'The optimizer target mode is not supported. Optimizers '
                      f'support "transport" and "sequencing" modes only')
        super().__init__(err_string)


class IllegalAction(Exception):
    def __init__(self):
        err_string = ()
        super().__init__(err_string)
# </editor-fold>


# <editor-fold desc="Utility Functions">
def create_folders(path):
    """
    Switches between '/' (POSIX) and '\'(windows) separated paths, depending on
    the current platform all non existing folders.

    :param path: A '/' separated *relative* path; the last entry is considered
        to be the file name and won't get created.
    :return: The platform specific file path.
    """
    segments = path.split('/')
    if not bool(segments):
        return path
    path_dir = join(*segments[:-1])
    file = segments[-1]
    if not exists(path_dir):
        makedirs(path_dir)
    return join(path_dir, file)


def decouple_view(arr: np.ndarray):
    """


    :param arr: The object to be copied.
    :return: A copy of the obj parameter with no shared objects in its
        structure.
    """
    immutable_types = {tuple, int, float, str, bool}
    decoupled_view = np.empty(shape=arr.shape, dtype=arr.dtype)
    for j in range(arr.shape[0]):
        if type(arr[j]) in immutable_types:
            continue
        if type(arr[j]) == dict:
            decoupled_view[j] = deepcopy(arr[j])
    return decoupled_view


def faster_deepcopy(obj, memo):
    cls = obj.__class__
    result = cls.__new__(cls)
    marshal_set = {list, tuple, set, str}
    basic_types = {int, bool, float, str}
    memo[id(obj)] = result
    for k, v in obj.__dict__.items():
        if id(v) in memo:
            setattr(result, k, memo[id(v)])
        elif type(v) == np.ndarray:
            arr_cp = v.copy()
            memo[id(v)] = arr_cp
            setattr(result, k, arr_cp)
        elif type(v) in marshal_set:
            if (len(v) == 0 or
                    (type(v) != set
                     and type(v[-1]) in basic_types
                     and type(v[0]) in basic_types)):
                collection_cp = marshal.loads(marshal.dumps(v))
                memo[id(v)] = collection_cp
                setattr(result, k, collection_cp)
            else:  # len(v) != 0 and next(iter(v)).__class__ not in basic_types
                # noinspection PyArgumentList
                collection_cp = deepcopy(v, memo)
                memo[id(v)] = collection_cp
                setattr(result, k, collection_cp)
        else:
            # noinspection PyArgumentList
            setattr(result, k, deepcopy(v, memo))
    return result
# </editor-fold>


# <editor-fold desc="STATIC PRECEDENCE GRAPH MANIPULATION CLASS">
class GraphUtils:
    """
    Methods for precedence graph generation and and transformation.
    """

    # <editor-fold desc="Transformation Functions">
    @staticmethod
    def graph_adjacency_list_to_matrix(graph_adjacency_list: dict,
                                       max_n_ops: int,
                                       # TODO: more explicit typing!
                                       current_job: 'T' = -1) -> np.ndarray:
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
    def get_random_precedence_relation(
            n_ops: int, n_ops_max: int, rng: np.random.Generator) \
            -> (dict, np.ndarray):
        """
        DEPRECATED COMMENT
        Creates random hasse diagrams representing the operation precedence
        relation. It works by iteratively sampling random integers smaller than
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
        :param rng: A numpy random number generator.
        :return: The Hasse diagram of the the job operation precedence
            constraints in its matrix and adjacency list representation.
        """
        divisors = GraphUtils.__get_divisors_nodes(n_ops, rng)
        graph = GraphUtils.__get_adjacency(divisors)
        al_hasse = GraphUtils.__transitive_reduction(graph)
        am = GraphUtils.graph_adjacency_list_to_matrix(
            al_hasse, n_ops_max, -1)  # -1 is the generic job root node
        return al_hasse, am

    @staticmethod
    def __get_divisors_nodes(n_ops: int, rng: np.random.Generator):
        divisors = set([])
        while len(divisors) < n_ops:
            new_int = rng.integers(1000000)
            divisors |= set(GraphUtils.__get_divisors(new_int))
            if len(divisors) > n_ops:
                while len(divisors) > n_ops:
                    divisors.pop()
                break
        return divisors

    @staticmethod
    def __get_divisors(n):
        """
        Finds the divisors of a number. O(sqrt(n))
        Source: https://github.com/tnaftali/hasse-diagram-processing-py
        """
        divisors = []
        limit = int(str(math.sqrt(n)).split('.')[0])
        for i in range(2, limit + 2):
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
        node = 1
        n_divisors = len(divisors)
        for _ in range(n_divisors + 1):
            if node not in sequential_names:
                sequential_names[node] = latest_node_nr
                latest_node_nr += 1
            neighbors = set([])
            for j in divisors:
                if j not in sequential_names:
                    sequential_names[j] = latest_node_nr
                    latest_node_nr += 1
                if j % node == 0 and j != node:
                    neighbors.add(sequential_names[j])
            graph[sequential_names[node]] = neighbors
            if divisors:
                node = divisors.pop()
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
            graph_dict = GraphUtils.graph_chain_precedence(
                list(range(n_ops[i])))
            graph_dict[(i,)] = [0]  # dummy element for job root
            graphs.append(graph_dict)
        return graphs

    @staticmethod
    def graph_chain_precedence(operations_range: list) -> dict:
        adjacency_dict = {}
        start_node = operations_range[0]
        for node in operations_range[1:]:
            adjacency_dict[start_node] = [node]
            start_node = node
        return adjacency_dict
    # </editor-fold>
# </editor-fold desc="STATIC PRECEDENCE GRAPH MANIPULATION CLASS">
