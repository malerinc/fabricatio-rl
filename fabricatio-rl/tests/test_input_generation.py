from fabricatio_rl.env_utils import GraphUtils
import numpy as np


def test__get_divisors_nodes():
    """
    The tested method of the GraphUtils class samples n numbers at random and
    creates the union set of the divisors thereof with the exception of 1.
    The process of sampling, computing the divisors and uniting the new set with
    the existing divisor set repeats until exactly a number of divisors
    indicated by the class method's parameter is reached.

    :return:
    """
    n_ops_list = np.random.randint(4, 200, 1000)
    for n_ops in n_ops_list:
        gu = GraphUtils()
        # noinspection
        divisors = gu._GraphUtils__get_divisors_nodes(
            n_ops, np.random.default_rng())
        assert len(divisors) == n_ops
        assert 1 not in divisors


def test__get_adjacency():
    """
    Tests for the __get_adjacency method of the graph generator class. The
    tested method renames the integers in the division relation graph, which is
    represented as a dictionary, such that only the numbers 1 to n_operations
    occur. An exception to this is the number -1 representing the job root,
    which should hence occur exactly once.

    :return: None
    """
    n_ops_list = np.random.randint(4, 200, 1000)
    for n_ops in n_ops_list:
        gu = GraphUtils()
        # noinspection
        divisors = gu._GraphUtils__get_divisors_nodes(
            n_ops, np.random.default_rng())
        graph = gu._GraphUtils__get_adjacency(divisors)
        assert len(graph) == n_ops + 1
        n_k_negative_1 = 0
        for k, val in graph.items():
            if k != -1:
                assert k in range(0, n_ops)
            else:
                n_k_negative_1 += 1
            for op_node in val:
                assert op_node != -1
                assert op_node in range(0, n_ops)
        assert n_k_negative_1 == 1

# TODO: test transitive closure generation; assert that if a-> b then the is
#  no c such that a -> c -> b
