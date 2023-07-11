from tests.t_helpers import get_envs_selectable_seq_opt
from copy import deepcopy
import numpy as np


def test_reset_invariance():
    env1, _ = get_envs_selectable_seq_opt(1)
    d0: np.ndarray = deepcopy(
        env1.parameters.matrices_j.operation_durations)
    done = False
    while not done:
        actions = env1.get_legal_actions()
        _, _, done, _ = env1.step(actions[0])
    env1.reset()
    d1: np.ndarray = deepcopy(
        env1.parameters.matrices_j.operation_durations)
    assert (d1 == d0).all()
