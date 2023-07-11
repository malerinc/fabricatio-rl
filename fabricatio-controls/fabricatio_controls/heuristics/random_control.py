import random
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from fabricatio_controls import Control
from fabricatio_rl.interface_templates import Optimizer

if TYPE_CHECKING:
    from fabricatio_rl.interface import FabricatioRL
    from fabricatio_rl.core_state import State


class Random(Optimizer):
    def __init__(self, nowait=True, target_mode='universal', seed=100):
        self.nowait = nowait
        np.random.seed(seed)
        super().__init__(target_mode)

    def get_action(self, state: 'State') -> int:
        if self.nowait:
            wait_flag = (state.params.max_n_operations *
                         state.params.max_jobs_visible)
            if wait_flag in state.legal_actions:
                # wait is always the last action
                return random.choice(state.legal_actions[:-1])
        return random.choice(state.legal_actions)


class RandomControl(Control):
    def __init__(self, nowait=True):
        super(RandomControl, self).__init__(f'RandomNowait{nowait}')
        self.optimizer = Random(nowait)

    def play_game(self, env: 'FabricatioRL', initial_state=None) -> 'State':
        env_c = deepcopy(env)
        done, state = False, None
        while not done:
            act = self.optimizer.get_action(env_c.core.state)
            state, done = env_c.core.step(act)
        return state