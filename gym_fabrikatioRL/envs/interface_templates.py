from gym_fabrikatioRL.envs.core_state import State
import numpy as np


class Optimizer:
    def __init__(self, target_mode):
        assert target_mode in ['sequencing', 'transport']
        self.target_mode = target_mode

    def get_action(self, state: State) -> int:
        pass


class ReturnTransformer:
    @staticmethod
    def transform_state(state: State) -> np.ndarray:
        """
        Transforms the environment state into the agent state. Note that a
        numpy array has to be returned for compatibility with the environment
        interface state shape inference.

        :param state: The state as defined by the environment Core.
        :return: The state representation as a numpy array.
        """
        raise NotImplementedError

    @staticmethod
    def transform_reward(state: State):
        return state.system_time
