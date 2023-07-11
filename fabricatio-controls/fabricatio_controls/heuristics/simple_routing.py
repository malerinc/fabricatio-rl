from fabricatio_rl.interface_templates import Optimizer

from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fabricatio_rl.core_state import State


class RoutingHeuristic(Optimizer):
    def __init__(self):
        self._routes = []
        super().__init__('transport')

    def _get_routes(self, state: 'State'):
        self._routes = []
        legal_actions = state.legal_actions
        src_machine = state.current_machine_nr
        transport_matrix = state.matrices.transport_times
        Route = namedtuple('Route', 'index distance')
        for tgt_machine in legal_actions:
            self._routes.append(
                Route(index=tgt_machine,
                      distance=transport_matrix[src_machine][tgt_machine]))

    def get_action(self, state: 'State') -> int:
        pass


class LDOptimizer(RoutingHeuristic):
    """
    Least Distance Heuristic: Select the closest machine for transport.
    """
    def get_action(self, state: 'State'):
        super()._get_routes(state)
        action = sorted(self._routes, key=lambda x: x.distance)[0]
        return action.index


class MDOptimizer(RoutingHeuristic):
    """
    Most Distance Heuristic: Select the furthest machine for transport.
    """

    def get_action(self, state: 'State'):
        super()._get_routes(state)
        action = sorted(self._routes, key=lambda x: x.distance)[-1]
        return action.index


class LQO(RoutingHeuristic):
    """
    Least queued operation: Prefer machines with less queued operations.
    """
    def get_action(self, state: 'State') -> int:
        self._routes = []
        tgt_machines = state.legal_actions
        buffer_lengths = state.trackers.buffer_lengths[
            np.array(tgt_machines) - 1]
        ret = np.argmin(buffer_lengths)
        if len(ret.shape) == 0:
            return tgt_machines[int(ret)]
        else:
            return tgt_machines[ret[0]]


class MBD(RoutingHeuristic):
    """
    NOTE: Bad naming! Should be MQO!

    Most queued operations: Prefer machines with less queued operations.
    """
    def get_action(self, state: 'State') -> int:
        self._routes = []
        tgt_machines = state.legal_actions
        src_machine = state.current_machine_nr
        Route = namedtuple('Route', 'index distance')
        buffer_lengths = state.trackers.buffer_lengths[
            np.array(tgt_machines) - 1]
        ret = np.argmax(buffer_lengths)
        if len(ret.shape) == 0:
            return tgt_machines[int(ret)]
        else:
            return tgt_machines[ret[0]]


class LQT(RoutingHeuristic):
    """
    Least busy discrete heuristic: Prefer machines with less queued operations.
    """
    def get_action(self, state: 'State') -> int:
        self._routes = []
        tgt_machines = state.legal_actions
        src_machine = state.current_machine_nr
        Route = namedtuple('Route', 'index distance')
        buffer_times = state.trackers.buffer_times[
            np.array(tgt_machines) - 1]
        ret = np.argmin(buffer_times)
        if len(ret.shape) == 0:
            return tgt_machines[int(ret)]
        else:
            return tgt_machines[ret[0]]


class MQT(RoutingHeuristic):
    """
    Least busy heuristic: Prefer machines with less queued operations.
    """
    def get_action(self, state: 'State') -> int:
        self._routes = []
        tgt_machines = state.legal_actions
        src_machine = state.current_machine_nr
        Route = namedtuple('Route', 'index distance')
        buffer_times = state.trackers.buffer_times[
            np.array(tgt_machines) - 1]
        ret = np.argmax(buffer_times)
        if len(ret.shape) == 0:
            return tgt_machines[int(ret)]
        else:
            return tgt_machines[ret[0]]
