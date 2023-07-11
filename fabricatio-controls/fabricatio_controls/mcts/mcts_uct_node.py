# from __future__ import annotations
# to use the Node type inside the Node class

from fabricatio_rl.interface import FabricatioRL
from fabricatio_rl.core_state import State

from math import sqrt, log
from copy import deepcopy
from typing import Union

import array
import hashlib
import numpy as np


class MCTSHelpers:
    @staticmethod
    def get_moves(simulation: FabricatioRL):
        state_core = simulation.core.state
        legal_actions = simulation.get_legal_actions()
        if legal_actions:
            wait_sig = (state_core.params.max_jobs_visible *
                        state_core.params.max_n_operations)
            if legal_actions[-1] == wait_sig:
                return legal_actions[:-1]
            else:
                return legal_actions
        else:
            return []

    @staticmethod
    def get_id(state: State):
        la = deepcopy(state.legal_actions)
        id_info = list(sorted(la))
        return MCTSHelpers.get_digest(id_info)

    @staticmethod
    def get_digest(legal_actions):
        return hashlib.md5(array.array('h', legal_actions)).hexdigest()


class EndgamePatchError(Exception):
    def __init__(self, node):
        super().__init__(f"Endgame patch on node {repr(node)}")


class Node:
    """
    A node in the game tree.
    """
    def __init__(self, simulation: FabricatioRL, move: Union[int, None] = None,
                 parent=None, uctk: float = 2):
        # the move that got us to this node - "None" for the root node
        self.move = move
        moves = MCTSHelpers.get_moves(simulation)
        self.moves = moves
        self.node_id = MCTSHelpers.get_id(simulation.core.state)
        self.parentNode = parent  # "None" for the root node
        self.child_nodes = []
        self.value = 0
        self.visits = np.finfo(float).eps
        self.untried_moves = deepcopy(moves)  # future child nodes
        self.uctk = uctk

    def multiply_branch(self, value):
        self.value *= value
        for child in self.child_nodes:
            child.multiply_branch(value)

    def patch(self, simulation: FabricatioRL):
        moves = MCTSHelpers.get_moves(simulation)
        # print(f"Patching Node with moves: {self.moves} "
        #       f"--> {moves}...")
        # if not moves:
        #     raise EndgamePatchError(self)
        self.moves = moves
        self.node_id = MCTSHelpers.get_id(simulation.core.state)
        self.child_nodes = []
        self.value = 0
        self.visits = np.finfo(float).eps
        self.untried_moves = deepcopy(moves)  # future child nodes
        self.uctk = self.uctk  # 2

    def uct_select_child(self):
        """
        Use the UCB1 formula to select a child node. Often a constant UCTK is
        applied so we have
        lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits
        to vary the amount of exploration versus exploitation.
        """
        s = sorted(self.child_nodes,
                   key=lambda c: c.value / c.visits + self.uctk * sqrt(
                       2 * log(self.visits) / c.visits))[-1]
        # print(s)
        return s

    def add_child(self, simulation: FabricatioRL, m: int, uctk: float):
        """
        Remove m from untriedMoves and add a new child node for this move.
        Return the added child node
        """
        n = Node(simulation, move=int(m), parent=self, uctk=uctk)
        self.untried_moves.remove(m)
        self.child_nodes.append(n)
        return n

    def update(self, result, no_impact_update=False):
        """
        Update this node - one additional visit and result additional wins.
        result must be from the viewpoint of playerJustmoved.
        """
        if no_impact_update:
            self.value = (self.value * (self.visits + 1)) / self.visits
            self.visits += 1
        else:
            self.visits += 1
            self.value += result

    def __repr__(self):
        return f"[Move:{self.move}; " \
               f"Value/Visits:{self.value}/{self.visits}; " \
               f"Untried Moves:{self.untried_moves}]"

    def tree_to_string(self, indent):
        s = Node.indent_string(indent) + str(self)
        for c in self.child_nodes:
            s += c.tree_to_string(indent + 1)
        return s

    @staticmethod
    def indent_string(indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def children_to_string(self):
        s = ""
        for c in self.child_nodes:
            s += str(c) + "\n"
        return s
