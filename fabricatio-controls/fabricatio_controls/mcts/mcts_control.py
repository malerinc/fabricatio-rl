# This is a very simple implementation of the UCT Monte Carlo Tree Search
#
# The basic MCTS algorithm is adapted from the code authored by Ed Powley and
# Daniel Whitehouse (University of York, UK) and published September 2012.
# The original code is available here: http://mcts.ai/code/python.html


import numpy as np
from copy import deepcopy

from fabricatio_controls import Control
from fabricatio_controls.heuristics import HeuristicControl, RandomControl
from fabricatio_controls.heuristics import SequencingHeuristic

from fabricatio_rl.interface import FabricatioRL
from fabricatio_rl.core_state import State

from fabricatio_controls.mcts.mcts_uct_node import Node, MCTSHelpers


class MCTS:
    @staticmethod
    def extract_utilization(state: State):
        utl_total = state.trackers.utilization_times.sum()
        return utl_total / state.system_time

    @staticmethod
    def extract_throughput(state: State):
        overal_throughput = state.trackers.job_completed_time.sum()
        return overal_throughput / state.system_time

    def extract_makespan(self, state: State):
        dt = state.system_time - self.t_decision
        assert dt > 0
        if self.normed:
            makespan_normed = dt / self.normalization_factor
            return 1 - makespan_normed
        else:
            return - state.system_time + self.t_decision

    def __init__(self, env: FabricatioRL = None, itermax: int = 2,
                 heuristic: SequencingHeuristic = None,
                 verbose: bool = False, criterium: str = 'makespan',
                 model=None, normed=False, uctk=2):
        # TODO: YOU SHOULDN'T FUCKING DO THIS!!!! In doing so you fuck up the
        #  internal simulation RNG!!!
        # np.random.seed(100)
        self.env = env
        self.igdrasil = None
        self.rootnode = None
        self.itermax = itermax
        self.heuristic = heuristic
        self.verbose = verbose
        self.n_jobs_completed = 0
        self.autoplay = True
        self.model_v = model
        self.utck = uctk
        self.normed = normed
        self.n_predictions = 0
        self.n_state_reads = 0
        self.use_model = True
        if criterium == 'tardiness':
            self.criterium = lambda s: s.trackers.tardiness.mean()
        elif criterium == 'throughput':
            self.criterium = (lambda s: MCTS.extract_throughput(s))
        elif criterium == 'utilization':
            self.criterium = (lambda s: MCTS.extract_utilization(s))
        elif criterium == 'makespan':
            self.normalization_factor = 1
            self.t_decision = 0
            self.criterium = (lambda s: self.extract_makespan(s))
        else:
            raise ValueError("Only 'makespan', 'tardiness', 'utilization' and "
                             "'throughput' are currently accepted criteria.")

    def mcts_select(self, simulation: FabricatioRL) -> Node:
        # Select
        # node is fully expanded (no untried moves) and
        # non-terminal (there are child nodes)
        # TODO: try out different expansion strategies [NO PRIO]
        node_src = self.rootnode
        selection_chain = ''
        # test_node_env_consistency(
        #     node_src, simulation.core.state, self.rootnode)
        while node_src.untried_moves == [] and node_src.child_nodes != []:
            node = node_src.uct_select_child()
            selection_chain += f'{node.move}; '
            simulation.step(node.move)
            if node.node_id != MCTSHelpers.get_id(simulation.core.state):
                # patch the node ;)
                # try:
                node.patch(simulation)
                # except EndgamePatchError:
                #     print('Sim should have ended. what up?')
            node_src = node
            # test_node_env_consistency(node_src, simulation.core.state)
        return node_src

    def mcts_expand(self, node, simulation: FabricatioRL):
        # if we can expand (i.e. state/node is non-terminal)
        if node.untried_moves and not simulation.core.simulation_has_ended():
            # test_node_env_consistency(node, simulation.core.state)
            core_actions = simulation.core.state.legal_actions
            if self.heuristic \
                    and simulation.optimizer_configuration in {0, 1, 2} \
                    and len(node.untried_moves) == len(core_actions):
                # expand with SPT if this is the first child we expand!
                m = self.heuristic.get_action(simulation.core.state)
            else:
                # otherwise pick at random
                m = int(np.random.choice(node.untried_moves))
            # try:
            # stepped_sim = deepcopy(simulation)
            simulation.step(m)
            # except AttributeError:
            #     node = None
            # except IndexError:
            #     node = None
            leaf = node.add_child(simulation, m, self.utck)
        else:
            leaf = node
        return leaf

    def mcts_rollout(self, stepped_sim, node):
        vs = []
        if not stepped_sim.core.has_ended():
            self.n_predictions += 1
            if self.model_v is not None and self.use_model:
                features = []
                state = stepped_sim.core.state
                transformer = stepped_sim.get_state_transformer()
                fs = transformer(state)
                assert np.isfinite(fs).all()
                s_rel_norm_makespan = self.model_v.predict(
                    np.array([fs]))
                if not self.normed:
                    return self.t_decision - (s_rel_norm_makespan
                                              * self.normalization_factor)
                else:
                    return 1 - s_rel_norm_makespan
            elif self.heuristic:
                end_state = HeuristicControl(self.heuristic).play_game(
                    stepped_sim)
                return self.criterium(end_state)
            else:
                end_state = RandomControl().play_game(stepped_sim)
                return self.criterium(end_state)
        else:
            # if self.model_v is not None:
            #     return None, True
            assert not stepped_sim.core.state.legal_actions
            end_state = stepped_sim.core.state
            self.n_state_reads += 1
            if self.model_v is not None:
                # if node.parentNode.value != 0:
                #     pv = node.parentNode.value / node.parentNode.visits
                #     multiplier = (1 - self.criterium(end_state)) / pv
                #     if multiplier <= 2:
                #         self.neural_correction = multiplier
                #         self.rootnode.multiply_branch(self.neural_correction)
                #         print(self.neural_correction)
                return self.criterium(end_state)
            return self.criterium(end_state)

    @staticmethod
    def mcts_backpropagate(node, v):
        while node is not None:
            # state is terminal. Update node with result
            node.update(v)
            # val_scale = val_scale * 0.90 if val_scale > 1 else 1
            node = node.parentNode

    def get_action(self, state: State):
        """
        Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        with game results in the range [0.0, 1.0].
        """
        # go through the 4 mcts stages self.itermax times and update the tree
        # test = None
        for i in range(self.itermax):
            # test_node_env_consistency(self.rootnode, state)
            # test_node_env_consistency(self.rootnode, self.env.core.state)
            # if test is None:
            #     test = deepcopy(self.env)
            simulation = deepcopy(self.env)
            # test_env_consistency(simulation, test)
            simulation.make_deterministic()
            # test_env_consistency(simulation, self.env)
            # Select node for expansion
            node = self.mcts_select(simulation)
            # test_node_env_consistency(self.rootnode, state)
            # test_node_env_consistency(self.rootnode, self.env.core.state)
            # Expand
            node = self.mcts_expand(node, simulation)
            # test_node_env_consistency(node, simulation.core.state)
            # test_node_env_consistency(self.rootnode, state)
            # test_node_env_consistency(self.rootnode, self.env.core.state)
            # Rollout
            vs = self.mcts_rollout(simulation, node)
            # test_node_env_consistency(self.rootnode, state)
            # test_node_env_consistency(self.rootnode, self.env.core.state)
            # Backpropagate
            MCTS.mcts_backpropagate(node, vs)
            # test_node_env_consistency(node, simulation.core.state)
        if self.verbose:
            print(self.rootnode.tree_to_string(0))
        # return the move that was most visited
        sorted_moves = list(sorted(
            self.rootnode.child_nodes, key=lambda c: c.value/c.visits))
        selected_move = sorted_moves[-1].move
        return selected_move

    def __update_tree(self, action, state):
        found_move = False
        for child in self.rootnode.child_nodes:
            if action == child.move:
                self.rootnode = child
                self.rootnode.value = 0
                self.rootnode.parentNode = None
                # self.rootnode.move = None
                found_move = True
                break
        if not found_move:
            # this happens if the root was never expanded or a parallel
            # optimizer took a child action that was not expanded yet;
            # also happens a lot because of the env skipping decisions ;)
            self.__cut_tree()

    def __cut_tree(self):
        self.rootnode = Node(self.env, uctk=self.utck)
        self.rootnode.parentNode = None

    def update(self, action: int, state: State):
        # test_node_env_consistency(self.rootnode, self.env.core.state)
        # self.env.step(action)
        # self.__cut_tree()
        if self.n_jobs_completed < state.trackers.n_completed_jobs:
            # new job completed, so we reinitialize the schedule ;)
            self.__cut_tree()
            self.n_jobs_completed = state.trackers.n_completed_jobs
        else:
            self.__update_tree(action, state)
        if self.verbose:
            print(f'Updated Tree on action: {action}')
            print(self.rootnode.tree_to_string(0))
        state_moves = MCTSHelpers.get_moves(self.env)
        if self.rootnode.node_id != MCTSHelpers.get_id(state):
            # patch the node ;)
            # print('ROOTNODE patch...')
            self.rootnode.patch(self.env)
        # test_node_env_consistency(self.rootnode, self.env.core.state)
        # test_node_env_consistency(self.rootnode, state)

    def initialize_tree(self, simulation: FabricatioRL):
        rt = Node(simulation, uctk=self.utck)
        self.rootnode = rt
        self.igdrasil = rt


class MCTSControl(Control):
    def __init__(self, optimizer: MCTS, transformer=None, heuristics=None,
                 autoplay=True, name=''):
        super().__init__(name)
        self.optimizer = optimizer
        if not name:
            neural_string = '_NN' if transformer is not None else ""
            h_string = (
                f'_h' 
                f'{"-".join([h.__class__.__name__ for h in heuristics])}'
                if heuristics is not None else ""
            )
            self.name = f'MCTS{self.optimizer.itermax}{neural_string}{h_string}'
        else:
            self.name = name
        self.autoplay = autoplay
        self.transformer = transformer
        self.heuristics = heuristics if heuristics is not None else []
        self.trace = []

    def play_game(self, env: FabricatioRL, initial_state=None) -> State:
        # configure env
        env.set_transformer(self.transformer)
        env.set_optimizers(self.heuristics)
        env.set_core_seq_autoplay(self.autoplay)
        env.set_core_rou_autoplay(True)
        # set mcts env and initialize tree
        self.optimizer.env = deepcopy(env)
        self.optimizer.initialize_tree(env)
        done, state, act = False, None, None
        move_trace = []
        while not done:
            # TODO: use this and step with core ;)
            # la_run_env = deepcopy(env.core.state.legal_actions)
            assert ((not env.core.wait_legal()) or
                    len(env.core.state.legal_actions) >= 2)
            #     # wait is always the last action
            #     test_env_consistency(self.optimizer.env, env)
            #     act = env.core.state.legal_actions[0]
            #     # test_env_consistency(self.optimizer.env, env)
            # else:
            # test_env_consistency(self.optimizer.env, env)
            if act is not None:
                self.optimizer.update(act, env.core.state)
            self.optimizer.t_decision = env.core.state.system_time
            wip = env.core.state.job_view_to_global_idx
            norm = env.core.state.matrices.op_duration[wip].sum()
            ops_completed = env.core.state.trackers.job_n_remaining_ops[wip].sum()
            ops_remaining = env.core.state.trackers.job_n_completed_ops[wip].sum()
            completion = ops_completed / (ops_completed + ops_remaining)
            self.optimizer.use_model = completion < 0.7
            self.optimizer.normalization_factor = norm
            act = self.optimizer.get_action(self.optimizer.env.core.state)
            # print(len(move_trace), env.core.state.scheduling_mode)
            # test_env_consistency(self.optimizer.env, env)
            if env.optimizer_configuration in {0, 1}:
                assert act in env.core.state.legal_actions
            # assert act in self.optimizer.env.core.state.legal_actions
            # assert la_run_env == env.core.state.legal_actions
            move_trace.append(act)
            state, _, done, _ = env.step(act)
            self.optimizer.env = deepcopy(env)
        print(f"Predict to read ratio: "
              f"{self.optimizer.n_predictions}/{self.optimizer.n_state_reads}")
        return env.core.state


def test_env_consistency(env_1, env_2):
    state_o_env = env_2.core.state
    state_i_env = env_1.core.state
    i_env_ot: np.ndarray = state_i_env.matrices.op_type
    o_env_ot: np.ndarray = state_o_env.matrices.op_type

    i_env_ol: np.ndarray = state_i_env.matrices.op_location
    o_env_ol: np.ndarray = state_o_env.matrices.op_location

    i_env_od: np.ndarray = state_i_env.matrices.op_duration
    o_env_od: np.ndarray = state_o_env.matrices.op_duration
    m_o_env = state_o_env.current_machine_nr
    m_i_env = state_i_env.current_machine_nr
    j_o_env = state_o_env.current_job
    j_i_env = state_i_env.current_job
    op_i_env = state_i_env.current_operation
    op_o_env = state_o_env.current_operation
    assert np.array_equal(i_env_ot, o_env_ot)
    assert np.array_equal(state_o_env.legal_actions, state_i_env.legal_actions)
    assert np.array_equal(i_env_od, o_env_od)
    assert np.array_equal(i_env_ol, o_env_ol)
    assert m_o_env == m_i_env
    assert j_i_env == j_o_env
    assert op_o_env == op_i_env


def test_node_env_consistency(node: Node, env_state: State, other_node=None):
    try:
        assert node.node_id == MCTSHelpers.get_id(env_state)
    except AssertionError:
        if other_node is not None:
            for on in other_node:
                print(on.tree_to_string(0))
        print(node.tree_to_string(0))
        print(env_state.legal_actions)
        raise AssertionError
