import numpy as np
from copy import deepcopy

from fabricatio_rl import FabricatioRL
from fabricatio_rl.core_state import State
from fabricatio_rl.interface_templates import (ReturnTransformer, Optimizer)

from fabricatio_controls import Control
from fabricatio_controls.comparison_utils.comparison_setup import (
    parallelize_heterogeneously)
from fabricatio_controls.heuristics import HeuristicControl


# DEFINE RETURN TRANSFORMATION OBJECT
class ModeTime(ReturnTransformer):
    def transform_state(self, state: State) -> np.ndarray:
        return np.array([state.scheduling_mode])

    def transform_reward(self, state: State, illegal=False, environment=None):
        return state.system_time
        # return state.trackers.tardiness.mean()


class SimulationSearch(Optimizer):
    """
    An implementation of the Optimizer interface using SimulationSearch to
    select an action based on the current state. The basic idea of simulation
    search is to use several heuristics to roll out a simulation copy starting
    at the current step. The simulation copy should only contain the
    deterministic information available. The heuristic leading to the best
    rollout results is then chosen as an action.

    Note that this class is a "universal" optimizer, meaning that it can be used
    for both sequencing and transport decisions.
    """
    def __init__(self, environ, seq_optimizers, tra_optimizers,
                 criterium='throughput', p_completion=0.1, n_threads=1):
        """
        Initializes the simulation search optimizer. Note that in order for this
        optimizer method to work, an indirect action environment corresponding
        to the one the optimizer will be embedded in needs to be passed as a
        parameter. The heuristics parameters need to mirror the environment
        copy heuristic optimizers.

        :param environ: The environment copy to use for search.
        :param seq_optimizers: The sequencing optimizers to use for search.
        :param tra_optimizers:  The transport optimizers to use for search.
        :param criterium: The optimizer evaluation criteria.
        :param p_completion: The percentage of total operation completion time
            simulate using the heuristics before evaluating them.
        """
        self.env = deepcopy(environ)
        self.h_controls = []
        self.seq_optimizers = seq_optimizers
        self.rou_optimizers = tra_optimizers
        self.n_threads = n_threads
        self.last_winner = None
        assert n_threads >= 1
        if criterium == 'makespan':
            self.criterium = lambda s: s.system_time
            self.cmp = min
        elif criterium == 'tardiness':
            self.criterium = lambda s: s.trackers.tardiness.mean()
            self.cmp = min
        elif criterium == 'throughput':
            self.criterium = (lambda s: s.trackers.job_completed_time.sum() /
                              s.system_time)
            self.cmp = max
        else:
            raise ValueError("Only 'tardiness', 'throughput' and 'makespan' "
                             "are currently accepted criteria.")
        for seq_opt in seq_optimizers:
            if tra_optimizers is not None:
                for tra_opt in tra_optimizers:
                    self.h_controls.append(
                        HeuristicControl(seq_opt, tra_opt, p_completion))
            else:
                self.h_controls.append(
                    HeuristicControl(seq_opt, None, p_completion))
        super().__init__('universal')

    def prep_parallel_exec(self, env):
        n_controls = len(self.h_controls)
        offset = min(self.n_threads, n_controls)
        start, end = 0, offset
        end_states = []
        while end != n_controls:
            end = min(start + offset, n_controls)
            end_states += parallelize_heterogeneously(
                fns=[h.play_game for h in self.h_controls[start:end]],
                args=[(deepcopy(env),) for _ in range(start, end)],
            )
            start = end

        results = []
        for state in end_states:
            results.append(self.criterium(state))
        return results

    def get_action(self, state: State) -> int:
        if state.scheduling_mode == 1 and len(self.rou_optimizers) == 1:
            return self.rou_optimizers[0].get_action(state)
        sim_c = self.__get_deterministic_copy()
        # t_s = time()
        winner = self.__get_winner(sim_c)
        if state.scheduling_mode == 0:
            act = winner.h_sequencing.get_action(state)
        else:  # environ.core.state.scheduling_mode == 1
            if winner.h_routing is None:  # purely sequencing game
                act = state.legal_actions[0]
            else:
                act = winner.h_routing.get_action(state)
        return act

    def __get_deterministic_copy(self):
        deterministic_copy = deepcopy(self.env)
        # deterministic_copy.set_optimizers(self.h_rou_optimizers)
        deterministic_copy.set_transformer(None)
        deterministic_copy.set_core_rou_autoplay(True)
        deterministic_copy.set_core_seq_autoplay(True)
        deterministic_copy.make_deterministic()
        return deterministic_copy

    def __get_winner(self, sim: FabricatioRL):
        """
        Runs all the parameter simulation with all the controls in the
        h_controls field and selects the one yielding the best result as the
        winner.

        The winning heuristic is additionally saved, such that actions can be
        replayed in the future.

        :param sim: The simulation to compare heuristics on.
        :return: The best heuristic.
        """
        if self.n_threads == 1:
            results = []
            for h in self.h_controls:
                results.append(self.criterium(h.play_game(sim)))
        else:
            results = self.prep_parallel_exec(sim)
        # print('Time sequential: {t_s}')
        result_best = self.cmp(results)
        winner = self.h_controls[results.index(result_best)]
        self.last_winner = winner
        return winner

    def update(self, action: int, state: State):
        self.env.core.step(action)

    def replay_action(self):
        s = self.env.core.state
        if self.last_winner is None:
            sim_c = self.__get_deterministic_copy()
            _ = self.__get_winner(sim_c)
        if s.scheduling_mode == 0:
            act = self.last_winner.h_sequencing.get_action(s)
        else:  # environ.core.state.scheduling_mode == 1
            if self.last_winner.h_routing is None:  # purely sequencing game
                act = s.legal_actions[0]
            else:
                act = self.last_winner.h_routing.get_action(s)
        return act


class SimulationSearchControl(Control):
    def __init__(self, optimizer: SimulationSearch, autoplay=True, n_steps=1):
        seq_opt_name = '_'.join([o.name for o in optimizer.h_controls])
        super().__init__(f"sim_search_{seq_opt_name}")
        self.optimizer = optimizer
        self.autoplay = autoplay
        self.sim_at_steps = n_steps

    def play_game(self, environ, initial_state=None):
        self.optimizer.env = deepcopy(environ)
        self.optimizer.env.set_core_rou_autoplay(True)
        done, state = False, None
        n_steps = 0
        while not done:
            if n_steps % self.sim_at_steps == 0:
                # print(n_steps)
                state = self.optimizer.env.core.state
                legal_actions = state.legal_actions
                wait_legal = self.optimizer.env.core.wait_legal()
                n_legal_actions = len(legal_actions)
                if self.autoplay:
                    if n_legal_actions == 1:
                        act = legal_actions[0]
                    elif wait_legal and n_legal_actions == 2:
                        # wait is always the last action
                        act = legal_actions[0]
                    else:
                        act = self.optimizer.get_action(state)
                else:
                    act = self.optimizer.get_action(state)
            else:
                act = self.optimizer.replay_action()
            state, done = self.optimizer.env.core.step(act)
            # self.optimizer.update(act, self.optimizer.env.core.state)
            n_steps += 1
        return state
