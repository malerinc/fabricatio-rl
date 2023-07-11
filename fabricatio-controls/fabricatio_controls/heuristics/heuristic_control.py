from copy import deepcopy
from typing import Union, TYPE_CHECKING

from fabricatio_controls import Control

if TYPE_CHECKING:
    from simple_routing import RoutingHeuristic
    from simple_sequencing import SequencingHeuristic
    from fabricatio_rl.interface import FabricatioRL
    from fabricatio_rl.core_state import State


class HeuristicControl(Control):
    def __init__(self, heuristic_sequencing: 'SequencingHeuristic',
                 heuristic_routing: Union['RoutingHeuristic', None] = None,
                 p_completion=1.0):
        self.name = f"{type(heuristic_sequencing).__name__}_" \
                    f"{type(heuristic_routing).__name__}"
        super().__init__(self.name)
        self.h_sequencing = heuristic_sequencing
        self.h_routing = heuristic_routing
        self.p_completion = p_completion
        self.full_run = True

    def play_game(self, environ: 'FabricatioRL', initial_state=None) -> 'State':
        env_c = deepcopy(environ)
        env_c.set_core_seq_autoplay(True)
        env_c.set_core_rou_autoplay(True)
        done, state = False, None
        # assert environ.deterministic_env and env_c.deterministic_env
        # TODO: completion should only consider WIP jobs;)
        p_completion_start = (
                env_c.core.state.trackers.job_completed_time.sum()
                / env_c.core.state.trackers.work_original_jobs.sum())
        p_completed = 0
        while not done and p_completed < self.p_completion:
            if env_c.core.state.scheduling_mode == 0:
                act = self.h_sequencing.get_action(env_c.core.state)
            else:  # environ.core.state.scheduling_mode == 1
                if len(env_c.core.state.legal_actions) == 1:
                    act = env_c.core.state.legal_actions[0]
                else:
                    act = self.h_routing.get_action(env_c.core.state)
            state, done = env_c.core.step(act)
            p_completed = (env_c.core.state.trackers.job_completed_time.sum()
                           / env_c.core.state.trackers.work_original_jobs.sum()
                           - p_completion_start)
        # TODO: move this to tests!
        # if environ.core.is_deterministic():
        #     wip = environ.core.state.job_view_to_global_idx
        #     upper_bound = environ.core.state.matrices.op_duration[wip].sum()
        #     makespan = env_c.core.state.system_time
        #     time_s = environ.core.state.system_time
        #     assert upper_bound >= (makespan - time_s)
        return state
