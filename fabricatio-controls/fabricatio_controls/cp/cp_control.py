import json
import os
from typing import Union, Tuple, TypeVar
from copy import deepcopy

import pandas as pd
import numpy as np

from fabricatio_controls import Control
from fabricatio_rl.core_state import State
from fabricatio_rl.interface_templates import Optimizer
from fabricatio_rl.interface import FabricatioRL

from fabricatio_controls.cp.cp_planner import CPPlanner

T = TypeVar('T')


class FixedCPPlanSequencing(Optimizer):
    def __init__(self, time_limit: int, schedule_path: str = ''):
        # TODO: create CPTransport optimizer that shares a schedule with the
        #  sequencing optimizer
        super().__init__('sequencing')
        self.time_limit = time_limit
        self.planner = None
        self.n_new_jobs = -1
        self.schedule_path = schedule_path

    def get_action(self, state: State) -> (int, int):
        if self.planner is None:
            self.planner = CPPlanner(state, time_limit=self.time_limit,
                                     save_schedule=self.schedule_path,
                                     wip_indices=np.arange(state.params.n_jobs))
            self.n_new_jobs = len(state.system_job_queue)
        assert self.n_new_jobs == 0  # only offline problems for now.
        legal_actions = state.legal_actions
        a = self.planner.choose_fixed_plan_action(state, legal_actions)
        return a

    def update(self, action: Union[int, Tuple], state: State):
        self.planner.remove_plan_operation(action, state)


class ReplanningCPSequencing(Optimizer):
    def __init__(self, time_limit: int,
                 n_ops_per_job: int = -1, schedule_path: str = ''):
        # TODO: create CPTransport optimizer that shares a schedule with the
        #  sequencing optimizer
        super().__init__('sequencing')
        self.planned_wip: Union[np.ndarray, None] = None
        self.time_limit = time_limit
        self.planner: Union[CPPlanner, None] = None
        self.schedule_path = schedule_path
        self.n_ops_per_job = n_ops_per_job
        self.get_action_call = 0

    def __reinitialize_plan(self, state: State, wip_indices: np.ndarray):
        self.planner = CPPlanner(state, time_limit=self.time_limit,
                                 save_schedule=self.schedule_path,
                                 wip_indices=wip_indices,
                                 n_ops_per_job=self.n_ops_per_job)

    def print_ndecisions(self):
        # if self.get_action_call != 0:
        #     CURSOR_UP_ONE = '\x1b[1A'
        #     ERASE_LINE = '\x1b[2K'
        #     print(CURSOR_UP_ONE + ERASE_LINE)
        print(self.get_action_call)
        self.get_action_call += 1

    def get_action(self, s: State) -> (int, int):
        wip_indices = s.job_view_to_global_idx
        # any of the job downstream ops
        m = s.current_machine_nr
        # self.print_ndecisions()
        if self.planner is None:
            self.__reinitialize_plan(s, wip_indices)
            self.planned_wip = np.array(wip_indices)
        if not (self.planned_wip == np.array(wip_indices)).all():
            # new job completed, so we reinitialize the schedule
            self.__reinitialize_plan(s, wip_indices)
            self.planned_wip = np.array(wip_indices)
        legal_actions = s.legal_actions
        if s.in_postbuffer_routing_mode():
            next_task = next(
                iter(s.operation_graph.get_next_ops(s.current_job)))
            if next_task not in self.planner.model_operations:
                # routing happens before sequencing; if the follow
                # up ops are not there reinit
                self.__reinitialize_plan(s, wip_indices)
        if self.n_ops_per_job != -1 and s.in_scheduling_mode():
            # this is done to avoid waiting ;)
            try:
                action = self.planner.machine_to_assigned_tasks[m][0].action
                if action not in legal_actions:
                    self.__reinitialize_plan(s, wip_indices)
            except IndexError:
                self.__reinitialize_plan(s, wip_indices)
        a = self.planner.choose_fixed_plan_action(s, legal_actions)
        assert a in legal_actions
        return a

    def update(self, action: Union[int, Tuple], state: State):
        if self.planner is not None:
            self.planner.remove_plan_operation(action, state)


class CPControl(Control):
    def __init__(self, optimizer: Union[FixedCPPlanSequencing,
                                        ReplanningCPSequencing],
                 name: str = 'CP', logging: bool = False):
        super().__init__(name)
        self.optimizer = optimizer
        self.logging = logging

    def play_game(self, env: FabricatioRL, initial_state: T = None) -> State:
        done, state = False, None
        env_c = deepcopy(env)
        env_c.set_core_seq_autoplay(False)
        env_c.set_core_rou_autoplay(False)
        step_nr = 0
        while not done:
            act = self.optimizer.get_action(env_c.core.state)
            state, done = env_c.core.step(act)
            if self.logging:
                self.__log_plan(env_c.core.logger.logdir,
                                env_c.core.state.job_view_to_global_idx,
                                step_nr)
            step_nr += 1
        self.optimizer.planner = None
        self.optimizer.planned_wip = 0
        return state

    def __log_plan(self, logfile_path: str, wip: np.ndarray, state_id: int):
        schedule = []
        plan = self.optimizer.planner.machine_to_assigned_tasks
        for machine_number, task_list in plan.items():
            for task in task_list:
                schedule.append((
                    f'M{machine_number}',
                    task.start,
                    task.start + task.duration,
                    wip[task.job_idx])
                )
        try:
            schedule_df = pd.DataFrame(schedule).sort_values(0, axis=0)
        except KeyError:  # empty dataframe ;)
            return
        filename = f'{logfile_path}/{state_id}.json'
        # while not os.path.isfile(filename):
        #     time.sleep(0.1)
        if os.path.isfile(filename):
            with open(filename) as f:
                data = json.load(f)
                data['current_plan'] = schedule_df.to_dict(orient="records")
            json.dump(data, open(filename, 'w'))
