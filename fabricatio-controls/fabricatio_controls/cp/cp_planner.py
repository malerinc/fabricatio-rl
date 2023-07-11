import collections
import numpy as np
import heapq as h
from typing import Union, List, Dict, Tuple

from fabricatio_rl.core_state import State

from ortools.sat.python import cp_model


class CPOperation:
    """
    Represents the state information required by the CP solver in a compact
    fashion.
    """
    def __init__(self, tgt_m_group: int, duration: int, status: int,
                 location: int, index: int):
        self.m_group = tgt_m_group
        self.duration = duration
        self.status = status
        self.location = location
        self.idx = index

    def __str__(self):
        return (f'OP{self.idx}:type{self.m_group}-loc{self.location}-' 
                f'stat{self.status}-d{self.duration}')


class ModelOperation:
    """
    References the model start and end variables of an operation so as to make
    the retrieval of the aforementioned variable values from the solver more
    convenient.
    """
    def __init__(self, start: Union[cp_model.IntVar, int],
                 end: Union[cp_model.IntVar, int], fixed: bool):
        self.start = start
        self.end = end
        self.fixed = fixed


class AssignedOperation:
    """
    Class used to bundle up solver values and (@see ModelOperation) and state
    values (@see CPOperation). The equality operators are implemented to
    enable heapsort.
    """
    def __init__(self, start, job_idx, op_idx, duration, n_j, n_o):
        self.start = start
        self.job_idx = job_idx
        self.op_idx = op_idx
        self.duration = duration
        self.action = n_o * job_idx + op_idx

    def __lt__(self, other):
        return self.start < other.start

    def __gt__(self, other):
        return self.start > other.start

    def __eq__(self, other):
        return self.start == other.start

    def __ge__(self, other):
        return self > other or self == other

    def __le__(self, other):
        return self < other or self == other


class CPPlanner:
    def __init__(self, state: State, time_limit: int, wip_indices: np.ndarray,
                 save_schedule: str = '', n_ops_per_job: int = -1) -> None:
        # required state info
        self.n_operations = state.params.max_n_operations
        self.n_jobs = state.params.max_jobs_visible
        self.jobs, self.horizon_start, self.horizon_end = self.__init_job_data(
            state, wip_indices, n_ops_per_job)
        self.machine_alternatives = {
            k: list(s) for (k, s) in state.matrices.machine_capab_dt.items()
        }
        # define model
        self.__model = cp_model.CpModel()
        # model variable collections
        self.__model_intervals = collections.defaultdict(list)
        self.model_operations: Dict[Tuple, ModelOperation] = {}
        self.__model_route_choices: Dict[Tuple, cp_model.IntVar] = {}
        self.__add_variables()
        self.__objective = self.__add_objective()
        # constraints
        self.__add_no_overlap_constraints(range(1, state.params.n_machines + 1))
        self.__add_precedence_constraints()
        # define solver and solution properties
        self.__solver = cp_model.CpSolver()
        self.__solver.parameters.max_time_in_seconds = time_limit
        self.__solution_status = None
        self.__objective_value = None
        # solve
        self.__minimize_makespan()
        # get information from the solution
        self.machine_to_assigned_tasks = self.__get_cp_plan()
        self.save_schedule(save_schedule)

    def save_schedule(self, schedule_path):
        if schedule_path != '':
            with open(schedule_path, 'w') as sln_file:
                for m_idx in self.machine_to_assigned_tasks.keys():
                    sln_file.write(
                        ' '.join(
                            [f'({o.job_idx},{o.op_idx},{m_idx},{o.start})'
                             for o in self.machine_to_assigned_tasks[m_idx]]))
                    sln_file.write('\n')

    @staticmethod                                       # full ;)
    def __init_job_data(state: State, wip, next_n_ops=-1) \
            -> Tuple[Dict[int, List[CPOperation]], int, int]:
        if next_n_ops == -1:
            return CPPlanner.__init_job_data_full(state, wip)
        else:
            return CPPlanner.__init_job_data_partial(state, wip, next_n_ops)

    @staticmethod
    def __init_job_data_partial(
            state: State,
            wip: List[int],
            next_n_ops: int) -> Tuple[Dict[int, List[CPOperation]], int, int]:
        jobs = collections.defaultdict(list)
        op_type = state.matrices.op_type
        op_duration = state.matrices.op_duration
        op_status = state.matrices.op_status
        op_location = state.matrices.op_location
        wip_inv = state.job_global_to_view_idx
        job_n_ops_pairs = CPPlanner.__distribute_planning_capa_to_jobs(
            state, wip, next_n_ops)
        op_indices = state.operation_graph.get_next_operations(job_n_ops_pairs)
        horizon_start = state.system_time
        horizon_end = state.system_time
        for (j, i) in op_indices:
            # j_view_idx = wip_inv[j]
            horizon_end += op_duration[j][i]
            jobs[wip_inv[j]].append(CPOperation(
                op_type[j][i], op_duration[j][i],
                op_status[j][i], op_location[j][i], i))
        # for j, j_ops in jobs.items():
        #     try:
        #         assert len(j_ops) == state.trackers.n_remaining_ops[j]
        #     except AssertionError:
        #         print(jobs, state.trackers.n_remaining_ops)
        return jobs, int(horizon_start), int(horizon_end)

    @staticmethod
    def __distribute_planning_capa_to_jobs(state: State, wip: list,
                                           j_planning_capa: int):
        """
        Calculates a fair distribution of planning capacity over jobs. The
        operation planning capacity is first accumulated over jobs with less
        than the oinitial j_planning_capa operations left. Then the excess
        capacity is distributed over all jobs using integer division, with an
        extra capacity of 1, depending on the integer division remainder, for
        the jobs with the least operations remaining.

        :param state: The environment state to read the information from.
        :param wip: The current wip indices.
        :param j_planning_capa: The planning capacity per job.
        :return: The list of tuples mapping job indices to the alotted planning
            capacity.
        """
        n_ops_remaining = state.trackers.job_n_remaining_ops[wip]
        sort_idxs = np.argsort(n_ops_remaining)
        n_jobs = sort_idxs.shape[0]
        capa_vec = np.repeat(j_planning_capa, n_jobs)
        pos = 0
        n_additional = 0
        ops_extra = 0
        start_idx = 0
        for i in sort_idxs:
            if n_ops_remaining[i] == 0:
                start_idx += 1
                ops_extra += j_planning_capa
            elif n_ops_remaining[i] <= j_planning_capa:
                ops_extra += j_planning_capa - n_ops_remaining[i]
                capa_vec[pos] = n_ops_remaining[i]
            elif n_ops_remaining[i] > j_planning_capa:
                n_additional = n_jobs - pos
                if ops_extra == 0:
                    break
                elif ops_extra < n_additional:
                    capa_vec[pos:pos + ops_extra] += 1
                    break
                else:
                    capa_vec[pos:n_additional] += 1
            pos += 1
            ops_extra -= n_additional
        j_to_capa = np.column_stack((np.array(wip)[sort_idxs],
                                     capa_vec))[start_idx:]
        return j_to_capa.tolist()

    @staticmethod
    def __init_job_data_full(
            state: State,
            wip: List[int]) -> Tuple[Dict[int, List[CPOperation]], int, int]:
        # TODO: use global job numbers here? YES MAN, lol :D
        op_type = state.matrices.op_type[wip]
        op_duration = state.matrices.op_duration[wip]
        op_status = state.matrices.op_status[wip]
        op_location = state.matrices.op_location[wip]
        jobs = {}
        horizon_start = state.system_time
        horizon_end = state.system_time
        for j in range(len(wip)):
            jobs[j] = []
            for i in range(op_type.shape[1]):
                op_tgt_m_group = op_type[j][i]
                if op_tgt_m_group == 0:   # op already finished
                    continue
                else:
                    horizon_end += op_duration[j][i]
                    jobs[j].append(CPOperation(
                        op_tgt_m_group, op_duration[j][i],
                        op_status[j][i], op_location[j][i], i))
            if not bool(jobs[j]):
                del jobs[j]
        return jobs, int(horizon_start), int(horizon_end)

    def __add_variables(self):
        # Creates job intervals and add to the corresponding machine lists.
        for job_id, job in self.jobs.items():
            for op in job:
                # Create main interval for the task.
                suffix = f'_j{job_id}_t{op.idx}'
                if op.status == 2:   # op is processing hence the start is fixed
                    start = self.horizon_start
                    fixed = True
                else:
                    start = self.__model.NewIntVar(self.horizon_start,
                                                   self.horizon_end,
                                                   f'start{suffix}')
                    fixed = False
                duration = int(op.duration)
                end = self.__model.NewIntVar(self.horizon_start,
                                             self.horizon_end,
                                             f'end{suffix}')
                interval = self.__model.NewIntervalVar(start, duration, end,
                                                       f'interval{suffix}')
                self.model_operations[job_id, op.idx] = ModelOperation(
                    start=start, end=end, fixed=fixed)
                if len(self.machine_alternatives[op.m_group]) == 1:
                    # only one route choice for the op
                    m = self.machine_alternatives[op.m_group][0]
                    self.__model_intervals[m].append(interval)
                    self.__model_route_choices[
                        (job_id, op.idx, m)] = self.__model.NewConstant(1)
                elif op.location != 0:
                    # op in buffer or processing; route fixed
                    m = op.location
                    self.__model_intervals[m].append(interval)
                    self.__model_route_choices[
                        (job_id, op.idx, m)] = self.__model.NewConstant(1)
                else:
                    self.__add_local_route_choice_variables(
                        job_id, op, start, duration, end)

    def __add_objective(self) -> cp_model.IntVar:
        # Makespan objective.
        return self.__model.NewIntVar(self.horizon_start,
                                      self.horizon_end,
                                      'makespan')

    def __add_local_route_choice_variables(self, job_id, op,
                                           start, duration, end):
        local_route_choices = []
        for alternative_machine in self.machine_alternatives[op.m_group]:
            alt_suffix = f'_j{job_id}_t{op.idx}_a{alternative_machine}'
            machine_choice = self.__model.NewBoolVar('choice' + alt_suffix)
            choice_start = self.__model.NewIntVar(
                self.horizon_start, self.horizon_end, f'start{alt_suffix}')
            choice_duration = int(op.duration)  # TODO: Implement diff proc ts
            choice_end = self.__model.NewIntVar(self.horizon_start,
                                                self.horizon_end,
                                                f'end{alt_suffix}')
            choice_interval = self.__model.NewOptionalIntervalVar(
                choice_start, choice_duration, choice_end, machine_choice,
                f'interval{alt_suffix}')
            local_route_choices.append(machine_choice)
            # Link the master variables with the local ones.
            self.__model.Add(start == choice_start).OnlyEnforceIf(
                machine_choice)
            self.__model.Add(duration == choice_duration).OnlyEnforceIf(
                machine_choice)
            self.__model.Add(end == choice_end).OnlyEnforceIf(machine_choice)
            # Add the local interval to the right machine.
            self.__model_intervals[alternative_machine].append(
                choice_interval)
            # Store the presences for the solution.
            self.__model_route_choices[
                (job_id, op.idx, alternative_machine)] = machine_choice
        # Exactly one machine alternative can be chosen
        self.__model.Add(sum(local_route_choices) == 1)

    def __add_no_overlap_constraints(self, machines: range):
        # Operations executed one after the other without preemption
        for machine in machines:
            self.__model.AddNoOverlap(self.__model_intervals[machine])
            # TODO: Add tooling times
            # TODO: Add transport times
            # TODO: Machine breakdowns

    def __add_precedence_constraints(self):
        # Operations within a job are executed one after the other
        for job_id, job in self.jobs.items():
            for op in job[:-1]:
                self.__model.Add(
                    self.model_operations[job_id, op.idx + 1].start >=
                    self.model_operations[job_id, op.idx].end)
        # TODO: precedence graphs
        # TODO: machine assignment here too

    def __minimize_makespan(self):
        # target_equation
        try:
            self.__model.AddMaxEquality(self.__objective, [
                self.model_operations[job_id, job[-1].idx].end
                for job_id, job in self.jobs.items()
            ])
        except IndexError:
            print(self.model_operations, self.jobs)
        self.__model.Minimize(self.__objective)
        # solve model
        # TODO: split into phases -> routing first
        #  self.__solver.Phase(self.__machine_to_next_actions.)
        self.__solution_status = self.__solver.Solve(self.__model)
        # get objective value
        self.__objective_value = self.__solver.ObjectiveValue()

    def __get_cp_plan(self):
        if self.__solution_status == cp_model.OPTIMAL:
            # Create one list of assigned tasks per machine.
            # print("OPTIMAL!")
            # print(self.__create_assigned_tasks_lists())
            return self.__create_assigned_tasks_lists()
        elif self.__solution_status == cp_model.FEASIBLE:
            print("FEASIBLE!")
            return self.__create_assigned_tasks_lists()
        elif self.__solution_status == cp_model.INFEASIBLE:
            print("The solution status was INFEASIBLE")
        elif self.__solution_status == cp_model.MODEL_INVALID:
            print("The solution status was MODEL_INVALID")
        else:
            print("The solution status was UNKNOWN")

    def __create_assigned_tasks_lists(self):
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in self.jobs.items():
            for op in job:
                selected = self.__get_machine_choice(job_id, op)
                assert selected != 0
                if not self.model_operations[job_id, op.idx].fixed:
                    machine_operation_list = assigned_jobs[selected]
                    h.heappush(machine_operation_list, AssignedOperation(
                        start=self.__solver.Value(
                            self.model_operations[job_id, op.idx].start),
                        job_idx=job_id,
                        op_idx=op.idx,
                        duration=op.duration,
                        n_j=self.n_jobs,
                        n_o=self.n_operations
                    ))
        return assigned_jobs

    def __get_machine_choice(self, job_id: int, op: CPOperation):
        if op.location != 0:  # already at a machine
            return op.location
        elif len(self.machine_alternatives[op.m_group]) > 1:
            # assert that there is only one set alternative !!!
            for alternative in self.machine_alternatives[op.m_group]:
                try:
                    altenative_set = self.__solver.Value(
                        self.__model_route_choices[
                            (job_id, op.idx, alternative)
                        ])
                    if altenative_set:
                        return alternative
                except IndexError:
                    rc = self.__model_route_choices
                    print(rc)
            return -1
        else:
            return self.machine_alternatives[op.m_group][0]

    def remove_plan_operation(self, action, state):
        if type(action[0]) == tuple:
            m_idx = action[1]
            act = action[0]
            if m_idx in self.machine_to_assigned_tasks:
                for i in range(len(self.machine_to_assigned_tasks[m_idx])):
                    if (self.machine_to_assigned_tasks[m_idx][i].action ==
                            action):
                        del self.machine_to_assigned_tasks[m_idx][i]
                        h.heapify(self.machine_to_assigned_tasks[m_idx])
                        break

    def get_solver_status(self):
        return self.__solution_status

    def choose_fixed_plan_action(self, state: State, legal_actions,
                                 nowait=False):
        if state.scheduling_mode == 1:  # routing
            # todo: what happens if a job finished?
            if len(legal_actions) == 1:
                return legal_actions[0]
            next_tasks = state.operation_graph.get_next_ops(state.current_job)
            wip = state.job_global_to_view_idx
            action = None
            for j_idx, op_idx in next_tasks:
                task = CPOperation(
                    state.matrices.op_type[j_idx][op_idx],
                    state.matrices.op_duration[j_idx][op_idx],
                    state.matrices.op_status[j_idx][op_idx],
                    state.matrices.op_location[j_idx][op_idx],
                    op_idx)
                try:
                    action = self.__get_machine_choice(wip[j_idx], task)
                except KeyError:
                    print('herehere!')
                if action in legal_actions:
                    # return a machine index i.e. m_nr - 1
                    return action
            return action
        else:
            m_idx = state.current_machine_nr
            n_ops = state.params.max_n_operations
            n_jobs = state.params.max_jobs_visible
            wait = n_jobs * n_ops
            if not any(self.machine_to_assigned_tasks[m_idx]):
                return legal_actions[0]
            action = self.machine_to_assigned_tasks[m_idx][0].action
            if action in legal_actions:
                h.heappop(self.machine_to_assigned_tasks[m_idx])
                return action
            if wait in legal_actions and not nowait:
                return wait
            else:
                return legal_actions[0]
        # todo: machine failures

    # def choose_replanning_action(self, state, legal_actions):
    #     self.__init__(state=state, time_limit=self.__time_limit)
    #     return self.choose_fixed_plan_action(state, legal_actions)

    def get_score(self):
        return self.__objective_value
