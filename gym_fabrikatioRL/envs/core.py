from gym_fabrikatioRL.envs.interface_input import Input
from gym_fabrikatioRL.envs.core_management import SimulationManager
from gym_fabrikatioRL.envs.core_state import State
from gym_fabrikatioRL.envs.env_utils import UndefinedSimulationModeError
from gym_fabrikatioRL.envs.core_logger import SchedulingLogger

import numpy as np


class Core:
    def __init__(self, parameters: Input, system_time=0):
        self.__end = False
        self.parameters = parameters
        # STATE
        self.state = State(parameters, system_time)
        # ENVIRONMENT CONTROL
        self.sim_manager = SimulationManager(parameters)
        # LOGGING
        self.logger = SchedulingLogger(parameters.logfile_path, parameters.seed,
                                       self.state, self.sim_manager)
        # EVENTS
        self.sim_manager.initialize_event_queues(
            parameters.matrices_m.machine_failures, self.state)
        # DECIDE
        if self.sim_manager.transport_decisions_required():
            m_nr, job = self.sim_manager.pop_routing_queue()
            self.state.setup_routing_decision(m_nr, job)
        else:
            # ROLLOUT SIMULATION
            self._trigger_events()
            self.__chmod()
        # POSSIBLE ACTIONS FROM THIS STATE
        self.state.legal_actions = self._get_legal_actions()
        # when making the environment deterministic, reinitialize everythin

    def make_deterministic(self):
        """
        Initializes the environment to a non stochastic version of itself from
        the current state. This allows for model based RL usage for stachstic
        environments.

        :return: None
        """
        # 1. requeue events without stochasticity (brakdowns and new arrivals
        # get eliminated here)
        self.sim_manager.event_heap.requeue_events(self.state)
        if bool(self.sim_manager.event_heap.event_heap):
            try:
                assert (self.sim_manager.event_heap.
                        event_heap[0].occurence_time >=
                        self.state.system_time)
            except AssertionError:
                # print('Something went wrong when reinitializing the queue...')
                pass
        # 2. eliminate stochasticity from matrices (op & deadline perturbations
        # get eliminated here)
        n_jobs = self.parameters.dims.n_jobs
        n_ops = self.parameters.dims.max_n_operations
        self.state.matrices.op_perturbations = np.zeros((n_jobs, n_ops))
        self.state.matrices.deadlines = np.zeros(n_jobs)
        # 3. turn off logging
        self.logger.turn_off()
        # Note that 1 has to come before 2, since requeueing uses the matrix
        # information to revert the stochasticity!

    def simulation_has_ended(self):
        return self.__end

    def _get_legal_actions(self):
        if self.__end:
            return []
        else:
            if self.state.in_scheduling_mode():
                return self._get_legal_machine_assignment_actions()
            elif self.state.in_postbuffer_routing_mode():
                return self.__get_legal_postbuffer_transport_actions()
            elif self.state.in_prebuffer_routing_mode():
                # same as above, but agents can choose to block the operation
                # behind the broken machine
                return ([self.sim_manager.wait_routing] +
                        self.__get_legal_prebuffer_transport_actions())
            else:
                raise UndefinedSimulationModeError()

    def wait_legal(self):
        if self.state.scheduling_mode == 0:
            return self.sim_manager.wait_legal(
                self.state.machines.machine_assignment_required())
        elif self.state.scheduling_mode == 1:
            return False
        else:
            return NotImplementedError

    def _get_legal_machine_assignment_actions(self):
        legal_actions = []
        # current_machine has to choose op from its queue
        assert self.state.machines[
                   self.state.current_machine_nr].queue is not None
        for op_idx in self.state.machines[self.state.current_machine_nr].queue:
            action = self.__get_action(op_idx)
            legal_actions.append(action)
        if (self.wait_legal() and
                len(self.state.operation_graph.root_visible.next) > 0):
            legal_actions.append(self.sim_manager.wait_scheduling)
        return legal_actions

    def __get_action(self, global_state_op_index):
        n_cols = self.parameters.dims.max_n_operations
        return (self.state.job_global_to_view_idx[global_state_op_index[0]] *
                n_cols + global_state_op_index[1])

    def __get_legal_postbuffer_transport_actions(self):
        self.sim_manager.purge_possible_transport_assignment()
        legal_actions = []
        # current_job has to be transported to a machine
        next_ops = self.state.operation_graph.get_next_ops(
            self.state.current_job)
        for op_idx in next_ops:
            legal_actions += self.__get_legal_tgt_machines(op_idx)
        return legal_actions

    def __get_legal_prebuffer_transport_actions(self):
        return self.__get_legal_tgt_machines(self.state.current_operation)

    def __get_legal_tgt_machines(self, op_idx):
        """
        Legal actions helper. Should only be called at the end of step when the
        mode is 1 (aka transport).

        Returns the legal target machines for the operation index passed.
        To this end the machines with the appropriate capabilities are appended
        to a list.

        This method additionally associates every target machine with the
        corresponding operation index, such that when the action
        (machine number) is passed in the next step call, the operation to be
        transported can be immediately retrieved.

        :param op_idx: The index of the operation for which to find eligible
            machines.
        :return: A list of eligible machines.
        """
        legal_tgt_machines = []
        op_t = self.state.matrices.op_type[op_idx]  # op idx is global
        eligible_ms = self.state.matrices.machine_capab_dt[op_t]
        for m_nr in eligible_ms:
            # machine is only eligible if there is space in the inpt buffer
            if (self.state.trackers.buffer_lengths[m_nr - 1] <
                    self.state.matrices.buffer_capacity[m_nr - 1]):
                self.sim_manager.mark_possible_transport_assignment(
                    m_nr, op_idx)
                legal_tgt_machines.append(m_nr)
        return legal_tgt_machines

    def _trigger_events(self):
        """
        Pops events from the event queue and handles them. If the event leads
        to a new scheduling or routing decision, the event handling loop is
        stopped.

        :return: True is the event queue is empty and False otherwise.
        """
        # assert len(self.machines.scheduling_queue) == 0
        # assert len(self.jobs_pending_transport) == 0
        # assert len(self.pending_transport_blocked) == 0
        while self.sim_manager.event_heap.events_available():
            event = self.sim_manager.event_heap.pop_next_event()
            event.handle(self.sim_manager, self.state)
            if not event.trigger_next:  # event leads to decision
                t_n = self.sim_manager.event_heap.peek_next_occurence_time()
                while event.occurence_time == t_n:
                    event = self.sim_manager.event_heap.pop_next_event()
                    event.handle(self.sim_manager, self.state)
                    t_n = self.sim_manager.event_heap.\
                        peek_next_occurence_time()
                break
        return self.sim_manager.event_heap.events_available()

    def step(self, action: int) -> (State, bool):
        self.logger.create_action_log(action, self.state.legal_actions)
        if self.state.in_scheduling_mode():
            if not self.sim_manager.is_wait_processing(action):
                self.__step_scheduling_mode(action)
            else:
                self.sim_manager.add_current_machine_to_visited(
                    self.state.current_machine_nr)
        elif self.state.in_postbuffer_routing_mode():
            # since the operation being transported now has already been,
            # finished, i.e. popped from its machine queue, all we need is
            # to update the states and create the transport event
            self.__step_transport_mode(action)
        else:  # self.scheduling_mode == 2
            broken_machine = self.state.machines[
                self.state.current_machine_nr]
            if self.sim_manager.is_wait_transport(action):
                # ops will now be blocked behind this ch.current_machine
                # mark operations as blocked behind the current machine
                for op_idx in broken_machine.queue:
                    self.state.matrices.change_op_status(op_idx, 4)
                self.sim_manager.purge_transport_from_failed_machine = []
            else:
                self.__step_transport_mode(action,
                                           failed_machine=broken_machine)
        # log decision and state
        self.logger.write_logs()
        # find next decision mode or rollout sim
        self.__chmod()
        # NOTE: reward & co get calculated outside the core!
        self.state.n_steps += 1
        if len(self.state.legal_actions) == 1:
            self.step(self.state.legal_actions[0])
        if not bool(self.state.legal_actions):
            return self.state, True
        return self.state, False

    def __chmod(self):
        # do not query the mode alone, since there may be deferred decisions
        # in the scheduling queue which will trigger a mode shift
        if self.state.machines.machine_assignment_required():  # decide schedule
            self.state.setup_machine_assignment_decision()
            # if the next job hasnt arrived yet, the key error will pop up!!!
        elif self.sim_manager.transport_decisions_required():
            m_idx, j_idx = self.sim_manager.pop_routing_queue()
            self.state.setup_routing_decision(m_idx, j_idx)
        elif self.sim_manager.failure_handling_required():
            m_idx, op_idx = self.sim_manager.pop_failure_routing_queue()
            self.state.setup_failure_routing_decision(m_idx, op_idx)
        else:
            events_available = self.sim_manager.event_heap.events_available()
            # TODO: establish if we need the following:
            #  operations_available = self.operation_graph.has_ops()
            if events_available:
                # add scheduling decisions back to queue and rollout simulation
                if self.sim_manager.has_deferred_scheduling_decisions():
                    self.state.machines.scheduling_queue |= (
                        self.sim_manager.deferred_scheduling_decisions)
                    self.sim_manager.reset_deferred()
                self._trigger_events()
                self.__chmod()
            else:
                if self.sim_manager.has_deferred_scheduling_decisions():
                    self.state.machines.scheduling_queue |= (
                        self.sim_manager.deferred_scheduling_decisions)
                    self.sim_manager.reset_deferred()
                    self.__chmod()
                else:
                    self.__end = True
        # update legal actions
        self.state.legal_actions = self._get_legal_actions()
        if not bool(self.state.legal_actions) and not self.__end:
            self.__chmod()

    def __step_transport_mode(self, action, failed_machine=None):
        if failed_machine is not None:
            # make sure to transport the current op, not the next one
            op_idx = self.state.current_operation  # todo!!!
            # since we are transporting from queue, we need to remove the op
            failed_machine.remove_op(op_idx)
            # TODO: handle buffer times
        else:
            op_idx = self.sim_manager.get_transport_tgt_op(action)
        m_tgt_nr = action
        m_src_nr = self.state.current_machine_nr
        # create transport event
        self.sim_manager.start_transport(m_src_nr, m_tgt_nr, op_idx, self.state)
        # mark op as being in transit to new target machine
        self.state.matrices.mark_transport_started(op_idx, m_tgt_nr)

    def __step_scheduling_mode(self, action):
        # mark that the scheduling decisions to follow are new
        self.sim_manager.reset_revisiting()
        # get internal (aka global) operation index
        op_idx = self.state.get_global_op_index(action)
        m_nr = self.state.current_machine_nr
        current_time = self.state.system_time
        # update machines
        op_finish_time, duration = self.state.machines.start_processing(
            m_nr, op_idx, current_time)
        # create event and add it to the heap
        self.state.trackers.buffer_lengths[m_nr - 1] -= 1
        self.sim_manager.create_processing_event(
            op_idx, m_nr, op_finish_time, duration)
        # set start times
        self.state.trackers.update_on_op_start(op_idx, current_time)
        # update processing status in state matrices
        self.state.matrices.change_op_status(op_idx, 2)
        # log decision for gantt schedule
        self.logger.add_to_schedule(
            m_nr, current_time, op_finish_time, op_idx[0])
