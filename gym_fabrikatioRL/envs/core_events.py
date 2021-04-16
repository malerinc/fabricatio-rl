from gym_fabrikatioRL.envs.core_state import State


class Event:
    def __init__(self, time: int):
        """
        :param time: Occurrence time.
        """
        self.occurence_time = time
        # whether to stop and ask agent for action (False) or go to next event
        self.trigger_next = True

    def __eq__(self, other):
        return self.occurence_time == other.occurence_time

    def __le__(self, other):
        return self.occurence_time <= other.occurence_time

    def __lt__(self, other):
        return self.occurence_time < other.occurence_time

    def __ge__(self, other):
        return self.occurence_time >= other.occurence_time

    def __gt__(self, other):
        return self.occurence_time > other.occurence_time

    def log_creation(self):
        message = (f"Event of type {str(self.__class__).split('.')[-1][:-2]} "
                   f"was created and set to occur at time "
                   f"{self.occurence_time}.")
        return message

    def log_occurence(self):
        return (f"Event of type {str(self.__class__).split('.')[-1][:-2]} "
                f"occurred at time {self.occurence_time}.")

    def handle(self, sim_manager, state: State):
        """

        :param sim_manager: gym_fabrikatioRL.envs.env_control.SimulationManager
        :type state: gym_fabrikatioRL.envs.jssp_env.State
        """
        elapsed_time = state.update_system_time(self.occurence_time)
        state.matrices.update_running_ops_duration(elapsed_time)

    def __str__(self):
        return f"'{self.__class__.__name__}':{vars(self)}"

    def get_deterministic_copy(self, state: State):
        pass


class TransportArrivalEvent(Event):
    def __init__(self, time, op_idx, machine_nr):
        super().__init__(time)
        self.op_idx = op_idx
        self.tgt_machine_nr = machine_nr

    def handle(self, sim_manager, state: State):
        super().handle(sim_manager, state)
        state.matrices.change_op_status(self.op_idx, 1)
        sim_manager.event_heap.update_event_heap_counts('transport', False)
        decision_needed = state.machines.finish_op_transport(
            self.tgt_machine_nr, self.op_idx)
        if decision_needed:
            self.trigger_next = False

    def log_creation(self):
        message = (f'{super().log_creation()} Operation {self.op_idx} with '
                   f'target machine {self.tgt_machine_nr} is under way.')
        return message

    def log_occurence(self):
        return f'{super().log_occurence()} ' \
               f'Op {self.op_idx} arrived at machine ' \
               f'{self.tgt_machine_nr}.'

    def get_deterministic_copy(self, state):
        return True


class JobArrivalEvent(Event):
    """
    Class defining job arrival events in the JSSP simulation.
    """
    def __init__(self, time, job_index):
        """
        Event constructor.

        :param time: The time at which the event occurs.
        :param job_index: The index of the arriving job from the list of all
            jobs considered in this simulation.
        """
        super().__init__(time)
        self.job_index = job_index

    def handle(self, sim_manager, state):
        """
        Event handler for job arrivals. When jobs arrive, they are added either
        to the planning horizon, or to the queue of pending jobs.
        """
        super().handle(sim_manager, state)
        state.queue_job(self.job_index)
        if state.has_free_view_slots():
            # todo: revisit this...
            self.trigger_next = False
            job_idx = state.get_next_job()
            sim_manager.start_new_job(job_idx, state)
        sim_manager.event_heap.update_event_heap_counts('job_arrivals', False)

    def log_creation(self):
        return super().log_creation()

    def log_occurence(self):
        return f'{super().log_occurence()} ' \
               f'Job {self.job_index} has arrived.'

    def get_deterministic_copy(self, state):
        return False


class MachineAvailabilityEvent(Event):
    def __init__(self, time, machine_idx, available=False):
        super().__init__(time)
        self.machine_nr = machine_idx
        self.available = available
        # this event always requires a decision step since either:
        # 1. the operations in the machine queue need to be transported away or
        # 2. the machine has become available and needs op assignment
        self.trigger_next = False

    def handle(self, sim_manager, state):
        super().handle(sim_manager, state)
        machine = state.machines[self.machine_nr]
        if not self.available:  # breakdown event
            # put the processing operation back in the queue
            ejected = machine.eject_operation()
            sim_manager.cancel_op_finished_event(ejected)
            # mark queue operations as needing transport
            for op_idx in machine.queue:
                # mark op as behind broken machine pending transport decision
                state.matrices.change_op_status(op_idx, 3)
                sim_manager.queue_blocked_operation(self.machine_nr, op_idx)
            sim_manager.event_heap.update_event_heap_counts('breakdown', False)
            if bool(machine.queue):
                self.trigger_next = False
        else:                   # repair event
            # mark queue operations as available again
            for op_idx in machine.queue:
                state.matrices.change_op_status(op_idx, 1)
            sim_manager.event_heap.update_event_heap_counts('repair', False)
        state.machines.make_available(self.machine_nr, self.available)

    def log_creation(self):
        return super().log_creation()

    def log_occurence(self):
        if not self.available:  # breakdown event
            return f'{super().log_occurence()} ' \
                   f'Machine {self.machine_nr} broke down.'
        else:
            return f'{super().log_occurence()} ' \
                   f'Machine {self.machine_nr} was repaired.'

    def get_deterministic_copy(self, state):
        return False


class OperationFinishedEvent(Event):
    def __init__(self, time, op_idx, machine_nr, duration):
        super().__init__(time)
        self.operation_index = op_idx
        self.machine_nr = machine_nr
        self.duration = duration
        # need to
        # 1. decide where to move finished op
        # 2. which op to start on this machine
        self.trigger_next = False

    def handle(self, sim_manager, state):
        super().handle(sim_manager, state)
        # if the event was invalidated by an eject during breakdown...
        if state.matrices.op_status[self.operation_index] == 4:
            self.trigger_next = True
            return
        # update machinery; needs to happen before matrices.mark_finished
        state.machines.finish_processing(self.machine_nr, self.occurence_time)
        # dont' call this after next statement: op duration != 0 required
        state.trackers.update_on_op_completion(self.operation_index,
                                               self.occurence_time,
                                               self.duration)
        # mark operation as finished by zeroing the corresponding matrix entries
        state.matrices.mark_finished(self.operation_index)
        # handle what happens to the operation graph
        j_idx = self.operation_index[0]  # global idx
        if state.job_finished(j_idx):  # job just finished ^^
            sim_manager.finish_job(state, j_idx)
            # only trigger next if no decision is necessary for this machine
            if not state.machines[self.machine_nr].has_queued_ops():
                self.trigger_next = True
        else:
            # shift all possible next ops and hence machines to job root
            try:
                # TODO: implement this in OperationGraph
                state.operation_graph.root_visible.next[(j_idx,)].next[
                    self.operation_index].delete()
            except KeyError:
                print(f'Could not find {j_idx} in the operation '
                      f'graph {state.operation_graph}')
            # add to transport decision queue
            sim_manager.queue_transport_decision(
                self.machine_nr, self.operation_index[0])
        # add to scheduling decision queue
        state.machines.make_available(self.machine_nr, True)
        # update heap counts
        sim_manager.event_heap.update_event_heap_counts('processing', False)

    def log_creation(self):
        return super().log_creation()

    def log_occurence(self):
        return f'{super().log_occurence()} ' \
               f'Finished operation {self.operation_index} on machine ' \
               f'{self.machine_nr}!'

    def get_deterministic_copy(self, state):
        perturbation = state.matrices.op_perturbations[self.operation_index]
        self.occurence_time -= perturbation
        self.duration -= perturbation
        return True
