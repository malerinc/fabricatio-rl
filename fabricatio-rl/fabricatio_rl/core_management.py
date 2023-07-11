import heapq as h
from collections import deque

from fabricatio_rl.core_events import *
from fabricatio_rl.core_state import State
from fabricatio_rl.env_utils import faster_deepcopy
from fabricatio_rl.interface_input import Input


class EventHeap:
    def __init__(self):
        self.__event_heap_counts = {
            # helps establish whether wait flags are legal; this is the case if
            # 1. there are processing or
            # 2. transport events in the queue or
            # 3. the number of repairs is larger than the number of breakdowns
            'transport': 0,
            'processing': 0,
            'breakdown': 0,
            'repair': 0,
            'job_arrivals': 0
        }  # todo: enums
        self.__event_heap = []
        self.__event_log = deque([])

    def requeue_events(self, state: State):
        deterministic_event_heap = []
        while len(self.__event_heap) > 0:
            event = h.heappop(self.event_heap)
            assert event.occurence_time >= state.system_time
            is_deterministic = event.get_deterministic_copy(state)
            if is_deterministic:
                h.heappush(deterministic_event_heap, event)
        self.__event_heap = deterministic_event_heap

    @property
    def event_heap_counts(self):
        return self.__event_heap_counts

    @property
    def event_heap(self):
        return self.__event_heap

    @property
    def event_log(self):
        return self.__event_log

    def pop_next_event(self):
        event = h.heappop(self.__event_heap)
        self.__event_log.append(event.log_occurence())
        return event

    def peek_next_event(self):
        return self.__event_heap[0]

    def get_event_logs(self):
        return '|'.join(self.__event_log)

    def flush_event_log(self):
        self.__event_log = []

    def get_heap_representation(self):
        return f"{'|'.join([str(event) for event in self.__event_heap])}"

    def add_event(self, event):
        h.heappush(self.__event_heap, event)
        self.__event_log.append(event.log_creation())

    def peek_next_occurence_time(self):
        if len(self.__event_heap) == 0:
            return -1
        else:
            return self.__event_heap[0].occurence_time

    def events_available(self):
        return bool(self.__event_heap)

    def has_blocking_events(self):
        return (self.__event_heap_counts['processing'] > 0 or
                self.__event_heap_counts['transport'] > 0 or
                self.__event_heap_counts['repair'] >
                self.__event_heap_counts['breakdown'])

    def update_event_heap_counts(self, event_type, increment=True):
        # TODO: add to add/pop!!!
        if increment:
            self.__event_heap_counts[event_type] += 1
        else:
            self.__event_heap_counts[event_type] -= 1

    def get_str_heap_counts(self):
        return f'Processing event created! Heap counts: ' \
               f'{self.__event_heap_counts}'


class SimulationManager:
    def __init__(self, simulation_parameters: Input):
        self.__params = simulation_parameters
        # flags
        self.__wait_scheduling = (simulation_parameters.dims.max_jobs_visible *
                                  simulation_parameters.dims.max_n_operations)
        self.__wait_routing = simulation_parameters.dims.n_machines + 1

        # temp mappings
        self.__machine_to_ops = {}  # temporary mapping for each routing step
        self.__current_transport_machine_to_op = {}  # upd. on get_legal_action
        # scheduling and routing queues
        self.__routing_queue = deque([])
        self.__deferred_scheduling_decisions = set({})
        self.__pending_transport_from_failed_machine = deque([])
        # the event heap
        self.__event_heap = EventHeap()
        self.__scheduling_decision_revisited = False

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)

    # <editor-fold desc="Getters">
    @property
    def params(self):
        return self.__params

    @property
    def wait_scheduling(self):
        return self.__wait_scheduling

    @property
    def wait_routing(self):
        return self.__wait_routing

    @property
    def pending_transport_from_failed_machine(self):
        return self.__pending_transport_from_failed_machine

    @property
    def routing_queue(self):
        return self.__routing_queue

    @property
    def machine_to_ops(self):
        return self.__machine_to_ops

    @property
    def current_transport_machine_to_op(self):
        return self.__current_transport_machine_to_op

    @property
    def deferred_scheduling_decisions(self):
        return self.__deferred_scheduling_decisions

    @property
    def event_heap(self):
        return self.__event_heap
    # </editor-fold>

    def add_current_machine_to_visited(self, m_nr):
        self.__deferred_scheduling_decisions.add(m_nr)

    def has_deferred_scheduling_decisions(self):
        return bool(self.__deferred_scheduling_decisions)

    def reset_deferred(self):
        if bool(self.__deferred_scheduling_decisions):
            self.__deferred_scheduling_decisions = set({})
            self.__scheduling_decision_revisited = True

    # for core
    def is_wait_transport(self, action):
        return action == self.__wait_routing

    def is_wait_processing(self, action):
        return action == self.__wait_scheduling

    def mark_possible_transport_assignment(self, tgt_machine_nr, op_idx):
        self.__current_transport_machine_to_op[tgt_machine_nr] = op_idx

    def purge_possible_transport_assignment(self):
        self.__current_transport_machine_to_op = {}

    def get_transport_tgt_op(self, tgt_m_nr):
        return self.__current_transport_machine_to_op[tgt_m_nr]

    def wait_legal(self, machine_assignment_required):
        # TODO: solve situation where all ops are in queue and no other events
        #  are available
        if self.__event_heap.has_blocking_events():
            return True
        elif machine_assignment_required:
            if self.__scheduling_decision_revisited:
                return False
            else:
                return True
        else:
            return False

    def queue_transport_decision(
            self, originating_machine_nr, job_idx):
        self.__routing_queue.append(
            (originating_machine_nr, job_idx))

    def transport_decisions_required(self):
        return len(self.__routing_queue) > 0

    def failure_handling_required(self):
        return len(self.__pending_transport_from_failed_machine) > 0

    def queue_blocked_operation(self, machine_nr, op_idx):
        self.__pending_transport_from_failed_machine.append(
            (machine_nr, op_idx))

    def purge_transport_from_failed_machine(self):
        self.__pending_transport_from_failed_machine = []

    def pop_routing_queue(self):
        """
        :return: m_idx, j_idx
        """
        return self.__routing_queue.popleft()

    def pop_failure_routing_queue(self):
        """
        :return: m_idx, o_idx
        """
        return self.__pending_transport_from_failed_machine.popleft()

    def initialize_event_queues(self, machine_breakdowns, state: State):
        """
        Creates all the job arrival and machine availability change events that
        are to occur during the simulation and adds them to the event heap.
        :return: None
        """
        # TODO: assert that the state is initialized before this
        # job arrival events
        job_index = 0
        for job_arrival_time in state.matrices.job_arrivals:
            if job_arrival_time == 0:
                job_index += 1
                continue
            jae = JobArrivalEvent(job_arrival_time, job_index)
            job_index += 1
            self.__event_heap.update_event_heap_counts('job_arrivals',
                                                       increment=True)
            self.__event_heap.add_event(jae)
        # breakdown & repair
        for machine_idx, breakdowns in machine_breakdowns.items():
            for breakdown in breakdowns:
                # machine breakdown event
                # TODO: consider moving to stateMatrices, such that all the
                #  times defining the simulation are encapsulated in one place
                breakdown_time = int(breakdown[0])
                mbe = MachineAvailabilityEvent(breakdown_time, machine_idx)
                self.__event_heap.update_event_heap_counts('breakdown', True)
                self.__event_heap.add_event(mbe)
                # machine repair events
                breakdown_duration = int(breakdown[1])
                repair_time = breakdown_time + breakdown_duration
                mre = MachineAvailabilityEvent(repair_time, machine_idx, True)
                self.__event_heap.update_event_heap_counts('repair', True)
                self.__event_heap.add_event(mre)
        # transport
        for job, j_root_node in state.operation_graph.root_visible.next.items():
            j = job[0]  # global idx
            self.initialize_job_transports(j, j_root_node, state)

    def finish_job(self, state: State, finished_job_idx: int):
        """
        Notes down the finished job and starts a new one if the latter is
        present in the job queue.

        :param state: The environment state.
        :param finished_job_idx: The index of the finished job.
        :return: None
        """
        j_next = state.mark_completed_job(finished_job_idx)
        if j_next != -1:  # next job is here
            # update view <-> global mappings
            self.start_new_job(j_next, state)

    def start_new_job(self, j_new: int, state: State) -> None:
        """
        Pulls a new job to view and initializes job transports for it, if these
        can be executed without a decision (see initialize_job_transports).

        Raises an IndexError if called when the system view slots are all full.

        :param j_new: The index
        :param state: The environment state object.
        :return: None
        """
        view_index = state.system_free_job_slots.popleft()
        state.pull_to_view(j_new, view_index)
        # update OperationGraph
        node = state.operation_graph.make_visible(j_new, view_index)
        # create appropriate transport events / decisions
        self.initialize_job_transports(j_new, node, state)
        state.trackers.update_on_job_in_wip(j_new, state.system_time)

    def initialize_job_transports(self, j, j_root_node, state: State):
        next_ops = j_root_node.next.keys()
        if len(next_ops) != 1:
            # at this point the operation is not relevant
            self.queue_transport_decision(-1, j)
        else:
            op_idx = next(iter(next_ops))   # global op index
            op_type = state.matrices.op_type[op_idx]
            eligible_machines = state.matrices.machine_capab_dt[op_type]
            if len(eligible_machines) > 1:
                # we need to decide where to transport these ops
                self.queue_transport_decision(0, j)
            else:
                # only one transport option; start transport
                tgt_m_nr = list(eligible_machines)[0]
                src_m_nr = 0  # the transport starts from the prod src
                # mark op as being in transit to new target machine
                state.matrices.mark_transport_started(
                    op_idx, tgt_m_nr)
                self.start_transport(src_m_nr, tgt_m_nr, op_idx, state)

    def start_transport(self, src_machine_nr: int, tgt_machine_nr: int,
                        op_idx: tuple, state: State) -> None:
        """
        Starts the transport event by booking the target machine buffer slot,
        calculating the event occurrence time, creating the transport event,
        and eliminating the machine to operation mapping. Should be called
        during core.init or at the beginning of core.step.

        :param src_machine_nr: The machine the transport starts from.
        :param tgt_machine_nr: The machine the operation is transported to.
        :param op_idx: The operation transported.
        :param state: The environment state for slot booking and duration
            computation.
        :return: None
        """
        # book the buffer slot at transport start!!
        state.trackers.buffer_lengths[tgt_machine_nr - 1] += 1
        # calculate occurrence time
        transport_time = state.matrices.transport_times[
            src_machine_nr][tgt_machine_nr]
        event_occurrence_time = state.system_time + transport_time
        # create transport event
        tae = TransportArrivalEvent(
            event_occurrence_time, op_idx, tgt_machine_nr)
        self.__event_heap.add_event(tae)
        self.__event_heap.update_event_heap_counts('transport', True)
        # flush machine to op mappings
        self.__current_transport_machine_to_op = {}

    def create_processing_event(self, op_idx, m_nr, op_finish_time, duration):
        # add OperationFinishedEvent
        ofe = OperationFinishedEvent(op_finish_time, op_idx, m_nr, duration)
        self.event_heap.update_event_heap_counts('processing', True)
        self.event_heap.add_event(ofe)

    def cancel_op_finished_event(self, operation_index):
        if operation_index is not None:
            # find OpFinishedEvent
            event_buffer = deque([])
            while True:
                event = self.__event_heap.pop_next_event()
                if (type(event) == OperationFinishedEvent and
                        event.operation_index == operation_index):
                    break
                else:
                    event_buffer.append(event)
            for event in event_buffer:
                self.__event_heap.add_event(event)
            # delete OpFinishedEvent
            # reestablish heap invariant

    def reset_revisiting(self):
        self.__scheduling_decision_revisited = False
