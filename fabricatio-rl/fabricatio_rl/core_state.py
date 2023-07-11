from collections import deque
from copy import deepcopy
from json import dumps
from typing import List

import numpy as np

from fabricatio_rl.env_utils import faster_deepcopy
from fabricatio_rl.interface_input import Input, SchedulingDimensions, \
    JobMatrices, MachineMatrices


class StateMatrices:
    """
    Class containing all the raw state information, The information within this
    object both determines the simulation logic and, if presented to an agent,
    the decision logic. The entirety of this information, if presented to an
    agent, would yield a fully observable MDP setup.

    The fact that most of the information here is stored in a numpy array,
    explains the class naming.

    Note that the machine_capab_d... instance variables are the only exceptions,
    given that they are dictionaries. These dictionaries encore redundant
    information with respect to the machine capability n_machines x n_types
    matrix machine_capab_cm. This is done for convenience and O(1) access of
    either the type of operations that can be ran on machine i
    (machine_capab_dm[i]) or the machine that can execute operation of type T
    (machine_capab_dt[T]).

    TODO: COMMENT ON ALL PARAMETERS
    """
    def __init__(self, sched_input: Input):
        # operation information
        ij_matrices: JobMatrices = sched_input.matrices_j
        self.op_type = ij_matrices.operation_types
        self.op_duration = ij_matrices.operation_durations
        self.op_tooling_lvl = ij_matrices.operation_tool_sets
        self.op_prec_m = ij_matrices.operation_precedence_m
        self.op_perturbations = ij_matrices.operation_perturbations
        self.job_arrivals = ij_matrices.job_arrivals
        self.deadlines: np.ndarray = ij_matrices.job_due_dates
        n, o = sched_input.dims.n_jobs, sched_input.dims.max_n_operations
        # opertion tracking info
        self.op_status = np.zeros((n, o), dtype='uint8')
        self.op_location = np.zeros((n, o), dtype='uint8')
        # machinery&layout information
        im_matrices: MachineMatrices = sched_input.matrices_m
        self.tooling_times = im_matrices.tool_switch_times
        self.transport_times = im_matrices.machine_distances
        self.machine_speeds = im_matrices.machine_speeds
        self.buffer_capacity = im_matrices.machine_buffer_capa
        self.machine_capab_dm = im_matrices.machine_capabilities_dm
        self.machine_capab_dt = im_matrices.machine_capabilities_dt
        self.machine_capab_cm = im_matrices.machine_capabilities_m
        self.machine_failures = im_matrices.machine_failures

    def change_op_status(self, op_idx, new_status):
        """
        Changes the status of an operation. The operation status is defined
        relative to the operation location maintained in the op_location
        parameter. Relative to a target machine, an operation can be either
            - In transit to the target machine (op_status[op_idx] == 0)
            - In buffer at the target machine (op_status[op_idx] == 1)
            - Processing at the target machine (op_status[op_idx] == 2)
            - In Buffer behind a broken down machine pending transport
            (op_status[op_idx] == 3)
            - Locked in buffer behind a broken down machine
            (op_status[op_idx] == 4)

        :param op_idx: The index of the operation to change.
        :param new_status: The new operation status.
        :return: None.
        """
        self.op_status[op_idx] = new_status

    def change_op_location(self, op_idx, new_location):
        """
        Changes the operation target machine. Should be used on each transport
        event.

        :param op_idx: The operation to apply the location change to.
        :param new_location: The new target machine.
        :return: None.
        """
        self.op_location[op_idx] = new_location

    def mark_transport_started(self, op_idx, target_machine_number):
        self.change_op_status(op_idx, 0)
        self.change_op_location(op_idx, target_machine_number)

    def mark_finished(self, op_idx):
        self.op_type[op_idx] = 0
        self.op_duration[op_idx] = 0
        self.op_tooling_lvl[op_idx] = 0
        self.op_perturbations[op_idx] = 0
        self.op_status[op_idx] = 0
        self.op_location[op_idx] = 0

    def update_running_ops_duration(self, elapsed_time):
        j_idx, op_idx = self.op_status.nonzero()
        for i in range(j_idx.shape[0]):
            if self.op_status[j_idx[i]][op_idx[i]] == 2:
                self.op_duration[j_idx[i]][op_idx[i]] -= elapsed_time
                if self.op_duration[j_idx[i]][op_idx[i]] <= 0:
                    #
                    # because of the stochastic nature of processing times,
                    # the operation duration can be longer than planned; in such
                    # a case, updating the planned duration on a running op
                    # can fall below zero; set a unit duration for these case
                    self.op_duration[j_idx[i]][op_idx[i]] = 1

    def __str__(self):
        """
        Creates a string representation of the StateMatrices object.

        :return: String representation.
        """
        # machinery&layout information
        return dumps(self.to_dict())

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)

    @staticmethod
    def __fill_matrix_metadata(matrix: np.ndarray,
                               label_x='Operation Index',
                               label_y='Job Index',
                               nfo_type='jobs'):
        """
        Converts a particular state matrix to a list and ads metadata pertaining
        to the contained values, matrix structure and index names as specified
        by the function parameters. The list representation together with the
        metadata fields is stored within a dictionary which is returned.

        to_dict() helper method.

        :param matrix: The state matrix to add metadata to.
        :param label_x: The name of the column index.
        :param label_y: The name of the row index.
        :param nfo_type: The matrix category; can be either "jobs", "machines"
            or "tracking".
        :return: The dictionary containing the list representation and metadata
            for the matrix parameter.
        """

        if np.inf in matrix:
            matrix[matrix == np.inf] = 9999
        return {
            'x_label': label_x,
            'y_label': label_y,
            'data': matrix.tolist(),
            'min_value': int(matrix.min(initial=np.infty)),
            'max_value': int(matrix.max(initial=-np.infty)),
            'n_rows': matrix.shape[0],
            'n_cols': matrix.shape[1] if len(matrix.shape) == 2 else 1,
            'nfo_type': nfo_type
         }

    def to_dict(self):
        """
        Creates a serializable dictionary representation of the state matrices.

        :return: The dictionary representation of the state matrices.
        """
        return {
            'operation_type_matrix': StateMatrices.__fill_matrix_metadata(
                self.op_type),
            'operation_duration_matrix': StateMatrices.__fill_matrix_metadata(
                self.op_duration),
            'operation_tooling_level_matrix':
                StateMatrices.__fill_matrix_metadata(self.op_tooling_lvl),
            'operation_perturbation_matrix':
                StateMatrices.__fill_matrix_metadata(self.op_perturbations),
            'operation_status_matrix':
                StateMatrices.__fill_matrix_metadata(self.op_status,
                                                     nfo_type='tracking'),
            'operation_location_matrix':
                StateMatrices.__fill_matrix_metadata(self.op_location,
                                                     nfo_type='tracking'),
            'transport_matrix':
                StateMatrices.__fill_matrix_metadata(
                    self.transport_times, 'Machine Number', 'Machine Number',
                    'machines'),
            'tooling_matrix':
                StateMatrices.__fill_matrix_metadata(
                    self.tooling_times, 'Tool Set', 'Tool Set', 'machines'),
            'job_release_times': StateMatrices.__fill_matrix_metadata(
                    self.job_arrivals, ''),
            'job_deadlines': StateMatrices.__fill_matrix_metadata(
                    self.deadlines, ''),
            'machine_speeds': StateMatrices.__fill_matrix_metadata(
                    self.machine_speeds, '', 'Machine Number', 'machines'),
            'machine_buffer_capacity': StateMatrices.__fill_matrix_metadata(
                    self.buffer_capacity, '', 'Machine Number', 'machines')
        }


class OperationGraph:
    """
    @__view_indices: Private attribute that tracks the indices of the jobs
        currently visible.
    @root_visible: TODO
    @root_hidden: TODO
    """
    def __init__(self, adjacency_dicts, n_visible):
        """
        Builds an all job operation precedence (directed acyclic) graph from
        adjacency lists of the following structure:
            [{(0,): [3], 3: [0], 0: [4], 4: [1], 1: [2]},
             {(1,): [1, 0, 3, 4], 3: [2]},
             {(2,): [2, 0, 1], 0: [4], 4: [3]},
             {(3,): [1, 0, 3], 0: [2], 2: [4]},
             {(4,): [2], 2: [1], 1: [4], 4: [0], 0: [3]}]
        Each dictionary represents a different (unique!) job. The job roots are
        given by nodes labeled (job_ix,).

        Each node in the resulting graph keeps pointers to its predecessors and
        next nodes. This allows us to find node predecessors as well as next
        nodes in O(1) for any given GraphNode object. Since a node represents an
        operation, the operation index (j_idx, j_op_idx) from the StateMatrices
        is also stored for each inner node.

        The first n_visible nodes from the list of dicts will be stored under a
        root node labeled "visible" while all the jobs arriving in the future
        can be found under the 'hidden' root. Both of the 2 roots point to job
        precedence graph start nodes labeled (j_idx, ). No start node has
        predecessor pointers.

        :param adjacency_dicts: The operation precedence graphs for each job
            with start nodes denoted by (job_idx) as a dictionary representation
            of graph adjacency lists.
        :param n_visible: Number of adjacency dicts to use in the initial
            planning window.
        """
        self.__view_indices = {i: i for i in range(n_visible)}
        self.root_visible = GraphNode(None, 'visible')
        self.root_hidden = GraphNode(None, 'hidden')
        for j in range(len(adjacency_dicts)):
            new_root = GraphNode(None, None)
            new_root.op_index = (j,)
            OperationGraph.build_recursively(
                j, new_root, adjacency_dicts[j], {})
            if j < n_visible:
                self.root_visible.next[
                    new_root.op_index] = new_root
            else:
                self.root_hidden.next[
                    new_root.op_index] = new_root

    @staticmethod
    def build_recursively(job_index, current_node, adjacency_lists,
                          nodes_added):
        idx = (current_node.op_index if current_node.previous is None
               else current_node.op_index[1])
        try:
            neighbors = adjacency_lists[idx]
        except KeyError:  # got to leaf node
            # assert adjacency_lists == {}
            return
        for neighbor in neighbors:
            if neighbor in nodes_added:
                # add the edge and stop recursion
                current_node.add_next(nodes_added[neighbor])
            else:
                # add new node
                new_node = GraphNode(current_node, (job_index, neighbor))
                nodes_added[neighbor] = new_node
                current_node.add_next(new_node)

        del adjacency_lists[idx]

        for next_node in current_node.next.values():
            if next_node.op_index in nodes_added:
                continue   # no need to recurse
            else:
                OperationGraph.build_recursively(
                    job_index, next_node, adjacency_lists, nodes_added)

    def remove_job(self, job_idx, visible=True):
        if visible:
            del self.root_visible.next[(job_idx,)]
        else:
            del self.root_hidden.next[(job_idx,)]

    def make_visible(self, job_idx, view_index):
        node_to_move = self.root_hidden.next[(job_idx,)]
        self.__view_indices[view_index] = job_idx
        self.root_visible.next[(job_idx,)] = node_to_move
        self.remove_job(job_idx, visible=False)
        return node_to_move

    def __str__(self):
        tree_to_string = ''
        for k, v in self.root_visible.next.items():
            tree_to_string += (str(self.root_visible.op_index) + '->' +
                               str(k) + ';' + str(v))
        for k, v in self.root_hidden.next.items():
            tree_to_string += (str(self.root_hidden.op_index) + '->' +
                               str(k) + ';' + str(v))
        return tree_to_string

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)

    def to_dict(self):
        rt_v_name = self.root_visible.op_index
        rt_h_name = self.root_hidden.op_index
        nodes = [{"id": hash(rt_v_name), "name": str(rt_v_name)},
                 {"id": hash(rt_h_name), "name": str(rt_h_name)}]
        links = []
        visited = {rt_v_name, rt_h_name}
        for k, v in self.root_visible.next.items():
            links.append({"source": hash(rt_v_name), "target": hash(k)})
            v.build_graph_dict(nodes, links, visited)
        for k, v in self.root_hidden.next.items():
            links.append({"source": hash(rt_h_name), "target": hash(k)})
            v.build_graph_dict(nodes, links, visited)
        return {"nodes": nodes, "links": links}

    def has_ops(self):
        return bool(self.root_visible.next) or bool(self.root_hidden.next)

    def get_next_ops(self, job):
        try:
            return self.root_visible.next[(job,)].next.keys()
        except KeyError:
            print(f"Could not find requested job {job}. The operation graph has"
                  f" the following contents: {str(self)}")

    def get_next_ops_view(self, job_view_index):
        return [(job_view_index, o) for _, o in self.get_next_ops(
            self.__view_indices[job_view_index])]

    def get_next_operations(self, job_n_ops_pairs: List[List[int]]):
        """
        Gets the next wip_size BFS sorted job operations from job j as specified by
        the parameter tuples.

        :param job_n_ops_pairs: The list of tuples (j, wip_size) with j specifying
            the job index and n the number of BFS soted operation indexes to
            return.
        :return: The list of sum(wip_size) operation indexes.
        """
        op_indexes = deque([])
        for j, n in job_n_ops_pairs:
            op_indexes += self.root_visible.next[(j,)].get_next_operations(
                n, True)
        return op_indexes


class GraphNode:
    def __init__(self, previous, op_index):
        if previous is not None:
            self.previous = {previous.op_index: previous}
        else:
            self.previous = None  # <-- dict!!!
        self.op_index = op_index
        self.next = {}

    def get_next_operations(self, n: int, job_root: bool = False):
        """
        Returns a list of the first n operations under the current node in
        breadth first search (BFS) order.

        :param job_root: Whether or not the current node is a job root.
        :param n: The number of operations to retrieve
        :return: The BFS sorted operations.
        """
        queue = deque([self])
        seen = {self.op_index}
        ops_sorted = deque([self.op_index]) if not job_root else deque([])
        while queue and len(ops_sorted) < n:
            node = queue.popleft()
            for op_idx, child_node in node.next.items():
                if op_idx not in seen:
                    seen.add(op_idx)
                    queue.append(child_node)
                    ops_sorted.append(op_idx)
        # try:
        #     assert len(ops_sorted) == n
        # except AssertionError:
        #     print(job_root, ops_sorted, n)
        return ops_sorted

    def add_next(self, node):
        self.next[node.op_index] = node
        node.previous[self.op_index] = self

    def delete(self):
        # we can only process nodes if directly under the root
        assert len(self.previous) == 1
        # delete oneself from prev
        for _, node in self.previous.items():
            del node.next[self.op_index]
        for _, node in self.next.items():
            # delete oneself from next
            del node.previous[self.op_index]
            if len(node.previous) == 0:
                # next has only one incidence edge --> conect to prev
                root_key = next(iter(self.previous))
                self.previous[root_key].add_next(node)

    def __str__(self):
        return GraphNode.__unique_path_str(self, visited=set({}))

    def __deepcopy__(self, memodict):
        return faster_deepcopy(self, memodict)

    @staticmethod
    def __unique_path_str(node, visited):
        tree_to_string = ''
        for k, v in node.next.items():
            if k in visited:
                tree_to_string += str(node.op_index) + '->' + str(k) + ';'
            else:
                visited.add(k)
                tree_to_string += (str(node.op_index) + '->' + str(k) + ';' +
                                   GraphNode.__unique_path_str(v, visited))
        return tree_to_string

    def build_graph_dict(self, nodes, links, visited):
        if self.op_index in visited:
            return
        nodes.append({"id": hash(self.op_index), "name": str(self.op_index)})
        visited.add(self.op_index)
        for k, v in self.next.items():
            links.append({"source": hash(self.op_index), "target": hash(k)})
            v.build_graph_dict(nodes, links, visited)


class Machinery:
    def __init__(self, sim_params: SchedulingDimensions,
                 state_matrices: StateMatrices):
        # TODO: refactor! Potentially keep machine params as matrices?
        # pointer to the state matrices
        self.state_matrices = state_matrices
        # the dict of dicts of machine objects indexed by nr
        self.machine_list = dict({})
        # available machines
        self.machine_available_idxs = set({})
        # available machines with sth in the queue; indexed by nr
        self.scheduling_queue = set({})
        for m in range(1, sim_params.n_machines + 1):
            self.add_machine(self.state_matrices.machine_capab_dm[m],
                             proc_speed=state_matrices.machine_speeds[m - 1])
        # self.invalid_op_finished = set({})
        # utilization times
        self.__utilization_times = np.zeros(sim_params.n_machines)
        self.__buffer_times = np.zeros(sim_params.n_machines)
        self.__setup_times = np.zeros(sim_params.n_machines)
        # the following field stores fast access pointers to the duration of the
        # operations currently running on the machines. to that end we trick
        # python into allowing pointers by wrapping the target future
        # state_matrices.op_duration references in a list,
        # i.e. mutable datastructure
        self.__running_op_indices = {}
        # keeps track of running machines for utilization updates
        self.__machine_active = np.zeros(sim_params.n_machines).astype(bool)
        self.__last_utilization_update_time = 0

    @property
    def setup_times(self):
        return self.__setup_times

    @property
    def remaining_processing_times(self):
        proc_times = np.zeros(self.__setup_times.shape[0])
        for m, running_op in self.__running_op_indices.items():
            proc_times[m] = (
                self.state_matrices.op_duration[running_op])
        return proc_times

    @property
    def machine_active(self):
        return self.__machine_active

    @property
    def utilization_times(self):
        return self.__utilization_times

    @property
    def buffer_times(self):
        return self.__buffer_times

    def add_machine(self, capabilities, proc_speed):
        nr = len(self.machine_list) + 1
        self.machine_list[nr] = Machine(nr, capabilities, self.state_matrices,
                                        proc_speed)
        self.machine_available_idxs.add(nr)

    def machine_assignment_required(self):
        return len(self.scheduling_queue) > 0

    def get_next_machine_assignment_tgt(self):
        next_machine_nr = self.scheduling_queue.pop()
        assert self[next_machine_nr].queue is not None
        return next_machine_nr

    def make_available(self, machine_nr, available):
        if available:
            machine = self[machine_nr]
            machine.status = 0
            # assert machine_nr not in self.machine_available_idxs or breakdown
            self.machine_available_idxs.add(machine_nr)
            if machine.has_queued_ops():
                self.scheduling_queue.add(machine_nr)
        else:
            machine = self[machine_nr]
            machine.status = 2
            if machine_nr in self.machine_available_idxs:
                self.machine_available_idxs.remove(machine_nr)
            if machine_nr in self.scheduling_queue:
                self.scheduling_queue.remove(machine_nr)

    def change_tooling_lvl(self, tgt_machine_nr, op_idx):
        tgt_op_tooling_lvl = self.state_matrices.op_tooling_lvl[op_idx]
        tgt_machine = self[tgt_machine_nr]
        current_tl_lvl = tgt_machine.tooling_lvl
        tooling_time = self.state_matrices.tooling_times[
            current_tl_lvl][tgt_op_tooling_lvl]
        tgt_machine.tooling_lvl = tgt_op_tooling_lvl
        self.__setup_times[tgt_machine_nr - 1] += tooling_time
        return tooling_time

    def start_processing(self, m_nr, op_idx, system_time):
        exec_time_planned = self.state_matrices.op_duration[op_idx]
        exec_time_real = (self.state_matrices.op_perturbations[op_idx] *
                          exec_time_planned)
        try:
            assert exec_time_real > 0
        except AssertionError:
            print(f"Trying to start operation with a zero duration: "
                  f"\nremaining time planned {exec_time_planned}; "
                  f"\ncurrent state matrices \n\n{self.state_matrices}")
            raise AssertionError
        # change tooling lvl
        tooling_time = self.change_tooling_lvl(m_nr, op_idx)
        duration = tooling_time + exec_time_real
        op_finish_time = system_time + duration
        # mark machine as busy
        self.make_available(m_nr, False)
        # remove from machine queue
        self[m_nr].queue.remove(op_idx)
        assert op_idx not in self[m_nr].queue
        # add to the machine's processing slot
        self[m_nr].operation_idx = op_idx
        # mark the processing start time
        self[m_nr].operation_start_time = system_time + tooling_time
        # update buffer times
        self.__buffer_times[m_nr - 1] -= exec_time_planned
        # op remaining time ptr
        self.__running_op_indices[m_nr - 1] = list(op_idx)
        # mark as active
        self.__machine_active[m_nr - 1] = True
        return op_finish_time, duration

    def finish_op_transport(self, m_nr, op_idx):
        target_machine = self[m_nr]
        target_machine.queue_op(op_idx)
        planned_duration = self.state_matrices.op_duration[op_idx[0]][op_idx[1]]
        self.__buffer_times[m_nr - 1] += planned_duration
        scheduling_decision_needed = False
        if m_nr in self.machine_available_idxs:
            # if arriving at available machine, ask agent for decision
            scheduling_decision_needed = True
            self.scheduling_queue.add(m_nr)
        return scheduling_decision_needed

    def finish_processing(self, m_nr: int, operation_finish_time: int):
        """
        Updates utilization times, removes the running operation from the
        machine and sets the machine to inactive.

        :param m_nr: The number of the machine that just finished processing.
        :param operation_finish_time: The time of the operation finish.
        :return: None.
        """
        self.update_utilization(operation_finish_time)
        self[m_nr].operation_idx = None
        self.__machine_active[m_nr - 1] = False

    def update_utilization(self, current_time: int):
        """
        Updates the time elapsed since the last utilization time update to the
        elements of the __utilization_times attribute that have been active
        since the last update. The __machine_active attribute is used to filter
        the machines requiring utilization time updates.

        :param current_time: The current system time.
        :return: None.
        """
        timedelta = current_time - self.__last_utilization_update_time
        self.__utilization_times[self.__machine_active] += timedelta
        self.__last_utilization_update_time = current_time

    def __delitem__(self, key):
        del self.machine_list[key]

    def __getitem__(self, key):
        return self.machine_list[key]

    def __setitem__(self, key, value):
        self.machine_list[key] = value

    def __len__(self):
        return len(self.machine_list)

    def __str__(self):
        return dumps({int(k): str(m) for k, m in self.machine_list.items()})

    def __deepcopy__(self, memodict):
        return faster_deepcopy(self, memodict)

    def to_dict(self):
        return {"nodes": [{
            "id": 0,
            "operation_processed": [],
            "operation_queue": [],
            "machine_group": 0
        }] + [m.to_dict() for m in self.machine_list.values()]
                }


class Machine:
    def __init__(self, machine_nr, capabilities, state_matrices,
                 processing_speed_scaler=1):
        """

        :param machine_nr: Index in the transport matrix
        :param processing_speed_scaler: Scaler for operation duration on this
            machine.
        """
        self.machine_nr = machine_nr
        self.capabilities = capabilities
        self.status = 0
        self.queue = deque([])
        self.operation_idx = None
        self.operation_start_time = -1
        self.tooling_lvl = 0
        self.processing_speed_scaler = processing_speed_scaler
        self.state_matrices = state_matrices

    def get_op_processing(self):
        return self.operation_idx

    def get_processing_time(self, estimated_time):
        # todo: check the processing finished event generation
        return estimated_time * self.processing_speed_scaler

    def eject_operation(self):
        ejected = self.operation_idx
        if ejected is not None:
            self.queue_op(ejected)
            self.operation_idx = None
        return ejected

    def change_tooling_lvl(self, new_tooling_lvl):
        self.tooling_lvl = new_tooling_lvl

    def has_queued_ops(self):
        return bool(self.queue)

    def queue_op(self, op_idx):
        self.queue.append(op_idx)

    def remove_op(self, op_idx):
        self.queue.remove(op_idx)

    def get_queue_length(self):
        return (len(self.queue) + 1 if self.operation_idx is not None else
                len(self.queue))

    @staticmethod
    def __to_int_tup(tup):
        return int(tup[0]), int(tup[1])

    def __str__(self):
        repr_str = dumps({
            'operation_processed': (Machine.__to_int_tup(self.operation_idx)
                                    if self.operation_idx is not None
                                    else ()),
            'operation_queue': [Machine.__to_int_tup(o) for o in self.queue],
            'machine_group':
                f'[{",".join([str(c) for c in self.capabilities])}]'
        })
        return repr_str

    def __deepcopy__(self, memodict):
        return faster_deepcopy(self, memodict)

    def to_dict(self):
        return {
            "id": int(self.machine_nr),
            "operation_processed": [Machine.__to_int_tup(self.operation_idx)
                                    if self.operation_idx is not None
                                    else ()],
            "operation_queue": [Machine.__to_int_tup(o)
                                for o in self.queue],
            "machine_group":
                f'[{",".join([str(c) for c in self.capabilities])}]'
        }


class Trackers:
    def __init__(self, dims: SchedulingDimensions,
                 s_matrices: StateMatrices, s_machines: Machinery,
                 wip: list, s_original_durations: np.ndarray):
        """
        DO NOT MODIFY OR REASSIGN THE WIP!

        :param dims:
        :param s_matrices:
        :param s_machines:
        :param wip:
        :param s_original_durations:
        """
        job_durations = s_matrices.op_duration.sum(axis=1)
        # SIMPLE TRACKERS
        self.__n_completed_jobs = 0
        self.__n_completed_ops = 0
        self.__n_running_ops = 0
        self.__running_jobs = np.zeros(dims.n_jobs, dtype=bool)
        # JOBS PROGRESS TRACKERS
        self.__n_jobs_in_window = dims.n_jobs_initial
        self.__job_n_completed_ops = np.zeros(dims.n_jobs)
        # TODO: check these out!
        self.__job_n_remaining_ops = dims.n_operations.copy()
        self.__job_last_processed_time = np.zeros(dims.n_jobs)
        self.__job_first_processed_time = np.zeros(dims.n_jobs)
        self.__job_completed_time = np.zeros(dims.n_jobs)
        self.__job_completed_time_planned = np.zeros(dims.n_jobs)
        self.__job_remaining_time = job_durations
        self.__job_total_time = job_durations.copy()
        self.__job_wip_dates = np.zeros(dims.n_jobs)
        # MATRICES AND MACHINERY FOR UPDATES
        self.__matrices, self.__machines = s_matrices, s_machines
        self.__wip = wip
        # RESOURCE CENTRIC INTERMEDIARY GOAL VARIABLES
        self.__buffer_lengths = np.zeros(dims.n_machines)
        self.__buffer_times = self.__machines.buffer_times  # readonly!
        self.__utilization_times = self.__machines.utilization_times  # readonly
        self.__setup_times = self.__machines.setup_times  # readonly
        self.__remaining_processing_times = (
            self.__machines.remaining_processing_times)
        # JOB CENTRIC INTERMEDIARY GOAL VARIABLES
        self.__operation_throughput = 0
        self.__job_throughput = 0                                  # Tpt_t
        self.__job_completion = np.repeat(-1, dims.n_jobs)         # C
        self.__flow_time = np.repeat(-1, dims.n_jobs)              # F
        self.__flow_time_start_relative = np.repeat(-1, dims.n_jobs)
        self.__flow_time_view_relative = np.repeat(-1, dims.n_jobs)
        self.__tardiness = np.zeros(dims.n_jobs)                   # T
        self.__earliness = np.zeros(dims.n_jobs)                   # E
        self.__unit_cost = np.repeat(-1, dims.n_jobs)              # U
        self.__idle_time = np.repeat(-1, dims.n_jobs)              # I
        # index of completed jobs to quickly select target vars
        self.__completed_j_idxs = deque([])
        # decision trackers
        self.__n_decision_skips = 0
        self.__n_total_decisions = 0
        # other: counts of op per type
        self.__type_counts = np.bincount(
            self.__matrices.op_type[wip].ravel())
        D = self.__matrices.op_duration
        self.__duration_bins = np.linspace(0, D.max() + 1, dims.n_types + 1)
        self.__duration_counts = np.bincount(
            np.digitize(D[wip], self.__duration_bins, right=True).ravel())
        self.__initial_durations = s_original_durations
        self.__dims = dims

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)

    # <editor-fold desc="Getters">
    @property
    def initial_durations(self):
        return self.__initial_durations

    @property
    def remaining_processing_times(self):
        return self.__remaining_processing_times

    @property
    def utilization_times(self):
        return self.__utilization_times

    @property
    def buffer_times(self):
        return self.__buffer_times

    @property
    def buffer_lengths(self):
        return self.__buffer_lengths

    @property
    def setup_times(self):
        return self.__setup_times

    @property
    def operation_throughput(self):
        return self.__operation_throughput

    @property
    def job_throughput(self):
        return self.__job_throughput

    @property
    def job_completion(self):
        return self.__job_completion

    @property
    def flow_time(self):
        return self.__flow_time

    @property
    def tardiness(self):
        return self.__tardiness

    @property
    def earlyness(self):
        return self.__earliness

    @property
    def unit_cost(self):
        return self.__unit_cost

    @property
    def idle_time(self):
        return self.__idle_time

    @property
    def job_visible_dates(self):
        return self.__job_wip_dates

    @property
    def n_jobs_in_window(self):
        return self.__n_jobs_in_window

    @property
    def n_completed_jobs(self):
        return self.__n_completed_jobs

    @property
    def job_completed_time_planned(self):
        return self.__job_completed_time_planned

    @property
    def work_original_jobs(self):
        return self.__job_total_time

    @property
    def job_n_completed_ops(self):
        return self.__job_n_completed_ops

    @property
    def job_n_remaining_ops(self):
        return self.__job_n_remaining_ops

    @property
    def n_running_ops(self):
        return self.__n_running_ops

    @property
    def n_completed_ops(self):
        return self.__n_completed_ops

    @property
    def job_start_times(self):
        return self.__job_first_processed_time

    @property
    def job_last_processed_time(self):
        return self.__job_last_processed_time

    @property
    def job_completed_time(self):
        return self.__job_completed_time

    @property
    def job_remaining_time(self):
        return self.__job_remaining_time

    @property
    def type_counts(self):
        return self.__type_counts

    @property
    def duration_counts(self):
        return self.__duration_counts

    @property
    def n_decision_skips(self):
        return self.__n_decision_skips

    @property
    def n_total_decisions(self):
        return self.__n_total_decisions
    # </editor-fold>

    # <editor-fold desc="Setters">
    @n_decision_skips.setter
    def n_decision_skips(self, val):
        self.__n_decision_skips = val

    @n_total_decisions.setter
    def n_total_decisions(self, val):
        self.__n_total_decisions = val

    @buffer_lengths.setter
    def buffer_lengths(self, new_val):
        """
        Though semantically this fits with machines, this proberty will be
        available directly for setting sucht that the simulation manager can
        book a slot in the target machine buffer at transport start. Otherwise
        buffer overloads might ocur.

        :param new_val: The new parameter value
        :return: None
        """
        self.__buffer_lengths = new_val
    # </editor-fold>

    def update_on_op_completion(self, op_idx: tuple, system_time: int,
                                real_duration: int):
        # simple trackers
        self.__n_completed_ops += 1
        self.__n_running_ops -= 1
        self.__operation_throughput = (self.__n_completed_ops / system_time)
        # job trackers
        p = self.__initial_durations[op_idx]
        j_idx = op_idx[0]
        # TODO: Bugfix - in deterministic cases p != real_duration
        #  solution: track real duration in events, or better,
        #  update_elapsed_time just before zeroing out the durations
        self.__job_n_completed_ops[j_idx] += 1
        self.__job_n_remaining_ops[j_idx] -= 1
        self.__job_last_processed_time[j_idx] = system_time
        self.__job_completed_time[j_idx] += real_duration
        self.__job_completed_time_planned[j_idx] += p
        self.__job_remaining_time[j_idx] -= real_duration
        try:
            finished_type = self.__matrices.op_type[op_idx]
            self.__type_counts[finished_type] -= 1
        except IndexError:
            print('herehere')
        op_d_bin = np.digitize(self.__initial_durations[op_idx],
                               self.__duration_bins, right=True)
        self.__duration_counts[op_d_bin] -= 1
        if self.__duration_counts[op_d_bin] < 0:
            print('herehere!')
        # print("test!")
        self.__running_jobs[op_idx[0]] = False

    def update_on_job_completion(self, j_idx: int, system_time: int) -> None:
        self.__n_jobs_in_window -= 1
        self.__completed_j_idxs.append(j_idx)
        self.__n_completed_jobs += 1
        self.__job_throughput = self.__n_completed_jobs / system_time
        C, r = system_time, self.__matrices.job_arrivals[j_idx]
        s = self.__job_first_processed_time[j_idx]
        v = self.__job_wip_dates[j_idx]
        d = self.__matrices.deadlines[j_idx]
        self.__job_completion[j_idx] = C
        self.__job_last_processed_time[j_idx] = C
        self.__flow_time[j_idx] = C - r
        self.__flow_time_start_relative[j_idx] = C - s
        self.__flow_time_view_relative[j_idx] = C - v
        T, E = max(0, C - d), min(0, C - d)
        self.__tardiness[j_idx] = T
        self.__earliness[j_idx] = E
        self.__unit_cost[j_idx] = 1 if T != 0 else 0
        p = self.__job_completed_time[j_idx]
        self.__idle_time[j_idx] = C - r - p

    def update_on_op_start(self, op_idx: tuple, time: int) -> None:
        if op_idx[1] == 0:
            self.__job_first_processed_time[op_idx[0]] = time
        self.__n_running_ops += 1
        self.__job_last_processed_time[op_idx[0]] = time
        self.__running_jobs[op_idx[0]] = True

    def update_on_job_in_wip(self, j_idx: int, time: int) -> None:
        self.__job_wip_dates[j_idx] = time
        # this is the minimum pt in time where the processing could have started
        self.__job_last_processed_time[j_idx] = time
        self.__type_counts = np.bincount(
                self.__matrices.op_type[self.__wip].ravel())
        # for idx in range(self.__matrices.op_type.shape[1]):
        #     self.__type_counts[self.__matrices.op_type[(j_idx, idx)] - 1] += 1
        self.__n_jobs_in_window += 1
        filter_m = self.__matrices.op_duration.astype(bool)
        remaining_ops = (self.__initial_durations[self.__wip]
                         * filter_m[self.__wip])
        self.__duration_counts = np.bincount(
            np.digitize(
                remaining_ops,
                self.__duration_bins, right=True).ravel())

    def update_on_step_end(self, legal_actions, wait_legal,
                           in_scheduling_mode, t):
        if (in_scheduling_mode and
                (len(legal_actions) == 1 or
                 (len(legal_actions) == 2 and wait_legal))):
            self.n_decision_skips += 1
        self.n_total_decisions += 1
        self.__job_last_processed_time[self.__running_jobs] = np.repeat(
            t, self.__n_running_ops)

    def update_wip(self, new_wip):
        self.__wip = new_wip


class State:
    # noinspection DuplicatedCode
    def __init__(self, parameters: Input, system_time):
        self.__params: SchedulingDimensions = parameters.dims
        # WORK IN PROGRESS (WIP) WINDOW TRACKERS
        self.__job_view_to_global_idx = list(range(
            parameters.dims.max_jobs_visible))
        self.__job_global_to_view_idx = dict(
            zip(self.__job_view_to_global_idx, self.__job_view_to_global_idx))
        self.__system_free_job_slots = deque([])
        # MAIN STATE INFORMATION
        self.__matrices = StateMatrices(parameters)
        self.__operation_graph = OperationGraph(
            deepcopy(parameters.matrices_j.operation_precedence_l),
            parameters.dims.n_jobs_initial)
        self.__machines = Machinery(parameters.dims, self.__matrices)
        self.__trackers = Trackers(
            parameters.dims, self.__matrices, self.__machines,
            self.__job_view_to_global_idx,
            parameters.matrices_j.operation_durations.copy())
        self.__system_time = system_time
        self.__n_sequencing_steps = 0
        self.__n_routing_steps = 0
        # CURRENT DECISION MARKERS
        self.__current_machine_nr = None
        self.__current_operation = None
        self.__current_job = 0  # contains job nr in scheduling mode 1
        self.__scheduling_mode = 0  # TODO: Enums
        # KNOWN JOBS TRACKERS
        self.__system_job_queue = deque([])
        self.__system_job_nr = parameters.dims.n_jobs_initial
        # LEGAL ACTIONS
        self.__legal_actions = None

    # <editor-fold desc="Getters">
    @property
    def legal_actions(self):
        return self.__legal_actions

    @property
    def trackers(self):
        return self.__trackers

    @property
    def n_sequencing_steps(self):
        return self.__n_sequencing_steps

    @property
    def n_routing_steps(self):
        return self.__n_routing_steps

    @property
    def params(self):
        return self.__params

    @property
    def matrices(self):
        return self.__matrices

    @property
    def operation_graph(self):
        return self.__operation_graph

    @property
    def machines(self):
        return self.__machines

    @property
    def system_time(self):
        return self.__system_time

    @property
    def current_machine_nr(self):
        return self.__current_machine_nr

    @property
    def current_operation(self):
        return self.__current_operation

    @property
    def current_job(self):
        return self.__current_job

    @property
    def scheduling_mode(self):
        return self.__scheduling_mode

    @property
    def job_view_to_global_idx(self):
        return self.__job_view_to_global_idx

    @property
    def job_global_to_view_idx(self):
        return self.__job_global_to_view_idx

    @property
    def system_free_job_slots(self):
        return self.__system_free_job_slots

    @property
    def system_job_queue(self):
        return self.__system_job_queue

    @property
    def system_job_nr(self):
        return self.__system_job_nr
    # </editor-fold>

    # <editor-fold desc="Setters">
    @legal_actions.setter
    def legal_actions(self, new_val):
        self.__legal_actions = new_val

    @n_sequencing_steps.setter
    def n_sequencing_steps(self, new_val):
        self.__n_sequencing_steps = new_val

    @n_routing_steps.setter
    def n_routing_steps(self, new_val):
        self.__n_routing_steps = new_val

    @current_machine_nr.setter
    def current_machine_nr(self, new_machine_nr):
        self.__current_machine_nr = new_machine_nr

    @current_operation.setter
    def current_operation(self, new_op):
        self.__current_operation = new_op

    @current_job.setter
    def current_job(self, new_job):
        self.__current_job = new_job

    @scheduling_mode.setter
    def scheduling_mode(self, new_mode):
        self.__scheduling_mode = new_mode
    # </editor-fold>

    def update_system_time(self, time):
        try:
            assert time >= self.__system_time
        except AssertionError:
            # print('The event should have occured before the current one!')
            pass
        time_elapsed = time - self.__system_time
        self.__system_time = time
        return time_elapsed

    def get_global_op_index(self, action):
        job_idx_view = action // self.__params.max_n_operations
        job_op_idx = action % self.__params.max_n_operations
        job_idx_global = self.__job_view_to_global_idx[job_idx_view]
        return job_idx_global, job_op_idx

    def queue_job(self, j_idx):
        self.__system_job_queue.append(j_idx)

    def purge_job_queue(self):
        self.__system_job_queue = []

    def get_next_job(self):
        return self.__system_job_queue.popleft()

    def mark_completed_job(self, finished_job_idx: int) -> int:
        """
        Called upon job completion. Removes the finished job from the graph,
        frees the corresponding view slot and returns the index of the next job
        if one is present in the system.

        :param finished_job_idx: Index of the finished job.
        :return: The index of the next job from the job queue or -1 if none is
            present.
        """
        self.__operation_graph.remove_job(finished_job_idx)
        freed_view_slot = self.__job_global_to_view_idx[finished_job_idx]
        self.__system_free_job_slots.append(freed_view_slot)
        del self.__job_global_to_view_idx[finished_job_idx]
        self.__system_job_nr -= 1
        if len(self.__system_job_queue) != 0:        # next job is not here
            j_next = self.__system_job_queue.popleft()  # global idx
        else:
            j_next = -1
        self.trackers.update_on_job_completion(
            finished_job_idx, self.system_time)
        return j_next

    def job_finished(self, j_idx):
        """
        Checks whether the job finished by reading the corresponding remaining
        operations tracker.

        :param j_idx: The index of the job queried.
        :return: True if there are no more operations remaining false otherwise.
        """
        if self.trackers.job_n_remaining_ops[j_idx] == 0:
            return True
        else:
            return False

    def pull_to_view(self, j_next_idx, view_position):
        self.__job_view_to_global_idx[view_position] = j_next_idx
        self.__job_global_to_view_idx[j_next_idx] = view_position
        self.__trackers.update_wip(self.__job_view_to_global_idx)

    def get_view_indices(self):
        return self.__job_view_to_global_idx

    def get_current_job_view(self):
        try:
            return self.__job_global_to_view_idx[self.__current_job]
        except KeyError:
            if self.matrices.op_type[self.current_job][-1] != 0:
                raise KeyError

    def has_free_view_slots(self):
        return bool(self.__system_free_job_slots)

    def in_scheduling_mode(self):
        return self.__scheduling_mode == 0

    def in_postbuffer_routing_mode(self):
        return self.__scheduling_mode == 1

    def in_prebuffer_routing_mode(self):
        return self.__scheduling_mode == 2

    def setup_machine_assignment_decision(self):
        m_nr = self.__machines.get_next_machine_assignment_tgt()
        self.__current_operation = None
        self.__current_machine_nr = m_nr
        self.__scheduling_mode = 0

    def setup_routing_decision(self, next_m_nr, next_j):
        self.__current_machine_nr, self.__current_job = next_m_nr, next_j
        self.__current_operation = None
        self.__scheduling_mode = 1

    def setup_failure_routing_decision(self, next_m_nr, next_op):
        self.__current_machine_nr, self.__current_operation = next_m_nr, next_op
        self.__current_job = None
        self.__scheduling_mode = 2

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)
