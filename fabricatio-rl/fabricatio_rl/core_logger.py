import os
from json import dump
from pathlib import Path

import numpy as np
import pandas as pd

from fabricatio_rl.core_state import State
from fabricatio_rl.env_utils import create_folders


class SchedulingLogger:
    def __init__(self, logdir, seed, state: State, simulation_manager):
        self.logdir = logdir
        if self.logdir == 'internal':
            sim_data_dir = (f'{Path(__file__).resolve().parent}'
                            f'/visualization/sim_data')
            run_nr = len(os.listdir(sim_data_dir))
            self.logdir = f'{sim_data_dir}/run_{run_nr}'
            create_folders(f'{self.logdir}/dummy')
            self.on = True
        elif logdir != '':
            create_folders(f'{logdir}/dummy')
            self.on = True
        else:
            self.on = False
        self.seed = seed
        self.state = state
        self.sim_manager = simulation_manager
        self.action_log = None
        self.schedule = []
        # mgmt nfo
        self.legal_actions = []
        self.action_taken = None
        self.current_machine = None
        self.current_operation = None
        self.wait_scheduling = (state.params.max_jobs_visible *
                                state.params.max_n_operations)
        self.current_operation_alternatives = None

    def turn_off(self):
        self.on = False

    def turn_on(self):
        self.on = True

    def add_to_schedule(self, machine_nr, current_time,
                        operation_finish_time, job_nr):
        if not self.on:
            return
        self.schedule.append(
            (f'M{machine_nr}', int(current_time),
             int(operation_finish_time), int(job_nr)))

    def create_action_log(self, action, legal_actions):
        self.current_machine = self.state.current_machine_nr
        self.current_operation = self.state.current_operation
        if not self.on:
            return
        if self.state.in_scheduling_mode():
            self.legal_actions = []
            for a in legal_actions:
                if a == self.sim_manager.wait_scheduling:
                    self.legal_actions.append((0, a))
                else:
                    self.legal_actions.append(self.state.get_global_op_index(a))
            self.action_taken = (0, action) \
                if action == self.sim_manager.wait_scheduling \
                else self.state.get_global_op_index(action)
        elif self.state.in_postbuffer_routing_mode():
            self.current_operation = (list(
                self.state.operation_graph.get_next_ops(
                    self.state.current_job)))
            self.legal_actions = [str(a) for a in legal_actions]
            self.action_taken = action
            self.current_operation = self.sim_manager.get_transport_tgt_op(
                action)
            self.current_operation_alternatives = (list(
                self.state.operation_graph.get_next_ops(
                    self.state.current_job)))
        else:  # self.scheduling_mode == 2, transport from machine breakdown
            self.legal_actions = legal_actions
            self.action_taken = action

    def write_logs(self):
        if not self.on:
            return
        decision_queues = dict(
            scheduling_queue=[int(i) for i in
                              self.state.machines.scheduling_queue],
            routing_queue=[SchedulingLogger.__to_int_tup(t)
                           for t in self.sim_manager.routing_queue])
        machine_metrics, job_metrics = self.__write_kpis()
        state_representation = dict(
            management_info=self.__write_nfo(decision_queues),
            precedence_graphs=self.__write_graphs(),
            partial_schedule=self.__write_gantt(),
            machines=self.__write_machines(self.action_log),
            matrices=self.__write_heatmaps(),
            pending_events=self.__write_events(),
            metrics_machines=machine_metrics,
            metrics_jobs=job_metrics
        )
        f_name = (f'{self.logdir}/'
                  f'{str(self.state.trackers.n_total_decisions)}.json')
        state_file = open(create_folders(f_name), 'w')
        dump(state_representation, state_file)
        state_file.close()

    def __write_nfo(self, decision_queues):
        nfo_dict = decision_queues  # beware of mutability...
        sched_mode_map = {
            0: 'Sequencing', 1: 'Routing', 2: 'Breakdown Handling'}
        nfo_dict = dict(
            scheduling_queue=decision_queues['scheduling_queue'],
            routing_queue=decision_queues['routing_queue'],
            scheduling_mode=sched_mode_map[self.state.scheduling_mode],
            current_operation=self.current_operation,
            current_machine=self.current_machine,
            current_job=self.state.current_job,
            current_operation_alternatives=self.current_operation_alternatives,
            legal_actions=self.legal_actions,
            action_taken=self.action_taken,
            wait_scheduling=self.wait_scheduling,
            system_time=float(self.state.system_time),
            wip_indices=list(sorted(self.state.job_view_to_global_idx))
        )
        return nfo_dict

    def __write_kpis(self):
        t = self.state.system_time
        utl_total = self.state.machines.utilization_times
        utl_rate = utl_total / t if t > 0 else utl_total
        bft = self.state.machines.buffer_times
        bft_sum = self.state.machines.buffer_times.sum()
        bf_rel_load = bft / bft_sum if bft_sum > 0 else bft
        machines_df = pd.DataFrame({
            'Machine Index': list(range(1, utl_rate.shape[0] + 1)),
            'Utilization Rate': utl_rate,
            'Relative Buffer Times': bf_rel_load
        })
        w_remaining = self.state.trackers.job_remaining_time
        work_time_last = self.state.trackers.job_last_processed_time
        work_time_start = self.state.trackers.job_start_times
        work_release_time = self.state.trackers.job_visible_dates
        ops_left = self.state.trackers.job_n_remaining_ops
        minimum_completion_time = np.array(
            [t + w_remaining[i] if ops_left[i] > 0 else work_time_last[i]
             for i in range(w_remaining.shape[0])])
        start_relative_flow_time = minimum_completion_time - work_time_start
        release_relative_flow_time = minimum_completion_time - work_release_time
        jobs_released = release_relative_flow_time < minimum_completion_time
        jobs_df = pd.DataFrame({
            'Job Index': list(range(0, minimum_completion_time.shape[0])),
            'Estimated Completion': minimum_completion_time,
            'Start Rel. Flow Time': start_relative_flow_time,
            'Release Rel. Flow Time': release_relative_flow_time,
            'Jobs Visible': jobs_released
        })
        return (machines_df.to_dict(orient="records"),
                jobs_df.to_dict(orient="records"))

    def __write_gantt(self):
        try:
            schedule_df = pd.DataFrame(self.schedule).sort_values(0, axis=0)
        except KeyError:  # empty dataframe ;)
            return {}
        return schedule_df.to_dict(orient="records")

    def __write_graphs(self):
        rt_v_name = self.state.operation_graph.root_visible.op_index
        rt_h_name = self.state.operation_graph.root_hidden.op_index
        nodes_visible = [{"id": hash(rt_v_name), "name": str(rt_v_name)}]
        links_visible = []
        visited = {rt_v_name, rt_h_name}
        for k, v in self.state.operation_graph.root_visible.next.items():
            links_visible.append({"source": hash(rt_v_name), "target": hash(k)})
            v.build_graph_dict(nodes_visible, links_visible, visited)
        nodes_hidden = [{"id": hash(rt_h_name), "name": str(rt_h_name)}]
        links_hidden = []
        for k, v in self.state.operation_graph.root_hidden.next.items():
            links_hidden.append({"source": hash(rt_h_name), "target": hash(k)})
            v.build_graph_dict(nodes_hidden, links_hidden, visited)
        f_prec_data = self.state.operation_graph.to_dict()
        return dict(visible={"nodes": nodes_visible, "links": links_visible},
                    hidden={"nodes": nodes_hidden, "links": links_hidden})

    def __write_machines(self, action_log):
        f_mach_data = self.state.machines.to_dict()
        if self.state.in_postbuffer_routing_mode():  # routing
            job_nr = self.state.current_job
            next_ops = self.state.operation_graph.get_next_ops(job_nr)
            self.__fill_routing_data(next_ops, f_mach_data)
        elif self.state.in_scheduling_mode():  # scheduling
            f_mach_data["links"] = [dict(
                source=int(self.current_machine),
                target=int(self.current_machine),
                route_chosen=None,
                op_routed=self.action_taken
                if not self.sim_manager.is_wait_processing(self.action_taken[1])
                else "Wait")]
        else:  # routing at breakdown; mode contains the op to be routed
            op_routed = -1  # TODO: implement!
            self.__fill_routing_data([op_routed], f_mach_data)
        return f_mach_data

    def __fill_routing_data(self, next_ops: list, machine_dict: dict):
        machine_dict["links"] = []
        for op_idx in next_ops:
            # every op in transit at most once
            op_t = self.state.matrices.op_type[op_idx]
            eligible_ms = self.state.matrices.machine_capab_dt[op_t]
            op_acts = list(eligible_ms)
            for action in op_acts:
                try:
                    assert str(action) in self.legal_actions
                except AssertionError:
                    print('cast necessary')
                if action == self.action_taken:
                    machine_dict["links"].append(dict(
                        source=int(self.current_machine),
                        target=int(self.action_taken),
                        route_chosen=True,
                        op_routed=op_idx))
                else:
                    machine_dict["links"].append(dict(
                        source=int(self.current_machine),
                        target=int(action),
                        route_chosen=False,
                        op_routed=op_idx))

    def __write_heatmaps(self):
        """
        Creates a json file with the information from the state matrices. The
        file can be used to create heatmap visualizations for the individual
        matrices. The json structure is as follows:
        {matrix_name_1: {
            data: nested list of values
            min_value: minimum value over the lists
            max_value: max value over the lists
            x_label: the column title
            y_label: the row title
            n_rows: the number of rows in the matrix
            n_cols: the number of columns in the matrix
            nfo_type: the information category; either 'jobs', 'tracking',
                'machines'
            }
            matrix_name_1: {
                ...
            }
            ...
        }
        :return:
        """
        # f_name = f'{self.log_dir}/heatmaps/{str(self.state.n_steps)}.json'
        # f_matrices = open(create_folders(f_name), 'w')
        # dump(self.state.matrices.to_dict(), f_matrices)
        # f_matrices.close()
        return self.state.matrices.to_dict()

    def __write_events(self):
        log = []
        for event in self.sim_manager.event_heap.event_heap:
            e_type = event.__class__.__name__
            e_time = int(event.occurence_time)
            e_next = event.trigger_next
            e_j_idx = int(event.job_index) \
                if hasattr(event, 'job_index') else None
            if hasattr(event, 'operation_index'):
                e_o_idx = event.operation_index
                e_o_idx = tuple([int(idx) for idx in e_o_idx])
            else:
                e_o_idx = None
            if hasattr(event, 'machine_nr'):
                e_m_idx = int(event.machine_nr)
            elif hasattr(event, 'tgt_machine_number'):
                e_m_idx = int(event.tgt_machine_number)
            else:
                e_m_idx = None
            log.append(dict(type=e_type, time=e_time, job=e_j_idx,
                            op=e_o_idx,
                            m=e_m_idx, next=e_next))
        # TODO: figure out why i did this...
        self.sim_manager.event_heap.flush_event_log()
        return log

    @staticmethod
    def __to_int_tup(tup):
        return int(tup[0]), int(tup[1])
