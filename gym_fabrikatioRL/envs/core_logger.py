from gym_fabrikatioRL.envs.env_utils import create_folders
from ast import literal_eval as make_tuple
from json import dumps, dump
import os
import pandas as pd
import numpy as np


class SchedulingLogger:
    def __init__(self, filepath, seed, state, simulation_manager):
        if filepath != '':
            self.log_dir = '/'.join(filepath.split('/')[:-1])
            create_folders(self.log_dir)
            # write logger object
            self.logfile = f'{filepath}_s{seed}.log'
            self.on = True
        else:
            self.on = False
        self.seed = seed
        self.state = state
        self.sim_manager = simulation_manager
        self.action_log = None
        self.schedule = []

    def turn_off(self):
        self.on = False

    def add_to_schedule(self, machine_nr, current_time,
                        operation_finish_time, job_nr):
        if not self.on:
            return
        self.schedule.append(
            (f'M{machine_nr}', int(current_time),
             int(operation_finish_time), int(job_nr)))

    def create_action_log(self, action, legal_actions):
        if not self.on:
            return
        self.action_log = ''
        if self.state.in_scheduling_mode():
            legal_action_repr = []
            for a in legal_actions:
                if a == self.sim_manager.wait_scheduling:
                    legal_action_repr.append(str((0, a)))
                else:
                    legal_action_repr.append(str(
                        self.state.get_global_op_index(a)))
            action_taken = (self.state.get_global_op_index(action)
                            if action != self.sim_manager.wait_scheduling
                            else (0, action))
            self.action_log += ','.join(legal_action_repr)
            self.action_log += (f'|{action_taken}'
                                f'|{self.state.current_machine_nr}'
                                f'|None')
        elif self.state.in_postbuffer_routing_mode():
            self.action_log += ','.join([str(a) for a in legal_actions])
            self.action_log += (f'|{action}|{self.state.current_machine_nr}'
                                f'|{self.state.current_job}')
        else:  # self.scheduling_mode == 2, transport from machine breakdown
            self.action_log += ','.join([str(a) for a in legal_actions])
            self.action_log += (f'|{action}|{self.state.current_machine_nr}'
                                f'|{self.state.current_operation}')

    def write_logs(self):
        if not self.on:
            return
        decision_queues = {
            "scheduling_queue":
                [int(i) for i in self.state.machines.scheduling_queue],
            "routing_queue":
                [SchedulingLogger.__to_int_tup(t) for t in
                 self.sim_manager.routing_queue]
        }
        step = str(self.state.n_steps).zfill(3)
        self.__write_global_logs(step, self.action_log, decision_queues)
        self.__write_nfo(decision_queues)
        self.__write_graphs()
        self.__write_gantt()
        self.__write_machines(self.action_log)
        self.__write_heatmaps()
        self.__write_events()
        self.__write_kpis()

    def __write_nfo(self, decision_queues):
        nfo_dict = decision_queues  # beware of mutability...
        nfo_dict['system_time'] = int(self.state.system_time)
        sched_mode_map = {
            0: 'Sequencing', 1: 'Routing', 2: 'Breakdown Handling'}
        nfo_dict['scheduling_mode'] = sched_mode_map[self.state.scheduling_mode]
        f_name = f'{self.log_dir}/nfo/{str(self.state.n_steps)}.json'
        f_nfo = open(create_folders(f_name), 'w')
        dump(nfo_dict, f_nfo)
        f_nfo.close()

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
        f_name = f'{self.log_dir}/kpis/{str(self.state.n_steps)}_kpi_m.csv'
        machines_df.to_csv(create_folders(f_name), index=False, header=True)
        w_remaining = self.state.matrices.trackers.job_remaining_time
        work_time_last = self.state.matrices.trackers.job_last_processed_time
        work_time_start = self.state.matrices.trackers.job_start_times
        work_release_time = self.state.matrices.trackers.job_visible_dates
        ops_left = self.state.matrices.trackers.n_remaining_ops
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
        jobs_df.to_csv(
            f'{self.log_dir}/kpis/{str(self.state.n_steps)}_kpi_j.csv',
            index=False, header=True)

    def __write_gantt(self):
        f_name = f'{self.log_dir}/scheduling/{str(self.state.n_steps)}.csv'
        pd.DataFrame(self.schedule).to_csv(create_folders(f_name),
                                           index=False, header=False)

    def __write_graphs(self):
        f_name = f'{self.log_dir}/graphData/{str(self.state.n_steps)}.json'
        f_prec = open(create_folders(f_name), 'w')
        f_prec_data = self.state.operation_graph.to_dict()
        dump(f_prec_data, f_prec)
        f_prec.close()

    def __write_machines(self, action_log):
        f_name = f'{self.log_dir}/machines/{str(self.state.n_steps)}.json'
        f_mach = open(create_folders(f_name), 'w')
        f_mach_data = self.state.machines.to_dict()
        legal_acts, act, current_mach, mode_object = action_log.split('|')
        mode = make_tuple(mode_object)
        if type(mode) == int:  # routing
            job_nr = mode
            next_ops = self.state.operation_graph.get_next_ops(job_nr)
            self.__fill_routing_data(legal_acts, current_mach, next_ops,
                                     f_mach_data, act)
        elif mode is None:  # scheduling
            f_mach_data["links"] = {
                "source": int(current_mach),
                "target": int(current_mach),
                "label": f"Scheduled {act}"
                if not self.sim_manager.is_wait_processing(make_tuple(act)[1])
                else f"Decided to wait"
            }
        else:  # routing at breakdown; mode contains the op to be routed
            op_routed = mode
            self.__fill_routing_data(legal_acts, current_mach, [op_routed],
                                     f_mach_data, act)
        dump(f_mach_data, f_mach)
        f_mach.close()

    def __fill_routing_data(self, legal_actions_string: str,
                            current_machine: str, next_ops: list,
                            machine_dict: dict, action_chosen: str
                            ):
        legal_actions = legal_actions_string.split(',')
        machine_dict["links"] = []
        for op_idx in next_ops:
            # every op in transit at most once
            op_t = self.state.matrices.op_type[op_idx]
            eligible_ms = self.state.matrices.machine_capab_dt[op_t]
            op_acts = list(eligible_ms)
            for action in op_acts:
                try:
                    assert str(action) in legal_actions
                except AssertionError:
                    print('cast necessary')
                if action == action_chosen:
                    machine_dict["links"].append({
                        "source": int(current_machine),
                        "target": int(action_chosen),
                        "label": f"Route Chosen for {op_idx}"
                    })
                else:
                    machine_dict["links"].append({
                        "source": int(current_machine),
                        "target": int(action),
                        "label": f"Route Alternative for {op_idx}"
                    })

    def __write_heatmaps(self):
        for m_name, matrix in self.state.matrices.to_dict().items():
            f_base = f'{self.log_dir}/heatmaps/{m_name}/'
            if not os.path.exists(f_base):
                os.makedirs(f_base)
            pd.DataFrame(matrix).to_csv(
                f'{f_base}'
                f'{str(self.state.n_steps)}.csv',
                index=False, header=False)

    def __write_events(self):
        events_path = f'{self.log_dir}/events/{str(self.state.n_steps)}.txt'
        if os.path.exists(events_path):
            os.remove(events_path)
        f_events = open(create_folders(events_path), 'a')
        f_events_data = self.sim_manager.event_heap.get_heap_representation()
        for event in f_events_data.split('|'):
            f_events.write(event + '\n')
        f_events.close()
        self.sim_manager.event_heap.flush_event_log()

    def __write_global_logs(self, step, action_log, decision_queues):
        logfile = open(self.logfile, 'a')
        logfile.write(f'GRAPH{step}:::{self.state.operation_graph}\n')
        logfile.write(f'STATE{step}:::{self.state.matrices}\n')
        logfile.write(f'MACHINES{step}:::{self.state.machines}|{action_log}\n')
        logfile.write(f'DECISIONS{step}:::{dumps(decision_queues)}\n')
        logfile.write(f'SCHEDULE{step}:::{dumps(self.schedule)}|'
                      f'{self.state.system_time}\n')
        logfile.write(f'EVENTS{step}:::'
                      f'{self.sim_manager.event_heap.get_heap_representation()}'
                      f'\n')
        logfile.close()

    @staticmethod
    def __to_int_tup(tup):
        return int(tup[0]), int(tup[1])
