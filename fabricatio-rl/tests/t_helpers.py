from ast import literal_eval as make_tuple
from mmap import mmap

import gym
import numpy as np
from gym import register

from fabricatio_rl.core_state import State
from fabricatio_rl.env_utils import UndefinedLegalActionCall
from fabricatio_rl.interface_templates import ReturnTransformer, \
    Optimizer, SchedulingUserInputs


class StochasticityTestTransformer(ReturnTransformer):
    def transform_state(self, state: State):
        O_D = state.matrices.op_duration
        O_P = state.matrices.op_prec_m
        O_T = state.matrices.op_type
        M_Tr = state.matrices.transport_times
        M_Ty = state.matrices.machine_capab_cm
        L = state.matrices.op_location
        S = state.matrices.op_status
        machine_nr = state.current_machine_nr
        current_j_nr = state.current_job
        simulation_mode = state.scheduling_mode
        return np.concatenate([
            O_D.flatten(), O_P.flatten(), O_T.flatten(),  M_Tr.flatten(),
            M_Ty.flatten(),  L.flatten(), S.flatten(),
            np.array([machine_nr, current_j_nr, simulation_mode])])

    def transform_reward(self, state: State, illegal=False, environment=None):
        return state.trackers.utilization_times.mean()


class OperationDurationSequencingHeuristic(Optimizer):
    def __init__(self):
        self._duration_sorted_operations = []
        super().__init__('sequencing')

    def _sort_actions(self, state: State):
        legal_actions = state.legal_actions
        op_durations = state.matrices.op_duration
        n_jobs = state.params.n_jobs
        n_j_operations = state.params.max_n_operations
        from collections import namedtuple
        Operation = namedtuple('Oeration', 'index duration')
        ops = []
        wait_flag = n_j_operations * n_jobs
        for action_nr in legal_actions:
            if action_nr == wait_flag:
                continue
            j_idx = action_nr // n_j_operations
            op_idx = action_nr % n_j_operations
            d = op_durations[j_idx][op_idx]
            ops.append(
                Operation(index=j_idx * n_j_operations + op_idx, duration=d))
        self._duration_sorted_operations = sorted(
            ops, key=lambda x: x.duration)


class LPTOptimizer(OperationDurationSequencingHeuristic):
    def __init__(self):
        super().__init__()

    def get_action(self, state: State):
        self._sort_actions(state)
        if len(self._duration_sorted_operations) != 0:
            action = self._duration_sorted_operations[-1]
        else:
            return state.params.max_n_operations * state.params.max_jobs_visible
        return action.index


class SPTOptimizer(OperationDurationSequencingHeuristic):
    def __init__(self):
        super().__init__()

    def get_action(self, state: State):
        self._sort_actions(state)
        if len(self._duration_sorted_operations) != 0:
            action = self._duration_sorted_operations[-1]
        else:
            return state.params.max_n_operations * state.params.max_jobs_visible
        return action.index


class LDOptimizer(Optimizer):
    """
    Least Distance Heuristic: Select the closest machine for transport.
    """
    def __init__(self):
        super().__init__('transport')

    def get_action(self, state: State):
        legal_actions = state.legal_actions
        src_machine = state.current_machine_nr
        transport_matrix = state.matrices.transport_times
        from collections import namedtuple
        Operation = namedtuple('Route', 'index distance')
        ops = []
        for tgt_machine in legal_actions:
            ops.append(
                Operation(index=tgt_machine,
                          distance=transport_matrix[src_machine][tgt_machine]))
        action = sorted(ops, key=lambda x: x.distance)[0]
        if action.index is None:
            print('herehere!')
        return action.index


class MDOptimizer(Optimizer):
    """
    Most Distance Heuristic: Select the furthest machine for transport.
    """
    def __init__(self):
        super().__init__('transport')

    def get_action(self, state: State):
        legal_actions = state.legal_actions
        src_machine = state.current_machine_nr
        transport_matrix = state.matrices.transport_times
        from collections import namedtuple
        Operation = namedtuple('Route', 'index distance')
        ops = []
        for tgt_machine in legal_actions:
            ops.append(
                Operation(index=tgt_machine,
                          distance=transport_matrix[src_machine][tgt_machine]))
        action = sorted(ops, key=lambda x: x.distance)[-1]
        if action.index is None:
            print('herehere!')
        return action.index


def init_stochasticity_test_env(seeds, optimizers, name,
                                all_jobs_visible=False, logdir=''):
    if all_jobs_visible:
        n_jobs_initial = n_jobs = max_jobs_visible = 20
    else:
        n_jobs_initial = 10
        n_jobs = 20
        max_jobs_visible = 15
    env_args = dict(
        scheduling_inputs=[SchedulingUserInputs(
            n_jobs=n_jobs,  # n
            n_machines=7,  # m
            n_tooling_lvls=5,  # l
            n_types=5,  # tests_env
            min_n_operations=5,
            max_n_operations=5,  # o
            n_jobs_initial=n_jobs_initial,  # jobs with arrival time 0
            max_jobs_visible=max_jobs_visible,  # entries in {1 .. n}
            operation_precedence='POm',
            operation_types='Jm',  # deafault_sampling
            operation_tool_sets='default_sampling',
            machine_speeds='default_sampling',
            machine_distances='default_sampling',
            machine_buffer_capacities=np.array([5, 6, 9, 7, 15, 8, 1]),
            # '' or nxo matrix (entries in {1, 2 .. })
            machine_capabilities={
                1: {1}, 2: {1}, 3: {2, 3}, 4: {3}, 5: {4}, 6: {5}, 7: {3}},
            tool_switch_times='default_sampling',
            inter_arrival_time='balanced',
            due_dates='default_sampling',
            perturbation_processing_time='default_sampling',
            perturbation_due_date='default_sampling',
        )],
        seeds=seeds,
        return_transformer=StochasticityTestTransformer(),
        selectable_optimizers=optimizers,
        logfile_path=logdir)
    if name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[name]
    register(id=name,
             entry_point='fabricatio_rl:FabricatioRL', kwargs=env_args)
    env = gym.make(name)
    return env


def get_envs_selectable_seq_opt(seed):
    # MULTIPLE SEQUENCING OPTIMIZERS
    # # sequencing optimizer, fixed transport optimizer actions
    # env_0 = init_stochasticity_test_env(
    #     [seed],
    #     [LPTOptimizer(), SPTOptimizer(), LDOptimizer()],
    #     'fabricatio-v0')
    # sequencing optimizer, raw transport actions
    env_1 = init_stochasticity_test_env(
        [seed],
        [LPTOptimizer(), SPTOptimizer()],
        'fabricatio-v1')
    # sequencing optimizer, transport optimizer actions
    env_2 = init_stochasticity_test_env(
        [seed],
        [LPTOptimizer(), SPTOptimizer(), LDOptimizer(), MDOptimizer()],
        'fabricatio-v2')
    return [env_1, env_2]


def get_envs_fixed_seq_opt(seed):
    # SINGLE SEQUENCING OPTIMIZER
    # fixed sequencing optimizer, fixed transport optimizer actions
    # env_0 = init_stochasticity_test_env(
    #     [seed],
    #     [LPTOptimizer(), LDOptimizer()],
    #     'fabricatio-v3')
    # fixed sequencing optimizer, raw transport actions
    env_1 = init_stochasticity_test_env(
        [seed],
        [LPTOptimizer()],
        'fabricatio-v4')
    # fixed sequencing optimizer, transport optimizer actions
    env_2 = init_stochasticity_test_env(
        [seed],
        [LPTOptimizer(), LDOptimizer(), MDOptimizer()],
        'fabricatio-v5')
    return [env_1, env_2]


def get_envs_raw_seq(seed):
    # SINGLE SEQUENCING OPTIMIZER
    # fixed sequencing optimizer, fixed transport optimizer actions
    # env_0 = init_stochasticity_test_env(
    #     [seed],
    #     [LDOptimizer()],
    #     'fabricatio-v6')
    # fixed sequencing optimizer, raw transport actions
    env_1 = init_stochasticity_test_env(
        [seed],
        [],
        'fabricatio-v7')
    # fixed sequencing optimizer, transport optimizer actions
    # env_2 = init_stochasticity_test_env(
    #     [seed],
    #     [LDOptimizer(), MDOptimizer()],
    #     'fabricatio-v8')
    return [env_1]


def get_all_env_interface_configs(seed):
    all_envs = []
    all_envs += get_envs_selectable_seq_opt(seed)
    all_envs += get_envs_fixed_seq_opt(seed)
    all_envs += get_envs_raw_seq(seed)
    return all_envs


def record_sim_on_first_last_legal_action_choice(env):
    states = []
    rewards = []
    actions = []
    state, done = env.reset(), False
    while not done:
        try:
            action = env.get_legal_actions()[0]
            state, reward, done, _ = env.step(action)
        except UndefinedLegalActionCall(
                env.optimizer_configuration,
                env.core.state.scheduling_mode):
            # happens for fixed optimizers, in which case we step with
            # the only opssible action
            action = 0
            state, reward, done, _ = env.step(action)
        states.append(state)
        rewards.append(reward)
        actions.append(action)
    return states, actions, rewards


class MakespanExtractor(ReturnTransformer):
    def transform_state(self, state: State) -> np.ndarray:
        return np.zeros(1)

    def transform_reward(self, state: 'State', illegal=False, environment=None):
        return state.system_time


class AssignedOperation:
    """
    Class used to bundle up solver values and (@see ModelOperation) and state
    values (@see CPOperation). The equality operators are implemented to
    enable heapsort.
    """
    def __init__(self, start, job_idx, op_idx, n_o):
        self.start = start
        self.job_idx = job_idx
        self.op_idx = op_idx
        self.action = n_o * job_idx + op_idx

    def __str__(self):
        return str(self.action)

    def __repr__(self):
        return str(self.action)


class FixedPlanControl(Optimizer):
    @staticmethod
    def __count_lines(filename):
        with open(filename, "r+") as f:
            buf = mmap(f.fileno(), 0)
            lines = 0
            readline = buf.readline
            while readline():
                lines += 1
            return lines

    def __init__(self, schedule_path=''):
        super().__init__('sequencing')
        self.machine_to_assigned_tasks = {}
        # Operation = namedtuple(
        #     'Operation', [
        #         'j_index', 'op_index', 'm_idx', 'start_time'])
        n_machines = FixedPlanControl.__count_lines(schedule_path)
        with open(schedule_path) as schedule:
            j_idx = 0
            for line in schedule:
                ops = line.split(' ')
                for i in range(len(ops)):
                    op = make_tuple(ops[i])
                    m_idx = op[2]
                    # idxed_op = Operation(*op)
                    idxed_op = AssignedOperation(
                        start=op[3],
                        job_idx=op[0],
                        op_idx=op[1],
                        n_o=n_machines
                    )
                    if m_idx in self.machine_to_assigned_tasks:
                        self.machine_to_assigned_tasks[m_idx].append(idxed_op)
                    else:
                        self.machine_to_assigned_tasks[m_idx] = [idxed_op]
                j_idx += 1
        for m_idx in self.machine_to_assigned_tasks.keys():
            self.machine_to_assigned_tasks[m_idx] = sorted(
                self.machine_to_assigned_tasks[m_idx], key=lambda x: x.start)

    def get_action(self, state: State) -> (int, int):
        legal_actions = state.legal_actions
        if state.scheduling_mode == 1:  # routing
            # in this setup routing decisions should not be made
            raise ValueError
        else:
            m_idx = state.current_machine_nr
            n_ops = state.params.max_n_operations
            n_jobs = state.params.max_jobs_visible
            wait = n_jobs * n_ops
            if not any(self.machine_to_assigned_tasks[m_idx]):
                return legal_actions[0]
            action = self.machine_to_assigned_tasks[m_idx][0].action
            if action in legal_actions:
                self.machine_to_assigned_tasks[m_idx].pop(0)
                return action
            if wait in legal_actions:
                return wait
            else:
                raise ValueError


def get_env_parameters(env_seed, f_name, sln_name, logpath=''):
    env_args = dict(
        scheduling_inputs=[SchedulingUserInputs(
            path=f_name,
            operation_types='Jm',
            operation_precedence='Jm',
            machine_capabilities=None,
            fixed_instance=True
        )],
        return_transformer=MakespanExtractor(),
        selectable_optimizers=[
            # FixedCPPlanSequencing(time_limit=60, schedule_path=sln_name),
            FixedPlanControl(schedule_path=sln_name)
        ],
        seeds=[env_seed],  # not gonna count, since deterministic
        logfile_path=logpath
    )
    return env_args


def init_env(env_seed, benchmark_f_name, sln_f_name, logpath=''):
    args_dict = get_env_parameters(
        env_seed, benchmark_f_name, sln_f_name, logpath)
    env_name = 'fabricatio-v2'
    if env_name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[env_name]
    register(
        id=env_name,
        entry_point='fabricatio_rl:FabricatioRL',
        kwargs=args_dict
    )
    environment = gym.make(env_name)
    return environment
