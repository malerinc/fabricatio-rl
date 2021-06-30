import numpy as np

from gym_fabrikatioRL.envs.interface_templates import (
    ReturnTransformer, Optimizer)
from gym_fabrikatioRL.envs.core_state import State
from gym_fabrikatioRL.envs.env_utils import UndefinedLegalActionCall

from time import time
from os import listdir
from collections import namedtuple
import gym
from gym import register
from ast import literal_eval as make_tuple


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


class StochasticityTestTransformer(ReturnTransformer):
    @staticmethod
    def transform_state(state: State):
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

    @staticmethod
    def transform_reward(state: State):
        return state.trackers.utilization_times.mean()


class MakespanExtractor(ReturnTransformer):
    @staticmethod
    def transform_state(state: State) -> np.ndarray:
        return np.zeros(1)


class FixedPlanControl(Optimizer):
    def __init__(self, schedule_path=''):
        super().__init__('sequencing')
        self.planner = None
        self.n_new_jobs = -1
        self.machine_schedules = {}
        Operation = namedtuple(
            'Operation', [
                'j_index', 'op_index', 'm_idx', 'start_time'])
        with open(schedule_path) as schedule:
            j_idx = 0
            for line in schedule:
                ops = line.split(' ')
                for i in range(len(ops)):
                    op = make_tuple(ops[i])
                    idxed_op = Operation(*op)
                    if idxed_op.m_idx in self.machine_schedules:
                        self.machine_schedules[idxed_op.m_idx].append(idxed_op)
                    else:
                        self.machine_schedules[idxed_op.m_idx] = [idxed_op]
                j_idx += 1
        for m_idx in self.machine_schedules.keys():
            self.machine_schedules[m_idx] = sorted(
                self.machine_schedules[m_idx], key=lambda x: x.start_time)

    def get_action(self, state: State) -> (int, int):
        # automaticaly
        m_idx = state.current_machine_nr
        n_ops = state.params.n_types
        n_jobs = state.params.n_jobs
        wait = n_jobs * n_ops
        if m_idx in self.machine_schedules:
            next_ops = self.machine_schedules[m_idx]
            if not any(next_ops):
                return wait
            next_op = self.machine_schedules[m_idx][0]
            action = next_op.j_index * n_ops + next_op.op_index
            if action in state.legal_actions:
                if len(self.machine_schedules[m_idx]) > 0:
                    self.machine_schedules[m_idx].pop(0)
                return action
        # assert wait in legal_actions
        return wait


def get_env_parameters(env_seed, f_name, sln_name, logpath=''):
    info = extract_jssp_benchmark_info(f_name)
    n_types, n_jobs_total, op_sequences, op_durations = info
    f_name, f_extension = f_name[3:].split(".")  # ../ disconsidered
    env_args = {
        'scheduling_inputs': {
            'n_jobs': n_jobs_total,  # n
            'n_machines': n_types,  # m
            'operation_types': op_sequences,
            'operation_durations': op_durations,
            'n_operations': np.count_nonzero(op_durations, axis=1),
            'n_tooling_lvls': 0,  # l
            'n_types': n_types,  # tests
            'min_n_operations': n_types,
            'max_n_operations': n_types,  # o
            'n_jobs_initial': n_jobs_total,  # jobs with arrival time 0
            'max_jobs_visible': n_jobs_total,  # entries in {1 .. n}
        },
        'return_transformer': MakespanExtractor(),
        'selectable_optimizers': [
            # FixedCPPlanSequencing(time_limit=60, schedule_path=sln_name),
            FixedPlanControl(schedule_path=sln_name)
        ],
        'seeds': env_seed,  # not gonna count, since deterministic
        'logfile_path': logpath
    }
    return env_args


def init_env(env_seed, benchmark_f_name, sln_f_name, logpath=''):
    args_dict = get_env_parameters(
        env_seed, benchmark_f_name, sln_f_name, logpath)
    env_name = 'fabricatio-v0'
    if env_name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[env_name]
    register(
        id=env_name,
        entry_point='gym_fabrikatioRL.envs:FabricatioRL',
        kwargs=args_dict
    )
    environment = gym.make(env_name)
    return environment


def extract_jssp_benchmark_info(benchmark_f_name: str) -> \
        (int, int, np.array, np.array):
    """
    Reads JSSP file and extracts the duration and type matrices and transfers
    them to a format accepted by FabrikatioRL.

    The JSSP instances are expected to be in the standard specification (see
    http://jobshop.jjvh.nl/explanation.php), meaning that rows in the file
    correspond to jobs. The types of the machine to be visited alternate with
    the duration specification, with even row entries i corresponding to the
    machine types to be visited and the immediate odd entries i + 1 to the
    associated operation durations. The type numbering starts at 0. The first
    row in the file contains the total number of jobs and operations separated
    by a whitespace.

    FabrikatioRL indexes types starting at 1, so the read type matrix will be
    changed correspondingly. Note that some benchmarking instances (e.g. orb07)
    contain 0 durations and 0 types. So as not to confuse Fabrikatio, the types
    corersponding to 0 durations are zeroed out.

    :param benchmark_f_name: The path to the JSSP benchmark file.
    :return: The
    """
    instance_file = open(benchmark_f_name)
    n_jobs_total, n_machine_groups = map(
        int, instance_file.readline().strip().split(' '))
    op_sequences = np.zeros((n_jobs_total, n_machine_groups), dtype='int8')
    op_durations = np.zeros((n_jobs_total, n_machine_groups), dtype='int16')
    i = 0
    for line in instance_file:
        line_elements = line.strip().split('  ')
        if bool(line_elements):
            for j in range(len(line_elements)):
                if j % 2 == 0:
                    op_sequences[i, j // 2] = int(line_elements[j]) + 1
                else:
                    op_durations[i, j // 2] = int(line_elements[j])
                    if line_elements[j] == 0:
                        op_sequences[i, j // 2] = 0
            i += 1
    instance_file.close()
    return n_machine_groups, n_jobs_total, op_sequences, op_durations


def test_jssp_simulation_benchmark_consistency():
    benchmark_dir = '../benchmarks/jssp_test_instances'
    solution_dir = '../benchmarks/jssp_test_solutions'
    logpath = ''
    env_seed = np.random.randint(low=1, high=1000, size=1)
    f_names = []
    if benchmark_dir != '':
        for file_name in listdir(benchmark_dir):
            sln_base_name, sln_extension = file_name.split(".")
            instance_path = f'{benchmark_dir}/{file_name}'
            sln_path = f'{solution_dir}/{sln_base_name}_sln.{sln_extension}'
            env = init_env(env_seed, instance_path, sln_path, logpath=logpath)
            t_start = time()
            state_repr, done, curr_makespan = env.reset(), False, 0
            n_steps = 0
            print(f"Testing results on {file_name}")
            while not done:
                state_repr, curr_makespan, done, _ = env.step(0)
                n_steps += 1
            t = time() - t_start
            print(f"Testing results: {curr_makespan}")
            lower_bound = int(((file_name.split('__')[-1]).split('_'))[0][2:])
            assert lower_bound == int(curr_makespan)
            env.close()


def init_stochasticity_test_env(seed, optimizers, name, all_jobs_visible=False):
    if all_jobs_visible:
        n_jobs_initial = n_jobs = max_jobs_visible = 20
    else:
        n_jobs_initial = 10
        n_jobs = 20
        max_jobs_visible = 15
    env_args = {
        'scheduling_inputs': {
            'n_jobs': n_jobs,  # n
            'n_machines': 7,  # m
            'n_tooling_lvls': 5,  # l
            'n_types': 5,  # tests
            'min_n_operations': 5,
            'max_n_operations': 5,  # o
            'n_jobs_initial': n_jobs_initial,  # jobs with arrival time 0
            'max_jobs_visible': max_jobs_visible,  # entries in {1 .. n}
            'operation_precedence': 'POm',
            'operation_types': 'Jm',  # deafault_sampling
            'operation_tool_sets': 'default_sampling',
            'machine_speeds': 'default_sampling',
            'machine_distances': 'default_sampling',
            'machine_buffer_capa': np.array([1, 2, 3, 1, 2, 5, 3]),
            # '' or nxo matrix (entries in {1, 2 .. })
            'machine_capabilities': {
                1: {1}, 2: {1}, 3: {2, 3}, 4: {3}, 5: {4}, 6: {5}, 7: {3}},
            'tool_switch_times': 'default_sampling',
            'time_inter_release': 'default_sampling',
            'time_job_due': 'default_sampling',
            'perturbation_processing_time': 'default_sampling',
            'perturbation_due_date': 'default_sampling',
        },
        'seeds': seed,
        'return_transformer': StochasticityTestTransformer(),
        'selectable_optimizers': optimizers,
    }
    if name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[name]
    register(id=name,
             entry_point='gym_fabrikatioRL.envs:FabricatioRL', kwargs=env_args)
    env = gym.make(name)
    return env


def get_envs_selectable_seq_opt(seed):
    # MULTIPLE SEQUENCING OPTIMIZERS
    # sequencing optimizer, fixed transport optimizer actions
    env_0 = init_stochasticity_test_env(
        [seed],
        [LPTOptimizer(), SPTOptimizer(), LDOptimizer()],
        'fabricatio-v0')
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
    return [env_0, env_1, env_2]


def get_envs_fixed_seq_opt(seed):
    # SINGLE SEQUENCING OPTIMIZER
    # fixed sequencing optimizer, fixed transport optimizer actions
    env_0 = init_stochasticity_test_env(
        [seed],
        [LPTOptimizer(), LDOptimizer()],
        'fabricatio-v3')
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
    return [env_0, env_1, env_2]


def get_envs_raw_seq(seed):
    # SINGLE SEQUENCING OPTIMIZER
    # fixed sequencing optimizer, fixed transport optimizer actions
    env_0 = init_stochasticity_test_env(
        [seed],
        [LDOptimizer()],
        'fabricatio-v6')
    # fixed sequencing optimizer, raw transport actions
    env_1 = init_stochasticity_test_env(
        [seed],
        [],
        'fabricatio-v7')
    # fixed sequencing optimizer, transport optimizer actions
    env_2 = init_stochasticity_test_env(
        [seed],
        [LDOptimizer(), MDOptimizer()],
        'fabricatio-v8')
    return [env_0, env_1, env_2]


def get_all_env_interface_configs(seed):
    all_envs = []
    all_envs += get_envs_selectable_seq_opt(seed)
    all_envs += get_envs_fixed_seq_opt(seed)
    all_envs += get_envs_raw_seq(seed)
    return all_envs


def test_seed_interface_invariance():
    # DEFINE ENVIRONMENT ARGS AND BUILD ENV
    """
    general pattern: None, to disregard the scheduling aspect,
    'default_sampling' for the builtin sampling scheme, scalars to parameterize
    the builtin sampling scheme, sampling functions or direct input as
    matrix/vector/tensor/dict
    """
    env_seeds = [56513, 30200, 28174, 9792, 63446, 81531, 31016, 5161, 8664,
                 12399]
    for seed in env_seeds:
        envs = get_all_env_interface_configs(seed)
        for i in range(1, len(envs)):
            first_state_0 = envs[i - 1].reset()
            first_state_1 = envs[i].reset()
            cmp_01 = first_state_0 == first_state_1
            assert cmp_01.all()


def record_sim_on_first_last_legal_action_choice(env):
    states = []
    rewards = []
    actions = []
    state, done = env.reset(), False
    while not done:
        try:
            action = env.get_legal_actions()[0]
            state, reward, done, _ = env.step(action)
        except UndefinedLegalActionCall:
            # happens for fixed optimizers, in which case we step with
            # the only opssible action
            action = 0
            state, reward, done, _ = env.step(action)
        states.append(state)
        rewards.append(reward)
        actions.append(action)
    return states, actions, rewards


def test_seed_identical_runs():
    env_seeds = np.random.randint(low=10, high=10000, size=10)
    for seed in env_seeds:
        envs = get_all_env_interface_configs(seed)
        for i in range(len(envs)):
            ss1, ss2 = [], []
            rs1, rs2 = [], []
            as1, as2 = [], []
            print(f"Running episode reproducibility tests on seed {seed} "
                  f"env configuration {envs[i].optimizer_configuration}")
            ss1, rs1, as1 = record_sim_on_first_last_legal_action_choice(
                envs[i])
            ss2, rs2, as2 = record_sim_on_first_last_legal_action_choice(
                envs[i])
            # sequencing optimizer, fixed transport optimizer actions
            assert len(as1) == len(as2)
            for j in range(len(as1)):
                comp_state = ss1[j] == ss2[j]
                assert comp_state.all()
                assert rs1[j] == rs2[j]
                assert as1[j] == as2[j]


def test_seed_stability():
    """
    Tests whether calling reset on the environment cycles through the seeds
    properly.
    :return:
    """
    pass


test_jssp_simulation_benchmark_consistency()
test_seed_interface_invariance()
test_seed_identical_runs()
