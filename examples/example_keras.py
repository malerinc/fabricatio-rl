import numpy as np
import gym
from gym import register
from gym_fabrikatioRL.envs.core_state import State
from gym_fabrikatioRL.envs.interface_templates import (Optimizer,
                                                       ReturnTransformer)

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.callbacks import Callback

from copy import deepcopy


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
        action = self._duration_sorted_operations[-1]
        return action.index


class SPTOptimizer(OperationDurationSequencingHeuristic):
    def __init__(self):
        super().__init__()

    def get_action(self, state: State):
        self._sort_actions(state)
        action = self._duration_sorted_operations[0]
        return action.index


class LDOptimizer(Optimizer):
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
        return action.index


class DQNReturnTransformer(ReturnTransformer):
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


if __name__ == "__main__":
    # DEFINE ENVIRONMENT ARGS AND BUILD ENV
    """
    general pattern: None, to disregard the scheduling aspect, 
    'default_sampling' for the builtin sampling scheme, scalars to parameterize
    the builtin sampling scheme, sampling functions or direct input as 
    matrix/vector/tensor/dict
    """
    env_args = {
        'scheduling_inputs': {
            'n_jobs': 20,                # n
            'n_machines': 7,             # m
            'n_tooling_lvls': 0,         # l
            'n_types': 5,                # t
            'min_n_operations': 5,
            'max_n_operations': 5,       # o
            'n_jobs_initial': 10,        # jobs with arrival time 0
            'max_jobs_visible': 10,      # entries in {1 .. n}
            # 'Jm','Om', 'POm' or nxoxo adj. matrix + n_operations; default Jm
            'operation_precedence': 'POm',
            # '' or nxo matrix (entries in {0 .. t-1})
            'operation_types': 'Jm',  # deafault_sampling
            # '' or nxo matrix (entries in {1, 2 .. })
            'machine_distances': 'default_sampling',
            'machine_capabilities': {
                1: {1}, 2: {1}, 3: {2, 3}, 4: {3}, 5: {4}, 6: {5}, 7: {3}},
        },
        'seeds':
            [56513, 30200, 28174, 9792, 63446, 81531, 31016, 5161, 8664, 12399],
        'return_transformer': DQNReturnTransformer(),
        'selectable_optimizers':
            [LPTOptimizer(), SPTOptimizer(), LDOptimizer()],
    }

    register(id='fabricatio-v0',
             entry_point='gym_fabrikatioRL.envs:FabricatioRL', kwargs=env_args)
    env = gym.make('fabricatio-v0')
    nb_actions = env.action_space.n

    # DEFINE AGENT NETWORK
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    # DEFINE RL AGENT
    memory = SequentialMemory(limit=1000, window_length=1)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                   nb_steps_warmup=10, enable_dueling_network=True,
                   target_model_update=1e-3)
    dqn.compile(Adam(lr=1e-3))

    # TRAIN RL AGENT
    tbCallback = TensorBoard(log_dir='./Graph', write_graph=True)
    dqn.fit(env, nb_steps=5000, callbacks=[tbCallback], visualize=False,
            verbose=2)

    # TEST AGENT
    # calback to pick up env information at the end of a test episode
    class MakespanLogger(Callback):
        def __init__(self):
            super().__init__()
            self.env_copy = None

        def on_episode_begin(self, episode, logs=None):
            self.env_copy = deepcopy(env)
            self.env_copy.reset()

        def on_episode_end(self, episode, logs=None):
            env_copy_h2 = deepcopy(self.env_copy)
            while self.env_copy.get_legal_actions():
                self.env_copy.step(0)  # run using solely the 1st heuristic
            used_seed = self.env_copy.core.logger.seed
            makespan = self.env_copy.core.state.trackers.job_completion.max()
            print(f"1st Heuristic Makespan obtained on the randomely sampled "
                  f"environment seeded with {used_seed}: {makespan}")
            while env_copy_h2.get_legal_actions():
                env_copy_h2.step(1)  # run using solely the 1st heuristic
            used_seed = env_copy_h2.core.logger.seed
            makespan = env_copy_h2.core.state.trackers.job_completion.max()
            print(f"2nd Heuristic Makespan obtained on the randomely sampled "
                  f"environment seeded with {used_seed}: {makespan}")
            env_core = self.env.core
            used_seed = env_core.logger.seed
            makespan = env_core.state.trackers.job_completion.max()
            print(f"DQN Makespan obtained on the randomely sampled "
                  f"environment seeded with {used_seed}: {makespan}")

    # set evaluation seeds
    env.seed([98324865, 72938549, 810926345,
              248605, 297849, 922634,
              94524865, 849729423, 109223])
    # evaluate agent
    dqn.test(env, nb_episodes=9, action_repetition=1,
             callbacks=[MakespanLogger()],
             visualize=False, nb_max_episode_steps=None,
             nb_max_start_steps=0, verbose=1)
