import numpy as np
import gym
from gym import register
from gym_fabrikatioRL.envs.core_state import State
from gym_fabrikatioRL.envs.interface_templates import ReturnTransformer
from time import time


# DEFINE RETURN TRANSFORMATION OBJECT
class HeuristicInfo(ReturnTransformer):
    @staticmethod
    def transform_state(state: State) -> np.ndarray:
        la, d = state.legal_actions, state.matrices.op_duration
        n, n_j_ops = state.params.n_jobs, state.params.max_n_operations
        return np.array([la, d, n, n_j_ops], dtype=object)


# DEFINE METHOD IMPLEMENTING HEURISTIC
def select_action(legal_actions, op_durations, n_jobs, n_j_operations):
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
        ops.append(Operation(index=j_idx * n_j_operations + op_idx, duration=d))
    action = sorted(ops, key=lambda x: x.duration)[-1]
    return action.index


if __name__ == "__main__":
    # DEFINE ENVIRONMENT PARAMETERS
    seed = 63059
    env_args = {
        'scheduling_inputs': {
            'n_jobs': 100,                # n
            'n_machines': 20,             # m
            'n_tooling_lvls': 0,          # l
            'n_types': 20,                # t
            'min_n_operations': 20,
            'max_n_operations': 20,       # o
            'n_jobs_initial': 100,        # jobs with arrival time 0
            'max_jobs_visible': 100,      # entries in {1 .. n}
        },
        'return_transformer': HeuristicInfo(),
        'seeds': [seed]
    }
    # RUN USING HEURISTIC
    start_time = time()
    register(
        id='fabricatio-v0',
        entry_point='gym_fabrikatioRL.envs:FabricatioRL',
        kwargs=env_args
    )
    env = gym.make('fabricatio-v0')
    init_time = time() - start_time
    n_steps = 0
    state_repr, done = env.reset(), False
    while not done:
        action = select_action(*state_repr)
        state_repr, reward, done, _ = env.step(action)
        n_steps += 1
    runtime = time() - init_time - start_time
    print(f"Test run on {seed} seeded random JSSP sample finished after "
          f"{n_steps} steps in {runtime} seconds after an init time of "
          f"{init_time} seconds.")
