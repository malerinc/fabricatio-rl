import gym
import numpy as np

from gym import register


if __name__ == "__main__":
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
    }

    register(id='fabricatio-v0',
             entry_point='gym_fabrikatioRL.envs:FabricatioRL', kwargs=env_args)
    env = gym.make('fabricatio-v0')

    state, done = env.reset(), False
    while not done:
        legal_actions = env.get_legal_actions()
        state, reward, done, _ = env.step(np.random.choice(legal_actions))

    print(f'The makespan after a random run was {state.system_time}')
