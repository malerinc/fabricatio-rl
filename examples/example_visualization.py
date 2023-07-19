from os.path import join
from pathlib import Path

import gym
import numpy as np

from time import time

from gym import register
from fabricatio_rl.interface_templates import SchedulingUserInputs

import fabricatio_rl.visualization_app as va

test_seed = 84556
np.random.seed(test_seed)


if __name__ == "__main__":
    # if passed to the simulation, the directory below will be created if
    # it does not yet exist
    log_directory = join(f'{Path(__file__).resolve().parent}',
                         'vizualization', 'pom')
    env_args = dict(
        scheduling_inputs=[SchedulingUserInputs(
            n_jobs=20,                # n
            n_machines=10,            # m
            n_tooling_lvls=0,         # l
            n_types=7,                # t
            operation_types='default_sampling',
            n_operations='default_sampling',
            min_n_operations=5,
            max_n_operations=10,       # o
            n_jobs_initial=5,          # jobs with arrival time 0
            max_jobs_visible=5,        # wip size < n
            inter_arrival_time='balanced',
            operation_precedence='POm',
            machine_capabilities={
                1: {1, 6}, 2: {1}, 3: {1, 2, 3}, 4: {3, 7}, 5: {4, 7},
                6: {5, 3}, 7: {3, 6}, 8: {1, 7}, 9: {2, 5}, 10: {3, 4, 5, 6}},
            name='')],
        return_transformer=None,
        seeds=[test_seed],
        logfile_path=log_directory
    )

    register(id='fabricatio-v0',
             entry_point='fabricatio_rl:FabricatioRL',
             kwargs=env_args)
    env = gym.make('fabricatio-v0')
    start_time = time()
    init_time = time() - start_time
    n_steps = 0
    state_repr, done = env.reset(), False
    while not done:
        las = env.get_legal_actions()
        action = int(np.random.choice(las))
        _, _, done, _ = env.step(action)
        n_steps += 1
    runtime = time() - init_time - start_time
    print(f"Test run on {test_seed} seeded random JSSP sample finished after "
          f"{n_steps} steps in {runtime} seconds after an init time of "
          f"{init_time} seconds.")

    # now we need to point the visualization app towards the right log directory
    # and start it; note that the visualization app can record multiple runs
    # in the same parent directory, provided each run is in its own directory
    app = va.create_app(log_directory)
    app.run(debug=True, host="0.0.0.0")
