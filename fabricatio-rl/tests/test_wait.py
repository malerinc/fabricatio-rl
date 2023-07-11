from tests import t_helpers
import numpy as np


def test_wait_only_scheduling():
    seeds = np.random.randint(low=0, high=10000, size=10)
    for seed in seeds:
        print(f"Seed {seed}:")
        # TODO: use helper functions not test modeule for env init
        env = t_helpers.init_stochasticity_test_env(
            [seed], [], 'fabricatioRL-v0', True,
            logdir='')
        state, done = env.reset(), False
        env2 = t_helpers.init_stochasticity_test_env(
            [seed], [], 'fabricatioRL-v1', True,
            logdir='')
        state, done = env.reset(), False
        n_actions = 0
        while not done:
            actions = env.get_legal_actions()
            state, _, done, _ = env.step(actions[-1])
            n_actions += 1
            assert n_actions < 20 * 5 * 7
        assert n_actions > 100
        state, done = env2.reset(), False
        n_act_no_wait = 0
        while not done:
            actions = env2.get_legal_actions()
            state, _, done, _ = env2.step(actions[0])
            n_act_no_wait += 1
        print(f"\tNo wait n_actions: {n_act_no_wait}; "
              f"Wait actions: {n_actions}")
        assert n_actions >= n_act_no_wait

# TODO: test that when wait is legal, wait is always the last action
