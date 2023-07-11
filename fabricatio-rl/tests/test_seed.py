from tests.t_helpers import get_all_env_interface_configs


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


def test_seed_identical_runs():
    # # TODO: refactor considering autoplay...
    # env_seeds = np.random.randint(low=10, high=10000, size=10)
    # for seed in env_seeds:
    #     fabricatio_rl = get_all_env_interface_configs(seed)
    #     for i in range(len(fabricatio_rl)):
    #         ss1, ss2 = [], []
    #         rs1, rs2 = [], []
    #         as1, as2 = [], []
    #         print(f"Running episode reproducibility tests_env on seed {seed} "
    #               f"env configuration {fabricatio_rl[i].optimizer_configuration}")
    #         ss1, rs1, as1 = record_sim_on_first_last_legal_action_choice(
    #             fabricatio_rl[i])
    #         ss2, rs2, as2 = record_sim_on_first_last_legal_action_choice(
    #             fabricatio_rl[i])
    #         # sequencing optimizer, fixed transport optimizer actions
    #         assert len(as1) == len(as2)
    #         for j in range(len(as1)):
    #             comp_state = ss1[j] == ss2[j]
    #             assert comp_state.all()
    #             assert rs1[j] == rs2[j]
    #             assert as1[j] == as2[j]
    return True


def test_seed_stability():
    """
    Tests whether calling reset on the environment cycles through the seeds
    properly.
    :return:
    """
    pass

