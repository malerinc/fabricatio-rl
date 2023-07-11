from fabricatio_rl.interface_templates import ReturnTransformer
from fabricatio_rl.core_state import State

from fabricatio_controls.rl.transformer_rewards import (
    UtilScaledReward,
    MakespanNormedReward,
    SimulativeMakespanReward,
    UtilDeviationReward,
    UtilIntReward, BufferLenReward)
from fabricatio_controls.rl.transformer_state_representations import (
    JmFlatFullState, JmMCTSFeatureState, JmFlatFullStateNormalized,
    FeatureTransformer, JmFullBoxState, JmMainMatrices)


class JmFlatMCTSFeaturesNoReward(ReturnTransformer):
    def __init__(self):
        self.reward_transformer = UtilScaledReward()
        self.state_transformer = JmMCTSFeatureState()

    def transform_state(self, state: State):
        return self.state_transformer.transform_state(state)

    def transform_reward(self, state: State, illegal=False, environment=None):
        return None


# <editor-fold desc=Flat Feature State>
class JmFlatFeatureStateUtilReward(ReturnTransformer):
    def __init__(self):
        self.reward_transformer = UtilScaledReward()
        self.state_transformer = FeatureTransformer([
            'job_op_max_rel_completion_rate_avg',
            'job_work_completion_rate_avg', 'op_completion_rate',
            'work_completion_rate', 'wip_rel_sys_t', 'type_entropy', 'utl_avg',
            'duration_ave', 'makespan_lb_ub_ratio',
            'job_op_completion_rate_ave'
        ])

    def transform_state(self, state: State):
        return self.state_transformer.transform_state(state)

    def transform_reward(self, state: State, illegal=False, environment=None):
        return self.reward_transformer.transform_reward(state, illegal)

    def reset(self):
        self.reward_transformer = UtilScaledReward()


class JmFlatFeatureBufferLenReward(ReturnTransformer):
    def __init__(self):
        self.reward_transformer = BufferLenReward()
        self.state_transformer = FeatureTransformer([
            'decision_skip_ratio', 'wip_rel_sys_t', 'tardiness_rate',
            'wip_to_arrival_ratio', 'utl_avg', 'estimated_tardiness_rate',
            'legal_action_len_stream_ave', 'estimated_flow_time_std',
            'throughput_time_j_abs_avg', 'type_hamming_mean'
        ])

    def transform_state(self, state: State):
        return self.state_transformer.transform_state(state)

    def transform_reward(self, state: State, illegal=False, environment=None):
        return self.reward_transformer.transform_reward(state, illegal)

    def reset(self):
        self.reward_transformer = MakespanNormedReward()


class JmFlatFeatureNormedMakespanReward(ReturnTransformer):
    def __init__(self):
        self.reward_transformer = MakespanNormedReward()
        self.state_transformer = FeatureTransformer([
            'decision_skip_ratio', 'wip_rel_sys_t', 'tardiness_rate',
            'wip_to_arrival_ratio', 'utl_avg', 'estimated_tardiness_rate',
            'legal_action_len_stream_ave', 'estimated_flow_time_std',
            'throughput_time_j_abs_avg', 'type_hamming_mean'
        ])

    def transform_state(self, state: State):
        return self.state_transformer.transform_state(state)

    def transform_reward(self, state: State, illegal=False, environment=None):
        return self.reward_transformer.transform_reward(state, illegal)

    def reset(self):
        self.reward_transformer = MakespanNormedReward()


class JmFlatFeatureStateUtilIntReward(ReturnTransformer):
    def __init__(self):
        self.reward_transformer = UtilIntReward()
        self.state_transformer = FeatureTransformer([])

    def transform_state(self, state: State):
        return self.state_transformer.transform_state(state)

    def transform_reward(self, state: State, illegal=False, environment=None):
        return self.reward_transformer.transform_reward(state, illegal)


class JmFlatFeatureStateSimCmaxReward(ReturnTransformer):
    def __init__(self):
        self.reward_transformer = SimulativeMakespanReward()
        self.state_transformer = FeatureTransformer([])

    def transform_state(self, state: State):
        return self.state_transformer.transform_state(state)

    def transform_reward(self, state: State, illegal=False, environment=None):
        return self.reward_transformer.transform_reward(
            state, illegal, environment)


class JmFlatFeatureStateUtlDeviaitonReward(ReturnTransformer):
    def __init__(self):
        self.reward_transformer = UtilDeviationReward()
        self.state_transformer = FeatureTransformer([])

    def transform_state(self, state: State):
        return self.state_transformer.transform_state(state)

    def transform_reward(self, state: State, illegal=False, environment=None):
        return self.reward_transformer.transform_reward(
            state, illegal, environment)
# </editor-fold>


# <editor-fold desc=Full Flat State>
class JmFlatFullStateUtilReward(ReturnTransformer):
    def __init__(self):
        self.reward_transformer = UtilScaledReward()
        self.state_transformer = JmFlatFullState()

    def transform_state(self, state: State):
        return self.state_transformer.transform_state(state)

    def transform_reward(self, state: State, illegal=False, environment=None):
        return self.reward_transformer.transform_reward(state, illegal)


class JmFlatMainMatricesNormedMakespanReward(ReturnTransformer):
    def __init__(self):
        self.reward_transformer = MakespanNormedReward()
        self.state_transformer = JmMainMatrices()

    def transform_state(self, state: State):
        """
        Return the flattened representations of the main Jm/FJc matrices.
        State shape: (4, wip_size, max_n_operations)
        """
        return self.state_transformer.transform_state(state).flatten()

    def transform_reward(self, state: State, illegal=False, environment=None):
        return self.reward_transformer.transform_reward(state, illegal)

    def reset(self):
        self.reward_transformer = MakespanNormedReward()


class JmFlatFullStateMakespanReward(ReturnTransformer):
    """
    Misnomer! The state is a stack of operation matrices, namely
    O_D, O_T, O_L, O_S and the legal actions as a one-hot matrix.
    State shape: (5, wip_size, max_n_operations)
    """
    def __init__(self):
        self.reward_transformer = MakespanNormedReward()
        self.state_transformer = JmFullBoxState()

    def transform_state(self, state: State):
        return self.state_transformer.transform_state(
            state)

    def transform_reward(self, state: State, illegal=False, environment=None):
        return self.reward_transformer.transform_reward(state, illegal)


class JmFlatFullStateUtilRewardNormalized(ReturnTransformer):
    def __init__(self):
        self.reward_transformer = MakespanNormedReward()
        self.state_transformer = JmFlatFullStateNormalized()

    def transform_state(self, state: State):
        return self.state_transformer.transform_state(state)

    def transform_reward(self, state: State, illegal=False, environment=None):
        return self.reward_transformer.transform_reward(state, illegal)
# </editor-fold>


# <editor-fold desc=Full Box State>
class JmBoxFullStateUtilReward(ReturnTransformer):
    def __init__(self):
        self.reward_transformer = UtilScaledReward()
        self.state_transformer = JmFullBoxState()

    def transform_state(self, state: State):
        return self.state_transformer.transform_state(state)

    def transform_reward(self, state: State, illegal=False, environment=None):
        return self.reward_transformer.transform_reward(state)
# </editor-fold>
