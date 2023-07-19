from typing import List, Dict, Union, Any, Type, cast
from typing_extensions import TypedDict
from copy import deepcopy
from numpy import array, bincount, argmax, random

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.policies import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.utils import obs_as_tensor, get_device
from stable_baselines3.dqn.policies import DQNPolicy

import torch as th

from fabricatio_rl.interface import FabricatioRL
from fabricatio_rl.interface_templates import ReturnTransformer
from fabricatio_rl.core_state import State

from fabricatio_controls import Control

# <editor-fold desc="Dictionary Types">
FeatureExtractorParams = TypedDict('FeatureExtractorParams', {
    'features_dim': int
})

ActorCriticPolicyParams = TypedDict('ActorCriticPolicyParams', {
    'features_extractor_class': BaseFeaturesExtractor,
    'feature_extractor_kwargs': FeatureExtractorParams
})

PPOParams = TypedDict('PPOParams', {
    'tensorboard_log': str,
    'policy_kwargs': ActorCriticPolicyParams,
    'learning_rate': float,
    'n_steps': int,
    'ent_coef': float,
    'vf_coef': float,
})
# </editor-fold>


# class SummaryWriterCallback(BaseCallback):
#
#     def _on_training_start(self):
#         self._log_freq = 1000  # log every 1000 calls
#
#         output_formats = self.logger.Logger.CURRENT.output_formats
#         # Save reference to tensorboard formatter object
#         # note: the failure case (not formatter found) is not handled here, should be done with try/except.
#         self.tb_formatter = next(
#             formatter for formatter in output_formats if
#             isinstance(formatter, TensorBoardOutputFormat))
#
#     def _on_step(self) -> bool:
#         if self.n_calls % self._log_freq == 0:
#             self.tb_formatter.writer.add_text("direct_access",
#                                               "this is a value",
#                                               self.num_timesteps)
#             self.tb_formatter.writer.flush()


class StableBaselinesRLControl(Control):
    def __init__(self, initial_env: Union[FabricatioRL, None],
                 training_envs: FabricatioRL,
                 validation_envs: FabricatioRL,
                 optimizers: List, return_transformer: ReturnTransformer,
                 model_class: Union[Type[PPO], Type[DQN]],
                 model_policy: Union[Type[DQNPolicy], Type[ActorCriticPolicy]],
                 model_parameters,  #: PPOParams,
                 control_name: str,
                 retrain: bool = False,
                 model_path: str = '', eval_freq: int = 500,
                 eval_n_episodes: int = 10, eval_deterministic: bool = True,
                 learn_total_timesteps: int = 10000,
                 deterministic_play: bool = False, log_freq=4):
        super().__init__(control_name)
        # experiment data
        self.learn_total_timesteps = learn_total_timesteps
        self.eval_n_eps = eval_n_episodes
        self.eval_deterministic = eval_deterministic
        self.eval_freq = eval_freq
        self.retrain = retrain
        self.model_path = model_path
        self.return_transformer = return_transformer
        self.optimizers = optimizers
        self.determinisc_play = deterministic_play
        # training environment
        self.training_env: FabricatioRL = training_envs
        self.training_env.set_transformer(return_transformer)
        self.training_env.set_optimizers(optimizers)
        self.training_env.set_core_seq_autoplay(True)
        self.training_env.set_core_rou_autoplay(True)
        # validation environment
        self.validation_env = validation_envs
        self.validation_env.set_transformer(return_transformer)
        self.validation_env.set_optimizers(optimizers)
        self.validation_env.set_core_seq_autoplay(True)
        self.training_env.set_core_rou_autoplay(True)
        # agent model
        self.model_params = model_parameters
        self.model = model_class(model_policy, self.training_env,
                                 **cast(Dict[str, Any], model_parameters))
        self.log_freq = log_freq

    def learn(self):
        # setup validation env
        eval_callback = EvalCallback(self.validation_env,
                                     best_model_save_path=self.model_path,
                                     eval_freq=self.eval_freq,
                                     n_eval_episodes=self.eval_n_eps,
                                     deterministic=self.eval_deterministic,
                                     render=False)
        self.model.learn(total_timesteps=self.learn_total_timesteps,
                         callback=[eval_callback],
                         log_interval=self.log_freq)
        self.model.save(f'{self.model_path}/end_model.zip')

    def load(self, model_path: Union[None, str] = None):
        if model_path is None:
            self.model.load(f'{self.model_path}/best_model')
        else:
            self.model.load(f'{model_path}/best_model', env=None,
                            custom_objects=None)
            self.model_path = model_path

    def play_game(self, test_env: FabricatioRL,
                  initial_state: Union[array, None] = None):
        assert initial_state is not None
        assert type(initial_state) == State
        # self.return_transformer.transform_reward = (
        #     lambda state, environment: None)
        test_env.set_transformer(self.return_transformer)
        test_env.set_optimizers(self.optimizers)
        done, observations = False, self.return_transformer.transform_state(
            initial_state)
        env_i = deepcopy(test_env)
        self.load()
        # stabilize agent fine-tuning and future choices
        if self.retrain:
            deterministic_copy = deepcopy(env_i)
            deterministic_copy.make_deterministic()
            self.model.tensorboard_log = None           # disable tb logging
            self.model.set_env(deterministic_copy)
            self.model.learn(total_timesteps=int(1000))
            self.model.set_env(env_i)
        random.seed(100)
        action_trail = []
        while not done:
            n_done = env_i.core.state.trackers.n_completed_jobs
            n_unknown = (env_i.core.state.params.n_jobs -
                         env_i.core.state.params.n_jobs_initial)
            # self.model = cast(self.model, PPO)
            action, _states = self.get_action(observations,
                                              deepcopy(env_i))
            observations, rewards, done, info = env_i.step(int(action))
            action_trail.append(action)
        env_core = env_i.core
        return env_core.state

    def get_action(self, observations, env):
        # action, worst_value = self.__get_action_worst_case_val(observations)
        # action_vals = []
        # for alternative in range(env.action_space.n):
        #     ec = deepcopy(env)
        #     observations, _, _, _ = ec.step(action)
        #     next_act, next_val = self.__get_action_worst_case_val(
        #         observations)
        #     if alternative == action:
        #         action_vals.append((next_val * 0.6 + worst_value * 0.4))
        #     else:
        #         action_vals.append(next_val)
        action, _states = self.model.predict(
            observation=observations,
            deterministic=True)
        # choice = np.argmax(action_vals)
        # return choice, None
        return action, None

    def __get_action_worst_case_val(self, observations):
        obs = obs_as_tensor(observations, get_device("auto"))
        with th.no_grad():
            acts = self.model.policy.forward(obs)
        action_selected = acts[0].cpu().numpy()
        val = acts[2].numpy()
        return int(action_selected), float(val)

    # def to_dict(self):
    #     env: FabricatioRL = self.training_env
    #     seq_o = [o.__class__.__name__ for o in env.sequencing_optimizers]
    #     tr_o = [o.__class__.__name__ for o in env.transport_optimizers]
    #     return dict(
    #         optimizers_seq=seq_o,
    #         optimizers_tra=tr_o,
    #         return_transformer=env.return_transformer.__class__.__name__,
    #         model_class=self.model.__class__.__name__,
    #         model_policy_class=self.model.policy_class,
    #         model_parameters=dict(
    #             self.
    #         )
    #     )


class RlEnsemble(Control):
    def __init__(self, models: List, name):
        super().__init__('ppo_ensemble')
        self.rl_controls = models

    def play_game(self, test_env: FabricatioRL,
                  initial_state: Union[array, None] = None):
        # TODO assert that the transformer and optimizers of the test env are
        #  compatible with the train/val fabricatio_rl
        assert initial_state is not None
        done, observations = False, initial_state
        env_i = deepcopy(test_env)
        n_jobs_completed = 0
        while not done:
            actions = []
            for rl_control in self.rl_controls:
                # self.model = cast(self.model, PPO)
                action, _states = rl_control.get_action(observations)
                actions.append(action)
            bins = bincount(actions)
            action = argmax(bins)
            observations, rewards, done, info = env_i.step(int(action))
        env_core = env_i.core
        return env_core.state
