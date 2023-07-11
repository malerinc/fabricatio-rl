from typing import Tuple, Callable, Optional, List, Union, Dict, Type

import gym
import torch as th
from stable_baselines3.common.policies import (ActorCriticPolicy,
                                               BaseFeaturesExtractor)

from stable_baselines3.dqn.policies import DQNPolicy

from torch import nn, cat, mul
from torch.nn import functional as f


# <editor-fold desc="PPO Heads">
class SharedStackMlpHead(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the
        features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of
        the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of
        the value network
    """

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
    ):
        super(SharedStackMlpHead, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        # Policy network layers
        self.fc_pi1 = nn.Linear(feature_dim, 256)
        self.fc_pi2 = nn.Linear(256, last_layer_dim_pi)
        # Value network layers
        self.fc_v1 = nn.Linear(feature_dim, 64)
        self.fc_v2 = nn.Linear(64, last_layer_dim_vf)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the
            specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        # policy head
        pi = f.relu(self.fc_pi1(features))
        pi = f.relu(self.fc_pi2(pi))
        # value head
        v = f.relu(self.fc_v1(features))
        v = f.relu(self.fc_v2(v))
        return pi, v


class SharedStackMlpPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[
                List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
    ):
        super(SharedStackMlpPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SharedStackMlpHead(self.features_dim)
# </editor-fold>


# <editor-fold desc="DQN Heads">
class QNet(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the
        features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of
        the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of
        the value network
    """

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
    ):
        super(QNet, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        # Value network layers
        self.fc_v1 = nn.Linear(feature_dim, 128)
        self.fc_v2 = nn.Linear(128, last_layer_dim_vf)

    def forward(self, features: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the
            specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        # value head
        v = f.relu(self.fc_v1(features))
        v = self.fc_v2(v)
        return v


class FCDQNPolicy(DQNPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[
                List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            *args,
            **kwargs,
    ):
        super(FCDQNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        # self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.q_net = QNet(self.features_dim)
# </editor-fold>


# <editor-fold desc="Extractors">
class ConvNetExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: Union[int, str] = 256):
        n_output_channels = 100
        if features_dim == 'auto':
            n_output_channels = int((observation_space.shape[1] *
                                     observation_space.shape[2]))
        super(ConvNetExtractor, self).__init__(
            observation_space, n_output_channels)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.conv_11 = nn.Conv2d(
            n_input_channels, 4, kernel_size=(3, 3), padding='same')
        self.conv_12 = nn.Conv2d(
            n_input_channels, 4, kernel_size=(14, 14), padding='same')
        self.conv_13 = nn.Conv2d(
            n_input_channels, 4, kernel_size=(7, 7), padding='same')
        self.conv_2 = nn.Conv2d(
            12, 8, kernel_size=(3, 3), padding='same')
        self.conv_3 = nn.Conv2d(
            8, 1, kernel_size=(3, 3), padding='same')
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x1 = self.conv_11(observations)
        x1 = self.relu(x1)
        x2 = self.conv_12(observations)
        x2 = self.relu(x2)
        x3 = self.conv_13(observations)
        x3 = self.relu(x3)
        x = cat([x1, x2, x3], dim=1)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.conv_3(x)
        x = self.relu(x)
        x = mul(observations[0][4], x)
        x = self.flatten(x)
        return self.relu(x)


class MlpBoxExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 256):
        super(MlpBoxExtractor, self).__init__(observation_space, features_dim)
        input_size = observation_space.shape[1] * observation_space.shape[2] * 2
        self.fc1 = nn.Linear(input_size, features_dim)
        self.flatten_all = nn.Flatten(start_dim=0)
        self.a1 = nn.Sigmoid()
        self.flatten_one = nn.Flatten(start_dim=0)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # wrt to modifications here ;)
        # https://discuss.pytorch.org/t/effect-of-tensor-modification-in-forward-pass-on-gradient-calculation/132332/3
        i1 = self.flatten_all(observations[0][0:2])
        i2 = observations[0][2]
        i2 = self.flatten_one(i2)
        x1 = self.fc1(i1)
        x2 = mul(i2, x1)
        x2 = self.a1(x2).reshape(1, -1)
        return x2


class MlpExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 256):
        super(MlpExtractor, self).__init__(observation_space, features_dim)
        input_size = observation_space.shape[0]
        self.fc1 = nn.Linear(input_size, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.fc1(observations)
        x = self.relu(x)
        return x
# </editor-fold>
