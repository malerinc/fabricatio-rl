import gym
import numpy as np

from copy import deepcopy
from gym_fabrikatioRL.envs.core import Core
from gym_fabrikatioRL.envs.core_state import State
from gym_fabrikatioRL.envs.interface_input import Input
from gym_fabrikatioRL.envs.env_utils import UndefinedOptimizerConfiguration
from gym_fabrikatioRL.envs.env_utils import UndefinedOptimizerTargetMode
from gym_fabrikatioRL.envs.env_utils import IllegalAction


class FabricatioRL(gym.Env):
    def __init__(self, scheduling_inputs, seeds='', logfile_path='',
                 return_transformer=None, selectable_optimizers=None):
        # SEED DEFINITION
        if bool(seeds):
            self.__seeds_remaining = seeds[1:]
            self.__seeds_used = [seeds[0]]
            init_seed = seeds[0]
        else:
            self.__seeds_remaining = []
            self.__seeds_used = []
            init_seed = -1
        # SETUP DEFINITION
        self.__parameters = Input(scheduling_inputs, init_seed, logfile_path)
        # CORE
        self.__core = Core(deepcopy(self.__parameters))
        # INTERFACE OBJECTS
        # return transformer
        self.__return_transformer = return_transformer
        # otimizers
        self.__optimizer_configuration = -1  # becomes value in {0, .., 11}
        self.__sequencing_optimizers = None
        self.__transport_optimizers = None
        self.__setup_optimizers(selectable_optimizers
                                if selectable_optimizers is not None else [])
        # needed when transport and routing decisions are made by the same agent
        self.__transport_decision_offset = None
        # action and state space
        self.action_space = None
        self.observation_space = None
        self.__get_action_space()
        self.__get_observation_space()

    # <editor-fold desc="Environment Interface">
    def step(self, action: int):
        try:
            direct_action = self.__transform_action(action)
        except IllegalAction:
            state_repr = self.__return_transformer.transform_state(
                self.__core.state)
            return state_repr, -1, True, {}
        state, done = self.__core.step(direct_action)
        if self.__return_transformer is None:
            # something other than RL is using this simulation
            return state, None, done, {}
        state_repr = self.__return_transformer.transform_state(state)
        reward = self.__return_transformer.transform_reward(state)
        return state_repr, reward, done, {}

    def reset(self) -> State:
        # seed cycling if seeds were passed
        if bool(self.__seeds_remaining) or bool(self.__seeds_used):
            if len(self.__seeds_remaining) > 0:
                seed = self.__seeds_remaining.pop(0)
                self.__seeds_used.append(seed)
            else:
                self.__seeds_remaining = self.__seeds_used[1:]
                seed = self.__seeds_used[0]
                self.__seeds_used = [seed]
            self.__parameters = Input(self.__parameters.scheduling_inputs,
                                      seed, self.__parameters.logfile_path)
        else:
            self.__parameters = Input(
                self.__parameters.scheduling_inputs,
                logfile_path=self.__parameters.logfile_path)
        self.__core = Core(self.__parameters)
        if self.__return_transformer is not None:
            return self.__return_transformer.transform_state(self.__core.state)
        else:
            return self.__core.state

    def render(self, mode='dummy'):
        raise NotImplementedError

    def get_legal_actions(self):
        """
        Returns a list of legal actions as interpreted by the core module.

        :return: (legel_actions, legal_action_mask)
        """
        # TODO: implement masking
        return self.__core.state.legal_actions

    def make_deterministic(self):
        """
        Purges all stochasticity from the simulation.

        This breaks the environment in that one cannot recover the initial
        stochastic events purged by this method.
        :return: None.
        """
        self.__core.make_deterministic()

    def seed(self, seed=-1):
        self.__seeds_remaining = seed
        self.__seeds_used = []
    # </editor-fold>

    # <editor-fold desc="Optimizer Configuration">
    def __setup_optimizers(self, selectable_opt: list):
        """
        Splits the transport and sequencing optimizers according to their type
        parameter, and initializes the optimizer_configuration parameter
        defining the action space definition and action selection schemes.

        :param selectable_opt: The list of optimizers.
        :return: None
        """
        seq_opt, tra_opt = [], []
        for optimizer in selectable_opt:
            if optimizer.target_mode == 'sequencing':
                seq_opt.append(optimizer)
            elif optimizer.target_mode == 'transport':
                tra_opt.append(optimizer)
            else:
                raise UndefinedOptimizerTargetMode()
        self.__sequencing_optimizers = np.array(seq_opt)
        self.__transport_optimizers = np.array(tra_opt)
        self.__setup_optimizer_config()

    def __is_sequencing_only_simulation(self):
        """
        If all types can be executed on exactly one machine, and the operation
        ordering is sequential, then there is no transport decision to be made,
        since jobs have only one downstream machine to be routed to. In such a
        case, return True.

        :return: True, if no transport decisions need to be made.
        """
        type_to_machine = self.__parameters.matrices_m.machine_capabilities_dt
        prec_list = self.__parameters.matrices_j.operation_precedence_l
        for _, eligible_machines in type_to_machine.items():
            if len(eligible_machines) > 1:
                return False
        for node_to_neighbor_map in prec_list:
            for _, neighbors in node_to_neighbor_map.items():
                if len(neighbors) > 1:
                    return False
        return True

    def __setup_optimizer_config(self):
        """
        Initializes the optimizer_configuration parameter influencing the action
        space definition and action translation to one of 11 integer values
        defined as follows:

         0: Direct sequencing action and direct transport action
         1: Direct sequencing action (sequencing only simulation)
         2: Direct sequencing action and fixed transport optimizer
         3: Selectable sequencing optimizer and selectable transport optimizer
         4: Fixed sequencing optimizer and direct transport action
         5: Fixed sequencing optimizer run (sequencing only simulation)
         6: Fixed sequencing and routing optimizer run
         7: Fixed sequencing and selectable transport optimizer
         8: Selectable sequencing optimizer and direct transport action
         9: Selectable sequencing optimizer (sequencing only simulation)
         10: Selectable sequencing optimizer and fixed transport optimizer
         11: Selectable sequencing and transport optimizers

        :return: None
        """
        n_to = self.__transport_optimizers.shape[0]
        n_so = self.__sequencing_optimizers.shape[0]
        if n_so == 0 and n_to == 0:  # direct actions only
            if not self.__is_sequencing_only_simulation():
                self.__optimizer_configuration = 0
            else:
                self.__optimizer_configuration = 1
        elif n_so == 0 and n_to == 1:
            self.__optimizer_configuration = 2
        elif n_so == 0 and n_to > 1:
            self.__optimizer_configuration = 3
        elif n_so == 1 and n_to == 0:
            if not self.__is_sequencing_only_simulation():
                self.__optimizer_configuration = 4
            else:
                self.__optimizer_configuration = 5
        elif n_so == 1 and n_to == 1:
            self.__optimizer_configuration = 6
        elif n_so == 1 and n_to > 1:
            self.__optimizer_configuration = 7
        elif n_so > 1 and n_to == 0:
            if not self.__is_sequencing_only_simulation():
                self.__optimizer_configuration = 8
            else:
                self.__optimizer_configuration = 9
        elif n_so > 1 and n_to == 1:
            self.__optimizer_configuration = 10
        else:  # n_so > 1 and n_to > 1:
            self.__optimizer_configuration = 11
    # </editor-fold>

    # <editor-fold desc="Action and Observation Space Setup">
    def __get_action_space(self):
        """
        Initializes the action space parameter based on the
        optimizer_configuration. The following scheme is applied:
        1.) The agent action vector contains sequencing actions first,
            then transport, except when there are no sequencing actions,
            in which case only the transport options are actions
        2.) For direct sequencing action, the total number of *visible*
            operation indices constitute the actions + 1 for the wait signal
        3.) For direct transport the number of machines in the system + 1 for
            the wait signal constitute the actions
        4.) For indirect optimizer actions the index of the respective optimizer
            represents the action (here too 1. applies)
        5.) If both routing and scheduling actions come from the agent, an
            offset scalar (number of possible agent sequencing actions, n_s)
            is kept to distinguish between the two, e.g. for agent action n
            in transport mode transport action = n - n_s

        :return: None
        """
        assert -1 < self.__optimizer_configuration <= 11
        n = self.__core.state.params.n_jobs
        o = self.__core.state.params.max_n_operations
        m = self.__core.state.params.n_machines
        n_so = self.__sequencing_optimizers.shape[0]
        n_to = self.__transport_optimizers.shape[0]
        self.__transport_decision_offset = None
        if self.__optimizer_configuration == 0:
            self.__transport_decision_offset = n * o + 1
            self.action_space = gym.spaces.Discrete(n * o + 1 + m + 1)
        elif self.__optimizer_configuration in {1, 2}:
            self.action_space = gym.spaces.Discrete(n * o + 1)
        elif self.__optimizer_configuration == 3:
            self.__transport_decision_offset = n * o + 1
            self.action_space = gym.spaces.Discrete(n * o + 1 + n_to)
        elif self.__optimizer_configuration == 4:
            self.action_space = gym.spaces.Discrete(m + 1)
        elif self.__optimizer_configuration in {5, 6}:
            return  # not RL; leave action space None
        elif self.__optimizer_configuration == 7:
            self.action_space = gym.spaces.Discrete(n_to)
        elif self.__optimizer_configuration == 8:
            self.__transport_decision_offset = n_so
            self.action_space = gym.spaces.Discrete(n_so + m + 1)
        elif self.__optimizer_configuration in {9, 10}:
            self.action_space = gym.spaces.Discrete(n_so)
        else:  # self.__optimizer_configuration == 11:
            self.__transport_decision_offset = n_so
            self.action_space = gym.spaces.Discrete(n_so + n_to)

    def __get_observation_space(self):
        """
        Initializes the observation space required by gym to a Box object as
        defined by gym.

        The observation (i.e. state) space dimension is inferred from the state
        representation returned by the state_transformer on the initial state.
        :return: None
        """
        if self.__return_transformer is None:
            # something other than RL is using this simulation
            return
        state_repr = self.__return_transformer.transform_state(
            self.__core.state)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=state_repr.shape)

    # </editor-fold>

    # <editor-fold desc="Getters">
    @property
    def parameters(self):
        return self.__parameters

    @property
    def core(self):
        return self.__core
    # </editor-fold>

    # <editor-fold desc="Action Transformation">
    def __transform_action(self, agent_action):
        """
        Switches between the 11 available decision interfaces and transforms the
        agent action accordingly into an environment core compatible decision.

        :param agent_action: The action as chosen by the agent.
        :return: The action compatible with the core.
        """
        if self.__optimizer_configuration in {0, 1}:
            # both routing and sequencing direct actions
            return self.__transform_a_direct_action_run(agent_action)
        elif self.__optimizer_configuration == 2:
            return self.__transform_a_direct_sequencing_fixed_transport(
                agent_action)
        elif self.__optimizer_configuration == 3:
            return self.__transform_a_direct_sequencing_selectable_transport(
                agent_action)
        elif self.__optimizer_configuration == 4:
            return self.__transform_a_fixed_sequencing_direct_transport(
                agent_action)
        elif self.__optimizer_configuration in {5, 6}:
            return self.__transform_a_fixed_optimizer_run()
        elif self.__optimizer_configuration == 7:
            self.__transform_a_fixed_sequencing_selectable_transport(
                agent_action)
        elif self.__optimizer_configuration in {8, 9}:
            return self.__transform_a_selectable_sequencing_direct_transport(
                agent_action)
        elif self.__optimizer_configuration == 10:
            return self.__transform_a_selectable_sequencing_fixed_transport(
                agent_action)
        elif self.__optimizer_configuration == 11:
            return self.__transform_action_fully_selectable_optimizer_run(
                agent_action)
        else:  # should not be possible at this point;
            raise UndefinedOptimizerConfiguration()

    def __transform_a_selectable_sequencing_direct_transport(
            self, action: int) -> int:
        if self.__core.state.scheduling_mode == 0:
            if action >= self.__transport_decision_offset:
                raise IllegalAction()
            return self.__sequencing_optimizers[action].get_action(
                self.__core.state)
        else:
            core_action = action - self.__transport_decision_offset + 1
            if (action < self.__transport_decision_offset or
                    core_action not in self.__core.state.legal_actions):
                raise IllegalAction()
            # m starts from 1!
            return core_action

    def __transform_a_direct_sequencing_selectable_transport(
            self, action: int) -> int:
        if self.__core.state.scheduling_mode == 0:
            if (action >= self.__transport_decision_offset or
                    action not in self.__core.state.legal_actions):
                raise IllegalAction()
            return action
        else:
            if action < self.__transport_decision_offset:
                raise IllegalAction()
            return self.__transport_optimizers[
                action - self.__transport_decision_offset].get_action(
                self.__core.state)

    def __transform_a_fixed_optimizer_run(self) -> int:
        # pure optimizer run. action space not relevant
        # illegal actions not possible
        if self.__core.state.scheduling_mode == 0:
            direct_core_action = self.__sequencing_optimizers[0].get_action(
                self.__core.state)
        else:
            direct_core_action = self.__transport_optimizers[0].get_action(
                self.__core.state)
        return direct_core_action

    def __transform_a_selectable_sequencing_fixed_transport(
            self, agent_action: int) -> int:
        # illegal actions not possible
        if self.__core.state.scheduling_mode == 0:
            return self.__sequencing_optimizers[agent_action].get_action(
                self.__core.state)
        else:
            return self.__transport_optimizers[0].get_action(self.__core.state)

    def __transform_a_direct_sequencing_fixed_transport(
            self, agent_action: int) -> int:
        if self.__core.state.scheduling_mode == 0:
            if agent_action not in self.__core.state.legal_actions:
                raise IllegalAction()
            return agent_action
        else:
            return self.__transport_optimizers[0].get_action(
                self.__core.state)

    def __transform_a_fixed_sequencing_selectable_transport(
            self, agent_action: int) -> int:
        if self.__core.state.scheduling_mode == 0:
            return self.__sequencing_optimizers[0].get_action(
                self.__core.state)
        else:
            # illegal actions not possible
            return self.__transport_optimizers[agent_action].get_action(
                self.__core.state)

    def __transform_a_fixed_sequencing_direct_transport(
            self, agent_action: int) -> int:
        if self.__core.state.scheduling_mode == 0:
            return self.__sequencing_optimizers[0].get_action(
                self.__core.state)
        else:
            # illegal actions handled by the core?
            core_action = agent_action + 1
            if core_action not in self.__core.state.legal_actions:
                raise IllegalAction()
            return core_action

    def __transform_a_direct_action_run(self, agent_action: int) -> int:
        if self.__core.state.scheduling_mode == 0:
            if self.__transport_decision_offset is None:
                if agent_action not in self.__core.state.legal_actions:
                    raise IllegalAction()
            elif (agent_action >= self.__transport_decision_offset or
                    agent_action not in self.__core.state.legal_actions):
                raise IllegalAction()
            return agent_action
        else:
            core_action = agent_action - self.__transport_decision_offset + 1
            if (agent_action < self.__transport_decision_offset or
                    core_action not in self.__core.state.legal_actions):
                raise IllegalAction()
            return core_action

    def __transform_action_fully_selectable_optimizer_run(
            self, agent_action: int) -> int:
        """
        Transforms action in the selectable routing and sequencing mode
        (opt_conf==6).

        When the core is in sequencing mode, the agent action
        designates a sequencing optimizer index. When in routing mode, the agent
        action designates a transport optimizer index. The first
        self.__transport_decision_offset optimizers designate sequencing
        optimizers while the next indices pertain to transport optimizers.


        The get_action method of the optimizer selected by the agent is called
        with the core state to return the core compatible action.

        :param agent_action: The transport or sequencing optimizer index.
        :return: The core compatible action.
        """
        # Selectable Indirect Transport Action &
        # Selectable Indirect Sequencing Action
        if self.__core.state.scheduling_mode == 0:
            if agent_action >= self.__transport_decision_offset:
                raise IllegalAction()
            direct_core_action = self.__sequencing_optimizers[
                agent_action].get_action(self.__core.state)
        else:
            if agent_action < self.__transport_decision_offset:
                raise IllegalAction()
            direct_core_action = self.__transport_optimizers[
                agent_action - self.__transport_decision_offset].get_action(
                self.__core.state)
        return direct_core_action
    # </editor-fold>
