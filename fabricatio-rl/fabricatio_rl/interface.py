from typing import List, Union

import gym
import numpy as np

from fabricatio_rl.core import Core
from fabricatio_rl.env_utils import IllegalAction
from fabricatio_rl.env_utils import UndefinedLegalActionCall
from fabricatio_rl.env_utils import UndefinedOptimizerConfiguration, \
    faster_deepcopy
from fabricatio_rl.env_utils import UndefinedOptimizerTargetMode
from fabricatio_rl.interface_input import Input
from fabricatio_rl.interface_rng import FabricatioRNG
from fabricatio_rl.interface_templates import Optimizer, \
    SchedulingUserInputs
from fabricatio_rl.interface_templates import ReturnTransformer


class MultisetupManager:
    def __init__(self, seeds: Union[None, List[int]],
                 scheduling_inputs: List[SchedulingUserInputs]):
        # SEED SET DEFINITION
        if bool(seeds):
            self.__seeds = seeds
        else:
            self.__seeds = [-1]
        # INSTANCE DEFINITION
        self.__schedulling_inputs = scheduling_inputs

    def cycle_seeds(self):
        seed = self.__seeds.pop(0)
        self.__seeds.append(seed)
        return seed

    def cycle_scheduling_input(self):
        scheduling_input = self.__schedulling_inputs.pop(0)
        self.__schedulling_inputs.append(scheduling_input)
        return scheduling_input

    @property
    def seeds(self):
        return self.__seeds

    @seeds.setter
    def seeds(self, new_seeds: List[int]):
        self.__seeds = new_seeds

    @property
    def scheduling_inputs(self):
        return self.__schedulling_inputs

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)


class FabricatioRL(gym.Env):
    def __init__(self, scheduling_inputs: List[SchedulingUserInputs],
                 seeds: List[int] = None, logfile_path: str = '',
                 return_transformer: ReturnTransformer = None,
                 selectable_optimizers: List[Optimizer] = None):
        """
        Initializes the simulation in three stages: Interface property
        definition (1) setup definition and interface property finalization (3).

        TODO: describe 1, 2, 3 individually.

        The separation between (1) and (3) is needed because, on the one hand,
        __setup_core in stage (2) requires the ReturnTransformer object to be
        initialized so as to call reset on it. This happens because __setup_core
        is also called during simulation resets. During resets, all the data
        saved by the return transformer object needs to be cleared. On the
        other hand, the correct initialization of the state space during
        __setup_observation_space requires both the ReturnTransformer and the
        __core property to be initialized so as to infer the correct state
        representation dimensions by applying the transformer to the initial
        state.

        :param scheduling_inputs:
        :param seeds:
        :param logfile_path:
        :param return_transformer:
        :param selectable_optimizers:
        """
        # SEED DEFINITION
        self.deterministic_env = False
        sim_rng = FabricatioRNG()
        # SETUP DEFINITION & CORE INITIALIZATION
        self.__setup_manager = MultisetupManager(seeds, scheduling_inputs)
        self.__parameters = Input(self.__setup_manager.cycle_scheduling_input(),
                                  sim_rng, self.__setup_manager.cycle_seeds(),
                                  logfile_path)
        self.__core = self.__setup_core(sim_rng)
        self.__no_change_reset = False
        # INTERFACE PROPERTY INITIALIZATION
        # transformer
        self.__return_transformer: ReturnTransformer = return_transformer
        # otimizers
        self.__optimizer_configuration = -1  # becomes value in {0, .., 11}
        self.__sequencing_optimizers: Union[List[Optimizer], None] = None
        self.__transport_optimizers: Union[List[Optimizer], None] = None
        # needed when transport and routing decisions are made by the same agent
        self.__transport_decision_offset: Union[int, None] = None
        self.__setup_optimizers(selectable_optimizers)
        self.action_space: Union[gym.spaces.Discrete, None] = None
        self.observation_space: Union[gym.spaces.Box, None] = None
        self.__setup_action_space()
        self.__setup_observation_space()
        if self.autoplay():
            self.__skip_fixed_decisions(False)

    def __setup_core(self, sim_rng):
        """
        TODO: Explain this!

        If needed, a two step initialization of the core object is executed...

        :param sim_rng:
        :return:
        """
        if not self.__parameters.matrices_j.simulation_needed:
            core = Core(self.__parameters)
        else:
            core = Core(self.__parameters)
            parameters = self.__parameters.finalize_stochastic_inputs(core)
            core = Core(parameters)
        if hasattr(self, '__core'):
            ra, sa = self.__core.rou_autoplay, self.__core.seq_autoplay
            core.set_rou_autoplay(ra)
            core.set_seq_autoplay(sa)
        return core

    def set_optimizers(self, selectable_optimizers):
        self.__setup_optimizers(selectable_optimizers
                                if selectable_optimizers is not None else [])
        self.__setup_action_space()

    def set_transformer(self,
                        return_transformer: Union[ReturnTransformer, None]):
        self.__return_transformer = return_transformer
        self.__setup_observation_space()

    def get_state_transformer(self):
        return self.__return_transformer.transform_state

    def __transform_return(self, illegal):
        if self.__return_transformer is None:
            # something other than RL is using this simulation
            return self.__core.state, None
        else:
            state_repr = self.__return_transformer.transform_state(
                self.__core.state)
            reward = self.__return_transformer.transform_reward(
                self.__core.state, environment=self, illegal=illegal)
            return state_repr, reward

    # <editor-fold desc="Environment Interface">
    def step(self, action: int):
        try:
            direct_action = self.__transform_action(action)
        except IllegalAction:
            state_repr, reward = self.__transform_return(illegal=True)
            self.__no_change_reset = False
            return state_repr, reward, True, {}
        self.__update_optimizers(direct_action)
        try:
            _, done = self.__core.step(direct_action)
        except IllegalAction:
            state_repr, reward = self.__transform_return(illegal=True)
            self.__no_change_reset = False
            return state_repr, reward, True, {}
        # TODO: move to tests!!!
        # if self.core.seq_autoplay and not done:
        #     if self.core.wait_legal():
        #         assert len(self.__core.state.legal_actions) > 2
        #     else:
        #         assert len(self.__core.state.legal_actions) > 1
        if self.autoplay():
            done = self.__skip_fixed_decisions(done)
        state_repr, reward = self.__transform_return(illegal=False)
        return state_repr, reward, done, {}

    def __update_optimizers(self, direct_action):
        if self.sequencing_optimizers is not None:
            for optimizer in self.sequencing_optimizers:
                # I forgot what this does... TODO: figure it out!
                # optimizer.update(
                #     (direct_action, self.__core.state.current_machine_nr),
                #     self.core.state)
                # TODO: differentiate between type of skipped action in core!!!
                for skipped_action in self.__core.skipped_seq_actions:
                    optimizer.update(skipped_action, self.core.state)
                self.__core.skipped_seq_actions = []
        if self.transport_optimizers is not None:
            for optimizer in self.transport_optimizers:
                optimizer.update(direct_action, self.core.state)
                for action in self.__core.skipped_rou_actions:
                    optimizer.update(direct_action, self.core.state)
                self.__core.skipped_rou_actions = []

    def __skip_fixed_decisions(self, done):
        state_repr, reward = None, None
        while self.autoplay() and not done:
            if self.__core.state.scheduling_mode == 0:
                _, done = self.core.step(
                    self.__sequencing_optimizers[0].get_action(
                        self.__core.state))
            else:
                _, done = self.core.step(
                    self.__transport_optimizers[0].get_action(
                        self.__core.state))
        return done

    def autoplay(self):
        """
        Checks whether the next action action can be played automatically.
        :return: True if the next action does not require external intervention.
        """
        return ((self.__core.state.scheduling_mode == 0 and
                 self.__optimizer_configuration in {4, 6, 7}) or
                (self.__core.state.scheduling_mode == 1 and
                 self.__optimizer_configuration in {2, 6, 10}))

    def set_core_seq_autoplay(self, val):
        self.core.set_seq_autoplay(val)

    def set_core_rou_autoplay(self, val):
        self.core.set_rou_autoplay(val)

    def reset(self) -> np.ndarray:
        # cycle seeds and inputs on reset
        if self.__no_change_reset:
            self.__no_change_reset = False
        else:
            sim_rng = FabricatioRNG()
            self.__parameters = Input(
                self.__setup_manager.cycle_scheduling_input(),
                sim_rng, self.__setup_manager.cycle_seeds(),
                self.__parameters.logfile_path)
            if self.__return_transformer is not None:
                self.__return_transformer.reset()
            self.__core = self.__setup_core(sim_rng)
            if self.autoplay():
                self.__skip_fixed_decisions(False)
        if self.__return_transformer is not None:
            return self.__return_transformer.transform_state(self.__core.state)
        else:
            return self.__core.state

    def render(self, mode='dummy'):
        raise NotImplementedError

    def has_ended(self):
        return self.__core.has_ended()

    def get_legal_actions(self) -> List[int]:
        """
        Returns a list of legal actions for each simulation mode and optimizer
        mode combination.

        :return: The legal actions in this state.
        """
        # TODO: implement masking
        toffs = self.__transport_decision_offset
        n_to = self.__transport_optimizers.shape[0]
        if len(self.__core.state.legal_actions) == 0:
            return []
        if self.__optimizer_configuration == 0:
            if self.__core.state.scheduling_mode == 0:
                return self.__core.state.legal_actions
            else:
                return [a + toffs - 1 for a in self.__core.state.legal_actions]
        elif self.__optimizer_configuration in {1, 2}:
            if self.__core.state.scheduling_mode == 0:
                return self.__core.state.legal_actions
            else:
                raise UndefinedLegalActionCall(
                    self.__optimizer_configuration,
                    self.__core.state.scheduling_mode)
        elif self.__optimizer_configuration == 3:
            if self.__core.state.scheduling_mode == 0:
                return self.__core.state.legal_actions
            else:
                return [toffs + i for i in range(n_to)]
        elif self.__optimizer_configuration == 4:
            if self.__core.state.scheduling_mode == 0:
                raise UndefinedLegalActionCall(
                    self.__optimizer_configuration,
                    self.__core.state.scheduling_mode)
            else:
                return self.__core.state.legal_actions
        elif self.__optimizer_configuration in {5, 6}:
            raise UndefinedLegalActionCall(
                self.__optimizer_configuration,
                self.__core.state.scheduling_mode)
        elif self.__optimizer_configuration == 7:
            if self.__core.state.scheduling_mode == 0:
                raise UndefinedLegalActionCall(
                    self.__optimizer_configuration,
                    self.__core.state.scheduling_mode)
            else:
                return list(range(n_to))
        elif self.__optimizer_configuration == 8:
            if self.__core.state.scheduling_mode == 0:
                return list(range(toffs))
            else:
                return [a + toffs - 1 for a in self.__core.state.legal_actions]
        elif self.__optimizer_configuration in {9, 10}:
            if self.__core.state.scheduling_mode == 0:
                return list(range(len(self.sequencing_optimizers)))
            else:
                raise UndefinedLegalActionCall(
                    self.__optimizer_configuration,
                    self.__core.state.scheduling_mode)
        else:  # self.__optimizer_configuration == 11:
            if self.__core.state.scheduling_mode == 0:
                return list(range(toffs))
            else:
                return [toffs + i for i in range(n_to)]

    def make_deterministic(self, wip_only=True):
        """
        Purges all stochasticity from the simulation.

        This breaks the environment in that one cannot recover the initial
        stochastic events purged by this method.
        :return: None.
        """
        self.deterministic_env = True
        self.__core.make_deterministic(wip_only)

    def seed(self, seed=-1):
        self.__setup_manager.seeds = seed

    def repr(self):
        if self.return_transformer is not None:
            a = self.return_transformer.transform_state(self.core.state)
            a.flags.writeable = False
        else:
            M = self.core.state.matrices
            a = np.concatenate(M.op_status, M.op_location)
        return hash(a.tostring())

    def get_reward(self):
        if self.return_transformer is not None:
            return self.return_transformer.transform_reward(self.core.state)
        else:
            return self.core.state.system_time

    def get_state(self):
        if self.return_transformer is not None:
            return self.return_transformer.transform_state(self.core.state)
        else:
            return self.core.state
    # </editor-fold>

    # <editor-fold desc="Optimizer Configuration">
    def __setup_optimizers(self, selectable_opt: List[Optimizer]):
        """
        Splits the transport and sequencing optimizers according to their type
        parameter, and initializes the optimizer_configuration parameter
        defining the action space definition and action selection schemes.

        :param selectable_opt: The list of optimizers.
        :return: None
        """
        seq_opt, tra_opt = [], []
        if selectable_opt is not None:
            for optimizer in selectable_opt:
                if optimizer.target_mode == 'sequencing':
                    seq_opt.append(optimizer)
                elif optimizer.target_mode == 'transport':
                    tra_opt.append(optimizer)
                elif optimizer.target_mode == 'universal':
                    seq_opt.append(optimizer)
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
    def __setup_action_space(self):
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
        n = self.__core.state.params.max_jobs_visible
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

    def __setup_observation_space(self):
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
    def scheduling_inputs(self):
        return self.__setup_manager.scheduling_inputs

    @property
    def seeds(self):
        return self.__setup_manager.seeds

    @property
    def parameters(self):
        return self.__parameters

    @property
    def core(self):
        return self.__core

    @property
    def sequencing_optimizers(self):
        return self.__sequencing_optimizers

    @property
    def transport_optimizers(self):
        return self.__transport_optimizers

    @property
    def optimizer_configuration(self):
        return self.__optimizer_configuration

    @property
    def return_transformer(self):
        return self.__return_transformer
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
            return self.__transform_a_fixed_sequencing_selectable_transport(
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
        """
        Translates an agent action into a simulation core action when sequencing
        decisions (mode 0) are made indirectly through optimizers and transport
        decisions (mode 1) are taken directly by the agent.

        This function ensures that:
            1. No transport action is taken in sequencing mode
            (action > transport decision offset)
            2. No transport decisions are made at all, if the simulation
            instance only needs sequencing decisions (transport decision offset
            is None)
            3. The raw transport action passed by the agent is legal, as
            perceived by the simulation core.

        :param action: The action selected by the agent.
        :return: The corresponding simulation core action.
        """
        if self.__core.state.scheduling_mode == 0:  # sequencing
            if self.__transport_decision_offset is None:
                # no transport decisions available
                return self.__sequencing_optimizers[action].get_action(
                    self.__core.state)
            elif action >= self.__transport_decision_offset:
                # picked a transport action in sequencing mode
                raise IllegalAction()
            else:
                # all goode :)
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
            act = self.__sequencing_optimizers[agent_action].get_action(
                self.__core.state)
            if act not in self.__core.state.legal_actions:
                raise IllegalAction()
            else:
                return act
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

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)
