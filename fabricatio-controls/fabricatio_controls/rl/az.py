import math
import numpy as np

from typing import List, Union
from random import choices, sample
from copy import deepcopy
from collections import deque

from fabricatio_rl.interface import FabricatioRL
from fabricatio_rl.interface_templates import ReturnTransformer, Optimizer

from fabricatio_controls import Control
from fabricatio_controls.comparison_utils import import_tf
from fabricatio_controls.comparison_utils import parallelize_heterogeneously
from fabricatio_controls.comparison_utils import parallelize_homogeneously

from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import Add, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform

from pathos.helpers import mp as mph

# following https://web.stanford.edu/~surag/posts/alphazero.html
# and the corresponding code on
# https://github.com/suragnair/alpha-zero-general/tree/master/

EPS = 1e-8


def create_mask(environment):
    legal_actions = environment.get_legal_actions(),
    n_actions = environment.action_space.n
    mask = np.zeros(n_actions)
    mask[legal_actions] = [1] * len(legal_actions)
    return mask


class AZAgentArgs:
    def __init__(self, filepath='', memlen=1000, learning_rate=0.001,
                 tb_logdir='.', res_blocks=3,
                 res_filters=64, itermax=100, c_puct=2.5, temperature=1,
                 stack_size=3):
        self.__filepath = filepath
        # only for training
        self.__memlen = memlen
        self.__learning_rate = learning_rate
        self.__tb_logdir = tb_logdir
        self.__res_blocks = res_blocks
        self.__res_filters = res_filters
        self.__temperature = temperature  # set to 0 during eval
        # needed for training and eval
        self.__c_puct = c_puct
        self.__itermax = itermax
        self.__stack_size = stack_size

    # <editor-fold desc="Attribute Properties">
    @property
    def stack_size(self):
        return self.__stack_size

    @property
    def c_puct(self):
        return self.__c_puct

    @property
    def temperature(self):
        return self.__temperature

    @property
    def itermax(self):
        return self.__itermax

    @property
    def res_blocks(self):
        return self.__res_blocks

    @property
    def res_filters(self):
        return self.__res_filters

    @property
    def memlen(self):
        return self.__memlen

    @property
    def learning_rate(self):
        return self.__learning_rate

    @property
    def filepath(self):
        return self.__filepath

    @filepath.setter
    def filepath(self, fn):
        self.__filepath = fn

    @property
    def tb_logdir(self):
        return self.__tb_logdir
    # </editor-fold>


class RLSetupArgs:
    def __init__(self, agent_args: AZAgentArgs,
                 baseln_hs: str = '', baseln_hr: str = '',
                 reward_delay: int = '', models_base_dir: str = '.'):
        self.agent_args = agent_args
        self.baseln_hs = baseln_hs
        self.baseln_hr = baseln_hr
        self.models_base_dir = models_base_dir
        self.reward_delay = reward_delay


class MCTS:
    """
    This class handles the MCTS tree.

    This implementation is adapted from
    https://web.stanford.edu/~surag/posts/alphazero.html
    """

    def __init__(self, nnet, args):
        """
        Always clone the game instance before passing it to this class
        constructor!
        """
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}   # stores #times board s was visited
        self.Ps = {}   # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def get_action_prob(self, environment: FabricatioRL, state, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from the
        curent state

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        environment.make_deterministic()
        # state = deque(state)
        for i in range(self.args['numMCTSSims']):
            environment_snapshot = deepcopy(environment)
            self.search(environment_snapshot, state)

        s = environment.repr()
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in
                  # at the root
                  range(environment.action_space.n)]

        if temp == 0:
            # print('counts: {0}'.format(counts))
            best_a = np.argmax(counts)
            probs = [0] * len(counts)
            # noinspection PyTypeChecker
            probs[best_a] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = sum(counts)
        assert counts_sum != 0
        probs = [x / float(counts_sum) for x in counts]
        return np.array(probs)

    def search(self, environment: FabricatioRL, state):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.
            v: the value of the current state
        """
        s = environment.repr()
        if s not in self.Es:
            actions = environment.get_legal_actions()
            if not bool(actions):
                self.Es[s] = environment.get_reward()
            else:
                self.Es[s] = -2
        if self.Es[s] != -2:
            # terminal node
            return self.Es[s]

        if s not in self.Ps:
            # leaf node
            pi, val = self.nnet.predict(
                np.array(state).reshape((1, self.args['stack_size'], 1, -1)),
                verbose=0)
            self.Ps[s], v = pi[0], val[0]
            valids = create_mask(environment)
            assert sum(valids) != 0
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally
                # probable
                self.Ps[s] = self.Ps[s] + valids
                # invalid value in true divide!!!
                self.Ps[s] /= np.sum(self.Ps[s])
            self.Vs[s] = valids
            self.Ns[s] = 0
            return v
        # MCTS selection
        # TODO: make absolutely sure states are unique!
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        # pick the action with the highest upper confidence bound
        for a in range(environment.action_space.n):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = 2 * self.Qsa[(s, a)] + self.args['cpuct'] * self.Ps[s][
                        a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    # 0.5 is added because so that unvisited edges have an
                    # associated value of half the possible reward
                    u = self.args['cpuct'] * self.Ps[s][a] * math.sqrt(
                        self.Ns[s] + EPS)

                if u > cur_best:
                    cur_best = u
                    best_act = a
        # Expand and evaluate
        a = best_act
        state_frame, _, _, _ = environment.step(a)
        state.append(state_frame)
        v = self.search(environment, state)
        if (s, a) in self.Qsa:
            # print('picked a node priorly visited :)')
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                        self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v


class AZAgent:
    def __init__(self, env, agent_params: AZAgentArgs):
        """
        Always clone the game instance before passing it to this class
        constructor!
        """
        self.memory = deque(maxlen=agent_params.memlen)
        self.learning_rate = agent_params.learning_rate
        self.temperature = agent_params.temperature
        self.stack_size = agent_params.stack_size
        if agent_params.filepath == '':
            self.model = self._get_resnet_model(
                env, agent_params.res_blocks, agent_params.res_filters,
                agent_params.stack_size
            )
        else:
            self.model = load_model(agent_params.filepath)

        self.tb_logdir = agent_params.tb_logdir
        self.mcts = MCTS(
            self.model,
            {'numMCTSSims': agent_params.itermax,
             'cpuct': agent_params.c_puct,
             'stack_size': agent_params.stack_size})

    @staticmethod
    def _build(model):
        """
        'Using theano or tensorflow is a two step process: build and compile the
        function on the GPU, then run it as necessary.  _make_predict_function()
        performs that first step.

        Keras builds the GPU function the first time you call predict(). That
        way, if you never call predict, you save some time and resources.
        However, the first time you call predict is slightly slower than every
        other time.

        This isn't safe if you're calling predict from several threads, so you
        need to build the function ahead of time. That line gets everything
        ready to run on the GPU ahead of time.'

        Source: https://github.com/keras-team/keras/issues/6124

        :return: None
        """
        # noinspection PyProtectedMember
        # model._make_predict_function()
        pass

    def _get_resnet_model(self, env: FabricatioRL,
                          res_layers=20, n_filters=256, stack_size=3):
        state_shape = env.observation_space.shape[0]
        action_shape = env.action_space.n
        # Define the input as a tensor with shape input_shape
        x_input = Input((stack_size, 1, state_shape))
        # t_input = Input(self.input_t_size, name='input_t')
        # t_1 = Dense(1, activation='relu')(t_input)
        # Stage 1: initial convolution
        x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same',
                   name='conv1',
                   kernel_initializer=glorot_uniform(seed=0))(x_input)
        x = BatchNormalization(axis=3, name='bn_conv1')(x)
        x = Activation('relu')(x)
        # Stage 2: between 10 and 40 residual identity layers
        for i in range(res_layers):
            x = AZAgent._identity_block(
                x, [n_filters, n_filters], stage=2, kernel=(3, 3), s=(1, 1),
                block='res_' + str(i) + '_')
        # Policy head
        pi = Conv2D(2, (1, 1), strides=(1, 1), name='pi_conv')(x)
        pi = BatchNormalization(axis=3, name='bn_pi_conv')(pi)
        pi = Activation('relu')(pi)
        pi = Flatten()(pi)
        # pi_t = Concatenate(axis=1)([pi, t_1])
        pi_t = Dense(action_shape, activation='softmax', name='pi')(pi)
        # Value head
        v = Conv2D(1, (1, 1), strides=(1, 1), name='v_conv')(x)
        v = BatchNormalization(axis=3, name='bn_v_conv')(v)
        v = Activation('relu')(v)
        v = Flatten()(v)
        # v_t = Concatenate(axis=1)([v, t_1])
        # v_t = Dense(256, activation='relu', name='dense_1v')(v_t)
        v_t = Dense(n_filters, activation='relu', name='dense_1v')(v)
        v_t = Dense(1, activation='linear', name='v')(v_t)
        # Create model
        # model = Model(inputs=[x_input, t_input], outputs=[pi_t, v_t])
        model = Model(inputs=[x_input], outputs=[pi_t, v_t])
        AZAgent._build(model)
        model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'],
            optimizer=Adam(self.learning_rate),
            metrics={'pi': 'acc', 'v': 'mean_absolute_error'}
        )
        return model

    @staticmethod
    def _identity_block(x, filters, stage, block, kernel=(3, 3), s=(1, 1)):
        # Defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        # Retrieve Filters
        f1, f2 = filters
        # Save the input value
        x_shortcut = x
        # First component of main path
        x = AZAgent._main_path_block(x, bn_name_base, conv_name_base,
                                     f1, f2, s, kernel=kernel)
        # Final step: Add shortcut value to main path, and pass it
        # through a RELU activation
        x = Add()([x, x_shortcut])
        x = Activation('relu')(x)
        return x

    @staticmethod
    def _main_path_block(x, bn_name_base, conv_name_base,
                         f1, f2, s, kernel=(3, 3)):
        # First component of main path
        x = Conv2D(filters=f1, kernel_size=kernel, strides=s,
                   padding='same', name=conv_name_base + '2a',
                   kernel_initializer=glorot_uniform(seed=0))(x)
        BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)
        # Second component of main path
        x = Conv2D(filters=f2, kernel_size=kernel, strides=s,
                   padding='same', name=conv_name_base + '2b',
                   kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        return x

    def act(self, env, state):
        pi = self.mcts.get_action_prob(
            deepcopy(env), deepcopy(state), temp=self.temperature)
        action = np.random.choice(len(pi), p=pi)
        return action, pi

    def remember(self, state, pi, r):
        self.memory.append([state, pi, r])

    def history_replay(self):
        states = []
        policies = []
        rewards = []
        if len(self.memory) < 50:
            print(f'To few examples in memory ({len(self.memory)}); '
                  f'Waiting for more samples...')
            return
        for state, pi, r in self.memory:
            states.append(state)
            policies.append(pi)
            rewards.append(r)
        tb = TensorBoard(write_graph=True, histogram_freq=1, write_images=False,
                         log_dir=self.tb_logdir)
        self.model.fit(np.array(states),
                       [np.array(policies), np.array(rewards)],
                       epochs=3, verbose=0,
                       callbacks=[tb], validation_split=0.1)

    def save_model(self, fn):
        self.model.save(fn)


class AZEpisodeExecutor:
    def __init__(self, environment, args: RLSetupArgs):
        self.env = environment
        self.agent = AZAgent(environment, args.agent_args)
        # too print out some comparisons during training
        self.baseline_sh = args.baseln_hs
        self.baseline_rh = args.baseln_hr

    def __select_action(self, state, actions):
        if int(state[0][0]) == 1:
            action = actions[0]
            pi = None
        else:  # state[0] == 0: scheduling
            action, pi = self.agent.act(self.env, state)
        return action, pi

    def execute_episode(self, verbose=True):
        trail = []
        state_frame = self.env.get_state()
        actions = self.env.get_legal_actions()
        mask = create_mask(self.env)
        done = False
        examples = []
        reward = 0
        # env_baseline = deepcopy(self.env)
        d_size = self.agent.stack_size
        state = deque([state_frame] * d_size, maxlen=d_size)   # TODO: continue.
        while bool(actions) and not done:
            action, pi = self.__select_action(state, actions)
            state_frame, reward, done, _ = self.env.step(action)
            trail.append(action)
            actions = self.env.get_legal_actions()
            mask = create_mask(self.env)
            state_reshaped = np.array(state).reshape((d_size, 1, -1))
            example = (state_reshaped, pi)
            state.append(state_frame)
            if pi is not None:
                examples.append(example)
        return [(s, pi, reward) for s, pi in examples]


class AZControl(Control):
    def __init__(self, azsa: RLSetupArgs,
                 environment: Union[FabricatioRL, None] = None,
                 state_adapter: Union[ReturnTransformer, None] = None,
                 optimizers: Union[List[Optimizer], None] = None,
                 name: str = 'AZ'):
        """

        :param environment:
        """
        super().__init__(name)
        self.azsa = azsa
        self.env: FabricatioRL = environment
        self.state_adapter = state_adapter
        self.optimizers = optimizers
        if environment is not None:
            self.env.set_transformer(state_adapter)
            self.env.set_optimizers(optimizers)
            self.env.set_core_seq_autoplay(True)
            self.env.set_core_rou_autoplay(True)
        self.name = name
        if azsa.agent_args.filepath != '':
            self.agent = AZAgent(self.env, azsa.agent_args)
        else:
            self.agent = None

    def get_name(self):
        return self.name

    @staticmethod
    def __run_ep(env, azsa: RLSetupArgs):
        import sys
        import_tf(gpu=False)
        sys.setrecursionlimit(4000)
        env.reset()
        # TODO: return seeds!!!
        executor = AZEpisodeExecutor(env, azsa)
        examples = executor.execute_episode()
        return examples

    @staticmethod
    def self_play(env, azsa: RLSetupArgs, n_threads, it):
        if it is None:  # indicates last iteration
            return []
        curr_proc = mph.current_process()
        curr_proc.daemon = False
        ep_examples = parallelize_homogeneously(
            AZControl.__run_ep, (env, azsa), n_threads)
        return ep_examples

    @staticmethod
    def __teach_agent(env: FabricatioRL, dqnsa: RLSetupArgs,
                      examples_separate_eps: list, it: int, n_threads: int):
        if it == 0:
            print('Nothing to train on. Skipping...')
            return ''
        import_tf(gpu=True)
        agent = AZAgent(env, dqnsa.agent_args)
        all_examples = []
        for example_set in examples_separate_eps:
            all_examples += example_set
        # for example in AZControl.__create_sample(all_examples):
        #     agent.remember(*example)
        for example in all_examples:
            agent.remember(*example)
        agent.history_replay()
        fn_main = f"{dqnsa.models_base_dir}dqn_selfplay" \
                  f"{n_threads * (it + 1):04d}"
        agent.save_model(fn_main)
        return fn_main

    def train_agent_parallel(self, n_eps, n_threads):
        examples = []
        iterations = math.floor(n_eps / n_threads)
        curr_proc = mph.current_process()
        curr_proc.daemon = False
        for i in range(iterations):
            print(f'========================================= ITERATION {i}'
                  f'=========================================')
            it = i if i < iterations - 1 else None
            env, azsa = deepcopy(self.env), deepcopy(self.azsa)
            examples_new, fn = parallelize_heterogeneously(
                [AZControl.self_play, AZControl.__teach_agent],
                [(env, azsa, n_threads, it),
                 (env, azsa, examples, i, n_threads)]
            )
            examples += examples_new
            self.azsa.agent_args.filepath = fn
            self.env.reset()
        return self.azsa.agent_args.filepath

    @staticmethod
    def __create_sample(examples):
        examples_prepped = []
        action_indexed_examples = {}
        for example in examples:
            s, pi, r = example
            a = np.argmax(pi)
            if a in action_indexed_examples:
                action_indexed_examples[a].append((s, pi, r))
            else:
                action_indexed_examples[a] = deque([(s, pi, r)], maxlen=10)
        if len(action_indexed_examples) > 2:
            n_samples = max([len(l) for l in action_indexed_examples.values()])
            for _, exes in action_indexed_examples.items():
                if len(exes) < n_samples:
                    examples_prepped += choices(exes, k=n_samples)
                else:
                    examples_prepped += sample(exes, n_samples)
        return examples_prepped

    def play_game(self, env: FabricatioRL,
                  initial_state: Union[np.array, None] = None,
                  verbose=True):
        """
        Plays a game on the class' environment and returns the end game reward
        as well as the list of actions taken.

        :return: The tuple (r, trail) of the end game reward and the list of
            actions leading to it.
        """
        # setup env interface and autoplay
        env.set_transformer(self.state_adapter)
        env.set_optimizers(self.optimizers)
        env.set_core_seq_autoplay(True)
        env.set_core_rou_autoplay(True)
        env.reset()
        # eliminate exploration
        self.agent.temperature = 0
        # tracking variables
        trail = []
        done, state_frame = False, env.get_state()
        actions = env.get_legal_actions()
        n_decisions = 0
        d_size = self.agent.stack_size
        state = deque([state_frame] * d_size, maxlen=d_size)
        rewards, times = [], []
        while bool(actions) and not done:
            action, _ = self.agent.act(env, state)
            n_decisions += 1
            trail.append(action)
            state_frame, reward, done, _ = env.step(action)
            state.append(state_frame)
            rewards.append(reward)
            times.append(env.core.state.system_time)
            actions = env.get_legal_actions()
        if verbose:
            print("Episode ended after {0} decisions".format(len(trail)))
            print([env.core.state.system_time], trail)
        return env.core.state
