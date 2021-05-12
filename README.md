## FabricatioRL

FabrikatioRL is an reinforcement learning (RL) compatible event discrete simulation framework for production scheduling  problems implementing the [OpenAI Gym](https://gym.openai.com/) standard. As such RL agents from Libraries such as [KerasRL](https://github.com/keras-rl/keras-rl) or [Stable Baselines](https://github.com/hill-a/stable-baselines) can be tested within the realm of scheduling. 

The simulation guarantees reproducibility within stochastic scheduling setups by seeding the numpy random number generator and allows the explicit sepparation of training and test instances by means of seed sets. 

FabrikatioRL is written with extensibility in mind and is highly configurable in terms of 
1. Scheduling problem setup and
2. Markov decision process (MDP) modeling  

## Features
### Setup Features
The following production scheduling problem components can be explicitly considered when instantiating the simulation (planned extensions are marked with unticked boxes):
1. Job characteristics
   - [x] Partially ordered job operations (DAG precedence constraints) 
   - [x] Variable number of operations per job
   - [x] Recurrent operation types within jobs (recirculation)
2. Resource characteristics
   - [x] Machine capabilities
   - [x] Machine input buffer capacities
   - [ ] Machine output buffers
   - [x] Machine dependent speeds
   - [x] Sequence dependent setup times
   - [ ] Machine and sequence dependent setup times
   - [ ] Batch processing capabilities (fixed and dynamic)
   - [x] Transport times
   - [ ] Limited number of transport resources
   - [ ] Mutiple job source and sinks
3. Stochsticity sources
   - [x] Operation duration perturbations
   - [x] Job arrival times
   - [x] Machine breakdowns (needs testing)

The following trackers are maintained for optimization goal measurement (and reward construction!) per machine or job respectively: 
1. Variables for machine-centric goals (per machine)
   - [x] Number of buffered operations
   - [x] Total processing time of buffered operations
   - [x] Machine utilization
   - [x] Total setup times overhead
2. Variables for job-centric goals (per job)
   - [x] Job completion times
   - [x] Job arrival relative flow time
   - [x] Job processing start relative flow time
   - [x] Tardiness
   - [x] Earliness
   - [x] Unit cost
   - [x] Idle time
3. Throughput variables
   - [x] Number of finished operations 
   - [x] Number of finished jobs 

Production scheduling setups are usually described using the parameters <img src="https://render.githubusercontent.com/render/math?math=\alpha, \beta"> for production setups and <img src="https://render.githubusercontent.com/render/math?math=\gamma"> for optimization goal [[1]](#1). The figures below present an overview of the parameters covered by our simulation. The images from [[2]](#2) were adapted for this purpose. There you can read more about the different setups in RL scheduling literature.

<figure class="image" align="center">
        <img src="figures/fabRL_alphas.png" alt="dec_modes" width="500"/>
    	<figcaption align="justify">Machine setup (<img src="https://render.githubusercontent.com/render/math?math=\alpha">) hierarchy in relationtion with FabrikatioRL. Arrows define a generalization relation. Green rectangles were introduced by Pinedo. Red rectangeles were defined in in RL scheduling literature. Filled in rectangles were experimented with in RL literature. The hatched rectangle represents our simulation.
    </figcaption>
</figure>

<figure align="center">
        <img src="figures/fabRL_betas.png" alt="dec_modes" width="500"/>
    <figcaption align="justify">Additional constraints (<img src="https://render.githubusercontent.com/render/math?math=\beta">) covered by FabrikatioRL (hatched). Arrows define a generalization relation. Green rectangles were introduced by Pinedo. Red rectangeles were defined in in RL scheduling literature. Filled in rectangles were experimented with in RL literature. Note that currently, FabrikatioRL can only simulate <img src="https://render.githubusercontent.com/render/math?math=tr(\infty)"> environments.
    </figcaption>
</figure>
<figure align="center">
        <img src="figures/fabRL_gammas.png" alt="dec_modes" width="500"/>
    <figcaption align="justify">Optimization goal (<img src="https://render.githubusercontent.com/render/math?math=\gamma">) intermediary variables covered by FabrikatioRL (hatched). Arrows indicate an "is used by" relation. Green rectangles are described by Pinedo. Red rectangeles were defined in in RL scheduling literature. Filled in rectangles (red or green) were experimented with in RL literature. Boxes with gray filling represent intermediary variables.
    </figcaption>
</figure>




### MDP Features
Agents are tasked with decision making (mainly) when operation processing finishes on  machines. The simulation supports the configuration of the following production scheduling MDP components:
1. Decisions
    - [x] Operation sequencing on machines (What operation is picked from the buffer?)
    - [x] Job transport (To which downstream machine is the job sent?)
    - [ ] Transport vehicle selection (Which transport resource carries the operation?)
    <p align="center">
    	<img src="figures/decisions_v2-1.png" alt="decisions" width="700"/>
    </p>

2. Action Space: The action space configuration is dependent on the decisions present as dictated by the chosen setup and is inferred by the number of optimizer objects (either sequencing, transport) present. The simulation can completely defer a decision type to a fixed optimizer. Agent action spaces are any combination of the following:
   - [x] Direct operation sequencing (agent action is an operation index)
   - [x] Direct transport target selection (agent action is a machine index)
   - [x] Indirect operation sequencing using optimizers (agent action is a sequencing optimizer index)
   - [x] Indirect transport target selection using optimizers (agent action is a transport optimizer index)
   - [ ] Direct vehicle selection
   - [ ] Indirect vehicle selection <br/>
	
	The supported direct/indirect action combinations are listed in the overview below
    <p align="center">
        <img src="figures/overview_simulation_modes.png" alt="dec_modes" width="500"/>
    </p>
3. Observation space: The observation space can be configured by means of a  `ReturnTransformer` object having access to the entire state representation. Selectable information is <br/>
	* System time<br/>
	* Raw state information <br/>
		* Operation durations matrix<br/>
		* Operation types matrix<br/>
		* Operation precedence graph matrices<br/>
		* Operation location matrix <br/>
		* Operation status matrix<br/>
		* ...<br/>
	* The legal next actions<br/>
	* Trackers<br/>

4. Reward: Same configuration mechanism as with observation space

## Getting Started
### Installation
The project is not yet uploaded to PyPI pending thorough debuging and documentation. You can use it by
1. Cloning the repository:
	```
	git clone https://github.com/malerinc/fabricatio-rl.git
	```
2. Installing the package in development mode:
	```
	cd fabricatio-rl
	pip install -e .
	```
3. Defining the environment arguments
	```
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
   ```
4. Registering and building the environment
   ``` 
    register(id='fabricatio-v0',
             entry_point='gym_fabrikatioRL.envs:FabricatioRL', kwargs=env_args)
    env = gym.make('fabricatio-v0')
   ```
5. Running the simulation
    ```
    state, done = env.reset(), False
    while not done:
        legal_actions = env.get_legal_actions()
        state, reward, done, _ = env.step(np.random.choice(legal_actions))
    print(f'The makespan after a random run was {state.system_time}')
    ```
### Examples
This repository contains three simulation usage examples namely
1. Random action selection 
2. A simple heuristic run on randomely sampled JSSPs
3. Training and testing a Double DQN Agent from `keras-rl` with networks defined in `keras` on stochastic dynamic JSSPs with 
	* 5 operations per job
	* 20 jobs total
	* 10 initial jobs 
	* partially ordered job operations
	* custom fixed machine capabilities

To run the third example you will need to additionally install keras 2.2, tensorflow 1.4 and keras-rl
```
pip install keras==2.2
pip install tensorflow==1.4
pip install keras-rl
```

## Citing the Project
If you use `FabricatioRL` in your research, you can cite it as follows:

```
@misc{rinciog2020fabricatio-rl,
    author = {Rinciog, Alexandru and Meyer Anne},
    title = {FabricatioRL},
    year = {2021},
    publisher = {GitHub},
    journal = {GitHub Repository},
    howpublished = {\url{https://github.com/malerinc/fabricatio-rl.git}},
}
```

## References
<a id="1">[1]</a> Pinedo, Michael. Scheduling. Vol. 29. New York: Springer, 2012.
<a id="2">[2]</a> Rinciog, Alexandru, and Anne Meyer. "Towards Standardizing Reinforcement Learning Approaches for Stochastic Production Scheduling." arXiv preprint arXiv:2104.08196 (2021).