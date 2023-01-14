# CIMA - Collective Intrinsic Motivation of Agents
Combining the principles of intrinsic motivation and multi-agent learning! :robot::pray::robot:

CIMA is written in PyTorch and uses [SMAC](https://github.com/oxwhirl/smac) as its environment. MADDPG implementation is inherited from [philtabor](https://github.com/philtabor/Multi-Agent-Deep-Deterministic-Policy-Gradients). For experiments we used [LIIR](https://github.com/yalidu/liir) original implementation.

## Setup
Clone repository:
```
git clone https://github.com/multiflexor/cima.git
cd cima/
```

Create virtual environment and install dependencues:
```
python -m venv venv_cima
source venv_cima/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run an experiment
```
python cima/main.py --scenario MADDPG --map_name 2m_vs_2zg_IM --n_steps 2000000
```

## Paper
#### Collective intrinsic motivation of a multi-agent system based on reinforcement learning algorithms
Link, description, citations will be added later.

### Abstract
One of the great challenges in reinforcement learning is learning an optimal behavior in environments with sparse rewards. Solving tasks in such setting require effective exploration methods that are often based on intrinsic rewards. Plenty of real-world problems involve sparse rewards and many of them are further complicated by multi-agent setting, where the majority of intrinsic motivation methods are ineffective. In this paper we address the problem of multi-agent environments with sparse rewards and propose to combine intrinsic rewards and multi-agent reinforcement learning (MARL) technics to create the **Collective Intrinsic Motivation of Agents (CIMA)** method. CIMA uses both the external reward and the intrinsic collective reward from the cooperative multi-agent system. The proposed method can be used along with any MARL method as base reinforcement learning algorithm. We compare CIMA with several state-of-the-art MARL methods within multi-agent environment with sparse rewards designed in StarCraft II

### Methods
One way to apply intrinsic motivation in reinforcement learning is to gain knowledge about the environment an agent interacts with through exploration. We consider three sources of intrinsic motivation that can help the agent gather information about the environment with sparse rewards: state novelty, discrepancy towards other states, prediction error.

#### State Novelty
Since the interaction environment considered within StarCraft II can be represented as a two-dimensional grid world, we divided the state space into equal cells and thus were able to calculate the state novelty using the frequency of the agent ending up in that state. Lower intrinsic reward corresponds to more frequently visited states. 

We consider two versions of the implementation of this multi-agent state novelty (MASN) method - individual (**MASNi** and collective (**MASNc**). In the first one we count visits for every agent individually, and in the second one, all agents share the total number of visits.

#### Discrepancy Towards Other States
Intrinsic motivation based on discrepancy towards other states can be efficiently implemented using the Variational Autoencoder (VAE). When using VAE to project the state space into a probabilistic latent representation that reflects the internal structure of the environment, one can naturally obtain some measure of discrepancy. This measure is determined by how much the posterior distribution of the latent representation deviates from the prior assumption. Since it is difficult and impractical to determine the exact posterior distribution, it can be approximated by the variational distribution.

The Multi-agent discrepancy towards other states (MADOS) method was also implemented in two versions. In the individual version **MADOSi** each agent has its own variational autoencoder which receives input observations and action only from this particular agent and the intrinsic rewards are individual. The collective version **MADOSc** has a single common and centralized variational autoencoder for all agents. In this case the input vector contains observations and recent actions of all agents and the intrinsic reward becomes collective.


#### Prediction Error
This group of intrinsic motivation methods use the next state prediction error as the intrinsic reward. For building such intrinsic motivation we choose an Autoencoder neural network, for which the input consists of agent observations and actions. 

Two versions of Multi-agent prediction error (MAPE) method were tested for modeling individual and collective intrinsic rewards: individual autoencoder neural network for each agent (**MAPEi**) and one shared autoencoder for all agents (**CIMA**).

### Architecture of CIMA
Centralized critics use the external reward $r^{ex}$ and the intrinsic reward $r^{in}$ for learning, while the decentralized actors by collective actions $a$ and collective observations $o$ affect the training of the intrinsic motivation module.

<p align="center">
  <img src="/resources/arch.png" width=70% height=70%>
</p>

### Experiments
We evaluate proposed CIMA method on multi-agent reinforcement learning task in StarCraft II using SMAC. It provides the possibility of decentralized management of multiple agents. Each allied unit is controlled by an independently learning agent that has access only to local observation of the environment. Agents have access to such observation information as health points, mutual position and unit types of the agent itself and other units in visibility range. In this sense the system is multi-agent and partially observable.

### Multi-agent IM Environment
An IM environment was created for experiments with the collective intrinsic motivation that included a special interaction element. When any agent (or several agents) approaches the gate for the first time, it will open to let him in, after which the gate will close and remain closed for the rest of the episode no matter what happens in the environment. Agents inside this zone are safe and enemies cannot attack them. This behavior greatly increases the chances of winning in episode, but requires consistent cooperative actions in the environment with sparse rewards.

The rewards in this environment are sparse because the agents receive rewards for winning the episode (i.e. destroying enemy team) but the victory itself in some cases needs a long chain of cooperative actions (the more agents are in safe zone, the higher the chances of winning). Episode time steps are limited. The allied units were ranged marines and the enemy units were melee zerglings.

<p align="center">
  <img src="/resources/env_overview.png" width=60% />
</p>

#### Equal forces
**The Plan**: Just detroy the enemy!

<p align="center">
  <img src="/resources/2v2_attack.gif" width=50% />
</p>

#### There is numerical superiority of the enemy
**The Plan** doest'n work anymore...

<p align="center">
  <img src="/resources/2v10_attack.gif" width=50% />
</p>

#### Hide when having disadvantage
**New Plan**: Get to the safe zone!

<p align="center">
  <img src="/resources/2v10_hide.gif" width=50% />
</p>

### Results
In scenarios where the number of allied and enemy agents were equal without a need for intrinsic rewards, almost all methods show good results and achieve highest possible score. However, with an advantage in the number of agents in favor of the opponent, only the CIMA method can learn the optimal strategy (hiding to the safe zone).

#### Mean rewards in 2m_vs_2z scenario
<p align="center">
  <img src="/resources/graph_1.png" width=40% />
  <img src="/resources/graph_2.png" width=40% />
</p>
<h4 align="center">Individual IM (left), Collective IM (right)</h4>

#### Mean rewards in 2m_vs_10z scenario
<p align="center">
  <img src="/resources/graph_3.png" width=40% />
  <img src="/resources/graph_4.png" width=40% />
</p>
<h4 align="center">Individual IM (left), Collective IM (right)</h4>

#### Heatmaps
It is also important to analyze the states visited by agents in terms of the completeness of the exploration of the environment. To do this, we have built heatmaps, which show the positions of agents at different time steps. The trace of agents on heat maps gradually evaporates over time. It can be noticed that agents using the MADDPG method and a random noise for environment exploration only visit regions close to their starting position. Everything gets even worse in **2m_vs_10z** scenario because the prevailing number of opposing agents almost instantly destroys the allied agents. At the same time the agents using the CIMA learning method visit much broader regions of the map which allows them to find the safe zone and reach maximum reward parameters and number of victories.

<p align="center">
  <img src="/resources/maddpg_2v2.gif" width=20% title="maddpg_2v2" />
  <img src="/resources/maddpg_2v10.gif" width=20% title="maddpg_2v10" />
  <img src="/resources/cima_2v2.gif" width=20% title="cima_2v2" />
  <img src="/resources/cima_2v10.gif" width=20% title="cima_2v10" />
</p>

<p align="center">
  From left to right: <b>maddpg_2v2</b>, <b>maddpg_2v10</b>, <b>cima_2v2</b>, <b>cima_2v10</b>
</p>


## Licence

The MIT License
