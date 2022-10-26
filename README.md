# CIMA - Collective Intrinsic Motivation of Agents
Combining the principles of intrinsic motivation and multi-agent learning! :robot::pray::robot:

## Install
Clone repository:
```
git clone https://github.com/bolshakoVofficial/cima.git
cd cima/
```

Create virtual environment and install dependencues:
```
python -m venv venv_cima
source venv_cima/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run example
```
python cima/main.py --scenario MADDPG --map_name 2m_vs_2zg_IM --n_steps 2000000
```

## Paper
#### Collective intrinsic motivation of a multi-agent system based on reinforcement learning algorithms
Link, description, citations will be added later.

### Abstract
One of the great challenges in reinforcement learning is learning an optimal behavior in environments with sparse rewards. Solving tasks in such setting require effective exploration methods that are often based on intrinsic rewards. Plenty of real-world problems involve sparse rewards and many of them are further complicated by multi-agent setting, where the majority of intrinsic motivation methods are ineffective. In this paper we address the problem of multi-agent environments with sparse rewards and propose to combine intrinsic rewards and multi-agent reinforcement learning (MARL) technics to create the Collective Intrinsic Motivation of Agents (CIMA) method. CIMA uses both the external reward and the intrinsic collective reward from the cooperative multi-agent system. The proposed method can be used along with any MARL method as base reinforcement learning algorithm. We compare CIMA with several state-of-the-art MARL methods within multi-agent environment with sparse rewards designed in StarCraft II

### Methods

### Architecture of CIMA
<p align="center">
  <img src="/resources/arch.png" width=70% height=70%>
</p>

### Experiments
We evaluate proposed CIMA method on multi-agent reinforcement learning task in StarCraft II using SMAC. It provides the possibility of decentralized management of multiple agents. Each allied unit is controlled by an independently learning agent that has access only to local observation of the environment. Agents have access to such observation information as health points, mutual position and unit types of the agent itself and other units in visibility range. In this sense the system is multi-agent and partially observable.

### Multi-agent IM Environment


### Results
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
<p align="center">
  <img src="/resources/maddpg_2v2.gif" width=20% />
  <img src="/resources/maddpg_2v10.gif" width=20% />
  <img src="/resources/cima_2v2.gif" width=20% />
  <img src="/resources/cima_2v10.gif" width=20% />
</p>
