# CoMetaVanetEnv: Heterogeneous Multi-Agent Reinforcement Learning Environment in Mobile Computing Power Networking for the Metaverse Using SUMO

![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Version](https://img.shields.io/badge/version-1.0.0-blue) 

This project aims to create multi vanet env, using [SUMO](https://github.com/eclipse-sumo/sumo), rMAPPO-TS model based on [CADP](https://github.com/zyh1999/CADP).
This project:
- Provide a simple interface to work with Reinforcement Learning in Mobile Computing Power Networking for the Metaverse using SUMO API
- Support Multiagent Reinforcement Learning
- Compatibility (some modifications are required) with gymnasium.Env, PettingZoo.Env and popular RL libraries such as [RLlib](https://docs.ray.io/en/main/rllib.html) and [tianshou](https://github.com/thu-ml/tianshou)
- Easy customisation: modify anything in decoupled method.

This code is part of my academic research for the graduation thesis.

Random policy simulation in Seattle (Linden Avenue North 80-84th Street):
![Random simulation](random_139.gif)

rMAPPO-TS (a marl method in my thesis) policy evaluation in London (Golden Square):
![rMAPPO_London simulation](London-rMAPPO-TS.gif)

## Usage
### Init
Install requirements:
```bash
pip install -r requirements.txt
```

Run train.py to train marl model:
```bash
python train.py
```
Run eval.py to evaluation marl model:
```bash
python eval.py
```

### **Table 5.1: Applicability of Different Platforms**  
| Platform Name | Traffic Simulation | Communication Simulation | Resource Allocation | Extensibility |  
|--------------|------------------|------------------|------------------|--------------|  
| **mobile-env** | ❌ Not Supported | ✅ Supported | ⚠️ Only Supports Network Resource Allocation | ⚠️ Partially Extensible |  
| **sumo-rl** | ✅ Supported | ❌ Not Supported | ❌ Not Supported | ❌ Not Extensible |  
| **CoMetaVanetEnv** | ✅ Supported | ✅ Supported | ✅ Supports Multi-Dimensional Resource Allocation | ✅ Highly Extensible |  
