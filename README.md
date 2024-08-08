# Highway Agent 401

## Install

```bash
# install latest dev highway-env 
 pip install git+https://github.com/eleurent/highway-env
 pip install -r requirements.txt
```

# This repo contains two scripts to train/test agents.
# src/rl_agents_framework.py -> Uses rl_agents repo to create models 
# src/sb3_framework.py -> Uses SB3 repo to create models 

# To train with rl_agent framework:
cd src
py -m rl_agents_framework evaluate rl_agents_config/[ENVIRONMENT_CONFIG_JSON_PATH] --train --episodes=NUM_EPISODES
# e.g.  py -m rl_agents_framework evaluate rl_agents_configs/401Env/env_multi_agent.json rl_agents_configs/401Env/agents/DQNAgent/dqn_cnn.json --train --episodes=2500 --cnn
# To test with rl_agent_framework
py -m rl_agents_framework evaluate rl_agents_config/[ENVIRONMENT_CONFIG_JSON_PATH] --test --episodes=NUM_EPISODES --recover-from [CHKPT_PATH]
# e.g. py -m rl_agents_framework evaluate rl_agents_configs/401Env/env_multi_agent.json rl_agents_configs/401Env/agents/DQNAgent/dqn_cnn.json --test --repeat 5 --episodes=2 --cnn --recover-from chkpt/g80_t5000_lr1e3.tar

# To train/test with sb3 framework:
cd src
py -m sb3_framework [ENVIRONMENT_NAME] [AGENT_NAME] --train_steps TRAIN_STEPS --test_runs TEST_RUNS --network [MLP or CNN]
# e.g. py -m sb3_framework intersection-v0 PPO --train_steps 1000 --test_runs 10 --hyperparameter --network MLP