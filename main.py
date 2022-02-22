from env.ppoAviary import ppoAviary
import os
import time
import argparse
import pybullet as p
import gym
import torch
import torch.nn as nn
import ray
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib import agents
from ray.rllib.agents.ppo import PPOTrainer
from  ray.tune.logger import pretty_print

if __name__ == '__main__':
    filename = os.path.dirname(os.path.abspath(__file__))

    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    env_name = "xyz_aviary_v0"
    register_env(env_name,lambda _:ppoAviary(num_drones=3,gui=False))

    env = ppoAviary(num_drones=3,gui=False)
    observer_space = env.observation_space[0]
    action_space = env.action_space[0]
    env.close()
    config={
        "env":env_name,
        "num_gpus":1,
        "framework": "torch",
        'train_batch_size': 9000,
        "num_workers": 1,
    }
    config['model']={'fcnet_hiddens': [64,64],
                     "fcnet_activation": "tanh",
                     }
    config["multiagent"] = {
        "policies": {
            "pol_0": (None, observer_space, action_space, {}),

        },
        "policy_mapping_fn": lambda agent_id: "pol_0"  # if x == 0 else "pol1", # # Function mapping agent ids to policy ids

    }

    # ppo_config = ppo.DEFAULT_CONFIG.copy()
    # ppo_config.update(config)

    #trainer = agents.ppo.PPOTrainer(config=ppo_config, env=env_name)
    trainer = PPOTrainer(config=config, env=env_name)

    for i in range(100):
        result = trainer.train()
        if i%10==0:
            checkpoint = trainer.save(checkpoint_dir=filename+'/ppo_models')
            print("checkpoint saved at", checkpoint)
   # print(pretty_print(result))
