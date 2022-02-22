from ray.tune import register_env
from env.ppoAviary import ppoAviary

if __name__ == '__main__':
    env_name = "xyz_aviary_v0"
    register_env(env_name, lambda _: ppoAviary(num_drones=3, gui=False))

    env = ppoAviary(num_drones=3, gui=False)
    observer_space = env.observation_space[0]
    action_space = env.action_space[0]
    print("obs",observer_space)
    print("action",action_space)
