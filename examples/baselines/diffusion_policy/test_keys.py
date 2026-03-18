import gymnasium as gym
from diffusion_policy.make_env import make_eval_envs
env = gym.make("StackCube-v1", obs_mode="rgb", render_mode="rgb_array")
obs, _ = env.reset()
print(list(obs["sensor_data"].keys()))
env.close()
