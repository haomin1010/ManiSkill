import gymnasium as gym
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.make_env import make_eval_envs

env_kwargs = dict(
    control_mode="pd_ee_delta_pos",
    reward_mode="sparse",
    obs_mode="rgb",
    render_mode="rgb_array",
    human_render_camera_configs=dict(shader_pack="default"),
    sensor_configs=dict(width=128, height=128),
    max_episode_steps=200,
)
other_kwargs = dict(obs_horizon=2)

envs = make_eval_envs(
    "StackCube-v1",
    1,
    "physx_cpu",
    env_kwargs,
    other_kwargs,
    video_dir=None,
    wrappers=[FlattenRGBDObservationWrapper],
)

obs, _ = envs.reset()
for k, v in obs.items():
    print(f"Eval env obs key: {k}, shape: {v.shape}, dtype: {v.dtype}")

envs.close()
