ALGO_NAME = "BC_Diffusion_rgb_UNet_StackCube"


import os
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.prompt_wrappers import DynamicStackCubeGoalPromptWrapper
from diffusion_policy.plain_conv import PlainConv, ResNetEncoder
from diffusion_policy.utils import (IterationBasedBatchSampler,
                                    build_state_obs_extractor, convert_obs,
                                    worker_init_fn)
from diffusion_policy.heatmap_utils import bbox_centers_to_heatmap_multi_camera


CAMERA_SETTINGS = {
    "base_left": ["base_camera", "left_side_camera"],
    "left_right": ["left_side_camera", "right_side_camera"],
}
PROMPT_CAMERAS = CAMERA_SETTINGS["base_left"]


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "StackCube-v1"
    """the id of the environment"""
    demo_path: str = (
        "videos_rgbd/StackCube-v1/stackcube_expert.rgbd.pd_ee_delta_pos.physx_cpu.h5"
    )
    """the path of demo dataset, it is expected to be a ManiSkill dataset h5py format file"""
    num_demos: Optional[int] = 200
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 500_000
    """total timesteps of the experiment"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    obs_horizon: int = 2  # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = (
        16  # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    )
    diffusion_step_embed_dim: int = 64  # not very important
    unet_dims: List[int] = field(
        default_factory=lambda: [128, 256, 512]
    )  # default setting is about ~4.5M params
    n_groups: int = (
        8  # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are similar
    )
    visual_feature_dim: int = 512
    """output dimension of the visual encoder. Larger values give the UNet more rich visual conditioning."""
    encoder: str = "plainconv"
    """visual encoder type: 'plainconv' or 'resnet18'. resnet18 uses pretrained ImageNet weights."""

    # Environment/experiment specific arguments
    obs_mode: str = "depth"
    """Observation mode. Can be "rgb", "depth", or "rgb+depth"."""
    max_episode_steps: Optional[int] = 300
    """Override environment max_episode_steps. Set to 300 to exceed the longest demo (~231 steps)."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "physx_cpu" or "gpu" """
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_ee_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""
    close_camera: bool = False
    """Use closer camera view (e.g. for StackCube). Must match the camera config used when recording demonstrations."""

    camera_setting: str = "left_right"
    """Which camera pair to use for visual inputs and prompts. Options: base_left, left_right."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None

    # visual prompt conditioning (optional)
    use_visual_prompt: bool = True
    """Whether to condition policy on first-frame visual prompt (bbox/center) from prompt JSONs."""
    prompt_dir: Optional[str] = "videos_rgbd/StackCube-v1/screenshots"
    """Directory containing ep{idx}_boxes_with_corners.json / ep{idx}_boxes.json files."""
    use_heatmap_prompt: bool = True
    """Use heatmap representation for visual prompt instead of vector encoding."""
    heatmap_h: int = 48
    """Height of heatmap."""
    heatmap_w: int = 48
    """Width of heatmap."""
    heatmap_sigma: float = 0.0
    """Gaussian sigma for heatmap generation. Set <=0 to auto-compute from resolution."""
    heatmap_sigma_ratio: float = 0.1
    """When heatmap_sigma<=0, effective_sigma = max(1.0, heatmap_sigma_ratio * min(H, W))."""
    prompt_embed_dim: int = 128
    """Embedding dimension for visual prompt MLP (used if not using heatmap)."""
    prompt_dropout: float = 0.0
    """Dropout probability for prompt vector during training."""
    save_eval_prompt_viz: bool = False
    """Whether to save eval-time prompt overlay images for debugging."""


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


class SmallDemoDataset_DiffusionPolicy(Dataset):  # Load everything into memory
    def __init__(self, data_path, obs_process_fn, obs_space, include_rgb, include_depth, device, num_traj):
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.use_heatmap_prompt = args.use_heatmap_prompt
        self.heatmap_h = args.heatmap_h
        self.heatmap_w = args.heatmap_w
        self.heatmap_sigma = args.heatmap_sigma
        self._prompt_json_cache = {}
        self._prompt_episode_ids = None
        if args.use_visual_prompt:
            self._prompt_episode_ids = self._build_prompt_episode_id_map(
                data_path=data_path,
                num_traj=num_traj,
            )
        from diffusion_policy.utils import load_demo_dataset
        trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
        # trajectories['observations'] is a list of dict, each dict is a traj, with keys in obs_space, values with length L+1
        # trajectories['actions'] is a list of np.ndarray (L, act_dim)
        print("Raw trajectory loaded, beginning observation pre-processing...")

        # Pre-process the observations, make them align with the obs returned by the obs_wrapper
        obs_traj_dict_list = []
        goal_prompt_list = []
        kept_actions = []
        skipped_prompt_count = 0
        total_traj_before_filter = len(trajectories["observations"])
        for i, obs_traj_dict in enumerate(tqdm(
            trajectories["observations"],
            desc="Preprocessing demo observations",
            leave=False,
        )):
            _obs_traj_dict = reorder_keys(
                obs_traj_dict, obs_space
            )  # key order in demo is different from key order in env obs
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.Tensor(
                    _obs_traj_dict["depth"].astype(np.float32)
                ).to(device=device, dtype=torch.float16)
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"]).to(
                    device
                )  # still uint8
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).to(
                device
            )
            obs_traj_dict_list.append(_obs_traj_dict)

            # Optional visual prompt (first-frame, third-person camera only)
            if args.use_visual_prompt:
                assert self._prompt_episode_ids is not None
                prompt_episode_id = self._prompt_episode_ids[i]
                try:
                    goal_prompt = self._build_goal_prompt_from_files(
                        traj_idx=i,
                        prompt_episode_id=prompt_episode_id,
                        raw_obs_traj_dict=obs_traj_dict,
                        prompt_dir=args.prompt_dir,
                        device=device,
                    )
                except (FileNotFoundError, KeyError, ValueError) as e:
                    skipped_prompt_count += 1
                    # 回退本次 append，跳过该条轨迹，避免 obs/action/prompt 长度错位
                    obs_traj_dict_list.pop()
                    if skipped_prompt_count <= 10:
                        print(f"[prompt-skip] traj_idx={i}, episode_id={prompt_episode_id}: {e}")
                    continue
            else:
                goal_prompt = torch.zeros(4 * len(PROMPT_CAMERAS) + 1, dtype=torch.float32, device=device)

            goal_prompt_list.append(goal_prompt)
            kept_actions.append(trajectories["actions"][i])

        trajectories["observations"] = obs_traj_dict_list
        trajectories["goal_prompt"] = goal_prompt_list
        trajectories["actions"] = kept_actions
        if args.use_visual_prompt and skipped_prompt_count > 0:
            print(
                f"[prompt-skip] skipped {skipped_prompt_count}/{total_traj_before_filter} trajectories due to missing/invalid prompt JSON"
            )
        if len(trajectories["observations"]) == 0:
            raise RuntimeError("No trajectories left after prompt filtering. Check prompt_dir and prompt JSON files.")
        self.obs_keys = list(_obs_traj_dict.keys())
        # Pre-process the actions
        for i in tqdm(
            range(len(trajectories["actions"])),
            desc="Converting demo actions to tensor",
            leave=False,
        ):
            trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i]).to(
                device=device
            )

        # Normalize actions to [-1, 1] for diffusion compatibility.
        # This is needed when using absolute controllers (e.g. pd_joint_pos) whose
        # joint angles are not naturally bounded in [-1, 1].
        all_acts = torch.cat(trajectories["actions"], dim=0)  # (total_T, act_dim)
        self.action_min = all_acts.min(dim=0).values  # (act_dim,)
        self.action_max = all_acts.max(dim=0).values  # (act_dim,)
        act_range = (self.action_max - self.action_min).clamp(min=1e-8)
        for i in tqdm(
            range(len(trajectories["actions"])),
            desc="Normalizing demo actions",
            leave=False,
        ):
            a = trajectories["actions"][i]
            trajectories["actions"][i] = 2.0 * (a - self.action_min) / act_range - 1.0
        print(f"Actions normalized to [-1, 1] using per-DoF min/max from training data.")

        print(
            "Obs/action pre-processing is done, start to pre-compute the slice indices..."
        )

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if (
            "delta_pos" in args.control_mode
            or args.control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        ):
            print("Detected a delta controller type, padding with a zero action to ensure the arm stays still after solving tasks.")
            original_zero = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,), device=device
            )
            # Normalize the zero action to match the normalized action space
            self.pad_action_arm = 2.0 * (original_zero - self.action_min[:-1]) / act_range[:-1] - 1.0
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
            self.use_last_action_pad = False
        else:
            # For absolute joint pos control (e.g. pd_joint_pos), repeat the last action
            # so that the robot holds its final position
            print(f"Detected absolute controller type ({args.control_mode}), padding with last action to hold position.")
            self.pad_action_arm = None
            self.use_last_action_pad = True
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = (
            args.obs_horizon,
            args.pred_horizon,
        )
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in tqdm(
            range(num_traj),
            desc="Building training sequence indices",
            leave=False,
        ):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories

    def _build_prompt_episode_id_map(self, data_path: str, num_traj: Optional[int]):
        """Map traj index -> prompt episode id using sidecar metadata JSON.

        We prefer orig_episode_id (for datasets filtered/reindexed after generation).
        """
        if not data_path.endswith(".h5"):
            raise ValueError(f"Expected .h5 demo path, got: {data_path}")
        meta_path = data_path[:-2] + "json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing metadata json for prompt id mapping: {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        episodes = meta.get("episodes", [])
        if num_traj is not None:
            episodes = episodes[:num_traj]

        if len(episodes) == 0:
            raise ValueError("No episodes found in metadata json for prompt mapping")

        ids = []
        for i, ep in enumerate(episodes):
            if "orig_episode_id" in ep:
                ids.append(int(ep["orig_episode_id"]))
            elif "episode_id" in ep:
                ids.append(int(ep["episode_id"]))
            else:
                raise KeyError(f"Episode {i} missing both orig_episode_id and episode_id")
        return ids

    def _load_episode_prompt_json(self, prompt_dir: str, ep_idx: int):
        if prompt_dir is None:
            raise ValueError("prompt_dir must be provided when use_visual_prompt=True")
        if ep_idx in self._prompt_json_cache:
            return self._prompt_json_cache[ep_idx]

        p = os.path.join(prompt_dir, f"ep{ep_idx}_boxes.json")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing visual prompt file for episode {ep_idx}: {p}"
            )
        with open(p, "r") as f:
            data = json.load(f)
        self._prompt_json_cache[ep_idx] = data
        return data

    def _build_goal_prompt_from_files(
        self,
        traj_idx: int,
        prompt_episode_id: int,
        raw_obs_traj_dict: dict,
        prompt_dir: Optional[str],
        device,
    ):
        """
        Build visual prompt from files. Can return either:
        - Heatmap tensor (num_cameras*2, H, W) if use_heatmap_prompt=True
        - Vector (4*num_cameras + 1,) if use_heatmap_prompt=False
        
        For heatmap: first and last channels are init/goal for cam0, middle channels for cam1, etc.
        """
        if prompt_dir is None:
            raise ValueError("prompt_dir must be provided when use_visual_prompt=True")

        data = self._load_episode_prompt_json(prompt_dir, prompt_episode_id)
        cams = data.get("cameras", {})
        if len(cams) == 0:
            raise KeyError(f"No cameras found in prompt json for episode {traj_idx}")

        if self.use_heatmap_prompt:
            # Generate heatmap representation
            centers_per_camera = []
            for cam_name in PROMPT_CAMERAS:
                cam_data = cams.get(cam_name)
                if cam_data is None:
                    raise KeyError(
                        f"Camera '{cam_name}' not found in prompt json for episode {prompt_episode_id}"
                    )

                init_center = cam_data.get("init_center_px")
                goal_center = cam_data.get("goal_center_px")
                if init_center is None and isinstance(cam_data.get("init_box_corners"), dict):
                    init_center = cam_data["init_box_corners"].get("center")
                if goal_center is None and isinstance(cam_data.get("goal_box_corners"), dict):
                    goal_center = cam_data["goal_box_corners"].get("center")
                if init_center is None or goal_center is None:
                    raise KeyError(
                        f"Missing init/goal center for episode {prompt_episode_id} (traj_idx={traj_idx}), camera '{cam_name}'"
                    )

                H, W = self._get_camera_hw(raw_obs_traj_dict, cam_name)
                if H <= 0 or W <= 0:
                    raise ValueError(f"Invalid image size for prompt camera '{cam_name}': H={H}, W={W}")

                # Normalize to [0, 1]
                init_norm = torch.tensor(
                    [float(init_center[0]) / W, float(init_center[1]) / H],
                    dtype=torch.float32,
                    device=device,
                )
                goal_norm = torch.tensor(
                    [float(goal_center[0]) / W, float(goal_center[1]) / H],
                    dtype=torch.float32,
                    device=device,
                )
                centers_per_camera.append((init_norm.unsqueeze(0), goal_norm.unsqueeze(0)))

            # Generate multi-camera heatmaps (1, num_cameras*2, H, W)
            heatmaps = bbox_centers_to_heatmap_multi_camera(
                centers_per_camera,
                heatmap_h=self.heatmap_h,
                heatmap_w=self.heatmap_w,
                sigma=self.heatmap_sigma,
            )
            return heatmaps[0]  # Remove batch dim: (num_cameras*2, H, W)
        else:
            # Generate vector representation (legacy)
            vec = torch.zeros(4 * len(PROMPT_CAMERAS) + 1, dtype=torch.float32, device=device)
            write_idx = 0
            for cam_name in PROMPT_CAMERAS:
                cam_data = cams.get(cam_name)
                if cam_data is None:
                    raise KeyError(
                        f"Camera '{cam_name}' not found in prompt json for episode {prompt_episode_id} (traj_idx={traj_idx}). "
                        f"Available cameras: {list(cams.keys())}"
                    )

                init_center = cam_data.get("init_center_px")
                goal_center = cam_data.get("goal_center_px")
                if init_center is None and isinstance(cam_data.get("init_box_corners"), dict):
                    init_center = cam_data["init_box_corners"].get("center")
                if goal_center is None and isinstance(cam_data.get("goal_box_corners"), dict):
                    goal_center = cam_data["goal_box_corners"].get("center")
                if init_center is None or goal_center is None:
                    raise KeyError(
                        f"Missing init/goal center for episode {prompt_episode_id} (traj_idx={traj_idx}), camera '{cam_name}'"
                    )

                H, W = self._get_camera_hw(raw_obs_traj_dict, cam_name)
                if H <= 0 or W <= 0:
                    raise ValueError(f"Invalid image size for prompt camera '{cam_name}': H={H}, W={W}")

                vec[write_idx + 0] = float(init_center[0]) / W
                vec[write_idx + 1] = float(init_center[1]) / H
                vec[write_idx + 2] = float(goal_center[0]) / W
                vec[write_idx + 3] = float(goal_center[1]) / H
                write_idx += 4

            vec[-1] = 1.0
            return vec

    @staticmethod
    def _get_camera_hw(raw_obs_traj_dict: dict, cam_name: str):
        """Infer camera frame size from available sensor streams without requiring RGB."""
        cam_sensor = raw_obs_traj_dict["sensor_data"][cam_name]
        for key in ("depth", "rgb"):
            if key not in cam_sensor:
                continue
            arr = cam_sensor[key]
            if arr.ndim < 3:
                continue
            # (T, H, W, C?)
            return int(arr.shape[1]), int(arr.shape[2])
        raise KeyError(
            f"Camera '{cam_name}' has neither usable depth nor rgb stream for size inference"
        )

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[
                max(0, start) : start + self.obs_horizon
            ]  # start+self.obs_horizon is at least 1
            if start < 0:  # pad before the trajectory
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
            # don't need to pad obs after the trajectory, see the above char drawing

        act_seq = self.trajectories["actions"][traj_idx][max(0, start) : end]
        if start < 0:  # pad before the trajectory
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            if self.use_last_action_pad:
                # absolute control: hold the final joint positions
                pad_action = act_seq[-1]
            else:
                # delta control: zero arm delta, copy gripper state
                gripper_action = act_seq[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.unsqueeze(0).repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert (
            obs_seq["state"].shape[0] == self.obs_horizon
            and act_seq.shape[0] == self.pred_horizon
        )
        return {
            "observations": obs_seq,
            "actions": act_seq,
            "goal_prompt": self.trajectories["goal_prompt"][traj_idx],
        }

    def __len__(self):
        return len(self.slices)


class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: Args):
        super().__init__()
        self.args = args
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        self.use_visual_prompt = args.use_visual_prompt
        self.use_heatmap_prompt = args.use_heatmap_prompt if args.use_visual_prompt else False
        self.heatmap_h = args.heatmap_h
        self.heatmap_w = args.heatmap_w
        
        if self.use_heatmap_prompt:
            self.heatmap_channels = 2 * len(PROMPT_CAMERAS)  # init + goal for each camera
        else:
            self.prompt_raw_dim = 4 * len(PROMPT_CAMERAS) + 1
        
        assert (
            len(env.single_observation_space["state"].shape) == 2
        )  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )
        # Note: we normalize actions to [-1, 1] in the dataset so we do not require
        # the env action space to be bounded in [-1, 1] here.
        self.act_dim = env.single_action_space.shape[0]
        # Buffers for action denormalization (set from dataset after construction)
        self.register_buffer("action_min", torch.zeros(self.act_dim))
        self.register_buffer("action_max", torch.ones(self.act_dim))
        self.action_normed = True  # always normalize/denormalize
        obs_state_dim = env.single_observation_space["state"].shape[1]
        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        visual_feature_dim = args.visual_feature_dim
        encoder_type = getattr(args, "encoder", "plainconv").lower()
        if encoder_type == "resnet18":
            self.visual_encoder = ResNetEncoder(
                in_channels=total_visual_channels,
                out_dim=visual_feature_dim,
                pretrained=True,
            )
            print(f"[encoder] Using pretrained ResNet18 encoder (in_channels={total_visual_channels}, out_dim={visual_feature_dim})")
        else:
            self.visual_encoder = PlainConv(
                in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=True
            )
            print(f"[encoder] Using PlainConv encoder (in_channels={total_visual_channels}, out_dim={visual_feature_dim})")

        prompt_cond_dim = 0
        if self.use_visual_prompt:
            if self.use_heatmap_prompt:
                # Use a simple CNN to encode heatmaps into feature vectors
                # Heatmap shape: (heatmap_channels, heatmap_h, heatmap_w)
                self.heatmap_encoder = nn.Sequential(
                    nn.Conv2d(self.heatmap_channels, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),  # (H/2, W/2)
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),  # (H/4, W/4)
                    nn.AdaptiveAvgPool2d(1),  # (1, 1)
                    nn.Flatten(),
                    nn.Linear(64, args.prompt_embed_dim),
                )
                prompt_cond_dim = args.prompt_embed_dim
                print(f"[prompt] Using heatmap encoder (channels={self.heatmap_channels}, H={self.heatmap_h}, W={self.heatmap_w}, embed_dim={args.prompt_embed_dim})")
            else:
                # Use MLP to encode bbox vector
                self.prompt_encoder = nn.Sequential(
                    nn.Linear(self.prompt_raw_dim, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, args.prompt_embed_dim),
                )
                prompt_cond_dim = args.prompt_embed_dim
                print(f"[prompt] Using vector encoder (dim={self.prompt_raw_dim} -> {args.prompt_embed_dim})")

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim) + prompt_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )

    def encode_obs(self, obs_seq, eval_mode, goal_prompt=None):
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0  # (B, obs_horizon, 1*k, H, W)
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W), C=4*k
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)  # (B*obs_horizon, C, H, W)
        visual_feature = self.visual_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(
            batch_size, self.obs_horizon, visual_feature.shape[1]
        )  # (B, obs_horizon, D)
        feature = torch.cat(
            (visual_feature, obs_seq["state"]), dim=-1
        )  # (B, obs_horizon, D+obs_state_dim)
        obs_cond = feature.flatten(start_dim=1)  # (B, obs_horizon * (D+obs_state_dim))

        if not self.use_visual_prompt:
            return obs_cond

        if self.use_heatmap_prompt:
            # Handle heatmap prompt
            if goal_prompt is None:
                goal_prompt = torch.zeros(
                    (batch_size, self.heatmap_channels, self.heatmap_h, self.heatmap_w),
                    dtype=obs_cond.dtype,
                    device=obs_cond.device,
                )
            else:
                if goal_prompt.ndim == 3:
                    goal_prompt = goal_prompt.unsqueeze(0)
                goal_prompt = goal_prompt.to(device=obs_cond.device, dtype=obs_cond.dtype)

            # Apply heatmap encoder
            prompt_cond = self.heatmap_encoder(goal_prompt)  # (B, prompt_embed_dim)
        else:
            # Handle vector prompt
            if goal_prompt is None:
                goal_prompt = torch.zeros(
                    (batch_size, self.prompt_raw_dim),
                    dtype=obs_cond.dtype,
                    device=obs_cond.device,
                )
            else:
                if goal_prompt.ndim == 1:
                    goal_prompt = goal_prompt.unsqueeze(0)
                goal_prompt = goal_prompt.to(device=obs_cond.device, dtype=obs_cond.dtype)

            if (not eval_mode) and self.args.prompt_dropout > 0:
                keep_mask = (
                    torch.rand((batch_size, 1), device=obs_cond.device)
                    > self.args.prompt_dropout
                ).to(obs_cond.dtype)
                goal_prompt = goal_prompt * keep_mask

            prompt_cond = self.prompt_encoder(goal_prompt)  # (B, prompt_embed_dim)

        return torch.cat([obs_cond, prompt_cond], dim=-1)

    def compute_loss(self, obs_seq, action_seq, goal_prompt=None):
        B = obs_seq["state"].shape[0]

        # observation as FiLM conditioning
        obs_cond = self.encode_obs(
            obs_seq, eval_mode=False, goal_prompt=goal_prompt
        )  # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond
        )

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq, goal_prompt=None):
        # obs_seq['state']: (B, obs_horizon, obs_state_dim)
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            obs_cond = self.encode_obs(
                obs_seq, eval_mode=True, goal_prompt=goal_prompt
            )  # (B, obs_horizon * obs_dim)

            # initialize action from Gaussian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        actions = noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim), normalized
        if self.action_normed:
            # denormalize from [-1, 1] back to original action space
            act_range = (self.action_max - self.action_min).clamp(min=1e-8)
            actions = (actions + 1.0) / 2.0 * act_range + self.action_min
        return actions  # (B, act_horizon, act_dim)


def save_ckpt(run_name, tag):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save(
        {
            "agent": agent.state_dict(),
            "ema_agent": ema_agent.state_dict(),
        },
        f"runs/{run_name}/checkpoints/{tag}.pt",
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.camera_setting not in CAMERA_SETTINGS:
        raise ValueError(f"camera_setting must be one of {sorted(CAMERA_SETTINGS.keys())}, got {args.camera_setting}")
    PROMPT_CAMERAS = CAMERA_SETTINGS[args.camera_setting]

    if args.use_heatmap_prompt:
        if args.heatmap_h <= 0 or args.heatmap_w <= 0:
            raise ValueError("heatmap_h and heatmap_w must be positive")
        if args.heatmap_sigma <= 0:
            args.heatmap_sigma = max(
                1.0, float(args.heatmap_sigma_ratio) * float(min(args.heatmap_h, args.heatmap_w))
            )
            print(
                f"[heatmap] auto sigma enabled: sigma={args.heatmap_sigma:.3f} "
                f"for resolution {args.heatmap_h}x{args.heatmap_w} "
                f"(ratio={args.heatmap_sigma_ratio})"
            )

    if args.use_visual_prompt and not args.prompt_dir:
        raise ValueError("--use-visual-prompt requires --prompt-dir (strict mode, no fallback)")

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    demo_info = None
    if args.demo_path.endswith(".h5"):
        import json

        json_file = args.demo_path[:-2] + "json"
        with open(json_file, "r") as f:
            demo_info = json.load(f)
            if "control_mode" in demo_info["env_info"]["env_kwargs"]:
                control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
            elif "control_mode" in demo_info["episodes"][0]:
                control_mode = demo_info["episodes"][0]["control_mode"]
            else:
                raise Exception("Control mode not found in json")
            assert (
                control_mode == args.control_mode
            ), f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # create evaluation environment
    # sensor_configs resizes all cameras to 128×128 to avoid concat failures
    # when cameras have different resolutions (e.g. hand_camera=128 vs base_camera=512)
    excluded_camera_names = {"hand_camera"}
    all_camera_names = {"base_camera", "left_side_camera", "right_side_camera"}
    excluded_camera_names |= all_camera_names - set(PROMPT_CAMERAS)
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        sensor_configs=dict(width=128, height=128),
        robot_uids="panda_wristcam",
    )
    assert args.max_episode_steps is not None, "max_episode_steps must be specified as imitation learning algorithms task solve speed is dependent on the data you train on"
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    # Sync env-specific params from demo so eval env matches training data
    if demo_info is not None:
        demo_env_kwargs = demo_info.get("env_info", {}).get("env_kwargs", {})
        for key in ["close_camera", "num_distractor_cubes"]:
            if key in demo_env_kwargs:
                env_kwargs[key] = demo_env_kwargs[key]
        print(f"[env] Synced from demo: close_camera={env_kwargs.get('close_camera')}, "
              f"num_distractor_cubes={env_kwargs.get('num_distractor_cubes')}")
    if args.close_camera:
        env_kwargs["close_camera"] = True
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    eval_wrappers = [partial(FlattenRGBDObservationWrapper, exclude_camera_names=excluded_camera_names)]
    if args.use_visual_prompt:
        eval_prompt_viz_dir = f"runs/{run_name}/prompt_viz" if args.save_eval_prompt_viz else None
        eval_wrappers = [
            partial(FlattenRGBDObservationWrapper, exclude_camera_names=excluded_camera_names),
            partial(
                DynamicStackCubeGoalPromptWrapper,
                output_dir=eval_prompt_viz_dir,
                save_visualizations=args.save_eval_prompt_viz,
                use_heatmap_prompt=args.use_heatmap_prompt,
                heatmap_h=args.heatmap_h,
                heatmap_w=args.heatmap_w,
                heatmap_sigma=args.heatmap_sigma,
                prompt_cameras=PROMPT_CAMERAS,
            ),
        ]

    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=eval_wrappers,
    )

    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="DiffusionPolicy",
            tags=["diffusion_policy"],
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(
            np.transpose, axes=(0, 3, 1, 2)
        ),  # (B, H, W, C) -> (B, C, H, W)
        state_obs_extractor=build_state_obs_extractor(args.env_id),
        rgb=(args.obs_mode in {"rgb", "rgb+depth"}),
        depth=(args.obs_mode in {"depth", "rgb+depth"}),
        exclude_camera_names=excluded_camera_names,
    )

    # create temporary env to get original observation space as AsyncVectorEnv (CPU parallelization) doesn't permit that
    tmp_env = gym.make(args.env_id, **env_kwargs)
    orignal_obs_space = tmp_env.observation_space
    # determine whether the env will return rgb and/or depth data
    include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
    include_depth = tmp_env.unwrapped.obs_mode_struct.visual.depth
    tmp_env.close()

    dataset = SmallDemoDataset_DiffusionPolicy(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=orignal_obs_space,
        include_rgb=include_rgb,
        include_depth=include_depth,
        device=device,
        num_traj=args.num_demos
    )
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
    )

    agent = Agent(envs, args).to(device)
    # Wire action normalization stats from the training dataset into the agent
    agent.action_min.copy_(dataset.action_min)
    agent.action_max.copy_(dataset.action_max)

    optimizer = optim.AdamW(
        params=agent.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6
    )

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)
    # Wire action normalization stats into ema_agent too (not copied by EMAModel)
    ema_agent.action_min.copy_(dataset.action_min)
    ema_agent.action_max.copy_(dataset.action_max)

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    # define evaluation and logging functions
    def evaluate_and_save_best(iteration):
        if iteration % args.eval_freq == 0:
            last_tick = time.time()
            ema.copy_to(ema_agent.parameters())
            eval_metrics = evaluate(
                args.num_eval_episodes, ema_agent, envs, device, args.sim_backend
            )
            timings["eval"] += time.time() - last_tick

            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(
                        f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                    )
    def log_metrics(iteration):
        if iteration % args.log_freq == 0:
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    agent.train()
    pbar = tqdm(total=args.total_iters)
    last_tick = time.time()
    for iteration, data_batch in enumerate(train_dataloader):
        timings["data_loading"] += time.time() - last_tick

        # forward and compute loss
        last_tick = time.time()
        total_loss = agent.compute_loss(
            obs_seq=data_batch["observations"],  # (B, obs_horizon, ...)
            action_seq=data_batch["actions"],  # (B, pred_horizon, act_dim)
            goal_prompt=data_batch.get("goal_prompt"),
        )
        timings["forward"] += time.time() - last_tick

        # backward
        last_tick = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()  # step lr scheduler every batch, this is different from standard pytorch behavior
        timings["backward"] += time.time() - last_tick

        # ema step
        last_tick = time.time()
        ema.step(agent.parameters())
        timings["ema"] += time.time() - last_tick

        # Evaluation
        evaluate_and_save_best(iteration)
        log_metrics(iteration)

        # Checkpoint
        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration))
        pbar.update(1)
        pbar.set_postfix({"loss": total_loss.item()})
        last_tick = time.time()

    evaluate_and_save_best(args.total_iters)
    log_metrics(args.total_iters)

    envs.close()
    writer.close()
