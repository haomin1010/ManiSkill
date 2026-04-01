ALGO_NAME = 'BC_ACT_stackcube'

import os
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
import random
import time
import json
import copy
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Optional, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import tyro
from diffusers.training_utils import EMAModel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from act.evaluate import evaluate
from act.make_env import make_eval_envs
from act.utils import IterationBasedBatchSampler, worker_init_fn
from act.detr.backbone import build_backbone
from act.detr.transformer import build_transformer
from act.detr.detr_vae import build_encoder, DETRVAE
from act.prompt_wrappers import DynamicStackCubeGoalPromptWrapper
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", UserWarning)

SELECTED_CAMERAS = ["base_camera", "left_side_camera", "hand_camera"]
PROMPT_CAMERAS = ["base_camera", "left_side_camera"]
PROMPT_CENTER_DIM_PER_CAMERA = 4
PROMPT_RAW_DIM = PROMPT_CENTER_DIM_PER_CAMERA * len(PROMPT_CAMERAS) + 1


def prompt_raw_dim() -> int:
    return PROMPT_RAW_DIM


def _center_from_value(center):
    if center is None:
        return None
    return [float(center[0]), float(center[1])]


def _extract_prompt_center_xy(cam_data: dict, prefix: str):
    center = cam_data.get(f"{prefix}_center_px")
    return _center_from_value(center)


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    capture_video: bool = True

    env_id: str = "StackCube-v1"
    demo_path: str = "videos/StackCube-new/videos_0401/videos/StackCube-v1/stackcube_expert.rgb.pd_ee_delta_pos.physx_cpu.h5"
    num_demos: Optional[int] = 100
    total_iters: int = 500_000
    batch_size: int = 64

    lr: float = 1e-4
    kl_weight: float = 10
    temporal_agg: bool = False

    position_embedding: str = 'sine'
    backbone: str = 'resnet18'
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    include_depth: bool = False
    image_size: int = 512
    quantize_rgb_to_uint8: bool = True

    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 1024
    hidden_dim: int = 384
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 30
    pre_norm: bool = False

    max_episode_steps: Optional[int] = 300
    log_freq: int = 1000
    eval_freq: int = 5000
    save_freq: Optional[int] = None
    num_eval_episodes: int = 20
    num_eval_envs: int = 10
    sim_backend: str = "physx_cpu"
    num_dataload_workers: int = 0
    control_mode: str = 'pd_ee_delta_pos'
    close_camera: bool = False
    use_right_side_camera: bool = False

    use_amp: bool = True
    amp_dtype: str = "bf16"

    use_visual_prompt: bool = True
    prompt_dir: Optional[str] = "videos/StackCube-new/videos_0401/videos/StackCube-v1/screenshots"
    prompt_embed_dim: int = 64
    save_eval_prompt_viz: bool = False


class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, rgb=True, depth=False, state=True, image_size: int = 224, quantize_rgb_to_uint8: bool = True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.quantize_rgb_to_uint8 = quantize_rgb_to_uint8
        self.transforms = T.Compose([T.Resize((image_size, image_size), antialias=True)])
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        goal_prompt = observation.pop("goal_prompt", None)
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        images_rgb = []
        images_depth = []
        for cam_name in SELECTED_CAMERAS:
            cam_data = sensor_data[cam_name]
            if self.include_rgb:
                resized_rgb = self.transforms(cam_data["rgb"].permute(0, 3, 1, 2))
                if self.quantize_rgb_to_uint8:
                    resized_rgb = torch.clamp(resized_rgb.round(), 0, 255).to(torch.uint8)
                images_rgb.append(resized_rgb)
            if self.include_depth:
                depth = (cam_data["depth"].to(torch.float32) / 1024).to(torch.float16)
                resized_depth = self.transforms(depth.permute(0, 3, 1, 2))
                images_depth.append(resized_depth)

        ret = dict()
        if self.include_rgb:
            ret["rgb"] = torch.stack(images_rgb, dim=1)
        if self.include_depth:
            ret["depth"] = torch.stack(images_depth, dim=1)
        if self.include_state:
            ret["state"] = common.flatten_state_dict(observation, use_torch=True)
        if goal_prompt is not None:
            ret["goal_prompt"] = goal_prompt
        return ret


class SmallDemoDataset_ACTPolicy(Dataset):
    def __init__(self, data_path, num_queries, num_traj, include_depth=False, image_size: int = 224, prompt_dir: Optional[str] = None, quantize_rgb_to_uint8: bool = True):
        from act.utils import load_demo_dataset
        trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
        print('Raw trajectory loaded, start to pre-process the observations...')

        self.include_depth = include_depth
        self.image_size = image_size
        self.prompt_dir = prompt_dir
        self.quantize_rgb_to_uint8 = quantize_rgb_to_uint8
        self.transforms = T.Compose([T.Resize((image_size, image_size), antialias=True)])
        self._prompt_json_cache = {}
        self._prompt_episode_ids = self._build_prompt_episode_id_map(data_path, num_traj) if args.use_visual_prompt else None

        obs_traj_dict_list = []
        goal_prompt_list = []
        for traj_idx, obs_traj_dict in enumerate(
            tqdm(
                trajectories['observations'],
                desc="Preprocessing demo observations",
                leave=False,
            )
        ):
            raw_obs = copy.deepcopy(obs_traj_dict)
            obs_traj_dict = self.process_obs(copy.deepcopy(obs_traj_dict))
            obs_traj_dict_list.append(obs_traj_dict)
            if args.use_visual_prompt:
                prompt_episode_id = self._prompt_episode_ids[traj_idx]
                goal_prompt = self._build_goal_prompt_from_files(traj_idx, prompt_episode_id, raw_obs, prompt_dir, device='cpu')
            else:
                goal_prompt = torch.zeros(prompt_raw_dim(), dtype=torch.float32)
            goal_prompt_list.append(goal_prompt)
        trajectories['observations'] = obs_traj_dict_list
        trajectories['goal_prompt'] = goal_prompt_list
        self.obs_keys = list(obs_traj_dict.keys())

        for i in tqdm(
            range(len(trajectories['actions'])),
            desc="Converting demo actions to tensor",
            leave=False,
        ):
            trajectories['actions'][i] = torch.Tensor(trajectories['actions'][i])
        print('Obs/action pre-processing is done.')

        if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
            self.pad_action_arm = torch.zeros((trajectories['actions'][0].shape[1]-1,))

        self.slices = []
        self.num_traj = len(trajectories['actions'])
        for traj_idx in tqdm(
            range(self.num_traj),
            desc="Building training sequence indices",
            leave=False,
        ):
            episode_len = trajectories['actions'][traj_idx].shape[0]
            self.slices += [(traj_idx, ts) for ts in range(episode_len)]

        print(f"Length of Dataset: {len(self.slices)}")
        self.num_queries = num_queries
        self.trajectories = trajectories
        self.delta_control = 'delta' in args.control_mode
        self.norm_stats = self.get_norm_stats() if not self.delta_control else None

    def _build_prompt_episode_id_map(self, data_path, num_traj):
        json_file = data_path[:-2] + 'json' if data_path.endswith('.h5') else None
        if json_file is None or not os.path.exists(json_file):
            return list(range(num_traj))
        with open(json_file, 'r') as f:
            meta = json.load(f)
        ids = []
        for i, ep in enumerate(meta['episodes'][:num_traj]):
            if 'orig_episode_id' in ep:
                ids.append(int(ep['orig_episode_id']))
            elif 'episode_id' in ep:
                ids.append(int(ep['episode_id']))
            else:
                ids.append(i)
        return ids

    def _load_episode_prompt_json(self, prompt_dir: str, ep_idx: int):
        if ep_idx in self._prompt_json_cache:
            return self._prompt_json_cache[ep_idx]
        candidates = [
            os.path.join(prompt_dir, f"ep{ep_idx}_boxes_with_corners.json"),
            os.path.join(prompt_dir, f"ep{ep_idx}_boxes.json"),
        ]
        p = next((candidate for candidate in candidates if os.path.exists(candidate)), None)
        if p is None:
            raise FileNotFoundError(f"Missing visual prompt file for episode {ep_idx}. Checked: {candidates}")
        with open(p, 'r') as f:
            data = json.load(f)
        self._prompt_json_cache[ep_idx] = data
        return data

    def _build_goal_prompt_from_files(self, traj_idx: int, prompt_episode_id: int, raw_obs_traj_dict: dict, prompt_dir: Optional[str], device='cpu'):
        vec = torch.zeros(prompt_raw_dim(), dtype=torch.float32, device=device)
        if prompt_dir is None:
            raise ValueError('prompt_dir must be provided when use_visual_prompt=True')
        data = self._load_episode_prompt_json(prompt_dir, prompt_episode_id)
        cams = data.get('cameras', {})
        if len(cams) == 0:
            raise KeyError(f'No cameras found in prompt json for episode {traj_idx}')
        write_idx = 0
        for cam_name in PROMPT_CAMERAS:
            cam_data = cams.get(cam_name)
            if cam_data is None:
                raise KeyError(f"Camera '{cam_name}' missing in prompt json")
            init_center = _extract_prompt_center_xy(cam_data, 'init')
            goal_center = _extract_prompt_center_xy(cam_data, 'goal')
            if init_center is None or goal_center is None:
                raise KeyError(f"Missing init/goal center for episode {prompt_episode_id}, camera '{cam_name}'")
            rgb = raw_obs_traj_dict['sensor_data'][cam_name]['rgb']
            H, W = rgb.shape[1], rgb.shape[2]
            init_center = [
                min(max(float(init_center[0]) / W, 0.0), 1.0),
                min(max(float(init_center[1]) / H, 0.0), 1.0),
            ]
            goal_center = [
                min(max(float(goal_center[0]) / W, 0.0), 1.0),
                min(max(float(goal_center[1]) / H, 0.0), 1.0),
            ]
            vec[write_idx:write_idx + 2] = torch.tensor(init_center, dtype=torch.float32, device=device)
            vec[write_idx + 2:write_idx + 4] = torch.tensor(goal_center, dtype=torch.float32, device=device)
            write_idx += PROMPT_CENTER_DIM_PER_CAMERA
        vec[-1] = 1.0
        return vec

    def __getitem__(self, index):
        traj_idx, ts = self.slices[index]
        state = self.trajectories['observations'][traj_idx]['state'][ts]
        act_seq = self.trajectories['actions'][traj_idx][ts:ts+self.num_queries]
        action_len = act_seq.shape[0]
        if action_len < self.num_queries:
            if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
                gripper_action = act_seq[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
                act_seq = torch.cat([act_seq, pad_action.repeat(self.num_queries-action_len, 1)], dim=0)
            elif not self.delta_control:
                target = act_seq[-1]
                act_seq = torch.cat([act_seq, target.repeat(self.num_queries-action_len, 1)], dim=0)

        if not self.delta_control:
            state = (state - self.norm_stats['state_mean'][0]) / self.norm_stats['state_std'][0]
            act_seq = (act_seq - self.norm_stats['action_mean']) / self.norm_stats['action_std']

        goal_prompt = self.trajectories['goal_prompt'][traj_idx]
        rgb = self.trajectories['observations'][traj_idx]['rgb'][ts]
        obs = dict(state=state, rgb=rgb, goal_prompt=goal_prompt)
        if self.include_depth:
            obs['depth'] = self.trajectories['observations'][traj_idx]['depth'][ts]
        return {'observations': obs, 'actions': act_seq}

    def __len__(self):
        return len(self.slices)

    def process_obs(self, obs_dict):
        sensor_data = obs_dict.pop('sensor_data')
        del obs_dict['sensor_param']
        images_rgb = []
        images_depth = []
        for cam_name in SELECTED_CAMERAS:
            cam_data = sensor_data[cam_name]
            rgb = torch.from_numpy(cam_data['rgb'])
            resized_rgb = self.transforms(rgb.permute(0, 3, 1, 2))
            if self.quantize_rgb_to_uint8:
                resized_rgb = torch.clamp(resized_rgb.round(), 0, 255).to(torch.uint8)
            images_rgb.append(resized_rgb)
            if self.include_depth:
                depth = torch.Tensor(cam_data['depth'].astype(np.float32) / 1024).to(torch.float16)
                resized_depth = self.transforms(depth.permute(0, 3, 1, 2))
                images_depth.append(resized_depth)
        rgb = torch.stack(images_rgb, dim=1)
        processed_obs = dict(rgb=rgb)
        if self.include_depth:
            depth = torch.stack(images_depth, dim=1)
            processed_obs['depth'] = depth
        obs_dict['extra'] = {k: v[:, None] if len(v.shape) == 1 else v for k, v in obs_dict['extra'].items()}
        obs_dict = common.flatten_state_dict(obs_dict, use_torch=True)
        processed_obs['state'] = obs_dict
        return processed_obs

    def get_norm_stats(self):
        state = torch.cat([traj['state'] for traj in self.trajectories['observations']], dim=0)
        action = torch.cat(self.trajectories['actions'], dim=0)
        action_mean = action.mean(dim=0, keepdim=True)
        action_std = action.std(dim=0, keepdim=True)
        state_mean = state.mean(dim=0, keepdim=True)
        state_std = state.std(dim=0, keepdim=True)
        state_std = torch.clip(state_std, 1e-2, np.inf)
        action_std = torch.clip(action_std, 1e-2, np.inf)
        return {'action_mean': action_mean, 'action_std': action_std, 'state_mean': state_mean, 'state_std': state_std}


class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        assert len(env.single_observation_space['state'].shape) == 1
        assert len(env.single_observation_space['rgb'].shape) == 4
        assert len(env.single_action_space.shape) == 1

        self.base_state_dim = env.single_observation_space['state'].shape[0]
        self.goal_prompt_dim = env.single_observation_space['goal_prompt'].shape[0] if args.use_visual_prompt else 0
        self.prompt_embed_dim = args.prompt_embed_dim if args.use_visual_prompt else 0
        self.state_dim = self.base_state_dim + self.prompt_embed_dim
        self.act_dim = env.single_action_space.shape[0]
        self.kl_weight = args.kl_weight
        self.include_depth = args.include_depth
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if args.use_visual_prompt:
            self.prompt_encoder = nn.Sequential(
                nn.Linear(self.goal_prompt_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.prompt_embed_dim),
            )
        else:
            self.prompt_encoder = None

        backbones = [build_backbone(args)]
        transformer = build_transformer(args)
        encoder = build_encoder(args)
        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=self.state_dim,
            action_dim=self.act_dim,
            num_queries=args.num_queries,
        )
        if args.lr_backbone <= 0 and self.model.backbones is not None:
            for p in self.model.backbones.parameters():
                p.requires_grad_(False)

    def _augment_state(self, obs):
        if self.prompt_encoder is None:
            return obs
        prompt = obs['goal_prompt'].to(torch.float32)
        prompt_emb = self.prompt_encoder(prompt)
        obs = dict(obs)
        obs['state'] = torch.cat([obs['state'], prompt_emb], dim=-1)
        return obs

    def compute_loss(self, obs, action_seq):
        obs = dict(obs)
        obs['rgb'] = self.normalize(obs['rgb'].float() / 255.0)
        if self.include_depth and 'depth' in obs:
            obs['depth'] = obs['depth'].float()
        obs = self._augment_state(obs)
        a_hat, (mu, logvar) = self.model(obs, action_seq)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        all_l1 = F.l1_loss(action_seq, a_hat, reduction='none')
        l1 = all_l1.mean()
        loss_dict = dict()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
        return loss_dict

    def get_action(self, obs):
        obs = dict(obs)
        obs['rgb'] = self.normalize(obs['rgb'].float() / 255.0)
        if self.include_depth and 'depth' in obs:
            obs['depth'] = obs['depth'].float()
        obs = self._augment_state(obs)
        a_hat, (_, _) = self.model(obs)
        return a_hat


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld


def save_ckpt(run_name, tag):
    os.makedirs(f'runs/{run_name}/checkpoints', exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save({
        'norm_stats': dataset.norm_stats,
        'agent': agent.state_dict(),
        'ema_agent': ema_agent.state_dict(),
    }, f'runs/{run_name}/checkpoints/{tag}.pt')


if __name__ == '__main__':
    args = tyro.cli(Args)
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len('.py')]
        run_name = f"{args.env_id}__{args.exp_name}__{args.backbone}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.use_visual_prompt and not args.prompt_dir:
        raise ValueError('--use-visual-prompt requires --prompt-dir')
    if args.amp_dtype not in {"fp16", "bf16"}:
        raise ValueError('--amp-dtype must be one of: fp16, bf16')

    demo_info = None
    if args.demo_path.endswith('.h5'):
        json_file = args.demo_path[:-2] + 'json'
        with open(json_file, 'r') as f:
            demo_info = json.load(f)
            if 'control_mode' in demo_info['env_info']['env_kwargs']:
                control_mode = demo_info['env_info']['env_kwargs']['control_mode']
            elif 'control_mode' in demo_info['episodes'][0]:
                control_mode = demo_info['episodes'][0]['control_mode']
            else:
                raise Exception('Control mode not found in json')
            assert control_mode == args.control_mode, f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode='sparse',
        obs_mode='rgbd' if args.include_depth else 'rgb',
        render_mode='rgb_array',
        human_render_camera_configs=dict(shader_pack='default'),
        sensor_configs=dict(width=512, height=512),
        use_right_side_camera=args.use_right_side_camera,
    )
    if args.max_episode_steps is not None:
        env_kwargs['max_episode_steps'] = args.max_episode_steps
    if args.close_camera:
        env_kwargs['close_camera'] = True
    if demo_info is not None and 'env_info' in demo_info and 'env_kwargs' in demo_info['env_info']:
        demo_env_kwargs = demo_info['env_info']['env_kwargs']
        for key in ['close_camera', 'num_distractor_cubes']:
            if key in demo_env_kwargs and demo_env_kwargs[key] is not None:
                env_kwargs[key] = demo_env_kwargs[key]

    other_kwargs = None
    wrappers = [
        partial(
            FlattenRGBDObservationWrapper,
            depth=args.include_depth,
            image_size=args.image_size,
            quantize_rgb_to_uint8=args.quantize_rgb_to_uint8,
        )
    ]
    if args.use_visual_prompt:
        eval_prompt_viz_dir = f'runs/{run_name}/prompt_viz' if args.save_eval_prompt_viz else None
        wrappers = [
            partial(
                FlattenRGBDObservationWrapper,
                depth=args.include_depth,
                image_size=args.image_size,
                quantize_rgb_to_uint8=args.quantize_rgb_to_uint8,
            ),
            partial(
                DynamicStackCubeGoalPromptWrapper,
                prompt_type="center",
                output_dir=eval_prompt_viz_dir,
                save_visualizations=args.save_eval_prompt_viz,
            ),
        ]
    envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs, other_kwargs, video_dir=f'runs/{run_name}/videos' if args.capture_video else None, wrappers=wrappers)

    dataset = SmallDemoDataset_ACTPolicy(
        args.demo_path,
        args.num_queries,
        num_traj=args.num_demos,
        include_depth=args.include_depth,
        image_size=args.image_size,
        prompt_dir=args.prompt_dir,
        quantize_rgb_to_uint8=args.quantize_rgb_to_uint8,
    )
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_dataload_workers, worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed))
    if args.num_demos is None:
        args.num_demos = dataset.num_traj

    if args.track:
        import wandb
        config = vars(args)
        config['eval_env_cfg'] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=config, name=run_name, save_code=True, group='ACT', tags=['act', 'stackcube'])
    writer = SummaryWriter(f'runs/{run_name}')
    writer.add_text('hyperparameters', '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()])))

    agent = Agent(envs, args).to(device)
    param_dicts = [
        {'params': [p for n, p in agent.named_parameters() if 'backbone' not in n and p.requires_grad]},
        {'params': [p for n, p in agent.named_parameters() if 'backbone' in n and p.requires_grad], 'lr': args.lr_backbone},
    ]
    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=1e-4)
    lr_drop = max(1, int((2 / 3) * args.total_iters))
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_drop)
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)
    eval_kwargs = dict(stats=dataset.norm_stats, num_queries=args.num_queries, temporal_agg=args.temporal_agg, max_timesteps=args.max_episode_steps, device=device, sim_backend=args.sim_backend)

    amp_enabled = bool(args.use_amp and device.type == 'cuda')
    amp_dtype = torch.float16 if args.amp_dtype == 'fp16' else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    agent.train()
    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    train_pbar = tqdm(
        enumerate(train_dataloader),
        total=args.total_iters,
        desc="Training",
        dynamic_ncols=True,
    )
    for cur_iter, data_batch in train_pbar:
        last_tick = time.time()
        obs, action_seq = data_batch['observations'], data_batch['actions']
        obs = common.to_tensor(obs, device)
        action_seq = common.to_tensor(action_seq, device)
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
            loss_dict = agent.compute_loss(obs, action_seq)
        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss_dict['loss']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict['loss'].backward()
            optimizer.step()
        lr_scheduler.step()
        ema.step(agent.parameters())
        timings['train'] += time.time() - last_tick
        train_pbar.set_postfix(loss=f"{loss_dict['loss'].item():.4f}")

        if cur_iter % args.log_freq == 0:
            for k, v in loss_dict.items():
                writer.add_scalar(f'train/{k}', v.item(), cur_iter)
                print(f'{k}: {v.item():.4f}')

        if cur_iter % args.eval_freq == 0:
            ema.copy_to(ema_agent.parameters())
            eval_metrics = evaluate(args.num_eval_episodes, ema_agent, envs, eval_kwargs)
            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f'eval/{k}', eval_metrics[k], cur_iter)
                print(f'{k}: {eval_metrics[k]:.4f}')
            for k in ['success_once', 'success_at_end']:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f'best_eval_{k}')
            agent.train()

        if args.save_freq is not None and cur_iter % args.save_freq == 0:
            save_ckpt(run_name, str(cur_iter))

    save_ckpt(run_name, 'final')
    train_pbar.close()
    writer.close()
    envs.close()
