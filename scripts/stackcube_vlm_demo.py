#!/usr/bin/env python3
"""StackCube VLM client with multi-view prompt rendering and low-level DP loading."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from scripts.vlm_env_client import (
    LlamaFactoryVLMClient,
    compose_prompt_rgb,
    extract_stackcube_rgbs,
    get_available_cameras,
    group_boxes_by_camera,
    overlay_pick_place_boxes,
    parse_pick_place_boxes,
    save_image,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_POLICY_DIR = REPO_ROOT / "examples" / "baselines" / "diffusion_policy"
TRAIN_RGBD_DEMO_PATH = DIFFUSION_POLICY_DIR / "train_rgbd_demo.py"
DEFAULT_CAMERAS = ["base_camera", "left_side_camera"]


class ExternalVisualPromptWrapper:
    """
    Match the training-time VisualPromptWrapper logic, but replace the prompt image
    with an externally generated VLM prompt.
    """

    def __init__(self, env, prompt_rgb: np.ndarray):
        self.env = env
        self.initial_obs = None
        self.prompt_rgb = prompt_rgb

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _make_prompt_batch(self, obs_prompt_rgb: torch.Tensor | np.ndarray):
        if isinstance(obs_prompt_rgb, torch.Tensor):
            prompt = torch.as_tensor(
                self.prompt_rgb,
                device=obs_prompt_rgb.device,
                dtype=obs_prompt_rgb.dtype,
            )
            if prompt.dim() == 3:
                prompt = prompt.unsqueeze(0)
            return prompt

        prompt = np.asarray(self.prompt_rgb, dtype=obs_prompt_rgb.dtype)
        if prompt.ndim == 3:
            prompt = np.expand_dims(prompt, axis=0)
        return prompt

    def _overwrite_prompt(self, obs):
        if "prompt_rgb" not in obs:
            return obs

        prompt = self._make_prompt_batch(obs["prompt_rgb"])
        if isinstance(obs["prompt_rgb"], torch.Tensor):
            obs["prompt_rgb"][:] = prompt[:, None, ...]
        else:
            obs["prompt_rgb"][:] = prompt[:, None, ...]
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.initial_obs = {}
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                self.initial_obs[k] = v[:, 0].clone()
            else:
                self.initial_obs[k] = v[:, 0].copy()

        if "prompt_rgb" in self.initial_obs:
            prompt = self._make_prompt_batch(self.initial_obs["prompt_rgb"])
            self.initial_obs["prompt_rgb"] = prompt

        obs = self._overwrite_prompt(obs)
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        dones = term | trunc

        if hasattr(dones, "any") and dones.any():
            for k in obs.keys():
                if isinstance(obs[k], torch.Tensor):
                    _dones_t = torch.from_numpy(dones).to(obs[k].device) if isinstance(dones, np.ndarray) else dones
                    self.initial_obs[k][_dones_t] = obs[k][_dones_t, -1].clone()
                else:
                    _dones_np = dones.cpu().numpy() if isinstance(dones, torch.Tensor) else dones
                    self.initial_obs[k][_dones_np] = obs[k][_dones_np, -1].copy()

            if "prompt_rgb" in self.initial_obs:
                prompt = self._make_prompt_batch(self.initial_obs["prompt_rgb"])
                if isinstance(self.initial_obs["prompt_rgb"], torch.Tensor):
                    _dones_t = torch.from_numpy(dones).to(self.initial_obs["prompt_rgb"].device) if isinstance(dones, np.ndarray) else dones
                    self.initial_obs["prompt_rgb"][_dones_t] = prompt[_dones_t]
                else:
                    _dones_np = dones.cpu().numpy() if isinstance(dones, torch.Tensor) else dones
                    self.initial_obs["prompt_rgb"][_dones_np] = prompt[_dones_np]

        obs = self._overwrite_prompt(obs)
        return obs, rew, term, trunc, info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query a VLM from StackCube-v1, draw pick/place boxes, and load low-level DP weights."
    )
    parser.add_argument("--server-url", type=str, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "For each input camera image, return pick_box and place_box bounding boxes. "
            "Include the camera name for every box."
        ),
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Camera name to send. Can be passed multiple times. Defaults to base_camera and left_side_camera.",
    )
    parser.add_argument("--obs-mode", type=str, default="rgb")
    parser.add_argument("--robot", type=str, default="panda_wristcam")
    parser.add_argument("--close-camera", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "stackcube_vlm",
        help="Directory for annotated images and prompt artifacts.",
    )

    parser.add_argument("--dp-checkpoint", type=Path, default=None)
    parser.add_argument("--dp-use-ema", action="store_true")
    parser.add_argument("--dp-device", type=str, default="cuda")
    parser.add_argument("--dp-control-mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--dp-obs-horizon", type=int, default=2)
    parser.add_argument("--dp-act-horizon", type=int, default=8)
    parser.add_argument("--dp-pred-horizon", type=int, default=16)
    parser.add_argument("--dp-diffusion-step-embed-dim", type=int, default=64)
    parser.add_argument("--dp-unet-dims", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--dp-n-groups", type=int, default=8)
    parser.add_argument("--dp-max-episode-steps", type=int, default=200)
    parser.add_argument("--dp-run-inference", action="store_true")
    return parser


def _load_train_rgbd_demo_module():
    if str(DIFFUSION_POLICY_DIR) not in sys.path:
        sys.path.insert(0, str(DIFFUSION_POLICY_DIR))
    spec = importlib.util.spec_from_file_location("train_rgbd_demo_module", TRAIN_RGBD_DEMO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {TRAIN_RGBD_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_dp_agent(args: argparse.Namespace, prompt_rgb: np.ndarray):
    module = _load_train_rgbd_demo_module()
    device_name = args.dp_device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    env_kwargs = dict(
        control_mode=args.dp_control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        sensor_configs=dict(width=128, height=128),
        max_episode_steps=args.dp_max_episode_steps,
    )
    if args.close_camera:
        env_kwargs["close_camera"] = True

    envs = module.make_eval_envs(
        "StackCube-v1",
        1,
        "physx_cpu",
        env_kwargs,
        dict(obs_horizon=args.dp_obs_horizon),
        video_dir=None,
        wrappers=[module.FlattenRGBDAndPromptWrapper],
    )
    envs = ExternalVisualPromptWrapper(envs, prompt_rgb)

    agent_args = SimpleNamespace(
        obs_horizon=args.dp_obs_horizon,
        act_horizon=args.dp_act_horizon,
        pred_horizon=args.dp_pred_horizon,
        diffusion_step_embed_dim=args.dp_diffusion_step_embed_dim,
        unet_dims=args.dp_unet_dims,
        n_groups=args.dp_n_groups,
    )
    agent = module.Agent(envs, agent_args).to(device)
    return module, envs, agent, device


def load_dp_checkpoint(agent: torch.nn.Module, checkpoint_path: Path, use_ema: bool) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}")
    preferred_key = "ema_agent" if use_ema else "agent"
    state_dict = checkpoint.get(preferred_key)
    used_key = preferred_key
    if state_dict is None:
        fallback_key = "agent" if use_ema else "ema_agent"
        state_dict = checkpoint.get(fallback_key)
        used_key = fallback_key
    if state_dict is None:
        state_dict = checkpoint
        used_key = "<checkpoint-root>"
    agent.load_state_dict(state_dict, strict=True)
    agent.eval()
    return used_key


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_rendered_images(output_dir: Path, images: dict[str, np.ndarray]) -> dict[str, str]:
    saved = {}
    for camera_name, image in images.items():
        file_path = output_dir / f"{camera_name}_prompt.png"
        save_image(str(file_path), image)
        saved[camera_name] = str(file_path)
    return saved


def summarize_obs_shapes(obs: dict[str, Any]) -> dict[str, list[int]]:
    summary = {}
    for key, value in obs.items():
        if hasattr(value, "shape"):
            summary[key] = list(value.shape)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    env = gym.make(
        "StackCube-v1",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        robot_uids=args.robot,
        close_camera=args.close_camera,
    )
    obs, info = env.reset(seed=args.seed)

    selected_cameras = args.camera or DEFAULT_CAMERAS
    available_cameras = get_available_cameras(obs)
    rgb_images = extract_stackcube_rgbs(obs, camera_names=selected_cameras)

    client = LlamaFactoryVLMClient(
        server_url=args.server_url,
        prompt=args.prompt,
        timeout=args.timeout,
    )
    raw_vlm_response = client.query_multi(rgb_images)
    grouped = parse_pick_place_boxes(raw_vlm_response)
    all_boxes = grouped["pick"] + grouped["place"] + grouped["other"]
    boxes_by_camera = group_boxes_by_camera(all_boxes)
    rendered_images = overlay_pick_place_boxes(rgb_images, boxes_by_camera)
    prompt_rgb = compose_prompt_rgb(rendered_images, camera_order=selected_cameras)

    saved_images = save_rendered_images(output_dir, rendered_images)
    prompt_path = output_dir / "prompt_rgb.npy"
    np.save(prompt_path, prompt_rgb)

    dp_summary: dict[str, Any] | None = None
    dp_envs = None
    if args.dp_checkpoint is not None:
        module, dp_envs, dp_agent, dp_device = build_dp_agent(args, prompt_rgb)
        loaded_key = load_dp_checkpoint(dp_agent, args.dp_checkpoint, use_ema=args.dp_use_ema)
        dp_summary = {
            "checkpoint": str(args.dp_checkpoint),
            "state_dict_key": loaded_key,
            "device": str(dp_device),
            "obs_horizon": args.dp_obs_horizon,
            "act_horizon": args.dp_act_horizon,
            "pred_horizon": args.dp_pred_horizon,
            "unet_dims": args.dp_unet_dims,
            "prompt_replacement": "FlattenRGBDAndPromptWrapper + ExternalVisualPromptWrapper",
        }

        obs_dp, info_dp = dp_envs.reset(seed=args.seed)
        dp_summary["obs_shapes_after_prompt_replace"] = summarize_obs_shapes(obs_dp)

        if args.dp_run_inference:
            obs_for_agent = {
                k: (v.to(dp_device) if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=dp_device))
                for k, v in obs_dp.items()
            }
            pred_action = dp_agent.get_action(obs_for_agent)
            dp_summary["pred_action_shape"] = list(pred_action.shape)
            dp_summary["pred_action_preview"] = pred_action[0, 0].detach().cpu().tolist()

    result = {
        "requested_cameras": selected_cameras,
        "available_cameras": available_cameras,
        "image_shapes": {
            camera_name: list(np.asarray(image).shape)
            for camera_name, image in rgb_images.items()
        },
        "raw_vlm_response": raw_vlm_response,
        "pick_boxes": [box.to_dict() for box in grouped["pick"]],
        "place_boxes": [box.to_dict() for box in grouped["place"]],
        "other_boxes": [box.to_dict() for box in grouped["other"]],
        "saved_prompt_images": saved_images,
        "prompt_rgb_path": str(prompt_path),
        "prompt_rgb_shape": list(prompt_rgb.shape),
        "dp": dp_summary,
    }

    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))

    env.close()
    if dp_envs is not None:
        dp_envs.close()


if __name__ == "__main__":
    main()
