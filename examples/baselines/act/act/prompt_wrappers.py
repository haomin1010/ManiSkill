import copy
import os
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space


DEFAULT_PROMPT_CAMERAS = ["base_camera", "left_side_camera"]
PROMPT_BOX_DIM_PER_CAMERA = 8
PROMPT_CENTER_DIM_PER_CAMERA = 4


class DynamicStackCubeGoalPromptWrapper(gym.Wrapper):
    """Attach a per-episode visual goal prompt computed from the live env state."""

    def __init__(
        self,
        env,
        prompt_cameras=DEFAULT_PROMPT_CAMERAS,
        box_size_px: float = 32.0,
        prompt_type: str = "center",
        output_dir: str = None,
        save_visualizations: bool = True,
    ) -> None:
        super().__init__(env)
        self.prompt_cameras = list(prompt_cameras)
        self.box_size_px = float(box_size_px)
        if prompt_type not in {"box", "center"}:
            raise ValueError(
                f"Unsupported prompt_type={prompt_type!r}, expected 'box' or 'center'"
            )
        self.prompt_type = prompt_type
        dim_per_camera = (
            PROMPT_BOX_DIM_PER_CAMERA
            if self.prompt_type == "box"
            else PROMPT_CENTER_DIM_PER_CAMERA
        )
        self.prompt_raw_dim = dim_per_camera * len(self.prompt_cameras) + 1
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.save_visualizations = bool(save_visualizations)
        self._cached_goal_prompt = None
        self._episode_counter = 0

        prompt_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.prompt_raw_dim,),
            dtype=np.float32,
        )

        if hasattr(env, "single_observation_space") and isinstance(
            env.single_observation_space, spaces.Dict
        ):
            new_single_spaces = copy.copy(env.single_observation_space.spaces)
            new_single_spaces["goal_prompt"] = prompt_space
            self.single_observation_space = spaces.Dict(new_single_spaces)
            if getattr(env, "num_envs", 1) > 1:
                self.observation_space = batch_space(
                    self.single_observation_space, n=env.num_envs
                )
            else:
                self.observation_space = self.single_observation_space
        elif isinstance(env.observation_space, spaces.Dict):
            new_spaces = copy.copy(env.observation_space.spaces)
            new_spaces["goal_prompt"] = prompt_space
            self.observation_space = spaces.Dict(new_spaces)

        self.base_env._init_raw_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(self.base_env._init_raw_obs)

    @property
    def base_env(self):
        return self.env.unwrapped

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_counter += 1
        self._cached_goal_prompt = self._compute_goal_prompt(obs)
        obs = self.observation(obs)
        if self.save_visualizations:
            self._save_prompt_visualizations(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, observation):
        obs = dict(observation)
        obs["goal_prompt"] = self._compute_goal_prompt(obs).clone()
        self._cached_goal_prompt = obs["goal_prompt"]
        return obs

    def _compute_goal_prompt(self, observation: dict) -> torch.Tensor:
        device = self.base_env.device
        sensor_params = self.base_env.get_sensor_params()
        cube_a_pos = self.base_env.cubeA.pose.p
        cube_b_pos = self.base_env.cubeB.pose.p

        if cube_a_pos.ndim == 1:
            cube_a_pos = cube_a_pos.unsqueeze(0)
        if cube_b_pos.ndim == 1:
            cube_b_pos = cube_b_pos.unsqueeze(0)

        goal_pos = cube_b_pos.clone()
        goal_pos[:, 2] = goal_pos[:, 2] + float(self.base_env.cube_half_size[2] * 2.0)
        width, height = self._get_camera_size(observation)

        batch_size = cube_a_pos.shape[0]
        vec = torch.zeros(
            (batch_size, self.prompt_raw_dim), dtype=torch.float32, device=device
        )

        write_idx = 0
        for cam_name in self.prompt_cameras:
            if cam_name not in sensor_params:
                raise KeyError(
                    f"Camera '{cam_name}' not found in sensor params. "
                    f"Available cameras: {list(sensor_params.keys())}"
                )
            cam_param = sensor_params[cam_name]
            intrinsic = self._to_tensor(cam_param["intrinsic_cv"], device)
            extrinsic = self._to_tensor(cam_param["extrinsic_cv"], device)
            source_width, source_height = self._get_projection_size(intrinsic)

            init_center = self._project_world_points(cube_a_pos, intrinsic, extrinsic)
            goal_center = self._project_world_points(goal_pos, intrinsic, extrinsic)
            if self.prompt_type == "box":
                init_box = self._center_to_box_xyxy(init_center)
                goal_box = self._center_to_box_xyxy(goal_center)
                vec[:, write_idx + 0] = (init_box[:, 0] / source_width).clamp(0.0, 1.0)
                vec[:, write_idx + 1] = (init_box[:, 1] / source_height).clamp(0.0, 1.0)
                vec[:, write_idx + 2] = (init_box[:, 2] / source_width).clamp(0.0, 1.0)
                vec[:, write_idx + 3] = (init_box[:, 3] / source_height).clamp(0.0, 1.0)
                vec[:, write_idx + 4] = (goal_box[:, 0] / source_width).clamp(0.0, 1.0)
                vec[:, write_idx + 5] = (goal_box[:, 1] / source_height).clamp(0.0, 1.0)
                vec[:, write_idx + 6] = (goal_box[:, 2] / source_width).clamp(0.0, 1.0)
                vec[:, write_idx + 7] = (goal_box[:, 3] / source_height).clamp(0.0, 1.0)
                write_idx += PROMPT_BOX_DIM_PER_CAMERA
            else:
                vec[:, write_idx + 0] = (init_center[:, 0] / source_width).clamp(0.0, 1.0)
                vec[:, write_idx + 1] = (init_center[:, 1] / source_height).clamp(0.0, 1.0)
                vec[:, write_idx + 2] = (goal_center[:, 0] / source_width).clamp(0.0, 1.0)
                vec[:, write_idx + 3] = (goal_center[:, 1] / source_height).clamp(0.0, 1.0)
                write_idx += PROMPT_CENTER_DIM_PER_CAMERA

        vec[:, -1] = 1.0
        return vec

    def _save_prompt_visualizations(self, observation: dict) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sensor_params = self.base_env.get_sensor_params()
        cube_a_pos = self.base_env.cubeA.pose.p
        cube_b_pos = self.base_env.cubeB.pose.p
        if cube_a_pos.ndim == 1:
            cube_a_pos = cube_a_pos.unsqueeze(0)
        if cube_b_pos.ndim == 1:
            cube_b_pos = cube_b_pos.unsqueeze(0)
        goal_pos = cube_b_pos.clone()
        goal_pos[:, 2] = goal_pos[:, 2] + float(self.base_env.cube_half_size[2] * 2.0)

        rgb = observation.get("rgb")
        if rgb is None:
            return
        frames = self._split_rgb_cameras(rgb)

        for env_idx in range(cube_a_pos.shape[0]):
            for cam_name in self.prompt_cameras:
                if cam_name not in frames:
                    continue
                canvas = cv2.cvtColor(frames[cam_name][env_idx], cv2.COLOR_RGB2BGR)
                intrinsic = self._to_tensor(
                    sensor_params[cam_name]["intrinsic_cv"], cube_a_pos.device
                )
                extrinsic = self._to_tensor(
                    sensor_params[cam_name]["extrinsic_cv"], cube_a_pos.device
                )
                source_width, source_height = self._get_projection_size(
                    intrinsic[env_idx : env_idx + 1]
                )
                canvas_height, canvas_width = canvas.shape[:2]
                x_scale = canvas_width / float(source_width)
                y_scale = canvas_height / float(source_height)
                init_center = self._project_world_points(
                    cube_a_pos[env_idx : env_idx + 1],
                    intrinsic[env_idx : env_idx + 1],
                    extrinsic[env_idx : env_idx + 1],
                )[0]
                goal_center = self._project_world_points(
                    goal_pos[env_idx : env_idx + 1],
                    intrinsic[env_idx : env_idx + 1],
                    extrinsic[env_idx : env_idx + 1],
                )[0]
                init_center = self._scale_points(init_center, x_scale, y_scale)
                goal_center = self._scale_points(goal_center, x_scale, y_scale)
                if self.prompt_type == "box":
                    init_box = self._center_to_box_xyxy(init_center.unsqueeze(0))[0]
                    goal_box = self._center_to_box_xyxy(goal_center.unsqueeze(0))[0]
                    self._draw_box_prompt(canvas, init_box, goal_box)
                else:
                    self._draw_center_prompt(canvas, init_center, goal_center)
                stem = (
                    f"pid{os.getpid()}_ep{self._episode_counter:04d}"
                    f"_env{env_idx:02d}_{cam_name}"
                )
                cv2.imwrite(str(self.output_dir / f"{stem}.png"), canvas)

    def _get_camera_size(self, observation: dict):
        rgb = observation.get("rgb")
        if rgb is None:
            raise ValueError("DynamicStackCubeGoalPromptWrapper requires 'rgb' observation")
        if isinstance(rgb, torch.Tensor):
            shape = tuple(rgb.shape)
        else:
            shape = tuple(np.asarray(rgb).shape)
        if len(shape) == 5:
            height, width = shape[-2], shape[-1]
        elif len(shape) == 4:
            height, width = shape[1], shape[2]
        else:
            raise ValueError(f"Unexpected rgb shape for prompt wrapper: {shape}")
        return float(width), float(height)

    @staticmethod
    def _get_projection_size(intrinsic: torch.Tensor):
        if intrinsic.ndim == 3:
            intrinsic = intrinsic[0]
        width = float(intrinsic[0, 2] * 2.0)
        height = float(intrinsic[1, 2] * 2.0)
        return width, height

    @staticmethod
    def _scale_points(points: torch.Tensor, x_scale: float, y_scale: float) -> torch.Tensor:
        scaled = points.clone()
        scaled[..., 0] = scaled[..., 0] * float(x_scale)
        scaled[..., 1] = scaled[..., 1] * float(y_scale)
        return scaled

    def _split_rgb_cameras(self, rgb):
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu()
            if rgb.ndim == 4:
                rgb = rgb.unsqueeze(0)
            rgb = rgb.permute(0, 1, 3, 4, 2).numpy()
        rgb = np.asarray(rgb)
        if rgb.ndim != 5:
            raise ValueError(f"Unexpected rgb shape for visualization: {rgb.shape}")
        frames = {}
        for cam_idx, cam_name in enumerate(self.prompt_cameras):
            if cam_idx >= rgb.shape[1]:
                break
            cam_rgb = rgb[:, cam_idx]
            if cam_rgb.dtype != np.uint8:
                cam_rgb = np.clip(cam_rgb, 0, 255).astype(np.uint8)
            frames[cam_name] = cam_rgb
        return frames

    def _center_to_box_xyxy(self, center: torch.Tensor) -> torch.Tensor:
        half_box = self.box_size_px / 2.0
        x = center[:, 0]
        y = center[:, 1]
        return torch.stack([x - half_box, y - half_box, x + half_box, y + half_box], dim=-1)

    def _draw_box_prompt(self, canvas: np.ndarray, init_box: torch.Tensor, goal_box: torch.Tensor) -> None:
        init_box_xyxy = [int(round(v)) for v in init_box.detach().cpu().tolist()]
        goal_box_xyxy = [int(round(v)) for v in goal_box.detach().cpu().tolist()]
        init_xy = (
            int(round((init_box_xyxy[0] + init_box_xyxy[2]) / 2.0)),
            int(round((init_box_xyxy[1] + init_box_xyxy[3]) / 2.0)),
        )
        goal_xy = (
            int(round((goal_box_xyxy[0] + goal_box_xyxy[2]) / 2.0)),
            int(round((goal_box_xyxy[1] + goal_box_xyxy[3]) / 2.0)),
        )
        cv2.rectangle(canvas, (init_box_xyxy[0], init_box_xyxy[1]), (init_box_xyxy[2], init_box_xyxy[3]), (0, 255, 255), 2)
        cv2.rectangle(canvas, (goal_box_xyxy[0], goal_box_xyxy[1]), (goal_box_xyxy[2], goal_box_xyxy[3]), (0, 0, 255), 2)
        cv2.circle(canvas, init_xy, 4, (0, 255, 255), -1)
        cv2.circle(canvas, goal_xy, 4, (0, 0, 255), -1)
        cv2.line(canvas, init_xy, goal_xy, (255, 255, 255), 1)
        cv2.putText(canvas, "init", (init_xy[0] + 6, init_xy[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "goal", (goal_xy[0] + 6, goal_xy[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    def _draw_center_prompt(self, canvas: np.ndarray, init_center: torch.Tensor, goal_center: torch.Tensor) -> None:
        init_xy = tuple(int(round(v)) for v in init_center.detach().cpu().tolist())
        goal_xy = tuple(int(round(v)) for v in goal_center.detach().cpu().tolist())
        cv2.circle(canvas, init_xy, 6, (0, 255, 255), -1)
        cv2.circle(canvas, goal_xy, 6, (0, 0, 255), -1)
        cv2.line(canvas, init_xy, goal_xy, (255, 255, 255), 2)
        cv2.putText(canvas, "init", (init_xy[0] + 8, init_xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "goal", (goal_xy[0] + 8, goal_xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _to_tensor(value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    @staticmethod
    def _project_world_points(
        points_world: torch.Tensor,
        intrinsic: torch.Tensor,
        extrinsic: torch.Tensor,
    ) -> torch.Tensor:
        ones = torch.ones(
            (points_world.shape[0], 1),
            dtype=points_world.dtype,
            device=points_world.device,
        )
        points_h = torch.cat([points_world, ones], dim=-1)
        points_cam = (extrinsic @ points_h.unsqueeze(-1)).squeeze(-1)
        z = points_cam[:, 2].clamp(min=1e-6)
        u = intrinsic[:, 0, 0] * (points_cam[:, 0] / z) + intrinsic[:, 0, 2]
        v = intrinsic[:, 1, 1] * (points_cam[:, 1] / z) + intrinsic[:, 1, 2]
        return torch.stack([u, v], dim=-1)
