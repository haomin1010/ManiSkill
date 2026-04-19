#!/usr/bin/env python3
"""Save one demo heatmap sample from visual prompt JSONs.

This script reproduces the heatmap prompt generation logic used in
`train_stackcube.py` and exports a single sample for debugging.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diffusion_policy.heatmap_utils import bbox_centers_to_heatmap_multi_camera


CAMERA_SETTINGS = {
    "base_left": ["base_camera", "left_side_camera"],
    "left_right": ["left_side_camera", "right_side_camera"],
}


def read_png_size(path: Path) -> Tuple[int, int]:
    """Read PNG width/height from file header without external image libs."""
    with path.open("rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a valid PNG file: {path}")

        length_bytes = f.read(4)
        chunk_type = f.read(4)
        if len(length_bytes) != 4 or chunk_type != b"IHDR":
            raise ValueError(f"Invalid PNG IHDR chunk: {path}")

        ihdr_data = f.read(13)
        if len(ihdr_data) != 13:
            raise ValueError(f"Corrupted PNG IHDR data: {path}")

        width, height = struct.unpack(">II", ihdr_data[:8])
        return int(width), int(height)


def pick_episode(prompt_dir: Path, episode: int | None) -> int:
    if episode is not None:
        return int(episode)

    files = sorted(prompt_dir.glob("ep*_boxes.json"))
    if not files:
        raise FileNotFoundError(f"No ep*_boxes.json found in: {prompt_dir}")

    # files like ep12_boxes.json -> 12
    stem = files[0].stem
    return int(stem.split("_", 1)[0][2:])


def get_center(cam_data: Dict, key: str) -> List[float]:
    center = cam_data.get(key)
    if center is not None:
        return [float(center[0]), float(center[1])]

    # Fallback to *_box_corners.center format
    corners_key = "init_box_corners" if key.startswith("init") else "goal_box_corners"
    corners = cam_data.get(corners_key)
    if isinstance(corners, dict) and corners.get("center") is not None:
        c = corners["center"]
        return [float(c[0]), float(c[1])]

    raise KeyError(f"Missing {key} / {corners_key}.center in camera prompt data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Save one heatmap demo sample")
    parser.add_argument("--prompt-dir", type=str, default="videos/StackCube-v1/screenshots")
    parser.add_argument("--episode", type=int, default=None, help="episode id, e.g. 0; default: first available")
    parser.add_argument("--camera-setting", type=str, default="left_right", choices=sorted(CAMERA_SETTINGS.keys()))
    parser.add_argument("--heatmap-h", type=int, default=48)
    parser.add_argument("--heatmap-w", type=int, default=48)
    parser.add_argument("--heatmap-sigma", type=float, default=0.0)
    parser.add_argument("--heatmap-sigma-ratio", type=float, default=0.1)
    parser.add_argument("--out", type=str, default=None, help="output .npz path")
    args = parser.parse_args()

    prompt_dir = Path(args.prompt_dir)
    ep = pick_episode(prompt_dir, args.episode)
    prompt_json = prompt_dir / f"ep{ep}_boxes.json"
    if not prompt_json.exists():
        raise FileNotFoundError(f"Prompt JSON not found: {prompt_json}")

    with prompt_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cameras = data.get("cameras", {})
    if not cameras:
        raise KeyError(f"No cameras found in: {prompt_json}")

    prompt_cameras = CAMERA_SETTINGS[args.camera_setting]
    centers_per_camera = []
    centers_px: Dict[str, Dict[str, List[float]]] = {}
    centers_norm: Dict[str, Dict[str, List[float]]] = {}
    image_wh: Dict[str, List[int]] = {}
    channel_names: List[str] = []

    for cam_name in prompt_cameras:
        cam_data = cameras.get(cam_name)
        if cam_data is None:
            raise KeyError(f"Camera '{cam_name}' not found in {prompt_json}")

        image_path = cam_data.get("image_path")
        if image_path is None:
            raise KeyError(f"Camera '{cam_name}' missing image_path in {prompt_json}")

        png_path = prompt_dir / image_path
        width, height = read_png_size(png_path)

        init_px = get_center(cam_data, "init_center_px")
        goal_px = get_center(cam_data, "goal_center_px")

        init_norm = [init_px[0] / width, init_px[1] / height]
        goal_norm = [goal_px[0] / width, goal_px[1] / height]

        init_t = torch.tensor([init_norm], dtype=torch.float32)
        goal_t = torch.tensor([goal_norm], dtype=torch.float32)
        centers_per_camera.append((init_t, goal_t))

        centers_px[cam_name] = {"init": init_px, "goal": goal_px}
        centers_norm[cam_name] = {"init": init_norm, "goal": goal_norm}
        image_wh[cam_name] = [int(width), int(height)]
        channel_names.extend([f"{cam_name}_init", f"{cam_name}_goal"])

    sigma = float(args.heatmap_sigma)
    if sigma <= 0:
        sigma = max(1.0, float(args.heatmap_sigma_ratio) * float(min(args.heatmap_h, args.heatmap_w)))

    heatmaps = bbox_centers_to_heatmap_multi_camera(
        centers_per_camera,
        heatmap_h=args.heatmap_h,
        heatmap_w=args.heatmap_w,
        sigma=sigma,
    )
    heatmap = heatmaps[0].cpu().numpy().astype(np.float32)  # (C, H, W)

    if args.out:
        out_npz = Path(args.out)
    else:
        out_npz = prompt_dir / f"ep{ep}_heatmap_demo_{args.camera_setting}_{args.heatmap_h}x{args.heatmap_w}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "episode": int(ep),
        "prompt_json": str(prompt_json),
        "camera_setting": args.camera_setting,
        "prompt_cameras": prompt_cameras,
        "channel_names": channel_names,
        "heatmap_h": int(args.heatmap_h),
        "heatmap_w": int(args.heatmap_w),
        "sigma": float(sigma),
        "centers_px": centers_px,
        "centers_norm": centers_norm,
        "image_wh": image_wh,
    }

    np.savez_compressed(
        out_npz,
        heatmap=heatmap,
        channel_names=np.array(channel_names, dtype=object),
        episode=np.array([ep], dtype=np.int32),
    )

    out_json = out_npz.with_suffix(".json")
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved heatmap demo to: {out_npz}")
    print(f"Saved metadata to:    {out_json}")
    print(f"heatmap shape: {heatmap.shape}  (C,H,W)")
    print(f"channels: {channel_names}")


if __name__ == "__main__":
    main()
