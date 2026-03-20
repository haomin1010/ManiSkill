#!/usr/bin/env python3
"""
Minimal ManiSkill-side VLM client.

This module sends RGB observations to a LLaMA-Factory FastAPI endpoint and
expects bounding box predictions in return.

It is intentionally lightweight:
- no ManiSkill internals are required to import this file
- observations can come from env.reset()/env.step() outputs
- images are encoded as base64 PNG before being sent over HTTP
- StackCube convenience helpers are provided for common camera choices
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw

STACKCUBE_CAMERA_PRIORITY = [
    "base_camera",
    "left_side_camera",
]

PICK_ROLE_NAMES = {"pick", "pick_box", "pickbox", "grasp", "pickup"}
PLACE_ROLE_NAMES = {"place", "place_box", "placebox", "goal", "target"}


@dataclass
class BoundingBox:
    label: str | None
    score: float | None
    x1: float
    y1: float
    x2: float
    y2: float
    camera: str | None = None
    role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": self.score,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "camera": self.camera,
            "role": self.role,
        }


def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(
            f"Expected image with shape [H, W, 3/4], but got {image.shape}."
        )

    if image.shape[-1] == 4:
        image = image[..., :3]

    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    else:
        image = np.clip(image, 0, 255)

    return image.astype(np.uint8)


def _to_numpy(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach()
    if hasattr(image, "cpu"):
        image = image.cpu()
    return np.asarray(image)


def _strip_batch_dim(image: np.ndarray) -> np.ndarray:
    if image.ndim == 4 and image.shape[0] == 1:
        return image[0]
    return image


def _normalize_role(role: str | None) -> str | None:
    if role is None:
        return None
    role_l = role.lower().strip()
    if role_l in PICK_ROLE_NAMES:
        return "pick"
    if role_l in PLACE_ROLE_NAMES:
        return "place"
    return role_l


def encode_image_to_base64(image: np.ndarray) -> str:
    image_uint8 = _ensure_uint8_rgb(image)
    pil_image = Image.fromarray(image_uint8)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def save_image(path: str, image: np.ndarray) -> None:
    Image.fromarray(_ensure_uint8_rgb(image)).save(path)


def get_available_cameras(obs: dict[str, Any]) -> list[str]:
    sensor_data = obs.get("sensor_data")
    if isinstance(sensor_data, dict):
        return list(sensor_data.keys())
    image_data = obs.get("image")
    if isinstance(image_data, dict):
        return list(image_data.keys())
    return []


def extract_rgb_from_obs(
    obs: dict[str, Any],
    image_key: str = "rgb",
    camera_key: str | None = None,
) -> np.ndarray:
    if camera_key is None and image_key in obs:
        return _strip_batch_dim(_to_numpy(obs[image_key]))

    for top_level_key in ("sensor_data", "image", "images", "camera_obs"):
        if top_level_key not in obs:
            continue

        container = obs[top_level_key]
        if camera_key is None:
            if image_key in container:
                return _strip_batch_dim(_to_numpy(container[image_key]))
            continue

        if camera_key in container and image_key in container[camera_key]:
            return _strip_batch_dim(_to_numpy(container[camera_key][image_key]))

    raise KeyError(
        "Unable to find RGB image in observation. "
        f"camera_key={camera_key!r}, image_key={image_key!r}"
    )


def extract_stackcube_rgb(
    obs: dict[str, Any],
    preferred_cameras: list[str] | None = None,
    image_key: str = "rgb",
) -> tuple[np.ndarray, str]:
    camera_candidates = preferred_cameras or STACKCUBE_CAMERA_PRIORITY
    available_cameras = get_available_cameras(obs)

    for camera_name in camera_candidates:
        if camera_name in available_cameras:
            return (
                extract_rgb_from_obs(obs, image_key=image_key, camera_key=camera_name),
                camera_name,
            )

    if available_cameras:
        fallback_camera = available_cameras[0]
        return (
            extract_rgb_from_obs(obs, image_key=image_key, camera_key=fallback_camera),
            fallback_camera,
        )

    if image_key in obs:
        return extract_rgb_from_obs(obs, image_key=image_key, camera_key=None), "<top-level-rgb>"

    raise KeyError(
        "No camera RGB found in StackCube observation. "
        f"Available cameras: {available_cameras}"
    )


def extract_stackcube_rgbs(
    obs: dict[str, Any],
    camera_names: list[str] | None = None,
    image_key: str = "rgb",
) -> dict[str, np.ndarray]:
    selected_cameras = camera_names or STACKCUBE_CAMERA_PRIORITY
    available_cameras = get_available_cameras(obs)
    results: dict[str, np.ndarray] = {}

    for camera_name in selected_cameras:
        if camera_name in available_cameras:
            results[camera_name] = extract_rgb_from_obs(
                obs, image_key=image_key, camera_key=camera_name
            )

    if not results:
        raise KeyError(
            "None of the requested StackCube cameras were found. "
            f"requested={selected_cameras}, available={available_cameras}"
        )

    return results


class LlamaFactoryVLMClient:
    def __init__(
        self,
        server_url: str,
        timeout: float = 30.0,
        prompt: str = "Detect the target object and return bounding boxes.",
    ) -> None:
        self.server_url = server_url
        self.timeout = timeout
        self.prompt = prompt

    def build_payload(self, rgb_image: np.ndarray) -> dict[str, Any]:
        return {
            "image": encode_image_to_base64(rgb_image),
            "prompt": self.prompt,
        }

    def build_multi_payload(self, rgb_images: dict[str, np.ndarray]) -> dict[str, Any]:
        return {
            "images": [encode_image_to_base64(image) for image in rgb_images.values()],
            "camera_names": list(rgb_images.keys()),
            "prompt": self.prompt,
        }

    def query(self, rgb_image: np.ndarray) -> Any:
        payload = self.build_payload(rgb_image)
        response = requests.post(self.server_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def query_multi(self, rgb_images: dict[str, np.ndarray]) -> Any:
        payload = self.build_multi_payload(rgb_images)
        response = requests.post(self.server_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


def _coerce_box_item(item: dict[str, Any], role: str | None = None) -> BoundingBox:
    if "bbox" in item:
        x1, y1, x2, y2 = item["bbox"]
    else:
        x1 = item["x1"]
        y1 = item["y1"]
        x2 = item["x2"]
        y2 = item["y2"]

    inferred_role = _normalize_role(
        role or item.get("role") or item.get("type") or item.get("name")
    )
    label = item.get("label") or item.get("text") or inferred_role
    return BoundingBox(
        label=label,
        score=item.get("score"),
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        camera=item.get("camera") or item.get("camera_name"),
        role=inferred_role,
    )


def _coerce_box_mapping(value: Any, role: str) -> list[BoundingBox]:
    role = _normalize_role(role) or role
    if isinstance(value, dict):
        if "bbox" in value or {"x1", "y1", "x2", "y2"}.issubset(value.keys()):
            return [_coerce_box_item(value, role=role)]
        out = []
        for camera_name, camera_box in value.items():
            if isinstance(camera_box, list) and camera_box and not isinstance(camera_box[0], (int, float)):
                for item in camera_box:
                    item = dict(item)
                    item.setdefault("camera", camera_name)
                    out.append(_coerce_box_item(item, role=role))
            else:
                if isinstance(camera_box, dict):
                    item = dict(camera_box)
                else:
                    item = {"bbox": camera_box}
                item.setdefault("camera", camera_name)
                out.append(_coerce_box_item(item, role=role))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                out.append(_coerce_box_item(item, role=role))
            else:
                out.append(_coerce_box_item({"bbox": item}, role=role))
        return out
    raise ValueError(f"Unsupported {role} box format: {type(value)!r}")


def parse_pick_place_boxes(response_json: Any) -> dict[str, list[BoundingBox]]:
    result = {"pick": [], "place": [], "other": []}

    if isinstance(response_json, dict):
        if "pick_boxes" in response_json:
            result["pick"].extend(_coerce_box_mapping(response_json["pick_boxes"], "pick"))
        if "pick_box" in response_json:
            result["pick"].extend(_coerce_box_mapping(response_json["pick_box"], "pick"))
        if "place_boxes" in response_json:
            result["place"].extend(_coerce_box_mapping(response_json["place_boxes"], "place"))
        if "place_box" in response_json:
            result["place"].extend(_coerce_box_mapping(response_json["place_box"], "place"))

        generic_items = None
        if "boxes" in response_json:
            generic_items = response_json["boxes"]
        elif "bboxes" in response_json:
            generic_items = response_json["bboxes"]
        elif "data" in response_json and isinstance(response_json["data"], list):
            generic_items = response_json["data"]

        if generic_items is not None:
            for item in generic_items:
                box = _coerce_box_item(item)
                if box.role == "pick":
                    result["pick"].append(box)
                elif box.role == "place":
                    result["place"].append(box)
                else:
                    result["other"].append(box)
    elif isinstance(response_json, list):
        for item in response_json:
            box = _coerce_box_item(item)
            if box.role == "pick":
                result["pick"].append(box)
            elif box.role == "place":
                result["place"].append(box)
            else:
                result["other"].append(box)
    else:
        raise ValueError(f"Unsupported response type: {type(response_json)!r}")

    return result


def parse_bboxes(response_json: Any) -> list[BoundingBox]:
    grouped = parse_pick_place_boxes(response_json)
    return grouped["pick"] + grouped["place"] + grouped["other"]


def draw_boxes_on_image(
    image: np.ndarray,
    boxes: list[BoundingBox],
    role_to_color: dict[str, tuple[int, int, int]] | None = None,
    line_width: int = 3,
) -> np.ndarray:
    role_to_color = role_to_color or {
        "pick": (255, 64, 64),
        "place": (64, 220, 120),
        "other": (80, 160, 255),
    }
    canvas = Image.fromarray(_ensure_uint8_rgb(image).copy())
    draw = ImageDraw.Draw(canvas)

    for box in boxes:
        role = box.role or "other"
        color = role_to_color.get(role, role_to_color["other"])
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=line_width)
        text = role.upper()
        if box.label and box.label.lower() != role:
            text = f"{text}:{box.label}"
        text_pos = (max(0, int(box.x1) + 2), max(0, int(box.y1) - 14))
        draw.text(text_pos, text, fill=color)

    return np.asarray(canvas)


def group_boxes_by_camera(boxes: list[BoundingBox]) -> dict[str, list[BoundingBox]]:
    grouped: dict[str, list[BoundingBox]] = {}
    for box in boxes:
        camera = box.camera or ""
        grouped.setdefault(camera, []).append(box)
    return grouped


def overlay_pick_place_boxes(
    rgb_images: dict[str, np.ndarray],
    grouped_boxes: dict[str, list[BoundingBox]] | dict[str, list[Any]],
) -> dict[str, np.ndarray]:
    rendered: dict[str, np.ndarray] = {}
    for camera_name, image in rgb_images.items():
        boxes = grouped_boxes.get(camera_name, [])
        rendered[camera_name] = draw_boxes_on_image(image, boxes)
    return rendered


def compose_prompt_rgb(rgb_images: dict[str, np.ndarray], camera_order: list[str] | None = None) -> np.ndarray:
    ordered_names = camera_order or list(rgb_images.keys())
    arrays = [_ensure_uint8_rgb(rgb_images[name]) for name in ordered_names if name in rgb_images]
    if not arrays:
        raise ValueError("No images provided to compose_prompt_rgb.")
    return np.concatenate(arrays, axis=-1)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send RGB observations to a LLaMA-Factory FastAPI VLM service."
    )
    parser.add_argument("--server-url", type=str, required=True)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Detect the target object and return bounding boxes.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.image_path is None:
        raise ValueError("--image-path is required for standalone testing.")

    image = load_image(args.image_path)
    client = LlamaFactoryVLMClient(
        server_url=args.server_url,
        timeout=args.timeout,
        prompt=args.prompt,
    )
    result = client.query(image)
    boxes = [bbox.to_dict() for bbox in parse_bboxes(result)]

    if args.pretty:
        print(json.dumps(boxes, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(boxes, ensure_ascii=False))


if __name__ == "__main__":
    main()
