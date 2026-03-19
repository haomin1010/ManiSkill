import argparse
from pathlib import Path

import h5py
import json
import numpy as np
from PIL import Image, ImageDraw

from mani_skill.utils.visualization.misc import images_to_video


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 StackCube 轨迹文件生成带虚线框的多相机视频（base + 左右侧相机）。"
    )
    parser.add_argument(
        "--traj",
        type=str,
        required=True,
        help="RecordEpisode 生成的 .h5 轨迹文件路径，例如 videos/StackCube-v1/stackcube_expert.h5",
    )
    parser.add_argument(
        "--meta",
        type=str,
        required=False,
        help="与轨迹同名的 .json 元数据路径，默认与 --traj 同名（只改后缀）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="输出视频目录，默认与 --traj 同目录下的 boxed/ 子目录",
    )
    parser.add_argument(
        "--meta-trim",
        type=str,
        required=False,
        help="可选：包含每条轨迹 trim_head_steps 信息的 JSON 文件路径（例如 stackcube_expert_meta.json）。",
    )
    parser.add_argument(
        "--box-size",
        type=int,
        default=40,
        help="虚线框的宽高（像素），正方形。",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="base_camera,left_side_camera,right_side_camera",
        help="需要画框的相机名，用逗号分隔。",
    )
    parser.add_argument(
        "--skip-collision",
        action="store_true",
        help="跳过标记为碰撞的轨迹（需要 --meta-trim 中有 collision 字段）。",
    )
    parser.add_argument(
        "--save-boxed-video",
        action="store_true",
        help="是否生成带框的 mp4 视频（默认不生成，仅保存 screenshots 中的截图）。",
    )
    return parser.parse_args()


def draw_dashed_rect(
    img_np: np.ndarray,
    center_uv: tuple[int, int],
    size: int,
    color=(0, 0, 255),  # 蓝色（初始位置框）
    dash_len: int = 5,
    thickness: int = 2,
) -> np.ndarray:
    """在图像上以 center_uv 为中心画一个 size×size 的虚线矩形。"""
    u, v = center_uv
    x0 = int(u - size / 2)
    y0 = int(v - size / 2)
    x1 = int(u + size / 2)
    y1 = int(v + size / 2)

    im = Image.fromarray(img_np)
    draw = ImageDraw.Draw(im)

    def dashed_line(p0, p1):
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        length = max(abs(dx), abs(dy))
        if length == 0:
            return
        for i in range(0, length, dash_len * 2):
            t0 = i / length
            t1 = min(i + dash_len, length) / length
            sx = x0 + dx * t0
            sy = y0 + dy * t0
            ex = x0 + dx * t1
            ey = y0 + dy * t1
            draw.line((sx, sy, ex, ey), fill=color, width=thickness)

    dashed_line((x0, y0), (x1, y0))
    dashed_line((x1, y0), (x1, y1))
    dashed_line((x1, y1), (x0, y1))
    dashed_line((x0, y1), (x0, y0))

    return np.array(im)


def project_world_to_pixel(P_world: np.ndarray, K: np.ndarray, extrinsic_cv: np.ndarray):
    """用 sensor_param 里的 intrinsic_cv / extrinsic_cv 把 3D 点投影到像素坐标。"""
    Pw = np.append(P_world, 1.0)  # (4,)
    Pc = extrinsic_cv @ Pw  # (3,)
    x, y, z = Pc
    if z <= 0:
        return None
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return int(round(u)), int(round(v))


def process_one_traj(
    h5_file: h5py.File,
    traj_id: str,
    box_size: int,
    camera_names: list[str],
    output_dir: Path,
    trim_head_steps: int = 0,
    screenshot_dir: Path = None,
    ep_idx: int = None,
    save_boxed_video: bool = False,
):
    """
    对单条轨迹生成视频：
    - 从 obs/sensor_data 中读取每一步的 rgb
    - 从 obs/sensor_param 中读取相机参数
    - 从 env_states/actors 中读取 cubeA / cubeB 的世界坐标，构造目标点
    - 对指定相机画 初始(蓝) / 目标(黄) 虚线框
    - 为每个相机分别写一个独立的视频文件（不再拼接）
    - 如果 screenshot_dir 不为 None，保存第一帧的带框截图
    """
    group = h5_file[traj_id]

    # 这里假设 obs 结构与 ManiSkill RecordEpisode 默认一致：
    # obs/sensor_data/<cam>/rgb : (T, H, W, 3)
    # obs/sensor_param/<cam>/intrinsic_cv : (T, 3, 3)
    # obs/sensor_param/<cam>/extrinsic_cv : (T, 3, 4)
    obs_grp = group["obs"]

    # 读取 env_states 里的 cubeA / cubeB 位姿。
    # 典型结构：env_states/actors/cubeA/pose : (T, 4, 4)
    # 如果你本地的形状不同，可以根据实际情况稍作调整。
    env_states = group.get("env_states", None)
    if env_states is None:
        raise RuntimeError(
            "轨迹中没有 env_states，当前脚本依赖 env_states 中的 cubeA / cubeB 位姿。"
        )
    actors = env_states["actors"]
    if "cubeA" not in actors or "cubeB" not in actors:
        raise RuntimeError("env_states/actors 下没有 cubeA 或 cubeB 数据。")

    # 在目前 ManiSkill 的存储格式中，actors["cubeA"] / ["cubeB"] 本身就对应一个 pose 数组
    # （而不是一个包含 "pose" 字段的子 group 或 compound 类型），所以直接读取数据即可。
    cubeA_pose = actors["cubeA"][...]  # (T, 4, 4) 或 (T, 7) 等
    cubeB_pose = actors["cubeB"][...]

    # 根据常规模型，若是 4x4 齐次矩阵，则平移在 [:3, 3]。
    # 若是 7 维 (xyz + quat)，则前 3 维是平移。
    def extract_pos(pose_array):
        if pose_array.ndim == 3 and pose_array.shape[1:] == (4, 4):
            return pose_array[..., :3, 3]
        elif pose_array.ndim == 2 and pose_array.shape[1] >= 3:
            return pose_array[..., :3]
        else:
            raise RuntimeError(
                f"不认识的 pose 形状 {pose_array.shape}，请在 vis_stackcube_boxes.py 里根据实际数据调整 extract_pos。"
            )

    cubeA_pos = extract_pos(cubeA_pose)  # (T, 3)
    cubeB_pos = extract_pos(cubeB_pose)  # (T, 3)

    T = cubeA_pos.shape[0]
    # 按需要裁掉前面的若干步（例如回 home 用掉的步数）
    start_t = max(0, int(trim_head_steps))

    # 估计 cube 半边长（在当前任务里是 0.02，可以直接写死；这里尝试从轨迹里读，失败则退回 0.02）
    cube_half = 0.02
    # 目标点：cubeB 顶面上方一个 cube 高度的位置（大致与 solve() 里的 goal_pose 对齐）
    goal_pos = cubeB_pos + np.array([0.0, 0.0, cube_half * 2.0], dtype=np.float32)  # (T,3)

    sensor_data_grp = obs_grp["sensor_data"]
    sensor_param_grp = obs_grp["sensor_param"]

    # 只截图时仅处理第一帧，生成视频时处理全部
    frames_per_camera: dict[str, list[np.ndarray]] = {cam: [] for cam in camera_names}
    frame_indices = [start_t] if not save_boxed_video else list(range(start_t, T))

    for t in frame_indices:
        for cam in camera_names:
            if cam not in sensor_data_grp:
                # 如果该相机在这条轨迹里不存在，跳过
                continue
            cam_data = sensor_data_grp[cam]
            cam_param = sensor_param_grp[cam]

            # rgb: (T, H, W, 3)
            if "rgb" not in cam_data:
                continue
            rgb = cam_data["rgb"][t]  # (H, W, 3)

            # intrinsic_cv: (T, 3, 3), extrinsic_cv: (T, 3, 4)
            K = cam_param["intrinsic_cv"][t]
            extr = cam_param["extrinsic_cv"][t]

            # 初始 cubeA (当下时刻的位置) 画蓝框
            init_world = cubeA_pos[t]
            init_uv = project_world_to_pixel(init_world, K, extr)
            if init_uv is not None:
                rgb = draw_dashed_rect(
                    rgb,
                    init_uv,
                    size=box_size,
                    color=(0, 0, 255),  # 蓝色（初始位置框）
                )

            # 目标位置（使用预先估计的 goal_pos[t]）画黄框
            g_world = goal_pos[t]
            g_uv = project_world_to_pixel(g_world, K, extr)
            if g_uv is not None:
                rgb = draw_dashed_rect(
                    rgb,
                    g_uv,
                    size=box_size,
                    color=(255, 255, 0),  # 黄色（目标位置框）
                )

            frames_per_camera[cam].append(rgb.astype(np.uint8))

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存第一帧的带框截图
    if screenshot_dir is not None and ep_idx is not None:
        screenshot_dir = Path(screenshot_dir)
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        for cam, frames in frames_per_camera.items():
            if frames:
                # 保存第一帧作为截图
                first_frame = frames[0]
                img_path = screenshot_dir / f"ep{ep_idx}_{cam}_boxed.png"
                Image.fromarray(first_frame).save(img_path)
        print(f"  带框截图已保存到: {screenshot_dir}")
    
    # 每个相机单独写一个视频（仅当 save_boxed_video 时）
    if save_boxed_video:
        for cam, frames in frames_per_camera.items():
            if not frames:
                continue
            video_name = f"{traj_id}_{cam}_boxed"
            images_to_video(frames, str(output_dir), video_name=video_name, fps=10, verbose=True)


def compute_box_corners(center, box_size):
    """根据中心点和框大小计算四个角的像素坐标"""
    if center is None:
        return None
    cx, cy = center
    half = box_size // 2
    return {
        "center": [cx, cy],
        "top_left": [cx - half, cy - half],
        "top_right": [cx + half, cy - half],
        "bottom_left": [cx - half, cy + half],
        "bottom_right": [cx + half, cy + half],
    }


def update_screenshot_json_with_corners(screenshot_dir, ep_idx, box_size):
    """
    读取 screenshot JSON 文件，根据 box_size 计算8个角点坐标，保存更新后的 JSON。
    """
    screenshot_dir = Path(screenshot_dir)
    json_path = screenshot_dir / f"ep{ep_idx}_boxes.json"
    
    if not json_path.exists():
        return None
    
    with json_path.open("r") as f:
        data = json.load(f)
    
    # 添加 box_size 信息
    data["box_size"] = box_size
    
    # 为每个相机计算8个角点
    for cam_name, cam_data in data.get("cameras", {}).items():
        init_center = cam_data.get("init_center_px")
        goal_center = cam_data.get("goal_center_px")
        
        # 计算初始位置框的4个角（红色框）
        cam_data["init_box_corners"] = compute_box_corners(init_center, box_size)
        # 计算目标位置框的4个角（蓝色框）
        cam_data["goal_box_corners"] = compute_box_corners(goal_center, box_size)
    
    # 保存更新后的 JSON
    output_path = screenshot_dir / f"ep{ep_idx}_boxes_with_corners.json"
    with output_path.open("w") as f:
        json.dump(data, f, indent=2)
    
    print(f"  框角点坐标已保存到: {output_path}")
    return data


def main():
    args = parse_args()
    traj_path = Path(args.traj)
    meta_path = Path(args.meta) if args.meta is not None else traj_path.with_suffix(".json")
    output_dir = Path(args.output_dir) if args.output_dir is not None else traj_path.parent / "boxed"
    screenshot_dir = traj_path.parent / "screenshots"

    if not traj_path.exists():
        raise FileNotFoundError(traj_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    with meta_path.open("r") as f:
        _ = json.load(f)  # 目前并未用到 env 元数据，不过将来可以用来做一致性检查

    # 读取可选的 trim meta（记录每条轨迹前面需要裁掉的步数）
    trim_meta = {}
    if args.meta_trim is not None:
        try:
            with open(args.meta_trim, "r") as f:
                trim_meta = json.load(f)
        except FileNotFoundError:
            print(f"[vis] meta_trim file not found: {args.meta_trim}, ignoring trim info")

    camera_names = [c.strip() for c in args.cameras.split(",") if c.strip()]

    with h5py.File(traj_path, "r") as h5_file:
        for traj_id in h5_file.keys():
            if not traj_id.startswith("traj_"):
                continue
            # 轨迹名形如 "traj_0"、"traj_1"，我们用其中的索引到 trim_meta 中查找
            try:
                ep_idx = int(traj_id.split("_")[1])
            except (IndexError, ValueError):
                ep_idx = None

            trim_head = 0
            collision = False
            if ep_idx is not None:
                # save_record.py 中用的是整型 key 写入 meta（会在 json 里变成字符串）
                key = str(ep_idx)
                if key in trim_meta:
                    trim_head = int(trim_meta[key].get("trim_head_steps", 0))
                    collision = bool(trim_meta[key].get("collision", False))

            if args.skip_collision and collision:
                print(f"Skipping {traj_id} (collision=True)")
                continue

            print(f"Processing {traj_id} (trim_head_steps={trim_head}, collision={collision}) ...")
            process_one_traj(
                h5_file=h5_file,
                traj_id=traj_id,
                box_size=args.box_size,
                camera_names=camera_names,
                output_dir=output_dir,
                trim_head_steps=trim_head,
                screenshot_dir=screenshot_dir,
                ep_idx=ep_idx,
                save_boxed_video=args.save_boxed_video,
            )
            
            # 更新 screenshot JSON，添加8个角点坐标
            if ep_idx is not None and screenshot_dir.exists():
                update_screenshot_json_with_corners(screenshot_dir, ep_idx, args.box_size)


if __name__ == "__main__":
    main()

