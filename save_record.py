import argparse
from pathlib import Path
import json
import shutil

import numpy as np
from PIL import Image
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.examples.motionplanning.panda.solutions.stack_cube import solve


def get_static_cube_positions(env):
    """
    获取不应该被碰撞的静态方块位置。
    包括：
    - extra_green_cubes（堆叠塔中除 cubeB 之外的块）
    - extra_red_cubes（散落的背景红块）
    不包括 cubeB（目标块，红块放上去会有正常的物理交互）。
    """
    unwrapped = env.unwrapped
    positions = {}

    # extra_green_cubes（堆叠塔中除 cubeB 之外的块）
    if hasattr(unwrapped, "extra_green_cubes"):
        for i, cube in enumerate(unwrapped.extra_green_cubes):
            p = cube.pose.p
            positions[f"green_{i}"] = np.asarray(p).flatten()[:3].copy()

    # extra_red_cubes（散落的背景红块）
    if hasattr(unwrapped, "extra_red_cubes"):
        for i, cube in enumerate(unwrapped.extra_red_cubes):
            p = cube.pose.p
            positions[f"red_{i}"] = np.asarray(p).flatten()[:3].copy()

    return positions


def wait_physics_stable(env, steps=10):
    """
    执行若干步"保持当前位置"的动作，让物理引擎稳定。
    返回实际执行的步数。
    
    注意：对于 pd_joint_pos 控制模式，零动作会让机器人移到关节位置 0，
    所以我们需要使用当前关节位置作为动作。
    """
    unwrapped = env.unwrapped
    # 获取当前关节位置作为"保持不动"的动作
    qpos = unwrapped.agent.robot.qpos[0].cpu().numpy()
    # pd_joint_pos 动作维度是 arm joints + gripper，取前 action_dim 个
    action_dim = unwrapped.action_space.shape[0]
    hold_action = qpos[:action_dim].astype(np.float32)
    
    for _ in range(steps):
        env.step(hold_action)
    return steps


def check_collision(init_positions, final_positions, threshold=0.01):
    """
    检查静态方块是否被碰撞（位置变化超过阈值）。
    返回：(bool, dict) -> (是否发生碰撞, 每个方块的位移)
    """
    displacements = {}
    collision = False

    for name, init_p in init_positions.items():
        if name not in final_positions:
            continue
        final_p = final_positions[name]
        disp = np.linalg.norm(final_p - init_p)
        displacements[name] = float(disp)
        if disp > threshold:
            collision = True

    return collision, displacements


def project_world_to_pixel(world_point, intrinsic, extrinsic):
    """将 3D 世界坐标投影到 2D 像素坐标"""
    world_h = np.append(world_point, 1.0)
    cam_h = extrinsic @ world_h
    if cam_h[2] <= 0:
        return None
    uv_h = intrinsic @ cam_h[:3]
    u = uv_h[0] / uv_h[2]
    v = uv_h[1] / uv_h[2]
    return (int(round(u)), int(round(v)))


def save_initial_screenshots(env, ep_idx, output_dir, camera_names=None):
    """
    保存环境稳定后的初始截图和框中心点像素坐标。
    
    Args:
        env: 环境对象
        ep_idx: episode 索引
        output_dir: 输出目录
        camera_names: 要保存的相机名称列表
    """
    if camera_names is None:
        camera_names = ["base_camera", "left_side_camera", "right_side_camera"]
    
    unwrapped = env.unwrapped
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取 cubeA 的当前位置（初始位置）
    cubeA_pos = np.asarray(unwrapped.cubeA.pose.p).flatten()[:3]
    
    # 获取目标位置（cubeB 顶部）
    cubeB_pos = np.asarray(unwrapped.cubeB.pose.p).flatten()[:3]
    cube_half_size = float(unwrapped.cube_half_size[0])
    goal_pos = cubeB_pos.copy()
    goal_pos[2] += cube_half_size * 2  # 在 cubeB 顶部
    
    # 获取最新的观测（包含相机图像）
    obs = unwrapped.get_obs()
    
    result = {
        "episode": ep_idx,
        "cubeA_world_pos": cubeA_pos.tolist(),
        "goal_world_pos": goal_pos.tolist(),
        "cameras": {},
    }
    
    for cam_name in camera_names:
        if cam_name not in obs["sensor_data"]:
            continue
        
        cam_data = obs["sensor_data"][cam_name]
        
        # 获取 RGB 图像
        rgb = cam_data["rgb"][0].cpu().numpy()  # (H, W, 3)
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
        
        # 保存截图
        img_path = output_dir / f"ep{ep_idx}_{cam_name}.png"
        Image.fromarray(rgb).save(img_path)
        
        # 获取相机内参和外参
        cam_params = obs["sensor_param"][cam_name]
        intrinsic = cam_params["intrinsic_cv"][0].cpu().numpy()
        extrinsic = cam_params["extrinsic_cv"][0].cpu().numpy()
        
        # 计算框中心点的像素坐标（框大小可以在后处理时决定）
        init_center = project_world_to_pixel(cubeA_pos, intrinsic, extrinsic)
        goal_center = project_world_to_pixel(goal_pos, intrinsic, extrinsic)
        
        result["cameras"][cam_name] = {
            "image_path": str(img_path.name),
            "init_center_px": list(init_center) if init_center else None,  # 初始位置框中心（红色）
            "goal_center_px": list(goal_center) if goal_center else None,  # 目标位置框中心（蓝色）
        }
    
    # 保存框位置信息
    json_path = output_dir / f"ep{ep_idx}_boxes.json"
    with json_path.open("w") as f:
        json.dump(result, f, indent=2)
    
    print(f"  截图和框位置已保存到: {output_dir}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=3,
        help="要录制的 episode 数量。",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=1,
        help="基础随机种子。若不为 None，则第 i 条轨迹使用 seed = base_seed + i。",
    )
    parser.add_argument(
        "--collision-threshold",
        type=float,
        default=0.02,
        help="判定碰撞的位移阈值（米），默认 0.02m（2cm）。",
    )
    parser.add_argument(
        "--close-camera",
        action="store_true",
        help="使用更近的相机位置（聚焦工作区域）。",
    )
    args = parser.parse_args()

    output_dir = Path("videos/StackCube-v1")
    collision_dir = output_dir / "collision"
    collision_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir = output_dir / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "StackCube-v1",
        obs_mode="rgbd",
        reward_mode="sparse",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        close_camera=args.close_camera,
    )

    env = RecordEpisode(
        env,
        output_dir=str(output_dir),
        save_trajectory=True,
        trajectory_name="stackcube_expert",
        save_video=True,
        info_on_video=False,
        save_on_reset=True,
        record_reward=True,
        record_env_state=True,
        source_type="motionplanning",
        source_desc="Panda motion planner expert demonstrations",
    )

    meta = {}
    collision_episodes = []

    for ep in range(args.num_episodes):
        if args.base_seed is not None:
            ep_seed = int(args.base_seed) + ep
        else:
            ep_seed = None

        obs, info = env.reset(seed=ep_seed)

        # 等待物理引擎稳定（堆叠塔自然沉降），然后再记录初始位置
        stable_steps = wait_physics_stable(env, steps=10)

        # 记录初始位置（物理已稳定）
        init_positions = get_static_cube_positions(env)

        # 定义截图回调函数（在机械臂抬高后调用）
        def screenshot_callback():
            save_initial_screenshots(
                env, ep, screenshot_dir,
                camera_names=["base_camera", "left_side_camera", "right_side_camera"],
            )

        # do_reset=False 因为上面已经 reset 过了，避免重复 reset 导致 episode 被截断
        # after_home_callback 在机械臂抬高后截图，此时机械臂不遮挡视野
        res, home_steps = solve(
            env, seed=ep_seed, debug=False, vis=False,
            do_reset=False, after_home_callback=screenshot_callback
        )
        # 稳定步数也算在需要裁剪的前缀里
        home_steps += stable_steps

        # 检查碰撞
        final_positions = get_static_cube_positions(env)
        collision, displacements = check_collision(
            init_positions, final_positions, threshold=args.collision_threshold
        )

        print(
            f"episode {ep}, seed={ep_seed}, home_steps={home_steps}, "
            f"success={res}, collision={collision}"
        )
        if collision:
            collision_episodes.append(ep)
            print(f"  -> 碰撞检测：位移 = {displacements}")

        meta[ep] = {
            "seed": ep_seed,
            "trim_head_steps": int(home_steps),
            "success": bool(res),
            "collision": collision,
            "displacements": displacements,
        }

    env.close()

    # 保存 meta
    meta_path = output_dir / "stackcube_expert_meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    # 将碰撞 episode 的视频移动到 collision 目录
    if collision_episodes:
        print(f"\n发生碰撞的 episode: {collision_episodes}")
        print(f"正在将碰撞视频移动到 {collision_dir} ...")
        for ep in collision_episodes:
            video_file = output_dir / f"{ep}.mp4"
            if video_file.exists():
                dest = collision_dir / f"{ep}.mp4"
                shutil.move(str(video_file), str(dest))
                print(f"  移动: {video_file.name} -> collision/")
    else:
        print("\n没有检测到碰撞的 episode。")

    print(f"\n录制完成。meta 保存在: {meta_path}")


if __name__ == "__main__":
    main()

