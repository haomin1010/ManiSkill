import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import StackCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.wrappers.record import RecordEpisode

def solve(env: StackCubeEnv, seed=None, debug=False, vis=False, do_reset=True, after_home_callback=None):
    """
    执行堆叠任务的 motion planning。
    
    Args:
        do_reset: 是否在 solve 内部调用 env.reset()。
                  如果外部已经 reset 过，设为 False 避免重复 reset。
        after_home_callback: 可选的回调函数，在机械臂抬高（home pose）完成后调用。
                             用于在机械臂不遮挡视野时截图。
    """
    print(f"[solve] start, seed={seed}, do_reset={do_reset}")
    if do_reset:
        env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # 记录从本次 solve 开始的 step 计数，用于后面统计“home 段”用了多少步
    planner.elapsed_steps = 0

    # ---------------------------------------------------------------------- #
    # Home pose：在任务开始前先把机械臂从当前初始姿态纯粹抬高 30cm，
    # 这样起始帧中机械臂不会挡住桌面上的方块。只在开头真实执行一次，不再去远处复杂位置。
    # ---------------------------------------------------------------------- #
    tcp_pose = env.agent.tcp.pose
    # tcp_pose.p / tcp_pose.q 是 batched 的 torch.Tensor，这里取第 0 个并转成 numpy float32
    cur_p = tcp_pose.p[0].cpu().numpy().astype(np.float32)
    home_p = cur_p.copy()
    home_p[2] += 0.30  # 在当前姿态基础上抬高 30 cm
    home_q = tcp_pose.q[0].cpu().numpy().astype(np.float32)
    home_pose = sapien.Pose(p=home_p, q=home_q)
    print(f"[solve] move to simple home pose (+0.3m), target_p={home_p}, current_p={cur_p}")
    import sys; sys.stdout.flush()
    res_home = planner.move_to_pose_with_screw(home_pose)
    print(f"[solve] home pose planning done")
    home_steps = int(planner.elapsed_steps)
    print(f"[solve] simple home pose result={res_home}, home_steps={home_steps}")
    # 若规划失败，不强制中止 episode，继续尝试后续任务

    # 在机械臂抬高后调用回调（此时机械臂不遮挡视野，适合截图）
    if after_home_callback is not None:
        after_home_callback()

    obb = get_actor_obb(env.cubeA)

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Search a valid pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    print("[solve] search grasp pose by rotating around z")
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        print(f"[solve]   dry-run grasp angle={angle:.3f}, result={res}")
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("[solve] Fail to find a valid grasp pose, abort episode")
        planner.close()
        return False, home_steps

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    print("[solve] move to reach pose")
    res_move = planner.move_to_pose_with_screw(reach_pose)
    print(f"[solve] reach pose result={res_move}")
    if res_move == -1:
        print("[solve] fail to reach pre-grasp pose, abort episode")
        planner.close()
        return False, home_steps

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    print("[solve] move to grasp pose & close gripper")
    res_move = planner.move_to_pose_with_screw(grasp_pose)
    print(f"[solve] grasp pose result={res_move}")
    if res_move == -1:
        print("[solve] fail to move to grasp pose, abort episode")
        planner.close()
        return False, home_steps
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    # 把红块抬高到足够高度，避免在水平移动时碰到堆叠塔上层的其它方块
    lift_pose = sapien.Pose([0, 0, 0.25]) * grasp_pose
    print("[solve] lift after grasp")
    res_move = planner.move_to_pose_with_screw(lift_pose)
    print(f"[solve] lift pose result={res_move}")
    if res_move == -1:
        print("[solve] fail to lift, abort episode")
        planner.close()
        return False, home_steps

    # -------------------------------------------------------------------------- #
    # Stack
    # -------------------------------------------------------------------------- #
    # 使用 cubeB 顶部的位置对齐红块的中心，同时保留抓取后的姿态（lift_pose.q），
    # 避免在已经抓稳之后再对方块施加额外的扭转导致滑落。
    goal_pose = env.cubeB.pose * sapien.Pose(
        [0, 0, (env.cube_half_size[2] * 2).item()]
    )
    # 这里的 pose.p / pose.q 可能是 batched 的 torch.Tensor 或 numpy.ndarray，
    # 统一转成 numpy，并在有 batch 维时取第 0 个。
    def _to_vec(x, expected_dim):
        arr = np.asarray(x)
        if arr.ndim > 1:
            arr = arr[0]
        assert arr.shape[0] == expected_dim, f"Unexpected shape {arr.shape} for dim={expected_dim}"
        return arr

    goal_p = _to_vec(goal_pose.p, 3)
    lift_p = _to_vec(lift_pose.p, 3)
    lift_q = _to_vec(lift_pose.q, 4)

    # 计算红块需要旋转的角度，使其与 cubeB 对齐
    # 提取 cubeA 和 cubeB 的 yaw（绕 z 轴旋转角度）
    def quat_to_yaw(q):
        # q = [w, x, y, z]，假设主要旋转是绕 z 轴
        return 2.0 * np.arctan2(q[3], q[0])

    def quat_mul(q1, q2):
        # 四元数乘法，q = [w, x, y, z]
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=np.float32)

    cubeA_q = _to_vec(env.cubeA.pose.q, 4)
    cubeB_q = _to_vec(env.cubeB.pose.q, 4)
    cubeA_yaw = quat_to_yaw(cubeA_q)
    cubeB_yaw = quat_to_yaw(cubeB_q)
    yaw_diff = cubeB_yaw - cubeA_yaw

    # 正方体有 90° 旋转对称性，所以 yaw_diff 归一化到 [-45°, 45°] 范围
    # 这样可以避免多转半圈
    while yaw_diff > np.pi / 4:
        yaw_diff -= np.pi / 2
    while yaw_diff < -np.pi / 4:
        yaw_diff += np.pi / 2
    print(f"[solve] cubeA_yaw={np.degrees(cubeA_yaw):.1f}°, cubeB_yaw={np.degrees(cubeB_yaw):.1f}°, adjusted_diff={np.degrees(yaw_diff):.1f}°")

    # 构造绕世界 z 轴旋转 yaw_diff 的四元数
    q_rot = np.array([np.cos(yaw_diff / 2), 0, 0, np.sin(yaw_diff / 2)], dtype=np.float32)
    # 左乘：在世界坐标系中绕 z 轴旋转
    place_q = quat_mul(q_rot, lift_q)

    # 改为“到目标上方直接松爪”：
    # 先移动到目标 xy 上方的安全高度，不再下探接触，直接释放让方块自然下落，
    # 以减少周围有方块时的碰撞风险。

    # 用“方块中心位移补偿”消除抓取偏心导致的系统性落点偏差：
    # 先读取当前 cubeA 中心与 TCP 的相对结果，不直接假设 TCP 对到 goal 就能让方块到 goal。
    cubeA_p_now = _to_vec(env.cubeA.pose.p, 3).astype(np.float32)
    desired_cube_center = goal_p.astype(np.float32)

    # 目标上方的释放高度：约 1.2 个方块边长
    release_height = float(env.cube_half_size[2] * 2.0 * 1.2)
    desired_cube_center[2] = float(goal_p[2] + release_height)

    # 若当前 cubeA 不是正好在 TCP 下方中心，按“cubeA 当前中心 -> 目标中心”的位移来移动 TCP。
    # 这样能补偿抓取偏心、轻微滑移造成的恒定偏差。
    delta_cube = desired_cube_center - cubeA_p_now
    pre_place_p = (lift_p.astype(np.float32) + delta_cube).astype(np.float32)
    pre_place_pose = sapien.Pose(p=pre_place_p, q=place_q)
    print("[solve] move to release pose (above target, aligned)")
    res_move = planner.move_to_pose_with_screw(pre_place_pose)
    print(f"[solve] release pose result={res_move}")
    if res_move == -1:
        print("[solve] fail to move to release pose, abort episode")
        planner.close()
        return False, home_steps

    # 在目标上方先短暂停顿几步，再松爪，减少末端还在微动时释放带来的偏差。
    pause_steps = 6
    qpos_hold = planner.robot.get_qpos()[0, : len(planner.planner.joint_vel_limits)].cpu().numpy()
    for _ in range(pause_steps):
        if planner.control_mode == "pd_joint_pos":
            hold_action = np.hstack([qpos_hold, planner.CLOSED])
        else:
            hold_action = np.hstack([qpos_hold, qpos_hold * 0, planner.CLOSED])
        env.step(hold_action)
        planner.elapsed_steps += 1

    res = planner.open_gripper()
    print(f"[solve] open gripper above target, result={res}")

    # -------------------------------------------------------------------------- #
    # Reset：在成功放置并松爪之后，再次回到开头使用的简单 home pose
    # 这样可以在数据中记录一个“复位”过程，方便训练包含复位段的策略 / 视觉先验。
    # -------------------------------------------------------------------------- #
    print("[solve] move back to simple home pose for reset (+0.3m)")
    planner.elapsed_steps = 0
    res_reset = planner.move_to_pose_with_screw(home_pose)
    reset_steps = int(planner.elapsed_steps)
    print(f"[solve] reset home pose result={res_reset}, reset_steps={reset_steps}")

    print("[solve] end of task (with explicit reset home move)")
    planner.close()
    print("[solve] done")
    # 返回：任务是否成功 + home 段所占的步数（用于后处理时裁剪前缀）
    return res, home_steps
