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

def solve(env: StackCubeEnv, seed=None, debug=False, vis=False):
    print(f"[solve] start, seed={seed}")
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
    print("[solve] move to simple home pose (+0.3m)")
    res_home = planner.move_to_pose_with_screw(home_pose)
    home_steps = int(planner.elapsed_steps)
    print(f"[solve] simple home pose result={res_home}, home_steps={home_steps}")
    # 若规划失败，不强制中止 episode，继续尝试后续任务

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
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
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
    cubeA_p = _to_vec(env.cubeA.pose.p, 3)
    offset = goal_p - cubeA_p

    lift_p = _to_vec(lift_pose.p, 3)
    lift_q = _to_vec(lift_pose.q, 4)

    target_p = lift_p + offset
    target_q = lift_q
    align_pose = sapien.Pose(
        p=target_p.astype(np.float32),
        q=target_q.astype(np.float32),
    )
    print("[solve] move to align (stack) pose")
    res_move = planner.move_to_pose_with_screw(align_pose)
    print(f"[solve] align pose result={res_move}")
    if res_move == -1:
        print("[solve] fail to move to stack pose, abort episode")
        planner.close()
        return False, home_steps

    res = planner.open_gripper()
    print(f"[solve] open gripper, result={res}")

    # 任务完成后将机械臂复位到同一个 home pose，避免遮挡最后若干帧的视野
    # 结束时不再强制回 home，避免额外长轨迹；此处只打印一行标记结束。
    print("[solve] end of task (no explicit home move)")
    planner.close()
    print("[solve] done")
    # 返回：任务是否成功 + home 段所占的步数（用于后处理时裁剪前缀）
    return res, home_steps
