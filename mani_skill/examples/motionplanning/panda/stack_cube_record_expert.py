import argparse

import gymnasium as gym
import torch

from mani_skill.examples.motionplanning.panda.solutions.stack_cube import solve
from mani_skill.utils.wrappers.record import RecordEpisode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        default="StackCube-v1",
        help="Environment ID to generate expert demonstrations for.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of expert episodes to record.",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="rgbd",
        help="Observation mode to use when recording trajectories.",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        default="pd_joint_pos",
        help="Control mode used by the motion planner.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="demos/StackCube-v1/trajectory.rgbd.pd_joint_pos.physx_cpu.h5",
        help="Full output h5 path for recorded expert trajectories.",
    )
    parser.add_argument(
        "--num-distractor-cubes",
        type=int,
        default=4,
        help="Number of distractor cubes to enable in the environment.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="基础随机种子。若不为 None，则第 i 条轨迹使用 seed = base_seed + i。",
    )
    args = parser.parse_args()

    # We only need sensor-based observations (rgbd) for recording,
    # not human-viewer rendering. To reduce crashes from GPU / driver
    # issues, force CPU PhysX + CPU rendering backend and disable
    # human render images.
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode="sparse",
        control_mode=args.control_mode,
        sim_backend="physx_cpu",
        render_backend="cpu",
        render_mode=None,
        num_distractor_cubes=args.num_distractor_cubes,
    )

    # RecordEpisode expects an output directory + (optional) trajectory_name,
    # not a full output_path. We parse the user-given save_path accordingly.
    from pathlib import Path

    save_path = Path(args.save_path)
    output_dir = save_path.parent
    trajectory_name = save_path.stem

    env = RecordEpisode(
        env,
        output_dir=str(output_dir),
        save_trajectory=True,
        trajectory_name=trajectory_name,
        save_video=False,
        save_on_reset=True,
        record_reward=True,
        record_env_state=True,
        source_type="motionplanning",
        source_desc="Panda motion planner expert demonstrations with distractor cubes.",
    )

    num_success = 0
    for ep in range(args.num_episodes):
        # 为当前 episode 生成种子：base_seed + ep（若未指定 base_seed，则为 None，保持随机）
        if args.base_seed is not None:
            ep_seed = int(args.base_seed) + ep
        else:
            ep_seed = None

        obs, _ = env.reset(seed=ep_seed)
        # 注意：这里直接把带有 RecordEpisode wrapper 的 env 传给 solve，
        # 使轨迹与视频都从该 reset 开始记录。
        # solve 现在返回 (success, home_steps)，这里忽略 home_steps。
        res, _ = solve(env, seed=ep_seed, debug=False, vis=False)
        # solve returns True/False style result; also check env info success if needed
        if res:
            num_success += 1
        # force a reset/save via wrapper
        env.reset()
        print(f"Episode {ep + 1}/{args.num_episodes}, success={bool(res)}")

    print(f"Finished recording {args.num_episodes} episodes, successes={num_success}")
    env.close()


if __name__ == "__main__":
    # ensure torch does not create unnecessary cuda contexts if not needed
    if not torch.cuda.is_available():
        torch.set_num_threads(1)
    main()

