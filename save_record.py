import argparse

import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.examples.motionplanning.panda.solutions.stack_cube import solve


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
    args = parser.parse_args()

    env = gym.make(
        "StackCube-v1",
        obs_mode="rgbd",
        reward_mode="sparse",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",  # 为了录视频要开 human render
    )

    env = RecordEpisode(
        env,
        output_dir="videos/StackCube-v1",  # 这里会保存 .h5/.json + .mp4
        save_trajectory=True,
        trajectory_name="stackcube_expert",
        save_video=True,  # 开启视频保存
        info_on_video=False,
        save_on_reset=True,
        record_reward=True,
        record_env_state=True,
        source_type="motionplanning",
        source_desc="Panda motion planner expert demonstrations",
    )

    meta = {}

    for ep in range(args.num_episodes):
        if args.base_seed is not None:
            ep_seed = int(args.base_seed) + ep
        else:
            ep_seed = None

        obs, info = env.reset(seed=ep_seed)
        res, home_steps = solve(env, seed=ep_seed, debug=False, vis=False)
        print(
            f"episode {ep}, seed={ep_seed}, home_steps={home_steps}, solve result: {res}"
        )
        meta[ep] = {
            "seed": ep_seed,
            "trim_head_steps": int(home_steps),
            "success": bool(res),
        }

    env.close()

    # 将每条轨迹的 trim_head_steps 等信息保存到与轨迹同目录的 meta JSON 中
    from pathlib import Path
    import json

    output_dir = Path("videos/StackCube-v1")
    meta_path = output_dir / "stackcube_expert_meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

