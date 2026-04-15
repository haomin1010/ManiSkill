
# Step 1: 用运动规划生成专家演示（pd_joint_pos，原始格式）
# --delete-collision-videos: 检测到碰撞时删除视频，不保留到 collision/
python save_record.py --num-episodes 200 --base-seed 42 --delete-collision-videos
# StackCube：待抓取块 / 目标堆叠 / 散落块颜色每回合随机；散落块数量随机，总方块数（含 cubeA）不超过 10。

# Step 2: 重放轨迹，转换为 pd_ee_delta_pos + rgb 格式（供 diffusion policy 训练用）
# --skip-collision: 跳过碰撞 episode，不重放、不保存到训练 H5
# 输出文件：videos/StackCube-v1/stackcube_expert.rgb.pd_ee_delta_pos.physx_cpu.h5
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path videos/StackCube-v1/stackcube_expert.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o rgbd \
  --save-traj --num-envs 10 -b physx_cpu \
  --allow-failure --skip-collision

# Step 3: 生成带框截图（screenshots/）及可选的 boxed 视频
# --skip-collision: 跳过碰撞 episode
python tools/vis_stackcube_boxes.py \
  --traj videos/StackCube-v1/stackcube_expert.rgbd.pd_ee_delta_pos.physx_cpu.h5 \
  --meta videos/StackCube-v1/stackcube_expert.rgbd.pd_ee_delta_pos.physx_cpu.json \
  --output-dir videos/StackCube-v1/boxed \
  --box-size 25 \
  --cameras base_camera,left_side_camera,right_side_camera \
  --meta-trim videos/StackCube-v1/stackcube_expert_meta.json \
  --skip-collision
