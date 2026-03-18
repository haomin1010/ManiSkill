# Step 1: 用运动规划生成专家演示（pd_joint_pos，原始格式）
python save_record.py --num-episodes 30 --base-seed 15

# Step 2: 重放轨迹，转换为 pd_ee_delta_pos + rgb 格式（供 diffusion policy 训练用）
# 输出文件：videos/StackCube-v1/stackcube_expert.rgb.pd_ee_delta_pos.physx_cpu.h5
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path videos/StackCube-v1/stackcube_expert.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o rgb \
  --save-traj --num-envs 10 -b physx_cpu

# Step 3: 生成带 bounding box 的可视化视频（用于检查数据质量）
python tools/vis_stackcube_boxes.py \
  --traj videos/StackCube-v1/stackcube_expert.h5 \
  --meta videos/StackCube-v1/stackcube_expert.json \
  --output-dir videos/StackCube-v1/boxed \
  --box-size 20 \
  --cameras base_camera,left_side_camera,right_side_camera \
  --meta-trim videos/StackCube-v1/stackcube_expert_meta.json
