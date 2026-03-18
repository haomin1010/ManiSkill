python save_record.py --num-episodes 50 --base-seed 15

python tools/vis_stackcube_boxes.py   --traj videos/StackCube-v1/stackcube_expert.h5   --meta videos/StackCube-v1/stackcube_expert.json   --output-dir videos/StackCube-v1/boxed   --box-size 20 --cameras base_camera,left_side_camera,right_side_camera --meta-trim videos/StackCube-v1/stackcube_expert_meta.json
