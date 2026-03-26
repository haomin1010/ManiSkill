from huggingface_hub import HfApi

api = HfApi()

# api.upload_folder(
#     folder_path="./my_local_data",  # 本地文件夹路径
#     repo_id="username/my-dataset", # 目标仓库 ID
#     repo_type="dataset",           # 仓库类型
#     path_in_repo="data/train",     # (可选) 上传到仓库中的指定位置
#     commit_message="Upload training data folder"
# )

files_to_upload = {
    "/home/user/ManiSkill/videos/StackCube-v1/stackcube_expert.rgb.pd_ee_delta_pos.physx_cpu.h5": "stackcube_expert.rgb.pd_ee_delta_pos.physx_cpu.h5",
    "/home/user/ManiSkill/videos/StackCube-v1/stackcube_expert.rgb.pd_ee_delta_pos.physx_cpu.json": "stackcube_expert.rgb.pd_ee_delta_pos.physx_cpu.json",  
}

for local_path, repo_path in files_to_upload.items():
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id="Notyourbing/StackCubeDP",
        repo_type="dataset"
    )
print("所有文件上传完成！")