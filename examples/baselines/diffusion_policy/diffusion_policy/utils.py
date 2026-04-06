import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from h5py import Dataset, File, Group
from torch.utils.data.sampler import Sampler
from tqdm import tqdm


class IterationBasedBatchSampler(Sampler):
    """Wraps a BatchSampler.
    Resampling from it until a specified number of iterations have been sampled
    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter


def worker_init_fn(worker_id, base_seed=None):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.
    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    if base_seed is None:
        base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


TARGET_KEY_TO_SOURCE_KEY = {
    "states": "env_states",
    "observations": "obs",
    "success": "success",
    "next_observations": "obs",
    # 'dones': 'dones',
    # 'rewards': 'rewards',
    "actions": "actions",
}


def load_content_from_h5_file(file):
    if isinstance(file, (File, Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, Dataset):
        return file[()]
    else:
        raise NotImplementedError(f"Unspported h5 file type: {type(file)}")


def load_hdf5(
    path,
):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    ret = load_content_from_h5_file(file)
    file.close()
    print("Loaded")
    return ret


def load_traj_hdf5(path, num_traj=None):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    keys = list(file.keys())
    if num_traj is not None:
        assert num_traj <= len(keys), f"num_traj: {num_traj} > len(keys): {len(keys)}"
        keys = sorted(keys, key=lambda x: int(x.split("_")[-1]))
        keys = keys[:num_traj]
    ret = {}
    for key in tqdm(keys, desc="Loading HDF5 trajectories", leave=False):
        ret[key] = load_content_from_h5_file(file[key])
    file.close()
    print("Loaded")
    return ret


def load_demo_dataset(
    path, keys=["observations", "actions"], num_traj=None, concat=True
):
    # assert num_traj is None
    raw_data = load_traj_hdf5(path, num_traj)
    # raw_data has keys like: ['traj_0', 'traj_1', ...]
    # raw_data['traj_0'] has keys like: ['actions', 'dones', 'env_states', 'infos', ...]
    _traj = raw_data["traj_0"]
    for key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[key]
        assert source_key in _traj, f"key: {source_key} not in traj_0: {_traj.keys()}"
    dataset = {}
    for target_key in keys:
        # if 'next' in target_key:
        #     raise NotImplementedError('Please carefully deal with the length of trajectory')
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = [raw_data[idx][source_key] for idx in raw_data]
        if isinstance(dataset[target_key][0], np.ndarray) and concat:
            if target_key in ["observations", "states"] and len(
                dataset[target_key][0]
            ) > len(raw_data["traj_0"]["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[:-1] for t in dataset[target_key]], axis=0
                )
            elif target_key in ["next_observations", "next_states"] and len(
                dataset[target_key][0]
            ) > len(raw_data["traj_0"]["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[1:] for t in dataset[target_key]], axis=0
                )
            else:
                dataset[target_key] = np.concatenate(dataset[target_key], axis=0)

            print("Load", target_key, dataset[target_key].shape)
        else:
            print(
                "Load",
                target_key,
                len(dataset[target_key]),
                type(dataset[target_key][0]),
            )
    return dataset


def _resize_img_to(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a (H,W,C) or (T,H,W,C) uint8/float numpy array to (target_h, target_w, C)."""
    if img.ndim == 3:  # (H, W, C)
        if img.shape[0] == target_h and img.shape[1] == target_w:
            return img
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    elif img.ndim == 4:  # (T, H, W, C)
        if img.shape[1] == target_h and img.shape[2] == target_w:
            return img
        return np.stack(
            [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for f in img]
        )
    return img


def convert_obs(
    obs,
    concat_fn,
    transpose_fn,
    state_obs_extractor,
    depth=True,
    rgb=True,
    target_size=(128, 128),
    exclude_camera_names=None,
):
    img_dict = obs["sensor_data"]
    exclude_camera_names = set(exclude_camera_names or [])
    # [DEBUG] 首次调用时打印 sensor_data 的相机顺序（决定 concat 后的通道顺序）
    if not hasattr(convert_obs, "_logged"):
        cam_order = [k for k in img_dict.keys() if k not in exclude_camera_names]
        n_rgb_cams = len([v for k, v in img_dict.items() if k not in exclude_camera_names and isinstance(v, dict) and "rgb" in v])
        n_depth_cams = len([v for k, v in img_dict.items() if k not in exclude_camera_names and isinstance(v, dict) and "depth" in v])
        print(
            f"[DEBUG] convert_obs 首次调用: sensor_data 相机顺序 = {cam_order}, "
            f"含 rgb 的数量 = {n_rgb_cams}, 含 depth 的数量 = {n_depth_cams}"
        )
        convert_obs._logged = True

    ls = []
    if rgb:
        ls.append("rgb")
    if depth:
        ls.append("depth")
    if len(ls) == 0:
        raise ValueError("convert_obs requires at least one visual modality: rgb or depth")

    target_h, target_w = target_size

    new_img_dict = {}
    for key in ls:
        per_cam = []
        for cam_name, cam_data in img_dict.items():
            if cam_name in exclude_camera_names:
                continue
            if key not in cam_data:
                continue
            arr = cam_data[key]
            if key == "depth":
                arr = np.asarray(arr)
                # Normalize depth to (..., H, W, 1) before resizing/concatenation.
                # Common dataset layouts include (T,H,W) and (T,H,W,1).
                if arr.ndim == 3:
                    arr = arr[..., None]
                elif arr.ndim == 2:
                    arr = arr[None, ..., None]
            resized = _resize_img_to(arr, target_h, target_w)
            if key == "depth":
                resized = np.asarray(resized)
                # cv2 may squeeze single-channel depth to (..., H, W); restore channel dim.
                if resized.ndim == 3:
                    resized = resized[..., None]
                elif resized.ndim == 2:
                    resized = resized[None, ..., None]
            per_cam.append(resized)
        if len(per_cam) == 0:
            raise KeyError(
                f"Requested modality '{key}' but none of the cameras contain it. "
                f"Available per camera: { {k: list(v.keys()) for k, v in img_dict.items()} }"
            )
        new_img_dict[key] = transpose_fn(concat_fn(per_cam))
        # (C, H, W) or (B, C, H, W)

    if "depth" in new_img_dict and isinstance(new_img_dict['depth'], torch.Tensor): # MS2 vec env uses float16, but gym AsyncVecEnv uses float32
        new_img_dict['depth'] = new_img_dict['depth'].to(torch.float16)

    # Unified version
    states_to_stack = state_obs_extractor(obs)
    for j in range(len(states_to_stack)):
        if states_to_stack[j].dtype == np.float64:
            states_to_stack[j] = states_to_stack[j].astype(np.float32)
    try:
        state = np.hstack(states_to_stack)
    except:  # dirty fix for concat trajectory of states
        state = np.column_stack(states_to_stack)
    if state.dtype == np.float64:
        for x in states_to_stack:
            print(x.shape, x.dtype)
        import pdb

        pdb.set_trace()

    out_dict = {"state": state}

    if "rgb" in new_img_dict:
        out_dict["rgb"] = new_img_dict["rgb"]

    if "depth" in new_img_dict:
        out_dict["depth"] = new_img_dict["depth"]


    return out_dict


def build_obs_space(env, depth_dtype, state_obs_extractor):
    # NOTE: We have to use float32 for gym AsyncVecEnv since it does not support float16, but we can use float16 for MS2 vec env
    obs_space = env.observation_space

    # Unified version
    state_dim = sum([v.shape[0] for v in state_obs_extractor(obs_space)])

    single_img_space = next(iter(env.observation_space["image"].values()))
    h, w, _ = single_img_space["rgb"].shape
    n_images = len(env.observation_space["image"])

    return spaces.Dict(
        {
            "state": spaces.Box(
                -float("inf"), float("inf"), shape=(state_dim,), dtype=np.float32
            ),
            "rgb": spaces.Box(0, 255, shape=(n_images * 3, h, w), dtype=np.uint8),
            "depth": spaces.Box(
                -float("inf"), float("inf"), shape=(n_images, h, w), dtype=depth_dtype
            ),
        }
    )


def build_state_obs_extractor(env_id):
    # NOTE: You can tune/modify state observations specific to each environment here as you wish. By default we include all data
    # but in some use cases you might want to exclude e.g. obs["agent"]["qvel"] as qvel is not always something you query in the real world.
    # 针对 PickCube 和 StackCube 任务，精准提取需要的本体特征，过滤掉 is_grasped 等特权信息
    # if "StackCube" in env_id:
    #     return lambda obs: [
    #         obs["agent"]["qpos"],
    #         obs["agent"]["qvel"],
    #         obs["extra"]["tcp_pose"]
    #     ]
    return lambda obs: list(obs["agent"].values()) + list(obs["extra"].values())
