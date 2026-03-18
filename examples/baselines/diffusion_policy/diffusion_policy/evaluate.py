from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from mani_skill.utils import common

def evaluate(n: int, agent, eval_envs, device, sim_backend: str, progress_bar: bool = True):
    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        while eps_count < n:
            obs = common.to_tensor(obs, device)
            action_seq = agent.get_action(obs)
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break

            if truncated.any():
                assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                if "final_info" in info:
                    # physx_cuda / newer gymnasium vector env: final episode info stored in final_info
                    if isinstance(info["final_info"], dict):
                        for k, v in info["final_info"]["episode"].items():
                            eval_metrics[k].append(v.float().cpu().numpy())
                    else:
                        for final_info in info["final_info"]:
                            for k, v in final_info["episode"].items():
                                eval_metrics[k].append(v)
                elif "episode" in info:
                    # physx_cpu / CPUGymWrapper path: episode stats stored directly in info["episode"]
                    for k, v in info["episode"].items():
                        v_arr = np.array(v)
                        if v_arr.ndim == 0:
                            v_arr = v_arr.reshape(1)
                        eval_metrics[k].append(v_arr)
                else:
                    # Fallback: try to collect success from top-level info
                    if "success" in info:
                        v = np.array(info["success"])
                        eval_metrics["success_once"].append(v.reshape(1) if v.ndim == 0 else v)
                eps_count += eval_envs.num_envs
                if progress_bar:
                    pbar.update(eval_envs.num_envs)
    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.concatenate(eval_metrics[k])
    return eval_metrics
