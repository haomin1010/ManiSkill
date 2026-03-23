# Diffusion Policy — StackCube-v1 Training Summary

Script: `train_stackcube.py`

---

## Parameters

### Experiment

| Parameter | Default | Description |
|---|---|---|
| `--exp-name` | `None` (auto-generated) | Run name. Auto format: `StackCube-v1__train_stackcube__<seed>__<timestamp>` |
| `--seed` | `1` | Random seed for reproducibility |
| `--cuda` | `True` | Use GPU if available |
| `--track` | `False` | Enable Weights & Biases logging |
| `--capture-video` | `True` | Save evaluation rollout videos to `runs/<run_name>/videos/` |

### Data

| Parameter | Default | Description |
|---|---|---|
| `--demo-path` | `videos/StackCube-v1/stackcube_expert.rgb.pd_ee_delta_pos.physx_cpu.h5` | Path to the HDF5 demo dataset (relative to working directory) |
| `--num-demos` | `None` (all) | Number of trajectories to load. `None` loads all 48 (24 success + 24 failure). Recommend filtering to success-only first. |
| `--batch-size` | `256` | Training batch size |
| `--num-dataload-workers` | `0` | DataLoader worker processes (0 = main process) |

### Diffusion Policy Architecture

| Parameter | Default | Description |
|---|---|---|
| `--obs-horizon` | `2` | Number of consecutive observations fed as conditioning |
| `--act-horizon` | `8` | Number of actions actually executed per policy query |
| `--pred-horizon` | `16` | Number of actions the U-Net predicts per forward pass |
| `--lr` | `1e-4` | Learning rate (AdamW, cosine schedule with 500-step warmup) |
| `--diffusion-step-embed-dim` | `64` | Embedding dim for diffusion timestep sinusoidal encoding |
| `--unet-dims` | `[64, 128, 256]` | Channel sizes for the 1D U-Net down/up blocks (~4.5M params) |
| `--n-groups` | `8` | Group norm groups per block |

### Environment

| Parameter | Default | Description |
|---|---|---|
| `--env-id` | `StackCube-v1` | ManiSkill environment |
| `--control-mode` | `pd_ee_delta_pos` | Must match demo dataset. End-effector delta position, action dim = 4 (dx, dy, dz, gripper) |
| `--obs-mode` | `rgb` | No depth — dataset was replayed with `-o rgb` |
| `--max-episode-steps` | `300` | Episode length override. Demos are ~190–231 steps; 300 gives margin |
| `--sim-backend` | `physx_cpu` | CPU simulation backend, matches demo replay backend |
| `--close-camera` | `False` | Auto-synced from demo JSON (`close_camera=True` if used during recording) |

### Training Schedule

| Parameter | Default | Description |
|---|---|---|
| `--total-iters` | `1_000_000` | Total gradient steps. May converge earlier with small dataset — monitor TensorBoard |
| `--log-freq` | `1000` | Log loss and LR to TensorBoard every N iterations |
| `--eval-freq` | `5000` | Run evaluation every N iterations |
| `--save-freq` | `None` | Save checkpoint every N iterations. `None` = only save on best eval metric |
| `--num-eval-episodes` | `20` | Episodes per evaluation. 20 × 10 envs = 2 batches |
| `--num-eval-envs` | `10` | Parallel eval environments |

---

## Input / Output

### Training Input (from HDF5 dataset)

Each trajectory contains:

```
obs/
  agent/
    qpos       (T+1, 9)   — joint positions
    qvel       (T+1, 9)   — joint velocities
  extra/
    tcp_pose   (T+1, 7)   — end-effector pose (pos + quaternion)
  sensor_data/
    base_camera/rgb        (T+1, 512, 512, 3)   uint8
    left_side_camera/rgb   (T+1, 512, 512, 3)   uint8
    right_side_camera/rgb  (T+1, 512, 512, 3)   uint8
    hand_camera/rgb        (T+1, 512, 512, 3)   uint8
actions                    (T, 4)   — [dx, dy, dz, gripper]
```

After `obs_process_fn` preprocessing per trajectory:
- **RGB**: all 4 cameras concatenated channel-wise → `(T+1, 12, 128, 128)` uint8 (resized from 512→128)
- **State**: `[qpos(9), qvel(9), tcp_pose(7)]` → `(T+1, 25)` float32
- **Actions**: normalized to `[-1, 1]` per DoF using min/max from training data → `(T, 4)` float32

Each dataset item (one training sample) is a sliding window:
```
observations:
  rgb    (obs_horizon=2, 12, 128, 128)   — 2 stacked frames, 4 cams × 3 channels
  state  (obs_horizon=2, 25)             — 2 stacked state vectors
actions  (pred_horizon=16, 4)            — 16-step action chunk, normalized
```

### Model Architecture

```
Visual path:
  rgb (B, obs_horizon, 12, H, W)
    → flatten obs_horizon → (B*2, 12, 128, 128)
    → PlainConv (CNN encoder, pool_feature_map=True) → (B*2, 256)
    → reshape → (B, 2, 256)

State path:
  state (B, obs_horizon, 25)

Concat:
  (B, obs_horizon, 256+25) → flatten → (B, 2*281 = 562)   ← FiLM conditioning vector

Diffusion:
  ConditionalUnet1D(input_dim=4, global_cond_dim=562)
    → predicts noise over (B, pred_horizon=16, act_dim=4)
    → 100 DDPM denoising steps at inference
```

### Training Output (per iteration)

- Gradient update on `ConditionalUnet1D` + `PlainConv` weights
- EMA shadow copy updated every step
- Logged to TensorBoard every 1000 iters:
  - `losses/total_loss` — MSE between predicted and actual noise
  - `charts/learning_rate`

### Evaluation Output (every 5000 iters)

The EMA agent runs `num_eval_episodes=20` episodes across `num_eval_envs=10` parallel envs.

**Per query cycle:**
1. Stack last `obs_horizon=2` observations
2. Run 100-step DDPM denoising → `(num_envs, pred_horizon=16, 4)` normalized actions
3. Denormalize actions back to `pd_ee_delta_pos` space
4. Execute `act_horizon=8` actions in the environment
5. Repeat until episode truncation (`max_episode_steps=300`)

**Metrics logged to TensorBoard under `eval/`:**
- `success_once` — fraction of episodes where the cube was stacked at any point
- `success_at_end` — fraction of episodes where the cube was stacked at episode end

**Checkpoints saved to** `runs/<run_name>/checkpoints/`:
- `best_eval_success_once.pt` — saved when `success_once` improves
- `best_eval_success_at_end.pt` — saved when `success_at_end` improves
- `<iteration>.pt` — if `--save-freq` is set

Each checkpoint contains:
```python
{
    "agent": agent.state_dict(),       # online model weights
    "ema_agent": ema_agent.state_dict() # EMA model weights (used for eval)
}
```

---

## How to Run

**Important:** run from the directory where `videos/` exists (the ManiSkill repo root inside docker).

```bash
cd /home
python examples/baselines/diffusion_policy/train_stackcube.py \
    --demo-path videos/StackCube-v1/stackcube_expert.rgb.pd_ee_delta_pos.physx_cpu.h5 \
    --num-eval-episodes 20 \
    --total-iters 30000
```

Monitor training:
```bash
tensorboard --logdir runs/
```
