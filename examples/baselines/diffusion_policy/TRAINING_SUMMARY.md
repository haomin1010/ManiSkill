# 必须在宿主机环境运行

# Diffusion Policy Training & Evaluation Summary

**Task:** StackCube-v1 (RGB visual prompt with bounding box)
**Algorithm:** BC_Diffusion_rgbd_UNet — Behavior Cloning with DDPM-based Diffusion Policy

**Training Command:**
``` python
python examples/baselines/diffusion_policy/train_rgbd_demo.py --env-id StackCube-v1 \
  --demo-path videos/StackCube-v1/stackcube_expert.rgb.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" \
  --sim-backend "physx_cpu" \
  --max_episode_steps 300 \
  --total_iters 30000 \
  --obs-mode "rgb" \
  --exp-name diffusion_policy-StackCube-v1-rgb-delta-visual_prompt

```


## 1. Training Setup

### Data
- **Source:** Expert demonstration HDF5 file (e.g., `stackcube_expert.rgb.pd_ee_delta_pos.physx_cpu.h5`)
- **Loader:** `SmallDemoDataset_DiffusionPolicy` — loads all trajectories into GPU memory
- **Demos used:** 19 trajectories, 5178 total transitions, 5159 obs sequences
- **Observation processing:** `convert_obs()` resizes all camera RGB images to **128×128**, concatenates across cameras on the channel axis, and transposes to `(C, H, W)` layout

### Optimizer & Schedule
| Setting | Value |
|---|---|
| Optimizer | AdamW (`lr=1e-4`, betas=(0.95, 0.999), weight_decay=1e-6) |
| LR Schedule | Cosine with 500-step linear warmup |
| Total iterations | 30,000 |
| Batch size | 256 |
| EMA | power=0.75 (weights copied to `ema_agent` for evaluation) |

### Logging & Checkpointing
- TensorBoard logs every 1000 iterations
- Evaluation every 5000 iterations (100 episodes across 10 parallel envs)
- Best checkpoint saved on `success_once` and `success_at_end` metrics

---

## 2. Diffusion Policy: Input, Output, Condition

### Architecture Overview
```
Observations → PlainConv Visual Encoder → Condition Vector
                                              ↓
Gaussian Noise → ConditionalUnet1D (100 DDPM steps) → Denoised Action Sequence
                                              ↑
                                   Condition Vector (global_cond)
```

### Input (to the model at each training step)
| Key | Shape | Description |
|---|---|---|
| `rgb` | `(B, obs_horizon, C_rgb, 128, 128)` | Stacked RGB frames from all cameras; C_rgb = num_cameras × 3 = 12 for StackCube (4 cams × 3 channels), uint8 normalized to [0,1] |
| `state` | `(B, obs_horizon, obs_state_dim)` | Proprioceptive state (joint positions, velocities, TCP pose, gripper, object pose, etc.) |
| `prompt_rgb` | `(B, C_rgb, 128, 128)` | Single-frame visual prompt image with bounding boxes drawn (see Section 4) |
| `action_seq` | `(B, pred_horizon, act_dim)` | Ground-truth expert actions for supervised loss |

> **obs_horizon = 2** (2 stacked timesteps of observation)
> **pred_horizon = 16** (predict 16 future action steps)

### Output (from the model)
| Key | Shape | Description |
|---|---|---|
| `action_seq` | `(B, act_horizon, act_dim)` | Predicted actions to execute; `act_horizon=8` steps taken from positions `[obs_horizon-1 : obs_horizon-1+act_horizon]` of the denoised sequence |
| `act_dim` | 7 | 6-DOF EE delta position + 1 gripper (for `pd_ee_delta_pos` control mode) |

### Conditioning
The conditioning vector `obs_cond` passed as `global_cond` to the UNet is built in `encode_obs()`:

```
obs_cond = [prompt_feature | visual_feature_t0 | state_t0 | visual_feature_t1 | state_t1 | ...]
```

| Component | Computation | Dim |
|---|---|---|
| `prompt_feature` | `PlainConv(prompt_rgb)` → 1 feature vector | 256 |
| `visual_feature` (per timestep) | `PlainConv(rgb_t)` → feature per timestep | 256 × obs_horizon |
| `state` (per timestep) | raw proprioceptive state per timestep | obs_state_dim × obs_horizon |

**Total `global_cond_dim`:**
`obs_horizon × (256 + obs_state_dim) + 256` = `2 × (256 + obs_state_dim) + 256`

### Visual Encoder: PlainConv
```
Input: (B, C, 128, 128)
Conv2d(C→16, 3×3) → ReLU → MaxPool(2×2)   # 64×64
Conv2d(16→32, 3×3) → ReLU → MaxPool(2×2)  # 32×32
Conv2d(32→64, 3×3) → ReLU → MaxPool(2×2)  # 16×16
Conv2d(64→128, 3×3) → ReLU → MaxPool(2×2) # 8×8
Conv2d(128→128, 1×1) → ReLU
AdaptiveMaxPool2d(1×1) → flatten → Linear(128→256)
Output: (B, 256)
```
The **same PlainConv** is shared between the regular obs and the prompt_rgb — same architecture, same weights.

### Noise Scheduler (DDPM)
| Parameter | Value |
|---|---|
| num_train_timesteps | 100 |
| beta_schedule | `squaredcos_cap_v2` |
| clip_sample | True (clips to [-1, 1]) |
| prediction_type | `epsilon` (predict noise, not denoised action) |

### UNet: ConditionalUnet1D
| Parameter | Value |
|---|---|
| input_dim | act_dim (7) |
| global_cond_dim | obs_horizon × (256 + obs_state_dim) + 256 |
| diffusion_step_embed_dim | 64 |
| down_dims | [64, 128, 256] |
| n_groups | 8 |
| Total parameters | ~7.02M |

---

## 3. Evaluation Setup

### Environment Configuration
- **Env:** `StackCube-v1`, `physx_cpu` backend
- **Control mode:** `pd_ee_delta_pos`
- **Obs mode:** `rgb`
- **Camera resolution:** 128×128 (set via `sensor_configs`)
- **Max episode steps:** 200
- **Reward mode:** sparse

### Environment Wrappers (applied in order)
1. `FlattenRGBDAndPromptWrapper` — draws bounding boxes dynamically and returns `{state, rgb, prompt_rgb}`
2. `FrameStack(obs_horizon=2)` — stacks the last 2 observations
3. `CPUGymWrapper` — adds `ignore_terminations=True`, `record_metrics=True`
4. `VisualPromptWrapper` — fixes `prompt_rgb` to the **initial observation** of each episode

### Evaluation Loop
- 10 parallel environments (`AsyncVectorEnv` with forkserver context)
- 100 total evaluation episodes per eval cycle
- Agent uses **EMA weights** (not online weights) for evaluation
- At each step: encode obs → run full 100-step DDPM reverse diffusion → execute `act_horizon=8` actions open-loop → repeat
- Metrics: `success_once`, `success_at_end` (averaged across episodes)

---

## 4. Visual Prompt: Bounding Box Analysis

### Concept
The visual prompt is a single **reference image** annotated with **dashed bounding boxes** indicating the task-relevant objects. It gives the policy a "goal hint" — what objects to attend to and where — without changing the policy architecture.

### What is Drawn
For **StackCube-v1**, two bounding boxes are drawn per camera view:
| Box | Color | Target |
|---|---|---|
| Green dashed rectangle | `(0, 255, 0)` | `cubeA` (the cube to pick up) — projected world position |
| Blue dashed rectangle | `(0, 0, 255)` | Goal position = `cubeB.pose.p + [0, 0, 0.04]` (top of cubeB, where cubeA should be placed) |

### Drawing Method (`draw_dashed_rect`)
- Projects 3D world position to 2D pixel using camera intrinsics `K` and extrinsics
- Draws a **20×20 pixel dashed rectangle** centered on the projected pixel
- Dash length: 5 pixels, line thickness: 2 pixels
- Uses PIL for drawing, returns numpy array

### Projection Pipeline
```python
P_world (3D) → homogeneous → extrinsic_cv @ P_world → [x, y, z] camera space
→ u = K[0,0] * x/z + K[0,2]
→ v = K[1,1] * y/z + K[1,2]
→ pixel (u, v)
```

### Training: Offline Prompt from Boxed MP4
- Pre-generated boxed frames are stored in `<demo_dir>/boxed/<traj_id>_<cam>_boxed.mp4`
- At dataset construction time, the **first frame** of each boxed MP4 is read per camera
- All per-camera frames are resized to **128×128** and concatenated along channels → `(C, 128, 128)` tensor stored as `prompt_rgb` in the dataset
- One prompt per trajectory (static, not per timestep)
- Fallback: if no boxed MP4 exists, uses the raw frame 0 from HDF5 (no bounding box)

### Evaluation: Dynamic Online Prompt
- `FlattenRGBDAndPromptWrapper.observation()` draws boxes **live** every step using ground-truth object positions from the simulator
- `VisualPromptWrapper` captures `prompt_rgb` at episode **reset** (t=0) and **freezes** it for the entire episode
- On episode end (truncation), it updates `initial_obs` to the newest frame for the done environments

### Data Flow Summary
```
Training:
  HDF5 demo file
    └─ sensor_data[cam]["rgb"][0]  ─┐
    └─ boxed/<traj>_<cam>_boxed.mp4─┤→ resize to 128×128 → concat channels
                                    ↓
                          prompt_rgb: (C, 128, 128) stored in dataset
                                    ↓
                          __getitem__: prompt_rgb returned as-is (C, 128, 128)
                                    ↓
                          DataLoader batches to (B, C, 128, 128)
                                    ↓
                          encode_obs: PlainConv → prompt_feature (B, 256)
                                    ↓
                          Concat into global_cond for UNet

Evaluation:
  FlattenRGBDAndPromptWrapper
    └─ Live 3D pos from simulator → project to 2D → draw dashed boxes → prompt_rgb
  VisualPromptWrapper
    └─ Freeze prompt_rgb at t=0 for each episode
  get_action:
    └─ prompt_rgb (B, obs_horizon, H, W, C) → take [:, 0] → permute → PlainConv
```

### Design Rationale
- The bounding box serves as a **low-cost, interpretable visual goal specification** — no pretrained vision model needed
- By freezing the prompt at episode t=0, the policy gets a **consistent reference** of the initial object positions
- The shared `PlainConv` encoder means prompt features are in the same embedding space as the current observation features, enabling direct comparison
- Concatenating `prompt_feature` at the front of `global_cond` means the UNet denoising is **globally conditioned** on the task goal throughout all diffusion steps
