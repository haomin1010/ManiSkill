# Heatmap坐标获取方式总结

## 概览

Heatmap中的坐标来自两个不同的来源，取决于是训练还是评估：

| 阶段 | 来源 | 方式 | 频率 |
|------|------|------|------|
| **训练** | JSON文本文件 | 直接读取 | 固定（每个episode一次） |
| **评估** | 仿真环境 | 实时计算投影 | 动态（每一步更新） |

---

## 训练阶段：从JSON文件获取坐标

### 数据源
```
videos_rgbd/StackCube-v1/screenshots/
├── ep0_boxes.json
├── ep1_boxes.json
├── ...
└── ep999_boxes.json
```

### JSON文件结构
```json
{
  "cameras": {
    "base_camera": {
      "init_center_px": [320, 240],
      "goal_center_px": [380, 200],
      "init_box_corners": {
        "center": [320, 240],
        "top_left": [300, 220],
        "bottom_right": [340, 260]
      },
      "goal_box_corners": {
        "center": [380, 200],
        ...
      }
    },
    "left_side_camera": {...},
    "right_side_camera": {...}
  }
}
```

### 坐标提取流程

```
JSON文件
   ↓
[提取] init_center_px, goal_center_px (像素坐标)
   ↓
[读取] 图像尺寸 H, W (从观测数据)
   ↓
[归一化] center_px → [0, 1]
   ├── init_norm_x = init_center_px[0] / W
   └── init_norm_y = init_center_px[1] / H
   ↓
[生成] 多相机Heatmap堆栈
   └── 形状: (6, 32, 32) = 3相机 × 2(init+goal)
```

### 代码实现
**位置**: [train_stackcube.py](train_stackcube.py#L373-L450)

```python
def _build_goal_prompt_from_files(self, traj_idx, prompt_episode_id, ...):
    # 1. 加载JSON
    data = self._load_episode_prompt_json(prompt_dir, prompt_episode_id)
    cams = data.get("cameras", {})
    
    # 2. 对每个相机
    for cam_name in PROMPT_CAMERAS:
        cam_data = cams.get(cam_name)
        
        # 3. 提取像素坐标
        init_center = cam_data.get("init_center_px")  # [320, 240]
        goal_center = cam_data.get("goal_center_px")  # [380, 200]
        
        # 4. 获取图像尺寸
        H, W = self._get_camera_hw(raw_obs_traj_dict, cam_name)  # 512, 512
        
        # 5. 归一化到[0, 1]
        init_norm = torch.tensor([
            float(init_center[0]) / W,  # 320/512 = 0.625
            float(init_center[1]) / H   # 240/512 = 0.469
        ])
        
        # 6. 生成Heatmap
        # 坐标缩放到heatmap尺寸
        # init_x = 0.625 * 32 = 20px (在32×32网格中)
```

### 关键特点
- ✅ **快速加载**: 坐标已预计算，直接从JSON读取
- ✅ **固定值**: 同一episode的坐标在训练中不变
- ✅ **离线标注**: 由人工或其他工具预先生成
- ❌ **不支持动态场景**: 坐标值固定

---

## 评估阶段：从仿真环境实时计算

### 数据源
```
仿真环境中的动态对象位置
│
├─ cubeA_pos (3D世界坐标)
│  └─ [0.5, 0.3, 0.1]
│
└─ cubeB_pos (目标位置)
   └─ [0.4, 0.2, 0.15]
```

### 坐标计算管线

```
世界坐标 (3D)
   ├─ cubeA_pos = [0.5, 0.3, 0.1]
   └─ cubeB_pos = [0.4, 0.2, 0.15]
           ↓
    [相机参数]
    ├─ 内参矩阵 K (3×3)
    └─ 外参矩阵 [R|t] (3×4)
           ↓
    [投影] 世界 → 相机坐标
    ├─ p_cam = [R|t] @ p_world
    └─ 齐次坐标: p_cam_h = R @ p_world + t
           ↓
    [投影] 相机 → 像素坐标
    ├─ u = K[0,0] * (p_cam.x / p_cam.z) + K[0,2]
    └─ v = K[1,1] * (p_cam.y / p_cam.z) + K[1,2]
           ↓
    [归一化] 像素 → [0,1]
    ├─ u_norm = u / width, clamp(0, 1)
    └─ v_norm = v / height, clamp(0, 1)
           ↓
    [生成] 多相机Heatmap堆栈
    └─ 形状: (6, 32, 32)
```

### 数学公式

#### 1. 世界坐标 → 相机坐标
```
齐次坐标:
p_cam_h = [R | t] @ [p_world; 1]
        = R @ p_world + t

其中:
  R = 旋转矩阵 (3×3)
  t = 平移向量 (3×1)
```

#### 2. 相机坐标 → 像素坐标（透视投影）
```
内参矩阵:
    [f_x  0   c_x]
K = [ 0  f_y  c_y]
    [ 0   0    1 ]

投影公式:
   [u]   [f_x  0   c_x] [x/z]
   [v] = [ 0  f_y  c_y] [y/z]
   [1]   [ 0   0    1 ] [ 1 ]

则:
u = f_x * (x/z) + c_x
v = f_y * (y/z) + c_y
```

#### 3. 像素坐标 → 归一化坐标
```
u_norm = u / width ∈ [0, 1]
v_norm = v / height ∈ [0, 1]
```

### 代码实现
**位置**: [prompt_wrappers.py](diffusion_policy/prompt_wrappers.py#L110-L160)

```python
def _compute_goal_prompt(self, observation: dict) -> torch.Tensor:
    # 1. 获取世界坐标（仿真环境）
    cubeA_pos = self.base_env.cubeA.pose.p  # [0.5, 0.3, 0.1]
    cubeB_pos = self.base_env.cubeB.pose.p  # [0.4, 0.2, 0.15]
    
    # 目标是立方体B顶部
    goal_pos = cubeB_pos.clone()
    goal_pos[:, 2] += self.base_env.cube_half_size[2] * 2.0
    
    # 2. 对每个相机
    for cam_name in self.prompt_cameras:
        cam_param = sensor_params[cam_name]
        
        # 3. 获取相机参数
        intrinsic = torch.tensor(cam_param["intrinsic_cv"])  # K矩阵
        extrinsic = torch.tensor(cam_param["extrinsic_cv"]) # [R|t]矩阵
        
        # 4. 投影：世界 → 相机坐标（齐次）
        init_center = self._project_world_points(
            cubeA_pos, intrinsic, extrinsic
        )  # 返回(u, v)像素坐标
        
        # 5. 归一化到[0, 1]
        init_norm = torch.stack([
            (init_center[:, 0] / width).clamp(0.0, 1.0),
            (init_center[:, 1] / height).clamp(0.0, 1.0)
        ])
```

### _project_world_points方法的详细实现

```python
@staticmethod
def _project_world_points(points_world, intrinsic, extrinsic):
    # 1. 转为齐次坐标
    ones = torch.ones((points_world.shape[0], 1),...)
    points_h = torch.cat([points_world, ones], dim=-1)  # (B, 4)
    
    # 2. 相机坐标 = 外参 @ 世界坐标
    points_cam = (extrinsic @ points_h.unsqueeze(-1)).squeeze(-1)
    # extrinsic: (3, 4), points_h: (B, 4)
    # 结果: (B, 3) 在相机坐标系中
    
    # 3. 透视除法（深度归一化）
    z = points_cam[:, 2].clamp(min=1e-6)  # 避免除以0
    
    # 4. 像素坐标 = 内参 @ (相机坐标 / z)
    u = intrinsic[:, 0, 0] * (points_cam[:, 0] / z) + intrinsic[:, 0, 2]
    # u = f_x * (x/z) + c_x
    
    v = intrinsic[:, 1, 1] * (points_cam[:, 1] / z) + intrinsic[:, 1, 2]
    # v = f_y * (y/z) + c_y
    
    return torch.stack([u, v], dim=-1)  # (B, 2)像素坐标
```

### 关键特点
- ✅ **实时动态**: 坐标每一步都更新
- ✅ **准确投影**: 使用真实的相机参数
- ✅ **变动支持**: 适应移动的物体
- ❌ **计算成本**: 每一步都需要投影计算
- ❌ **依赖仿真**: 需要准确的相机标定

---

## 具体数值示例

### 训练示例
```
JSON中:
  init_center_px: [320, 240]
  goal_center_px: [380, 200]
  
图像尺寸: H=512, W=512

归一化过程:
  init_norm_x = 320 / 512 = 0.625
  init_norm_y = 240 / 512 = 0.469
  
  goal_norm_x = 380 / 512 = 0.742
  goal_norm_y = 200 / 512 = 0.391
  
转换到32×32 heatmap:
  init_hm_x = 0.625 * 32 = 20 像素
  init_hm_y = 0.469 * 32 = 15 像素
  
  goal_hm_x = 0.742 * 32 = 24 像素
  goal_hm_y = 0.391 * 32 = 12 像素
  
生成高斯heatmap:
  G(x,y) = exp(-(distance²)/(2σ²)), σ=3.0
```

### 评估示例
```
世界坐标:
  cubeA: [0.5, 0.3, 0.1]
  cubeB: [0.4, 0.2, 0.15]
  
外参矩阵 [R|t] (相对于base_camera):
  [ 0.9  0.0 -0.1 | 0.0]
  [ 0.0  1.0  0.0 | 0.5]
  [ 0.1  0.0  0.9 | 2.0]
  
内参矩阵 K (基于相机焦距):
  [500  0  256]
  [ 0  500  256]
  [ 0   0    1]
  
投影步骤:
  1. p_cam = extrinsic @ [0.5, 0.3, 0.1, 1]ᵀ
           = [0.04, 0.8, 2.08]ᵀ (相机坐标)
  
  2. 透视投影:
     u = 500 * (0.04/2.08) + 256 = 265.6
     v = 500 * (0.8/2.08) + 256 = 448.1
  
  3. 归一化 (假设512×512):
     u_norm = 265.6 / 512 = 0.519
     v_norm = 448.1 / 512 = 0.875
  
  4. 映射到32×32 heatmap:
     hm_x = 0.519 * 32 = 16.6
     hm_y = 0.875 * 32 = 28.0
```

---

## Heatmap生成

从归一化坐标 `(u_norm, v_norm)` 生成高斯热力图：

```
1. 坐标缩放
   x_hm = u_norm * heatmap_width    # ∈ [0, 32]
   y_hm = v_norm * heatmap_height   # ∈ [0, 32]

2. 高斯核计算
   G(x, y) = exp(-(distance² / (2σ²)))
   distance = √((x - x_hm)² + (y - y_hm)²)
   σ = 3.0 (可调参数)

3. 归一化
   G_normalized = G / G.max()  # ∈ [0, 1]
```

---

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `heatmap_h` | 32 | Heatmap高度（像素） |
| `heatmap_w` | 32 | Heatmap宽度（像素） |
| `heatmap_sigma` | 3.0 | 高斯核标准差 |
| `num_cameras` | 3 | 相机数量（base, left, right） |
| `total_channels` | 6 | 输出通道数（3相机 × 2） |

### 调参建议

- **更大的σ** (5-10): 更平滑的热力图，更容易学习但精度下降
- **更小的σ** (1-2): 更尖锐的热力图，更精确但可能过拟合
- **更大的Heatmap** (64×64): 更高的空间分辨率，更多计算
- **更小的Heatmap** (16×16): 更快速度，但信息损失

---

## 总结对比

| 特性 | 训练 | 评估 |
|------|------|------|
| 坐标源 | JSON文件 | 仿真环境 |
| 获取方式 | 直接读取 | 实时投影计算 |
| 更新频率 | 固定（预生成） | 每一步 |
| 计算复杂度 | O(1) 查表 | O(矩阵乘法) |
| 对应的代码 | `_build_goal_prompt_from_files()` | `_compute_goal_prompt()` |
| 使用场景 | 离线数据集 | 在线推理 |
| 动态适应 | ❌ 固定值 | ✅ 随环境变化 |

---

**生成日期**: 2026-04-04  
**相关文件**: 
- [train_stackcube.py](train_stackcube.py) - 训练时的坐标获取
- [prompt_wrappers.py](diffusion_policy/prompt_wrappers.py) - 评估时的坐标获取
- [heatmap_utils.py](diffusion_policy/heatmap_utils.py) - Heatmap生成工具
