from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("StackCube-v1", max_episode_steps=50)
class StackCubeEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to pick up a red cube and stack it on top of a green cube and let go of the cube without it falling

    **Randomizations:**
    - both cubes have their z-axis rotation randomized
    - both cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

    **Success Conditions:**
    - the red cube is on top of the green cube (to within half of the cube size)
    - the red cube is static
    - the red cube is not being grasped by the robot (robot must let go of the cube)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        num_distractor_cubes: int = 0,
        num_extra_red_cubes: int = 0,
        close_camera: bool = False,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.num_distractor_cubes = num_distractor_cubes
        self.num_extra_red_cubes = num_extra_red_cubes  # 额外的红色背景方块数量，默认 0 即仅 1 个红块（cubeA）
        self.distractor_cubes = []
        self.close_camera = close_camera  # 是否使用更近的相机位置
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # 根据 close_camera 参数选择相机位置
        if self.close_camera:
            # 更近的相机位置，聚焦在工作区域
            center_pose = sapien_utils.look_at(eye=[0.15, 0.0, 0.35], target=[0.0, 0.0, 0.04])
            left_pose = sapien_utils.look_at(eye=[0.15, 0.18, 0.35], target=[0.0, 0.0, 0.04])
            right_pose = sapien_utils.look_at(eye=[0.15, -0.18, 0.35], target=[0.0, 0.0, 0.04])
        else:
            # 原来的相机位置（较远）
            center_pose = sapien_utils.look_at(eye=[0.3, 0.0, 0.6], target=[-0.1, 0.0, 0.1])
            left_pose = sapien_utils.look_at(eye=[0.3, 0.25, 0.6], target=[-0.1, 0.0, 0.1])
            right_pose = sapien_utils.look_at(eye=[0.3, -0.25, 0.6], target=[-0.1, 0.0, 0.1])

        # 提高分辨率，避免后处理视频画面太小
        width = 512
        height = 512
        return [
            CameraConfig("base_camera", center_pose, width, height, np.pi / 2, 0.01, 100),
            CameraConfig("left_side_camera", left_pose, width, height, np.pi / 2, 0.01, 100),
            CameraConfig("right_side_camera", right_pose, width, height, np.pi / 2, 0.01, 100),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # 在桌面上画出一个固定的“作业范围”正方形（类似白色胶带圈出的区域）
        # 这里使用四条很细长的 box 作为边框，只添加可视几何体，不参与碰撞。
        # 将范围缩小到刚好容纳 4x4 个方块的正方形区域，以堆叠方块中心为正中。
        side = float(self.cube_half_size[0] * 8.0)  # 4 个方块直径 = 8 * half_size
        workspace_half_x = side / 2.0
        workspace_half_y = side / 2.0
        line_thickness = 0.002
        line_height = 0.001  # 稍微抬高一点避免与桌面共面导致闪烁
        # 上下两条边（沿 X 方向延伸）
        self.workspace_top_edge = actors.build_box(
            self.scene,
            half_sizes=[workspace_half_x, line_thickness, line_height],
            color=[1.0, 1.0, 1.0, 1.0],
            name="workspace_top_edge",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0.0, workspace_half_y, line_height]),
        )
        self.workspace_bottom_edge = actors.build_box(
            self.scene,
            half_sizes=[workspace_half_x, line_thickness, line_height],
            color=[1.0, 1.0, 1.0, 1.0],
            name="workspace_bottom_edge",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0.0, -workspace_half_y, line_height]),
        )
        # 左右两条边（沿 Y 方向延伸）
        self.workspace_left_edge = actors.build_box(
            self.scene,
            half_sizes=[line_thickness, workspace_half_y, line_height],
            color=[1.0, 1.0, 1.0, 1.0],
            name="workspace_left_edge",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[-workspace_half_x, 0.0, line_height]),
        )
        self.workspace_right_edge = actors.build_box(
            self.scene,
            half_sizes=[line_thickness, workspace_half_y, line_height],
            color=[1.0, 1.0, 1.0, 1.0],
            name="workspace_right_edge",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[workspace_half_x, 0.0, line_height]),
        )
        # 任务方块：
        # - cubeA：红色，可移动的“待摆放积木”（唯一红色目标块）
        # - cubeB + extra_green_cubes：堆叠在一起的目标积木，每个块一种鲜艳颜色，便于区分
        if self.num_extra_red_cubes > 0:
            self.distractor_colors = [
                [1.0, 0.0, 0.0, 1.0],   # 红
                [1.0, 0.0, 0.5, 1.0],   # 玫红
                [0.5, 0.0, 1.0, 1.0],   # 紫蓝
                [1.0, 0.5, 0.5, 1.0],   # 浅红
            ]
            self.cubeA = actors.build_cube(
                self.scene,
                half_size=0.02,
                color=self.distractor_colors[0],
                name="cubeA",
                initial_pose=sapien.Pose(p=[0, 0, 0.1]),
            )
            self.extra_red_cubes = []
            for i in range(self.num_extra_red_cubes):
                cube = actors.build_cube(
                    self.scene,
                    half_size=0.02,
                    color=self.distractor_colors[(i + 1) % len(self.distractor_colors)],
                    name=f"red_distractor_{i}",
                    initial_pose=sapien.Pose(p=[0.0, 0.0, -1.0]),
                )
                self.extra_red_cubes.append(cube)
        else:
            self.cubeA = actors.build_cube(
                self.scene,
                half_size=0.02,
                color=[1.0, 0.0, 0.0, 1.0],  # 红
                name="cubeA",
                initial_pose=sapien.Pose(p=[0, 0, 0.1]),
            )
            self.extra_red_cubes = []

        # 为堆叠积木准备一组不太浅、互相区分度高的颜色（不包含白色）
        stack_colors = [
            [0.0, 0.8, 0.0, 1.0],   # 绿
            [0.0, 0.4, 1.0, 1.0],   # 蓝
            [1.0, 0.8, 0.0, 1.0],   # 黄橙
            [0.8, 0.0, 0.8, 1.0],   # 紫
            [1.0, 0.4, 0.0, 1.0],   # 橙红
            [0.0, 0.8, 0.8, 1.0],   # 青
            [0.6, 0.3, 0.0, 1.0],   # 棕
            [0.5, 0.0, 0.0, 1.0],   # 深红
        ]

        # 主目标块 cubeB 使用堆叠颜色中的第一个
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=stack_colors[0],
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        )

        # 额外的堆叠方块，用来组成 1~3 层、总数 1~8 个的“堆叠场景”
        # 这里预先创建最多 7 个额外方块，实际每个 episode 中会根据采样需求启用其中一部分，其余放到桌子下方隐藏。
        self.max_green_cubes = 8  # 包括 cubeB 在内的总数上限
        self.extra_green_cubes = []
        for i in range(self.max_green_cubes - 1):
            cube = actors.build_cube(
                self.scene,
                half_size=0.02,
                color=stack_colors[i + 1],
                name=f"cubeB_extra_{i}",
                initial_pose=sapien.Pose(p=[0.0, 0.0, -1.0]),
            )
            self.extra_green_cubes.append(cube)

        # Optional distractor cubes (non-goal objects) for visual clutter / collisions.
        if self.num_distractor_cubes > 0:
            # Pre-create actors; their poses will be randomized in _initialize_episode.
            base_colors = [
                [1.0, 0.5, 0.0, 1.0],  # orange
                [0.6, 0.0, 0.8, 1.0],  # purple
                [0.0, 0.8, 0.8, 1.0],  # cyan
                [0.8, 0.8, 0.0, 1.0],  # yellow
            ]
            for i in range(self.num_distractor_cubes):
                color = base_colors[i % len(base_colors)]
                distractor = actors.build_cube(
                    self.scene,
                    half_size=0.02,
                    color=color,
                    name=f"distractor_cube_{i}",
                    initial_pose=sapien.Pose(p=[0.0, 0.0, -1.0]),
                )
                self.distractor_cubes.append(distractor)

    def _sample_distractor_cube_xy(self, num: int, cubeA_xy: torch.Tensor, cubeB_xy: torch.Tensor):
        """Sample XY positions for distractor cubes in a table region, avoiding heavy overlap with cubeA/B."""
        # Simple heuristic sampler in the same region used for task cubes.
        b = cubeA_xy.shape[0]
        device = cubeA_xy.device
        # Region consistent with main sampler
        region = torch.tensor([[-0.1, -0.2], [0.1, 0.2]], device=device)
        width = region[1] - region[0]
        # We currently assume single-env usage for motion planning / data generation,
        # so we take the first env index.
        xy_list = []
        max_trials = 50
        min_dist = torch.linalg.norm(torch.tensor([0.02, 0.02], device=device)) * 1.5
        for _ in range(num):
            for _ in range(max_trials):
                sample = region[0] + torch.rand(2, device=device) * width
                # Avoid being too close to cubeA or cubeB (env 0)
                if (
                    torch.linalg.norm(sample - cubeA_xy[0]) > min_dist
                    and torch.linalg.norm(sample - cubeB_xy[0]) > min_dist
                ):
                    xy_list.append(sample)
                    break
            else:
                xy_list.append(cubeA_xy[0])
        return torch.stack(xy_list, dim=0)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # -------------------------------
            # 1) 红色方块 cubeA 的初始位置、姿态（桌面上，允许轻微 yaw 偏转）
            # -------------------------------
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 2] = self.cube_half_size[2]
            # 将 cubeA 放置在 4x4 线框“外侧但不太远”的环形区域内，
            # 避免一开始就挤在堆叠塔附近，同时又不会离任务区太远。
            # 线框半边长与 _load_scene 中保持一致。
            side = float(self.cube_half_size[0] * 8.0)
            frame_half = side / 2.0
            max_dist = float(self.cube_half_size[0] * 10.0)  # 上界，约 5 个方块直径
            for _ in range(64):
                candidate_xy = torch.rand((b, 2), device=self.device) * 0.8 - 0.4  # [-0.4,0.4]
                # 只检查第 0 个 env（当前假设单 env）
                c0 = candidate_xy[0]
                # r 必须落在 (frame_half, max_dist) 之间，既在线框外，又不太远
                r = torch.linalg.norm(c0)
                if (torch.any(torch.abs(c0) > frame_half)) and (r < max_dist):
                    base_xy = candidate_xy
                    break
            else:
                base_xy = torch.rand((b, 2), device=self.device) * 0.8 - 0.4
            xyz[:, :2] = base_xy
            # 绕 z 轴添加一个小角度偏转（例如 ±30°），使方块看起来“歪一点”
            angle = (torch.rand(b, device=self.device) - 0.5) * (np.pi / 3.0)  # ±30°
            qw = torch.cos(angle / 2.0)
            qz = torch.sin(angle / 2.0)
            shared_qs = torch.zeros((b, 4), device=self.device)
            shared_qs[:, 0] = qw
            shared_qs[:, 3] = qz
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=shared_qs))

            # 为额外红块随机选择位置和轻微 yaw（线框外，与堆叠塔和 cubeA 拉开距离）
            if hasattr(self, "extra_red_cubes") and len(self.extra_red_cubes) > 0:
                min_dist_from_stack = float(self.cube_half_size[0] * 6.0)
                min_dist_from_cubeA = float(self.cube_half_size[0] * 4.0)
                cubeA_xy_np = base_xy[0].cpu().numpy()
                for cube in self.extra_red_cubes:
                    for _ in range(64):
                        xy = (torch.rand(2, device=self.device) * 0.8 - 0.4).cpu().numpy()
                        dist_stack = np.linalg.norm(xy)
                        dist_cubeA = np.linalg.norm(xy - cubeA_xy_np)
                        outside_frame = abs(xy[0]) > frame_half or abs(xy[1]) > frame_half
                        if outside_frame and dist_stack > min_dist_from_stack and dist_cubeA > min_dist_from_cubeA:
                            break
                    p = np.array([xy[0], xy[1], float(self.cube_half_size[2])], dtype=np.float32)
                    a = (np.random.rand() - 0.5) * (np.pi / 3.0)
                    qw_r = np.cos(a / 2.0)
                    qz_r = np.sin(a / 2.0)
                    q = np.array([qw_r, 0.0, 0.0, qz_r], dtype=np.float32)
                    pose = Pose.create_from_pq(
                        p=torch.tensor([p], device=self.device),
                        q=torch.tensor([q], device=self.device),
                    )
                    cube.set_pose(pose)

            # -------------------------------
            # 2) 绿色方块堆叠：总数 1~8 个，1~3 层，姿态对齐且规则栈叠
            #    - 所有绿色方块（包括 cubeB）共享同一个 yaw，整齐对齐
            #    - 采用 2x2 的网格，每一层最多 4 个方块，最多 3 层
            #    - 物理约束：如果第 n+1 层某位置有方块，则第 n 层同位置必须也有方块（不允许“悬空”）
            #    - 目标 cubeB 可以位于任意一层的任意方块上（不再强制选最上层），
            #      这样红块有时会放在最高层上方，有时会落在已有最高层之下。
            # -------------------------------
            # 第一步：为当前 episode 采样绿色方块总数和层数（至少 1 个）
            num_green = torch.randint(
                low=1,
                high=self.max_green_cubes + 1,
                size=(1,),
                device=self.device,
            ).item()
            max_layers = min(3, num_green)
            num_layers = torch.randint(
                low=1,
                high=max_layers + 1,
                size=(1,),
                device=self.device,
            ).item()

            # 为整堆绿色方块使用单位四元数，使所有块与桌面坐标轴完全对齐。
            green_q = torch.zeros((1, 4), device=self.device)
            green_q[:, 3] = 1.0

            # 每一层使用 2x2 网格，间距刚好是一个方块直径，使方块“挤在一起”
            spacing = float(self.cube_half_size[0] * 2.0)
            offsets_xy = torch.tensor(
                [
                    [-0.5 * spacing, -0.5 * spacing],
                    [0.5 * spacing, -0.5 * spacing],
                    [-0.5 * spacing, 0.5 * spacing],
                    [0.5 * spacing, 0.5 * spacing],
                ],
                device=self.device,
            )  # (4,2)

            # 整堆绿块的中心位置：固定在原点 (0, 0)，对应白色 4x4 作业框的正中心
            stack_center_xy = torch.zeros((1, 2), device=self.device)

            # 构造满足“上层必须有下层支撑”的槽位集合。
            # 先为第 0 层采样若干位置，再逐层向上，在每一层选择一部分下层已占用的位置继续堆叠。
            chosen_slots = []
            remaining = num_green

            # 第 0 层：至少 1 个，至多 4 个，同时预留出每一层至少 1 个的空间
            max_base = min(4, remaining - (num_layers - 1))
            base_count = torch.randint(
                low=1,
                high=max_base + 1,
                size=(1,),
                device=self.device,
            ).item()
            perm0 = torch.randperm(offsets_xy.shape[0], device=self.device)
            base_positions = [offsets_xy[i] for i in perm0[:base_count]]
            for pos in base_positions:
                chosen_slots.append((0, pos))
            remaining -= base_count

            prev_layer_positions = base_positions
            # 后续各层：只能在下层已有的位置上继续堆叠
            for layer_idx in range(1, num_layers):
                if remaining <= 0:
                    break
                max_here = min(len(prev_layer_positions), remaining - (num_layers - 1 - layer_idx))
                if max_here <= 0:
                    break
                layer_count = torch.randint(
                    low=1,
                    high=max_here + 1,
                    size=(1,),
                    device=self.device,
                ).item()
                perm_l = torch.randperm(len(prev_layer_positions), device=self.device)
                cur_positions = [prev_layer_positions[i] for i in perm_l[:layer_count]]
                for pos in cur_positions:
                    chosen_slots.append((layer_idx, pos))
                remaining -= layer_count
                prev_layer_positions = cur_positions

            # 若还有剩余方块（未分配），统一放在第 0 层的剩余网格位置上
            if remaining > 0:
                used_base = set(tuple(p.cpu().tolist()) for p in base_positions)
                extra_base_positions = [p for p in offsets_xy if tuple(p.cpu().tolist()) not in used_base]
                if extra_base_positions:
                    perm_extra = torch.randperm(len(extra_base_positions), device=self.device)
                    for idx in perm_extra[:remaining]:
                        chosen_slots.append((0, extra_base_positions[int(idx)]))

            # 从“最高一层”的槽位中随机选一个作为 cubeB 的位置，
            # 这样红块要么单独成为新的最上层一层，要么与该最高层的其他块处于同一高度。
            top_layer = max(s[0] for s in chosen_slots)
            top_slots = [s for s in chosen_slots if s[0] == top_layer]
            perm_top = torch.randperm(len(top_slots), device=self.device)
            target_slot = top_slots[int(perm_top[0])]

            # 其余槽位（包含同层和更低层）用于额外方块
            green_slots = [target_slot] + [s for s in chosen_slots if s is not target_slot]

            # 把不需要的额外绿块先放到桌子下面“隐藏”
            hide_pose = Pose.create_from_pq(
                p=torch.tensor([[0.0, 0.0, -1.0]], device=self.device),
                q=green_q,
            )
            for cube in self.extra_green_cubes:
                cube.set_pose(hide_pose)

            # 绿色方块高度：底层放在 z = cube_half_size[2]，每层在此基础上叠加 2 * half_size
            def _slot_to_pose(layer_idx: int, offset_xy: torch.Tensor) -> Pose:
                xy = stack_center_xy + offset_xy[None, :]
                z = self.cube_half_size[2] + layer_idx * self.cube_half_size[2] * 2.0
                p = torch.cat([xy, torch.full((1, 1), z, device=self.device)], dim=1)
                return Pose.create_from_pq(p=p, q=green_q)

            # 第一个槽位对应 cubeB（目标所在那一块）
            b_layer, b_offset = green_slots[0]
            self.cubeB.set_pose(_slot_to_pose(b_layer, b_offset))

            # 其余槽位分配给 extra_green_cubes
            for slot, cube in zip(green_slots[1:], self.extra_green_cubes):
                layer_idx, offset = slot
                cube.set_pose(_slot_to_pose(layer_idx, offset))

            # -------------------------------
            # 3) 干扰方块（保持原本逻辑）
            # -------------------------------
            # Randomize distractor cube poses if enabled.
            if self.num_distractor_cubes > 0 and len(self.distractor_cubes) > 0:
                distractor_xy = self._sample_distractor_cube_xy(
                    self.num_distractor_cubes, cubeA_xy, cubeB_xy
                )
                # Use same height as task cubes
                dz = 0.02
                for i, actor in enumerate(self.distractor_cubes):
                    if i >= distractor_xy.shape[0]:
                        break
                    px = distractor_xy[i, 0]
                    py = distractor_xy[i, 1]
                    pose = Pose.create_from_pq(
                        p=torch.tensor([[px, py, dz]], device=self.device),
                        q=randomization.random_quaternions(
                            1,
                            lock_x=True,
                            lock_y=True,
                            lock_z=False,
                        ),
                    )
                    actor.set_pose(pose)

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        # NOTE (stao): GPU sim can be fast but unstable. Angular velocity is rather high despite it not really rotating
        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        success = is_cubeA_on_cubeB * is_cubeA_static * (~is_cubeA_grasped)
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

        # grasp and place reward
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        goal_xyz = torch.hstack(
            [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)

        reward[info["is_cubeA_grasped"]] = (4 + place_reward)[info["is_cubeA_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda
        is_cubeA_grasped = info["is_cubeA_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_cubeA_grasped] = 1.0
        v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_cubeA_on_cubeB"]] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_cubeA_on_cubeB"]]

        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
