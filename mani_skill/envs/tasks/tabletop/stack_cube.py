from typing import Any, List, Union

import numpy as np
import sapien
import sapien.render
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
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
    Pick up the movable cube (cubeA) and stack it on top of the target cube (cubeB), then release without the stack collapsing.

    **Randomizations:**
    - cubeA / cubeB / stack blocks / scattered extras use distinct random colors each episode
    - optional scattered cubes (count depends on stack size) so total task cubes (cubeA + stack + scattered) is at most 10
    - both cubes have their z-axis rotation randomized
    - cube positions on the table are randomized without initial interpenetration

    **Success Conditions:**
    - cubeA is on top of cubeB (to within half of the cube size)
    - cubeA is static
    - cubeA is not being grasped by the robot
    """

    # cubeA(1) + stack + scattered extras <= _MAX_TASK_CUBES
    _MAX_TASK_CUBES = 11
    # 当堆叠只有 1 块时，最多再摆 8 个散落块
    _MAX_SCATTERED_CUBES = 8

    _CUBE_RGB_PALETTE: List[List[float]] = [
        [0.90, 0.14, 0.12],
        [0.12, 0.70, 0.22],
        [0.15, 0.35, 0.92],
        [0.95, 0.80, 0.10],
        [0.75, 0.12, 0.75],
        [0.10, 0.82, 0.82],
        [0.95, 0.45, 0.08],
        [0.45, 0.22, 0.90],
        [0.55, 0.90, 0.20],
        [0.92, 0.35, 0.55],
        [0.25, 0.55, 0.35],
        [0.55, 0.38, 0.22],
    ]

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        num_distractor_cubes: int = 0,
        close_camera: bool = False,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.num_distractor_cubes = num_distractor_cubes
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
        # 将范围设置为可容纳 4x4 个方块（含 1/5 边长缝隙）的正方形区域，以堆叠方块中心为正中。
        side = float(self.cube_half_size[0] * 8.0)  # 4 个方块直径 = 8 * half_size
        side *= 1.15  # 4x4 且有缝隙时，需要比纯“贴合直径”更大的包围框
        workspace_half_x = side / 2.0
        workspace_half_y = side / 2.0
        line_thickness = 0.004
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
        # 任务方块：具体颜色在 _initialize_episode 中按回合随机指定
        _placeholder = [0.55, 0.55, 0.55, 1.0]
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=_placeholder,
            name="cubeA",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=_placeholder,
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        )

        # 额外的堆叠方块，用来组成 1~3 层、总数 1~9 个的“堆叠场景”
        # 这里预先创建最多 8 个额外方块，实际每个 episode 中会根据采样需求启用其中一部分，其余放到桌子下方隐藏。
        self.max_green_cubes = 9  # 包括 cubeB 在内的总数上限
        self.extra_green_cubes = []
        for i in range(self.max_green_cubes - 1):
            cube = actors.build_cube(
                self.scene,
                half_size=0.02,
                color=_placeholder,
                name=f"cubeB_extra_{i}",
                initial_pose=sapien.Pose(p=[0.0, 0.0, -1.0]),
            )
            self.extra_green_cubes.append(cube)

        # 散落干扰块（与堆叠无关），每回合随机数量与颜色，预先创建满额
        self.extra_scattered_cubes = []
        for i in range(self._MAX_SCATTERED_CUBES):
            cube = actors.build_cube(
                self.scene,
                half_size=0.02,
                color=_placeholder,
                name=f"scattered_cube_{i}",
                initial_pose=sapien.Pose(p=[0.0, 0.0, -1.0]),
            )
            self.extra_scattered_cubes.append(cube)

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

    @staticmethod
    def _set_actor_base_color(actor, rgba):
        """rgba: length-4 列表，每回合更新方块外观。"""
        for obj in actor._objs:
            rb = obj.find_component_by_type(sapien.render.RenderBodyComponent)
            if rb is None:
                continue
            for render_shape in rb.render_shapes:
                for part in render_shape.parts:
                    part.material.set_base_color(
                        [float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])]
                    )

    def _sample_distinct_rgba_colors(self, n: int) -> List[List[float]]:
        palette = np.asarray(self._CUBE_RGB_PALETTE, dtype=np.float64)
        assert n <= palette.shape[0], "palette too small for requested cube count"
        idx = self._episode_rng.choice(palette.shape[0], size=n, replace=False)
        out = []
        for i in idx:
            rgb = palette[i]
            out.append([float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0])
        return out

    def _apply_episode_task_colors(self, num_green: int, num_extra_scattered: int):
        n = 1 + num_green + num_extra_scattered
        colors = self._sample_distinct_rgba_colors(n)
        k = 0
        self._set_actor_base_color(self.cubeA, colors[k])
        k += 1
        self._set_actor_base_color(self.cubeB, colors[k])
        k += 1
        for i in range(num_green - 1):
            self._set_actor_base_color(self.extra_green_cubes[i], colors[k])
            k += 1
        for i in range(num_extra_scattered):
            self._set_actor_base_color(self.extra_scattered_cubes[i], colors[k])
            k += 1

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            options = options or {}

            # 可选地由外部指定“额外已摆好方块数量”（不含 cubeA / cubeB），用于构造 0~9 的均匀分布数据。
            forced_preplaced = options.get("preplaced_count", None)
            if forced_preplaced is not None:
                extra_total = int(forced_preplaced)
                extra_total = max(0, min(9, extra_total))
                # 对于“已摆好数量”强约束，优先把数量放到堆叠塔中（更直观、也更符合你的数据控制目标）；
                # 只有超过堆叠上限时才分配到散落块。
                stack_extra = min(extra_total, self.max_green_cubes - 1)
                num_green = 1 + stack_extra
                num_extra_scattered = extra_total - stack_extra
            else:
                # 默认随机逻辑：采样堆叠规模与散落块数量，保证 cubeA + 堆叠 + 散落 <= _MAX_TASK_CUBES
                num_green = torch.randint(
                    low=1,
                    high=self.max_green_cubes + 1,
                    size=(1,),
                    device=self.device,
                ).item()
                max_extra = self._MAX_TASK_CUBES - 1 - num_green
                max_extra = max(0, min(max_extra, self._MAX_SCATTERED_CUBES))
                re = self._episode_rng
                if max_extra <= 0:
                    num_extra_scattered = 0
                elif max_extra == 1:
                    num_extra_scattered = int(re.randint(0, 2))
                else:
                    low = 2
                    num_extra_scattered = int(re.randint(low, max_extra + 1))

            max_layers = min(3, num_green)
            # 每层最多 4 个，为保证能容纳 num_green，层数下限为 ceil(num_green / 4)
            min_layers = max(1, int(np.ceil(num_green / 4.0)))
            num_layers = torch.randint(
                low=min_layers,
                high=max_layers + 1,
                size=(1,),
                device=self.device,
            ).item()

            # -------------------------------
            # 1) 待抓取方块 cubeA 的初始位置、姿态（桌面上，允许轻微 yaw 偏转）
            # -------------------------------
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 2] = self.cube_half_size[2]
            # 将 cubeA 放置在 4x4 线框“外侧但不太远”的环形区域内，
            # 避免一开始就挤在堆叠塔附近，同时又不会离任务区太远。
            # 线框半边长与 _load_scene 中保持一致。
            side = float(self.cube_half_size[0] * 8.0) * 1.15
            frame_half = side / 2.0
            max_dist = float(self.cube_half_size[0] * 10.0)  # 上界，约 5 个方块直径
            # 与散落块一致：限制待操作方块不要出现在“前方（远离机械臂）”区域。
            front_x_threshold = 0.0
            for _ in range(64):
                candidate_xy = torch.rand((b, 2), device=self.device) * 0.8 - 0.4  # [-0.4,0.4]
                # 只检查第 0 个 env（当前假设单 env）
                c0 = candidate_xy[0]
                # r 必须落在 (frame_half, max_dist) 之间，既在线框外，又不太远
                r = torch.linalg.norm(c0)
                not_front = c0[0] <= front_x_threshold
                if (torch.any(torch.abs(c0) > frame_half)) and (r < max_dist) and not_front:
                    base_xy = candidate_xy
                    break
            else:
                # 兜底也保持不在前方，避免采样失败时放宽约束。
                fallback = torch.rand((b, 2), device=self.device) * 0.8 - 0.4
                fallback[:, 0] = torch.clamp(fallback[:, 0], max=front_x_threshold - 1e-3)
                base_xy = fallback
            xyz[:, :2] = base_xy
            # 统一方块朝向：全部与世界坐标轴对齐（无随机 yaw）。
            shared_qs = torch.zeros((b, 4), device=self.device)
            shared_qs[:, 0] = 1.0
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=shared_qs))

            # -------------------------------
            # 2) 堆叠塔（cubeB + extra_green_cubes）：总数 1~9 个，1~3 层，姿态对齐且规则栈叠
            #    - 堆叠块共享同一个 yaw，整齐对齐
            #    - 采用 4x4 的网格，最多 3 层；第一层最多 6 个方块
            #    - 物理约束：如果第 n+1 层某位置有方块，则第 n 层同位置必须也有方块（不允许“悬空”）
            #    - 每层都采样为连通块，避免同层出现彼此分散的孤岛。
            # -------------------------------
            # 为整堆方块使用单位四元数（w=1），使所有块与桌面坐标轴完全对齐。
            green_q = torch.zeros((1, 4), device=self.device)
            green_q[:, 0] = 1.0

            # 每一层使用 4x4 网格，方块之间留约 1/5 边长缝隙。
            cube_side = float(self.cube_half_size[0] * 2.0)
            spacing = cube_side * 1.2
            grid_size = 4
            grid_coords = np.arange(grid_size, dtype=np.float32) - (grid_size - 1) / 2.0
            offsets_xy = []
            for r in range(grid_size):
                for c in range(grid_size):
                    offsets_xy.append([grid_coords[r] * spacing, grid_coords[c] * spacing])
            offsets_xy = torch.tensor(offsets_xy, device=self.device, dtype=torch.float32)  # (16,2)
            layer_capacity = offsets_xy.shape[0]
            first_layer_max = 6

            # 整堆绿块的中心位置：固定在原点 (0, 0)，对应白色 4x4 作业框的正中心
            stack_center_xy = torch.zeros((1, 2), device=self.device)

            def _cell_rc(cell_id: int):
                return (cell_id // grid_size, cell_id % grid_size)

            def _cell_id(r: int, c: int):
                return r * grid_size + c

            def _neighbors4(cell_id: int):
                r, c = _cell_rc(cell_id)
                out = []
                if r > 0:
                    out.append(_cell_id(r - 1, c))
                if r < grid_size - 1:
                    out.append(_cell_id(r + 1, c))
                if c > 0:
                    out.append(_cell_id(r, c - 1))
                if c < grid_size - 1:
                    out.append(_cell_id(r, c + 1))
                return out

            def _sample_connected_cells(allowed_cells: set[int], k: int):
                """在 allowed_cells 中采样 k 个 4-连通格子。"""
                allowed_list = sorted(list(allowed_cells))
                if k <= 0 or len(allowed_list) == 0:
                    return set()
                if k == 1:
                    idx = int(torch.randint(low=0, high=len(allowed_list), size=(1,), device=self.device).item())
                    return {allowed_list[idx]}
                # 尝试多次采样，尽量保证连通
                for _ in range(80):
                    seed_idx = int(torch.randint(low=0, high=len(allowed_list), size=(1,), device=self.device).item())
                    chosen = {allowed_list[seed_idx]}
                    while len(chosen) < k:
                        frontier = []
                        for c in chosen:
                            for nb in _neighbors4(c):
                                if nb in allowed_cells and nb not in chosen:
                                    frontier.append(nb)
                        if len(frontier) == 0:
                            break
                        pick = int(torch.randint(low=0, high=len(frontier), size=(1,), device=self.device).item())
                        chosen.add(frontier[pick])
                    if len(chosen) == k:
                        return chosen
                # 兜底：若连通采样失败，返回任意 k 个（极少发生）
                perm = torch.randperm(len(allowed_list), device=self.device)
                return set(allowed_list[int(i)] for i in perm[:k].tolist())

            # 重新采样层数，确保与“第一层最多 6 个”约束一致
            max_layers = min(3, num_green)
            min_layers = max(1, int(np.ceil(num_green / float(first_layer_max))))
            num_layers = int(
                torch.randint(low=min_layers, high=max_layers + 1, size=(1,), device=self.device).item()
            )

            # 先采样每层数量（每层非空、上层不超过下层；第 1 层最多 6）
            remaining = int(num_green)
            layer_counts = []
            for layer_idx in range(num_layers):
                layers_left = num_layers - layer_idx
                min_here = max(1, int(np.ceil(remaining / float(layers_left))))
                if layer_idx == 0:
                    max_here = min(first_layer_max, layer_capacity, remaining - (layers_left - 1))
                else:
                    max_here = min(layer_counts[-1], remaining - (layers_left - 1))
                if min_here > max_here:
                    min_here = max_here
                count = int(
                    torch.randint(low=min_here, high=max_here + 1, size=(1,), device=self.device).item()
                )
                layer_counts.append(count)
                remaining -= count

            # 根据每层数量采样具体格子：每层连通，上层仅能在下层同格位置中选（保证支撑）
            layer_cells = []
            for layer_idx, count in enumerate(layer_counts):
                if layer_idx == 0:
                    allowed = set(range(layer_capacity))
                else:
                    allowed = set(layer_cells[layer_idx - 1])
                cells = _sample_connected_cells(allowed, count)
                layer_cells.append(cells)

            chosen_slots = []
            for layer_idx, cells in enumerate(layer_cells):
                for cell in sorted(list(cells)):
                    chosen_slots.append((layer_idx, int(cell)))

            # 选择 cubeB 的目标槽位：
            # 1) 放置后目标层只能是 n 或 n+1（n 为当前最高层数）
            #    - n 层：选择 top-1 层中“上方为空”的槽位（补当前最高层空位）
            #    - n+1 层：选择 top 层槽位（单独新起一层）
            # 2) 优先“补最高层空位且周围三格都已有方块”的场景，增加目标周边拥挤程度。
            top_layer = max(s[0] for s in chosen_slots)

            def _xy_key(cell_id: int):
                r, c = _cell_rc(cell_id)
                return (int(r), int(c))

            occupied = set((layer, _xy_key(cell)) for layer, cell in chosen_slots)
            fill_top_candidates = []  # cubeB 在 top-1，放置后到 top 层（即 n 层）
            for layer, cell in chosen_slots:
                if layer != top_layer - 1:
                    continue
                above_key = (layer + 1, _xy_key(cell))
                if above_key not in occupied:
                    fill_top_candidates.append((layer, cell))

            top_slots = [s for s in chosen_slots if s[0] == top_layer]  # cubeB 在 top，放置后到 n+1 层

            crowded_fill_candidates = []
            if len(fill_top_candidates) > 0:
                # 在 4x4 网格中优先选择“L 型三邻居”场景：
                # 目标格周围满足两条正交边 + 一个角落邻居已占用。
                top_xy_keys = set(_xy_key(cell) for layer, cell in chosen_slots if layer == top_layer)
                for slot in fill_top_candidates:
                    _, cell = slot
                    r, c = _cell_rc(cell)
                    l_patterns = [
                        [(r - 1, c), (r, c - 1), (r - 1, c - 1)],
                        [(r - 1, c), (r, c + 1), (r - 1, c + 1)],
                        [(r + 1, c), (r, c - 1), (r + 1, c - 1)],
                        [(r + 1, c), (r, c + 1), (r + 1, c + 1)],
                    ]
                    ok = False
                    for pattern in l_patterns:
                        valid = True
                        for rr, cc in pattern:
                            if rr < 0 or rr >= grid_size or cc < 0 or cc >= grid_size:
                                valid = False
                                break
                            if (rr, cc) not in top_xy_keys:
                                valid = False
                                break
                        if valid:
                            ok = True
                            break
                    if ok:
                        crowded_fill_candidates.append(slot)

            if len(crowded_fill_candidates) > 0:
                perm = torch.randperm(len(crowded_fill_candidates), device=self.device)
                target_slot = crowded_fill_candidates[int(perm[0])]
            elif len(fill_top_candidates) > 0:
                perm = torch.randperm(len(fill_top_candidates), device=self.device)
                target_slot = fill_top_candidates[int(perm[0])]
            elif len(top_slots) > 0:
                perm = torch.randperm(len(top_slots), device=self.device)
                target_slot = top_slots[int(perm[0])]
            else:
                # 兜底：极端情况下至少选一个已有槽位
                target_slot = chosen_slots[0]

            # 其余槽位（包含同层和更低层）用于额外方块
            target_key = (target_slot[0], _xy_key(target_slot[1]))
            green_slots = [target_slot] + [
                s for s in chosen_slots
                if (s[0], _xy_key(s[1])) != target_key
            ]

            # 把不需要的额外绿块先放到桌子下面“隐藏”
            hide_pose = Pose.create_from_pq(
                p=torch.tensor([[0.0, 0.0, -1.0]], device=self.device),
                q=green_q,
            )
            for cube in self.extra_green_cubes:
                cube.set_pose(hide_pose)

            # 绿色方块高度：底层放在 z = cube_half_size[2]，每层在此基础上叠加 2 * half_size
            def _slot_to_pose(layer_idx: int, cell_id: int) -> Pose:
                offset_xy = offsets_xy[cell_id]
                xy = stack_center_xy + offset_xy[None, :]
                z = self.cube_half_size[2] + layer_idx * self.cube_half_size[2] * 2.0
                p = torch.cat([xy, torch.full((1, 1), z, device=self.device)], dim=1)
                return Pose.create_from_pq(p=p, q=green_q)

            # 第一个槽位对应 cubeB（目标所在那一块）
            b_layer, b_cell = green_slots[0]
            self.cubeB.set_pose(_slot_to_pose(b_layer, b_cell))

            # 其余槽位分配给 extra_green_cubes
            for slot, cube in zip(green_slots[1:], self.extra_green_cubes):
                layer_idx, cell_id = slot
                cube.set_pose(_slot_to_pose(layer_idx, cell_id))

            # -------------------------------
            # 2b) 散落额外方块（随机数量；线框外，与堆叠塔和 cubeA 拉开距离）
            # -------------------------------
            min_dist_from_stack = float(self.cube_half_size[0] * 6.0)
            min_dist_from_cubeA = float(self.cube_half_size[0] * 4.0)
            cubeA_xy_np = base_xy[0].cpu().numpy()
            hide_sc = Pose.create_from_pq(
                p=torch.tensor([[0.0, 0.0, -1.0]], device=self.device),
                q=green_q,
            )
            # “前方”定义为远离机械臂的一侧（x 更大的一侧），这里限制散落块只出现在 x <= 0 的半平面，
            # 同时仍保持在线框外，形成“后/左/右”分布。
            front_x_threshold = 0.0
            for i, cube in enumerate(self.extra_scattered_cubes):
                if i < num_extra_scattered:
                    valid_xy = None
                    for _ in range(64):
                        xy = (torch.rand(2, device=self.device) * 0.8 - 0.4).cpu().numpy()
                        dist_stack = np.linalg.norm(xy)
                        dist_cubeA = np.linalg.norm(xy - cubeA_xy_np)
                        outside_frame = abs(xy[0]) > frame_half or abs(xy[1]) > frame_half
                        not_front = xy[0] <= front_x_threshold
                        if (
                            outside_frame
                            and not_front
                            and dist_stack > min_dist_from_stack
                            and dist_cubeA > min_dist_from_cubeA
                        ):
                            valid_xy = xy
                            break
                    if valid_xy is None:
                        # 兜底：若采样失败，放在线框后方近处，避免落到前方不可达区域。
                        valid_xy = np.array([-frame_half - 0.03, 0.0], dtype=np.float32)
                    p = np.array(
                        [valid_xy[0], valid_xy[1], float(self.cube_half_size[2])],
                        dtype=np.float32,
                    )
                    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    pose = Pose.create_from_pq(
                        p=torch.tensor([p], device=self.device),
                        q=torch.tensor([q], device=self.device),
                    )
                    cube.set_pose(pose)
                else:
                    cube.set_pose(hide_sc)

            self._apply_episode_task_colors(num_green, num_extra_scattered)

            # -------------------------------
            # 3) 干扰方块（保持原本逻辑）
            # -------------------------------
            # Randomize distractor cube poses if enabled.
            if self.num_distractor_cubes > 0 and len(self.distractor_cubes) > 0:
                cubeA_xy = base_xy
                cubeB_xy = self.cubeB.pose.p[..., :2]
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
                        q=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device),
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
