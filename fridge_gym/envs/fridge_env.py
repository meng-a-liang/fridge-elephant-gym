"""
fridge_env.py
=================
这个文件定义了“把大象装进冰箱”的强化学习环境（Environment）。

对零基础来说，你只要先记住一句话：
- **环境（Env）**：负责“告诉智能体现在是什么状态（state/observation）”、接收智能体的动作（action），
  然后返回“新状态 + 奖励reward + 是否结束done”。

本项目里，我们保留了pygame可视化窗口，让你既能手动按键玩，也能让智能体自动执行，
从而更直观地理解“强化学习 = 试错 + 奖励反馈”的结构。
"""

import os
import pygame
import sys
import numpy as np

# 兼容导入：优先使用 gymnasium；如果用户只安装了 gym，也能运行（接口仍按Gymnasium风格返回）
try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover
    import gym  # type: ignore
    from gym import spaces  # type: ignore

from fridge_gym.elements.fridge import Fridge
from fridge_gym.elements.elephant import Elephant
from fridge_gym.utils.render_utils import draw_with_shadow


class FridgeGameEnv(gym.Env):
    """
    符合Gymnasium规范的强化学习环境（支持pygame渲染）。

    ### 你将会在这里看到强化学习的“三要素”
    - **状态/观测 observation**：`_get_obs()` 返回给智能体
    - **动作 action**：`step(action)` 接收智能体动作
    - **奖励 reward**：`step` 里根据动作是否有效、是否推进任务给出奖励
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}
    # 窗口尺寸
    DEFAULT_SCREEN_WIDTH = 1200
    DEFAULT_SCREEN_HEIGHT = 700
    FRIDGE_SIZE = (200, 180)
    ELEPHANT_SIZE = (100, 100)
    FRIDGE_INIT_Y_OFFSET = 150

    # -----------------------------
    # 以“米”为单位的可配置参数
    # -----------------------------
    # 为了让“0.8米、0.4米步长”这种需求更直观，我们引入一个简单的比例：
    # 1米 ≈ PIXELS_PER_METER 像素
    PIXELS_PER_METER = 100.0

    # 可改参数：大象初始离冰箱多远（单位：米）
    # 你反馈“太近了”，这里默认调远一些；后续想改只需要改这一行即可。
    # 默认起点不要太夸张，否则训练会变得很难、也更容易出现“上下抖动的局部策略”
    ELEPHANT_INIT_DISTANCE_M = 5

    # 可改参数：每次移动一步的距离（单位：米）
    # 你希望“步伐增大”，这里默认加大；后续想改只需要改这一行即可。
    MOVE_STEP_M = 0.2

    def __init__(self, render_mode="human", *, elephant_init_distance_m: float | None = None, move_step_m: float | None = None):
        super().__init__()
        self.render_mode = render_mode

        # 允许在创建环境时覆盖“起始距离/步长”
        # 重要：训练环境和可视化环境如果参数不一致，会出现“训练能成功但执行时卡住/乱跳”的错觉。
        self.elephant_init_distance_m = float(elephant_init_distance_m) if elephant_init_distance_m is not None else float(self.ELEPHANT_INIT_DISTANCE_M)
        self.move_step_m = float(move_step_m) if move_step_m is not None else float(self.MOVE_STEP_M)

        # Pygame初始化
        pygame.init()
        self.SCREEN_WIDTH = self.DEFAULT_SCREEN_WIDTH
        self.SCREEN_HEIGHT = self.DEFAULT_SCREEN_HEIGHT
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("把大象放进冰箱（键盘控制版）")

        # 把“米制参数”转换成像素步长（统一用于上/下/向前移动）
        # 如果你想改速度，只需要改 MOVE_STEP_M 或 PIXELS_PER_METER
        self.move_step_px = float(self.move_step_m * self.PIXELS_PER_METER)

        # 颜色优化：简洁无多余装饰
        self.colors = {
            "bg": (245, 245, 240),  # 浅米色背景
            "tip_text": (0, 0, 0),  # 常规提示（纯黑）
            "hint_text": (0, 100, 0),  # 引导/结束提示
        }

        # 字体初始化
        self._init_font()

        # 资源加载
        self._load_assets()

        # 初始化元素
        self._init_elements()

        # Gymnasium空间定义
        self._init_gym_spaces()

        # 游戏状态
        # phase: 0=需要开门, 1=需要移动到冰箱, 2=需要放入, 3=需要关门(完成)
        self.game_phase = 0
        self.done = False
        self.elephant_inside = False  # 大象是否已放入冰箱（逻辑状态）
        self.task_complete = False  # 任务是否完成（关冰箱后）
        self._reached_fridge_once = False  # 是否曾经移动到可放入位置（奖励整形用）
        self._opened_once = False  # 是否已经“首次开门”（防止刷开门奖励）
        self._prev_l1_dist_m = None  # 上一步与冰箱的L1距离（米），用于距离整形奖励

        # 阈值：判定“大象已到冰箱门口可放入”
        # 之前用像素阈值，配合较大的步长容易在目标附近震荡而无法满足条件。
        # 这里改成“米”单位，且阈值相对宽一些，让DQN更容易学到完整流程（先会做对，再逐步学更优）。
        self.put_distance_threshold_m = 0.8
        self.put_height_threshold_m = 0.8

    def _init_font(self):
        """初始化字体（去掉加粗，字号适中）"""
        try:
            self.font = pygame.font.SysFont(["SimHei", "Heiti TC"], 30)
            self.font_small = pygame.font.SysFont(["SimHei", "Heiti TC"], 24)
            self.font_big = pygame.font.SysFont(["SimHei", "Heiti TC"], 48)
        except:
            self.font = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 24)
            self.font_big = pygame.font.Font(None, 48)

    def _load_assets(self):
        """加载资源"""
        # 1. 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 2. 找到项目根目录（根据你的实际目录结构调整：比如当前脚本在 envs/ 下，根目录就是上两级）
        project_root = os.path.dirname(os.path.dirname(current_dir))
        # 3. 拼接 assets 路径（根目录下的 assets 文件夹）
        ASSETS_PATH = os.path.join(project_root, "assets")

        # 加载大象图片
        try:
            elephant_path = os.path.join(ASSETS_PATH, "elephant.png")
            self.elephant_img = pygame.image.load(elephant_path).convert_alpha()
            self.elephant_img = pygame.transform.scale(self.elephant_img, self.ELEPHANT_SIZE)
        except (FileNotFoundError, pygame.error):
            self.elephant_img = pygame.Surface(self.ELEPHANT_SIZE, pygame.SRCALPHA)
            self.elephant_img.fill((255, 0, 0, 180))  # 红色矩形

        # 加载冰箱图片
        try:
            fridge_closed_path = os.path.join(ASSETS_PATH, "fridge_closed.png")
            self.fridge_closed_img = pygame.image.load(fridge_closed_path).convert_alpha()
            self.fridge_closed_img = pygame.transform.scale(self.fridge_closed_img, self.FRIDGE_SIZE)

            fridge_open_path = os.path.join(ASSETS_PATH, "fridge_open.png")
            self.fridge_open_img = pygame.image.load(fridge_open_path).convert_alpha()
            self.fridge_open_img = pygame.transform.scale(self.fridge_open_img, self.FRIDGE_SIZE)
        except (FileNotFoundError, pygame.error):
            self.fridge_closed_img = pygame.Surface(self.FRIDGE_SIZE, pygame.SRCALPHA)
            self.fridge_closed_img.fill((0, 0, 255, 180))  # 蓝色（关闭）

            self.fridge_open_img = pygame.Surface(self.FRIDGE_SIZE, pygame.SRCALPHA)
            self.fridge_open_img.fill((0, 255, 0, 180))  # 绿色（打开）

    def _init_elements(self):
        """初始化冰箱和大象（居中不遮挡提示）"""
        self.fridge = Fridge(
            x=self.SCREEN_WIDTH * 0.7,
            y=self.SCREEN_HEIGHT * 0.7,
            width=self.FRIDGE_SIZE[0],
            height=self.FRIDGE_SIZE[1]
        )

        # 需求：大象初始距离冰箱从2.0m改为0.8m
        # 这里用 x 方向距离表达“离冰箱远近”：让大象在冰箱左侧 0.8m 的位置开始
        elephant_init_x = float(self.fridge.x - self.elephant_init_distance_m * self.PIXELS_PER_METER)
        elephant_init_x = max(self.ELEPHANT_SIZE[0] // 2 + 1, elephant_init_x)

        self.elephant = Elephant(
            x=elephant_init_x,
            y=self.SCREEN_HEIGHT * 0.7,
            width=self.ELEPHANT_SIZE[0],
            height=self.ELEPHANT_SIZE[1]
        )

    def _init_gym_spaces(self):
        """定义状态和动作空间（按需求：6维独热动作；状态包含门状态/相对位置/是否在冰箱内）"""
        # obs = [door_open(0/1), dx(m), dy(m), elephant_in_fridge(0/1)]
        # 把 dx/dy 从“像素”换成“米”有两个好处：
        # - 数值尺度更小（避免网络训练时因为输入过大而不稳定）
        # - 参数解释更直观（你可以直接说“离冰箱还差0.5米”）
        max_dx_m = float(self.SCREEN_WIDTH / self.PIXELS_PER_METER)
        max_dy_m = float(self.SCREEN_HEIGHT / self.PIXELS_PER_METER)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -max_dx_m, -max_dy_m, 0.0], dtype=np.float32),
            high=np.array([1.0, max_dx_m, max_dy_m, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # action(one-hot, 6):
        # [1,0,0,0,0,0] open
        # [0,1,0,0,0,0] close
        # [0,0,1,0,0,0] up
        # [0,0,0,1,0,0] forward (toward fridge)
        # [0,0,0,0,1,0] put into fridge
        # [0,0,0,0,0,1] down
        self.action_space = spaces.MultiBinary(6)

    def _get_obs(self):
        """获取状态向量"""
        # dx/dy 用“米”表示，便于学习和解释
        dx = float((self.fridge.x - self.elephant.x) / self.PIXELS_PER_METER)
        dy = float((self.fridge.y - self.elephant.y) / self.PIXELS_PER_METER)
        return np.array(
            [
                1.0 if self.fridge.is_open else 0.0,
                dx,
                dy,
                1.0 if self.elephant_inside else 0.0,
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        """返回额外信息"""
        return {
            "game_phase": self.game_phase,
            "done": self.done,
            "elephant_inside": self.elephant_inside,
            "task_complete": self.task_complete,
            "fridge_open": self.fridge.is_open,
            "elephant_pos": (float(self.elephant.x), float(self.elephant.y)),
            "fridge_pos": (float(self.fridge.x), float(self.fridge.y)),
        }

    def _is_elephant_aligned_for_put(self):
        """判定是否到达“可放入冰箱”的位置（与冰箱中心近似对齐）"""
        dx_m = abs(float((self.fridge.x - self.elephant.x) / self.PIXELS_PER_METER))
        dy_m = abs(float((self.fridge.y - self.elephant.y) / self.PIXELS_PER_METER))
        return dx_m <= self.put_distance_threshold_m and dy_m <= self.put_height_threshold_m

    def _l1_dist_m(self) -> float:
        """当前大象到冰箱的L1距离（米）：|dx|+|dy|，用于奖励整形。"""
        dx_m = abs(float((self.fridge.x - self.elephant.x) / self.PIXELS_PER_METER))
        dy_m = abs(float((self.fridge.y - self.elephant.y) / self.PIXELS_PER_METER))
        return float(dx_m + dy_m)

    def _dx_dy_m(self) -> tuple[float, float]:
        """返回 (abs_dx_m, abs_dy_m) 方便做“进度奖励”."""
        dx_m = abs(float((self.fridge.x - self.elephant.x) / self.PIXELS_PER_METER))
        dy_m = abs(float((self.fridge.y - self.elephant.y) / self.PIXELS_PER_METER))
        return float(dx_m), float(dy_m)

    @staticmethod
    def _action_index_from_input(action):
        """
        接受动作输入：
        - one-hot (shape=(6,)) -> index
        - int (0..5) -> index (兼容旧代码/训练脚本)
        """
        if isinstance(action, (int, np.integer)):
            idx = int(action)
            if 0 <= idx <= 5:
                return idx
            return None

        arr = np.asarray(action, dtype=np.int32).reshape(-1)
        if arr.shape[0] != 6:
            return None
        if arr.sum() != 1:
            return None
        idx = int(np.argmax(arr))
        return idx

    @staticmethod
    def _action_name(idx):
        return [
            "开冰箱门",
            "关冰箱门",
            "向上移动大象",
            "向前移动大象（靠近冰箱）",
            "将大象放入冰箱",
            "向下移动大象",
        ][idx]

    def format_state_text(self):
        """用于日志输出的简洁状态文本"""
        door = "开" if self.fridge.is_open else "关"
        inside = "是" if self.elephant_inside else "否"
        dx_m = abs(float((self.fridge.x - self.elephant.x) / self.PIXELS_PER_METER))
        dy_m = abs(float((self.fridge.y - self.elephant.y) / self.PIXELS_PER_METER))
        return f"冰箱门{door}，大象距冰箱水平{dx_m:.2f}m/垂直{dy_m:.2f}m，大象在冰箱内：{inside}"

    def manual_move_xy(self, dx_px: float, dy_px: float):
        """
        手动模式专用：允许用“↑↓←→”自由移动大象。

        重要说明：
        - 这不是强化学习动作空间的一部分（不改变你定义的6维独热动作）。
        - 仅用于“键盘测试”和“验证环境逻辑”更直观，避免被动作空间限制住。
        """
        new_x = float(self.elephant.x + dx_px)
        new_y = float(self.elephant.y + dy_px)

        # 边界裁剪：不让大象移出窗口
        half_w = self.ELEPHANT_SIZE[0] // 2
        half_h = self.ELEPHANT_SIZE[1] // 2
        new_x = max(half_w + 1, min(self.SCREEN_WIDTH - half_w - 1, new_x))
        new_y = max(half_h + 1, min(self.SCREEN_HEIGHT - half_h - 1, new_y))

        self.elephant.update_pos(new_x, new_y)

    def step(self, action):
        """执行动作"""
        reward = 0.0
        terminated = False
        truncated = False

        idx = self._action_index_from_input(action)
        if idx is None:
            # 非法动作输入：强负奖励
            return self._get_obs(), -5.0, self.done, False, self._get_info()

        if self.done:
            return self._get_obs(), 0.0, True, False, self._get_info()

        # 记录动作前的距离，用于“进度奖励”
        prev_dx_m, prev_dy_m = self._dx_dy_m()

        # 每一步都给一个小的时间惩罚，鼓励尽快完成任务而不是原地晃
        reward -= 0.02

        # -----------------------------
        # 关键约束：大象已放入冰箱后，只允许“关门”
        # -----------------------------
        # 目的：
        # - 防止出现“已经进冰箱但仍然上下跳/向前走”的奇怪行为
        # - 降低学习难度：进入冰箱后的最优动作只有一个 = 关门
        if self.elephant_inside:
            if idx == 1 and self.fridge.is_open:
                # 正常走下面 close 分支即可（这里不提前return，保持逻辑一致）
                pass
            else:
                # 进冰箱后还做别的动作：强惩罚，且不改变位置/门状态
                return self._get_obs(), -2.0, False, False, self._get_info()

        # -----------------------------
        # 阶段驱动奖励：门关时先开门，避免原地抖动
        # -----------------------------
        aligned_for_put = self._is_elephant_aligned_for_put()
        if (not self.fridge.is_open) and (not self.elephant_inside):
            # 门关着时，除“开门”以外的动作都额外扣分（仍允许你移动，但会更倾向先开门）
            if idx != 0:
                reward -= 0.5
            # 如果已经在门口附近还不去开门，扣更重，直接打掉“上下跳”局部策略
            if aligned_for_put and idx != 0:
                reward -= 1.0

        # 0=open,1=close,2=up,3=forward,4=put,5=down
        if idx == 0:  # open
            if not self.fridge.is_open:
                self.fridge.is_open = True
                # 只在“首次开门”时给正奖励，防止学到“原地刷开门”
                if not self._opened_once:
                    reward += 2.0
                    self._opened_once = True
                else:
                    reward -= 0.2
                if self.game_phase == 0:
                    self.game_phase = 1
            else:
                reward -= 1.0

        elif idx == 1:  # close
            if self.fridge.is_open:
                if self.elephant_inside:
                    self.fridge.is_open = False
                    self.task_complete = True
                    self.done = True
                    terminated = True
                    self.game_phase = 3
                    # 成功关门并且大象在冰箱里：给予最大的终局奖励
                    reward += 40.0
                else:
                    # 没放进去就关门：无效/反逻辑
                    self.fridge.is_open = False
                    reward -= 3.0
            else:
                reward -= 1.0

        elif idx == 2:  # up
            # 向上：y减小（屏幕坐标系）
            new_y = self.elephant.y - self.move_step_px
            if new_y - self.ELEPHANT_SIZE[1] // 2 > 0:
                self.elephant.update_pos(self.elephant.x, new_y)
            else:
                # 试图越界但位置不变：给额外惩罚，避免卡边界反复撞墙
                reward -= 0.5

        elif idx == 5:  # down
            # 向下：y增大
            new_y = self.elephant.y + self.move_step_px
            if new_y + self.ELEPHANT_SIZE[1] // 2 < self.SCREEN_HEIGHT:
                self.elephant.update_pos(self.elephant.x, new_y)
            else:
                reward -= 0.5

        elif idx == 3:  # forward (toward fridge)
            # 约定：向前=沿x方向朝冰箱移动
            direction = 1.0 if self.elephant.x < self.fridge.x else -1.0
            # 防止“跨过冰箱中心”导致来回震荡：如果离目标很近，就只走到目标附近，不要一脚跨过去
            dx_px = abs(float(self.fridge.x - self.elephant.x))
            step = min(float(self.move_step_px), dx_px)
            new_x = self.elephant.x + direction * step
            if (new_x - self.ELEPHANT_SIZE[0] // 2) > 0 and (new_x + self.ELEPHANT_SIZE[0] // 2) < self.SCREEN_WIDTH:
                self.elephant.update_pos(new_x, self.elephant.y)
            else:
                reward -= 0.5

        elif idx == 4:  # put
            if self.fridge.is_open and (not self.elephant_inside) and self._is_elephant_aligned_for_put():
                self.elephant_inside = True
                # 放入动作给较大的正奖励，引导学会“进冰箱”这一步
                reward += 20.0
                self.game_phase = 2
            else:
                reward -= 2.0

        # -----------------------------
        # 进度奖励（替换原先的L1距离差奖励，避免“上下抖动”钻空子）
        # -----------------------------
        # 直觉：
        # - 想完成任务，最重要的是让水平距离 |dx| 变小（向冰箱靠近）
        # - 高度对齐 |dy| 次之
        # 所以我们给 |dx| 更高权重，让智能体更愿意选择 “forward(向前)” 而不是一直 up/down。
        if not self.elephant_inside:
            curr_dx_m, curr_dy_m = self._dx_dy_m()
            dx_progress = float(prev_dx_m - curr_dx_m)  # 正=更靠近
            dy_progress = float(prev_dy_m - curr_dy_m)  # 正=更对齐
            reward += 1.2 * dx_progress + 0.3 * dy_progress

            # 额外塑形：当“门已开且水平已经很接近”时，主要目标就是把 dy 对齐。
            # 这样可以强力避免你看到的那种 1.0m↔1.2m 来回抖动：
            # - 往正确方向（dy变小）会更赚
            # - 往错误方向（dy变大）会更亏
            if self.fridge.is_open and (prev_dx_m <= (self.put_distance_threshold_m + 0.3)) and (prev_dy_m > self.put_height_threshold_m):
                reward += 1.0 * dy_progress

        # 如果已经“门开且对齐可放入”，但没有选择 put，则给予额外惩罚，强力引导学会关键动作
        if self.fridge.is_open and (not self.elephant_inside) and aligned_for_put and idx != 4:
            reward -= 1.0

        # （旧版“距离差奖励”已移除，统一用上面的进度奖励）

        # 阶段性奖励：第一次到达“可放入位置”给明显的正奖励
        if (not self._reached_fridge_once) and (not self.elephant_inside) and self._is_elephant_aligned_for_put():
            self._reached_fridge_once = True
            if self.game_phase in (0, 1):
                reward += 10.0
                if self.game_phase == 1:
                    self.game_phase = 2

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        # options 用于“自定义起点”：
        # - options["elephant_pos"] = (x_px, y_px)  # 用像素坐标指定大象起点
        # - options["fridge_open"] = bool           # 指定冰箱门初始开/关（默认关）
        options = options or {}

        self.fridge.is_open = bool(options.get("fridge_open", False))
        self.fridge.update_pos(
            self.SCREEN_WIDTH * 0.7,
            self.SCREEN_HEIGHT * 0.7
        )

        if "elephant_pos" in options and options["elephant_pos"] is not None:
            x0, y0 = options["elephant_pos"]
            # 边界裁剪，避免自定义起点把大象放到窗口外
            half_w = self.ELEPHANT_SIZE[0] // 2
            half_h = self.ELEPHANT_SIZE[1] // 2
            x0 = float(max(half_w + 1, min(self.SCREEN_WIDTH - half_w - 1, float(x0))))
            y0 = float(max(half_h + 1, min(self.SCREEN_HEIGHT - half_h - 1, float(y0))))
            self.elephant.update_pos(x0, y0)
        else:
            # 默认起点：按“离冰箱 elephant_init_distance_m 米”的位置开始
            elephant_init_x = float(self.fridge.x - self.elephant_init_distance_m * self.PIXELS_PER_METER)
            elephant_init_x = max(self.ELEPHANT_SIZE[0] // 2 + 1, elephant_init_x)
            self.elephant.update_pos(
                elephant_init_x,
                self.SCREEN_HEIGHT * 0.7
            )
        self.game_phase = 0
        self.done = False
        self.elephant_inside = False
        self.task_complete = False
        self._reached_fridge_once = False
        self._opened_once = False
        self._prev_l1_dist_m = self._l1_dist_m()
        return self._get_obs(), self._get_info()

    def render(self):
        """渲染界面（核心：无加粗字体+纯文字结束提示）"""
        if self.render_mode != "human":
            return

        self.screen.fill(self.colors["bg"])

        # 绘制冰箱和大象
        fridge_img = self.fridge_open_img if self.fridge.is_open else self.fridge_closed_img
        draw_with_shadow(
            fridge_img,
            (self.fridge.x - self.FRIDGE_SIZE[0] // 2, self.fridge.y - self.FRIDGE_SIZE[1] // 2),
            self.screen
        )
        if (not self.task_complete) and (not self.elephant_inside):
            draw_with_shadow(
                self.elephant_img,
                (self.elephant.x - self.ELEPHANT_SIZE[0] // 2, self.elephant.y - self.ELEPHANT_SIZE[1] // 2),
                self.screen
            )

        # 顶部操作提示（纯黑、常规字体、分行不重叠）
        op_tips = [
            "操作说明：W 上 | S 下 | D 向前(靠近冰箱) | O 开门 | P 放入 | C 关门 | R 重置",
            f"当前阶段：{self.game_phase} | {self.format_state_text()}",
            f"大象位置：({self.elephant.x:.0f}, {self.elephant.y:.0f}) | 冰箱位置：({self.fridge.x:.0f}, {self.fridge.y:.0f})"
        ]
        self.screen.blit(self.font.render(op_tips[0], True, self.colors["tip_text"]), (20, 20))
        self.screen.blit(self.font_small.render(op_tips[1], True, self.colors["tip_text"]), (20, 70))
        self.screen.blit(self.font_small.render(op_tips[2], True, self.colors["tip_text"]), (20, 110))

        # 引导提示（深绿、居中、纯文字）
        if self.elephant_inside and not self.task_complete:
            hint_text = self.font.render("大象已放入冰箱，请按 C 键关闭冰箱门完成任务", True, self.colors["hint_text"])
            hint_x = self.SCREEN_WIDTH // 2 - hint_text.get_width() // 2
            self.screen.blit(hint_text, (hint_x, 160))

        if self.task_complete:
            success_text1 = self.font_big.render("任务完成！大象已成功关进冰箱", True, self.colors["hint_text"])
            success_text2 = self.font.render("按 R 键重新开始游戏", True, self.colors["hint_text"])
            # 文字居中，放在窗口中上部，不遮挡元素
            text1_x = self.SCREEN_WIDTH // 2 - success_text1.get_width() // 2
            text1_y = self.SCREEN_HEIGHT // 2 - 50
            text2_x = self.SCREEN_WIDTH // 2 - success_text2.get_width() // 2
            text2_y = self.SCREEN_HEIGHT // 2 + 10
            self.screen.blit(success_text1, (text1_x, text1_y))
            self.screen.blit(success_text2, (text2_x, text2_y))

        pygame.display.flip()

    def close(self):
        """关闭环境

        注意：
        - 训练用的环境（render_mode != "human"）只需要释放自己的资源，不应该关闭整个pygame系统，
          否则会影响到已经打开的可视化窗口，出现“video system not initialized”错误。
        - 只有真正的人机交互窗口（render_mode == "human"）才调用 pygame.quit()。
        """
        if self.render_mode == "human":
            pygame.quit()