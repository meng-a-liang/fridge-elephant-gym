import os
import pygame
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from fridge_gym.elements.fridge import Fridge
from fridge_gym.elements.elephant import Elephant
from fridge_gym.utils.render_utils import draw_with_shadow


class FridgeGameEnv(gym.Env):
    """符合Gymnasium规范的强化学习环境（简洁文字版）"""
    metadata = {"render_modes": ["human"], "render_fps": 30}
    # 窗口尺寸
    DEFAULT_SCREEN_WIDTH = 1200
    DEFAULT_SCREEN_HEIGHT = 700
    FRIDGE_SIZE = (200, 180)
    ELEPHANT_SIZE = (100, 100)
    FRIDGE_INIT_Y_OFFSET = 150
    ELEPHANT_MOVE_STEP = 5

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode

        # Pygame初始化
        pygame.init()
        self.SCREEN_WIDTH = self.DEFAULT_SCREEN_WIDTH
        self.SCREEN_HEIGHT = self.DEFAULT_SCREEN_HEIGHT
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("把大象放进冰箱（键盘控制版）")

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

        # 阈值：判定“大象已到冰箱门口可放入”
        self.put_distance_threshold = 25.0
        self.put_height_threshold = 30.0

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
        self.elephant = Elephant(
            x=self.SCREEN_WIDTH * 0.2,
            y=self.SCREEN_HEIGHT * 0.7,
            width=self.ELEPHANT_SIZE[0],
            height=self.ELEPHANT_SIZE[1]
        )

    def _init_gym_spaces(self):
        """定义状态和动作空间（按需求：6维独热动作；状态包含门状态/相对位置/是否在冰箱内）"""
        # obs = [door_open(0/1), dx, dy, elephant_in_fridge(0/1)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -float(self.SCREEN_WIDTH), -float(self.SCREEN_HEIGHT), 0.0], dtype=np.float32),
            high=np.array([1.0, float(self.SCREEN_WIDTH), float(self.SCREEN_HEIGHT), 1.0], dtype=np.float32),
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
        dx = float(self.fridge.x - self.elephant.x)
        dy = float(self.fridge.y - self.elephant.y)
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
        dx = abs(float(self.fridge.x - self.elephant.x))
        dy = abs(float(self.fridge.y - self.elephant.y))
        return dx <= self.put_distance_threshold and dy <= self.put_height_threshold

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
        dx = abs(float(self.fridge.x - self.elephant.x))
        dy = abs(float(self.fridge.y - self.elephant.y))
        return f"冰箱门{door}，大象距冰箱水平{dx:.0f}px/垂直{dy:.0f}px，大象在冰箱内：{inside}"

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

        # 轻微步长惩罚，鼓励更短流程
        reward -= 0.01

        # 0=open,1=close,2=up,3=forward,4=put,5=down
        if idx == 0:  # open
            if not self.fridge.is_open:
                self.fridge.is_open = True
                reward += 2.0
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
                    reward += 20.0
                else:
                    # 没放进去就关门：无效/反逻辑
                    self.fridge.is_open = False
                    reward -= 1.0
            else:
                reward -= 1.0

        elif idx == 2:  # up
            new_y = self.elephant.y - self.ELEPHANT_MOVE_STEP
            if new_y - self.ELEPHANT_SIZE[1] // 2 > 0:
                self.elephant.update_pos(self.elephant.x, new_y)
                reward += 0.05

        elif idx == 5:  # down
            new_y = self.elephant.y + self.ELEPHANT_MOVE_STEP
            if new_y + self.ELEPHANT_SIZE[1] // 2 < self.SCREEN_HEIGHT:
                self.elephant.update_pos(self.elephant.x, new_y)
                reward += 0.05

        elif idx == 3:  # forward (toward fridge)
            # 约定：向前=沿x方向朝冰箱移动
            direction = 1.0 if self.elephant.x < self.fridge.x else -1.0
            new_x = self.elephant.x + direction * self.ELEPHANT_MOVE_STEP
            if (new_x - self.ELEPHANT_SIZE[0] // 2) > 0 and (new_x + self.ELEPHANT_SIZE[0] // 2) < self.SCREEN_WIDTH:
                self.elephant.update_pos(new_x, self.elephant.y)
                reward += 0.05

        elif idx == 4:  # put
            if self.fridge.is_open and (not self.elephant_inside) and self._is_elephant_aligned_for_put():
                self.elephant_inside = True
                reward += 8.0
                self.game_phase = 2
            else:
                reward -= 2.0

        # 阶段性奖励：第一次到达“可放入位置”给正奖励
        if (not self._reached_fridge_once) and (not self.elephant_inside) and self._is_elephant_aligned_for_put():
            self._reached_fridge_once = True
            if self.game_phase in (0, 1):
                reward += 2.0
                if self.game_phase == 1:
                    self.game_phase = 2

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        self.fridge.is_open = False
        self.fridge.update_pos(
            self.SCREEN_WIDTH * 0.7,
            self.SCREEN_HEIGHT * 0.7
        )
        self.elephant.update_pos(
            self.SCREEN_WIDTH * 0.2,
            self.SCREEN_HEIGHT * 0.7
        )
        self.game_phase = 0
        self.done = False
        self.elephant_inside = False
        self.task_complete = False
        self._reached_fridge_once = False
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
        """关闭环境"""
        pygame.quit()