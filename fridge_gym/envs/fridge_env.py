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
    DEFAULT_SCREEN_WIDTH = 1000
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
        self.game_phase = 1
        self.done = False
        self.elephant_inside = False  # 大象是否已进入开着的冰箱
        self.task_complete = False  # 任务是否完成（关冰箱后）

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
        """定义状态和动作空间"""
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, self.SCREEN_HEIGHT],
                          dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(6)  # 0=上 1=下 2=左 3=右 4=开冰箱 5=关冰箱

    def _get_obs(self):
        """获取状态向量"""
        return np.array([
            1.0 if self.fridge.is_open else 0.0,
            float(self.fridge.x),
            float(self.fridge.y),
            float(self.elephant.x),
            float(self.elephant.y)
        ], dtype=np.float32)

    def _get_info(self):
        """返回额外信息"""
        return {
            "game_phase": self.game_phase,
            "done": self.done,
            "elephant_inside": self.elephant_inside,
            "task_complete": self.task_complete
        }

    def _check_elephant_in_fridge(self):
        """检测大象是否完全在开着的冰箱内"""
        if not self.fridge.is_open:
            return False

        fridge_rect = pygame.Rect(
            self.fridge.x - self.FRIDGE_SIZE[0] // 2,
            self.fridge.y - self.FRIDGE_SIZE[1] // 2,
            self.FRIDGE_SIZE[0],
            self.FRIDGE_SIZE[1]
        )
        elephant_rect = pygame.Rect(
            self.elephant.x - self.ELEPHANT_SIZE[0] // 2,
            self.elephant.y - self.ELEPHANT_SIZE[1] // 2,
            self.ELEPHANT_SIZE[0],
            self.ELEPHANT_SIZE[1]
        )
        return fridge_rect.contains(elephant_rect)

    def step(self, action):
        """执行动作"""
        reward = 0.0
        if not self.done:
            if action == 0:  # 上移
                new_y = self.elephant.y - self.ELEPHANT_MOVE_STEP
                if new_y - self.ELEPHANT_SIZE[1] // 2 > 0:
                    self.elephant.update_pos(self.elephant.x, new_y)
                    reward = 0.1
            elif action == 1:  # 下移
                new_y = self.elephant.y + self.ELEPHANT_MOVE_STEP
                if new_y + self.ELEPHANT_SIZE[1] // 2 < self.SCREEN_HEIGHT:
                    self.elephant.update_pos(self.elephant.x, new_y)
                    reward = 0.1
            elif action == 2:  # 左移
                new_x = self.elephant.x - self.ELEPHANT_MOVE_STEP
                if new_x - self.ELEPHANT_SIZE[0] // 2 > 0:
                    self.elephant.update_pos(new_x, self.elephant.y)
                    reward = 0.1
            elif action == 3:  # 右移
                new_x = self.elephant.x + self.ELEPHANT_MOVE_STEP
                if new_x + self.ELEPHANT_SIZE[0] // 2 < self.SCREEN_WIDTH:
                    self.elephant.update_pos(new_x, self.elephant.y)
                    reward = 0.1
            elif action == 4:  # 开冰箱
                if not self.fridge.is_open:
                    self.fridge.is_open = True
                    reward = 1.0
                else:
                    reward = -0.1
            elif action == 5:  # 关冰箱
                if self.fridge.is_open:
                    self.fridge.is_open = False
                    if self.elephant_inside:
                        self.task_complete = True
                        self.done = True
                        reward = 20.0
                    else:
                        reward = 1.0
                else:
                    reward = -0.1

            self.elephant_inside = self._check_elephant_in_fridge()

        return self._get_obs(), reward, self.done, False, self._get_info()

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
        self.game_phase = 1
        self.done = False
        self.elephant_inside = False
        self.task_complete = False
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
        if not self.task_complete:
            draw_with_shadow(
                self.elephant_img,
                (self.elephant.x - self.ELEPHANT_SIZE[0] // 2, self.elephant.y - self.ELEPHANT_SIZE[1] // 2),
                self.screen
            )

        # 顶部操作提示（纯黑、常规字体、分行不重叠）
        op_tips = [
            "操作说明：↑↓←→ 移动大象 | O 开冰箱 | C 关冰箱 | R 重新开始",
            f"大象位置：({self.elephant.x:.0f}, {self.elephant.y:.0f}) | 冰箱状态：{'开' if self.fridge.is_open else '关'}",
            f"冰箱位置：({self.fridge.x:.0f}, {self.fridge.y:.0f}) | 大象是否在冰箱内：{'是' if self.elephant_inside else '否'}"
        ]
        self.screen.blit(self.font.render(op_tips[0], True, self.colors["tip_text"]), (20, 20))
        self.screen.blit(self.font_small.render(op_tips[1], True, self.colors["tip_text"]), (20, 70))
        self.screen.blit(self.font_small.render(op_tips[2], True, self.colors["tip_text"]), (20, 110))

        # 引导提示（深绿、居中、纯文字）
        if self.elephant_inside and not self.task_complete:
            hint_text = self.font.render("大象已进入冰箱，请按 C 键关闭冰箱门完成任务", True, self.colors["hint_text"])
            hint_x = self.SCREEN_WIDTH // 2 - hint_text.get_width() // 2
            self.screen.blit(hint_text, (hint_x, 160))

        # 结束提示（核心修改：去掉绿色背景，仅保留深绿大号文字）
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