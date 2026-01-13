import os

import pygame
import sys
import gym
from gym import spaces
import numpy as np
from fridge_gym.elements.fridge import Fridge
from fridge_gym.elements.elephant import Elephant
from fridge_gym.utils.render_utils import scale_bg, draw_with_shadow


class FridgeGameEnv(gym.Env):
    """
    符合OpenAI Gym规范的强化学习环境
    状态向量：[冰箱开关状态, 冰箱x坐标, 冰箱y坐标, 大象x坐标, 大象y坐标]
    动作空间：0-无操作, 1-开冰箱门, 2-放大象进冰箱, 3-关冰箱门
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}
    DEFAULT_SCREEN_WIDTH = 600
    DEFAULT_SCREEN_HEIGHT = 400
    FRIDGE_SIZE = (200, 180)
    ELEPHANT_SIZE = (100, 100)
    FRIDGE_INIT_Y_OFFSET = 100
    ELEPHANT_INIT_X = 150

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode

        # 初始化Pygame
        pygame.init()
        self.SCREEN_WIDTH = self.DEFAULT_SCREEN_WIDTH
        self.SCREEN_HEIGHT = self.DEFAULT_SCREEN_HEIGHT
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("把大象放进冰箱")

        # 颜色定义（新增柔和的按钮配色）
        self.colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "yellow": (255, 255, 0),
            "brown": (139, 69, 19),
            "gray": (200, 200, 200),
            "btn_bg": (245, 245, 240, 160),  # 米白色半透明（适配卡通背景）
            "btn_hover": (250, 250, 240, 190),  # 鼠标悬浮时更亮的米白
            "btn_text": (80, 60, 40)  # 深棕色文字（醒目不刺眼）
        }

        # 初始化字体（调用下面补全的方法）
        self._init_font()

        # 加载资源（调用下面补全的方法）
        self._load_assets()

        # 初始化游戏元素
        self.fridge = Fridge(
            x=self.SCREEN_WIDTH // 2 + 80,  # 冰箱移到厨房右侧空位（匹配第2张的布局）
            y=self.SCREEN_HEIGHT - self.FRIDGE_INIT_Y_OFFSET - 20,  # 稍微下移，贴合厨房地面
            width=self.FRIDGE_SIZE[0],
            height=self.FRIDGE_SIZE[1]
        )
        self.elephant = Elephant(
            x=self.SCREEN_WIDTH // 2 - 50,  # 大象放在冰箱左侧（和第2张一致）
            y=self.SCREEN_HEIGHT - self.FRIDGE_INIT_Y_OFFSET,
            width=self.ELEPHANT_SIZE[0],
            height=self.ELEPHANT_SIZE[1]
        )
        # 【修改1：调整按钮位置到屏幕下方，避开背景核心元素】
        self.start_button_rect = pygame.Rect(
            self.SCREEN_WIDTH // 2 - 120,  # 加宽按钮，更协调
            self.SCREEN_HEIGHT - 80,  # 距离底部80像素，不遮挡背景
            240, 70  # 按钮尺寸更舒展
        )

        # 定义Gym空间（核心）
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([1, self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, self.SCREEN_HEIGHT]),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # 0-3共4个动作

        # 游戏状态
        self.game_phase = 0  # 0-开始界面,1-开冰箱,2-放大象,3-关冰箱,4-完成
        self.done = False

    # 补全缺失的 _init_font 方法
    def _init_font(self):
        """初始化字体（兼容无自定义字体）"""
        FONT_PATH = r"C:\Users\20427\fridge-elephant-gym\assets\msyhl.ttc"
        try:
            self.font = pygame.font.Font(FONT_PATH, 24)
            self.font_big = pygame.font.Font(FONT_PATH, 40)  # 【修改2：字体稍大，更醒目】
        except FileNotFoundError:
            print("警告：找不到msyhl.ttc，使用系统默认黑体")
            self.font = pygame.font.SysFont("SimHei", 24)
            self.font_big = pygame.font.SysFont("SimHei", 40)

    # 补全缺失的 _load_assets 方法
    def _load_assets(self):
        """加载图片资源（兼容无图片场景）"""
        # 背景图
        ASSETS_PATH = r"C:\Users\20427\fridge-elephant-gym\assets"
        try:
            start_bg_path = os.path.join(ASSETS_PATH, "start.png")
            self.start_bg = pygame.image.load(start_bg_path).convert()
        except FileNotFoundError:
            print("警告：找不到start.png，使用灰色背景")
            self.start_bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.start_bg.fill(self.colors["gray"])

        try:
            game_bg_path = os.path.join(ASSETS_PATH, "background.png")
            self.game_bg = pygame.image.load(game_bg_path).convert()
        except FileNotFoundError:
            print("警告：找不到background.png，使用白色背景")
            self.game_bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.game_bg.fill(self.colors["white"])

        # 元素图
        try:
            elephant_path = os.path.join(ASSETS_PATH, "elephant.png")
            self.elephant_img = pygame.image.load(elephant_path).convert_alpha()
            self.elephant_img = pygame.transform.scale(self.elephant_img, self.ELEPHANT_SIZE)
        except FileNotFoundError:
            print("警告：找不到elephant.png，使用红色矩形代替")
            self.elephant_img = pygame.Surface(self.ELEPHANT_SIZE, pygame.SRCALPHA)
            self.elephant_img.fill((255, 0, 0, 150))

        try:
            fridge_closed_path = os.path.join(ASSETS_PATH, "fridge_closed.png")
            self.fridge_closed_img = pygame.image.load(fridge_closed_path).convert_alpha()
            self.fridge_closed_img = pygame.transform.scale(self.fridge_closed_img, self.FRIDGE_SIZE)

            fridge_open_path = os.path.join(ASSETS_PATH, "fridge_open.png")
            self.fridge_open_img = pygame.image.load(fridge_open_path).convert_alpha()
            self.fridge_open_img = pygame.transform.scale(self.fridge_open_img, self.FRIDGE_SIZE)
        except FileNotFoundError:
            print("警告：找不到冰箱图片，使用蓝/绿色矩形代替")
            self.fridge_closed_img = pygame.Surface(self.FRIDGE_SIZE, pygame.SRCALPHA)
            self.fridge_closed_img.fill((0, 0, 255, 150))
            self.fridge_open_img = pygame.Surface(self.FRIDGE_SIZE, pygame.SRCALPHA)
            self.fridge_open_img.fill((0, 255, 0, 150))

    # 补全 _get_obs 方法
    def _get_obs(self):
        """获取状态向量：[冰箱开关, 冰箱x, 冰箱y, 大象x, 大象y]"""
        return np.array([
            1.0 if self.fridge.is_open else 0.0,
            float(self.fridge.x),
            float(self.fridge.y),
            float(self.elephant.x),
            float(self.elephant.y)
        ], dtype=np.float32)

    # 补全 _get_info 方法
    def _get_info(self):
        """返回额外信息（拓展用）"""
        return {"game_phase": self.game_phase, "done": self.done}

    # 补全 step 方法（Gym核心）
    def step(self, action):
        """执行动作，返回(obs, reward, done, info)"""
        reward = 0.0

        if not self.done:
            # 正确动作奖励
            if self.game_phase == 1 and action == 1:
                self.fridge.is_open = True
                self.game_phase = 2
                reward = 1.0
            elif self.game_phase == 2 and action == 2:
                self.game_phase = 3
                reward = 2.0
            elif self.game_phase == 3 and action == 3:
                self.fridge.is_open = False
                self.game_phase = 4
                self.done = True
                reward = 10.0
            # 错误动作惩罚
            elif action != 0:
                reward = -0.1

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, self.done, info

    # 补全 reset 方法（Gym核心）
    def reset(self, seed=None, options=None):
        """重置环境，返回初始状态"""
        super().reset(seed=seed)
        np.random.seed(seed)

        # 重置元素状态
        self.fridge.is_open = False
        self.fridge.update_pos(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - self.FRIDGE_INIT_Y_OFFSET)
        self.elephant.update_pos(self.ELEPHANT_INIT_X, self.SCREEN_HEIGHT - self.FRIDGE_INIT_Y_OFFSET)
        self.game_phase = 0
        self.done = False

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # 补全 render 方法（渲染界面）
    def render(self):
        """渲染游戏界面"""
        if self.render_mode != "human":
            return

        # 填充背景
        self.screen.fill(self.colors["white"])

        if self.game_phase == 0:
            # 开始界面
            scaled_bg, bg_rect = scale_bg(self.start_bg, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            self.screen.blit(scaled_bg, bg_rect)

            # 【修改3：绘制融合感的开始按钮】
            # 1. 获取鼠标位置，判断是否悬浮在按钮上
            mouse_pos = pygame.mouse.get_pos()
            is_hover = self.start_button_rect.collidepoint(mouse_pos)

            # 2. 绘制按钮阴影（轻微偏移，增加层次感，不突兀）
            shadow_rect = self.start_button_rect.copy()
            shadow_rect.x += 3
            shadow_rect.y += 3
            pygame.draw.rect(self.screen, (0, 0, 0, 50), shadow_rect, border_radius=15)

            # 3. 绘制按钮主体（根据悬浮状态切换颜色，更自然）
            btn_color = self.colors["btn_hover"] if is_hover else self.colors["btn_bg"]
            pygame.draw.rect(self.screen, btn_color, self.start_button_rect, border_radius=15)  # 更大圆角

            # 4. 绘制按钮文字（深棕色，适配背景）
            text = self.font_big.render("开始游戏", True, self.colors["btn_text"])
            text_rect = text.get_rect(center=self.start_button_rect.center)
            self.screen.blit(text, text_rect)
        else:
            # 游戏界面
            scaled_bg, bg_rect = scale_bg(self.game_bg, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            self.screen.blit(scaled_bg, bg_rect)

            # 绘制冰箱
            fridge_img = self.fridge_open_img if self.fridge.is_open else self.fridge_closed_img
            draw_with_shadow(fridge_img,
                             (self.fridge.x - self.FRIDGE_SIZE[0] // 2, self.fridge.y - self.FRIDGE_SIZE[1] // 2),
                             self.screen)

            # 绘制大象（仅阶段1/2显示）
            if self.game_phase in [1, 2]:
                draw_with_shadow(self.elephant_img, (self.elephant.x - self.ELEPHANT_SIZE[0] // 2,
                                                     self.elephant.y - self.ELEPHANT_SIZE[1] // 2), self.screen)

            # 绘制提示文字
            tip_text = ""
            if self.game_phase == 1:
                tip_text = "第一步：按1/点击冰箱 开冰箱门"
            elif self.game_phase == 2:
                tip_text = "第二步：按2/点击大象 放大象进冰箱"
            elif self.game_phase == 3:
                tip_text = "第三步：按3/点击冰箱 关冰箱门"
            elif self.game_phase == 4:
                tip_text = "完成！按R重置游戏"

            tip_surf = self.font.render(tip_text, True, self.colors["black"])
            tip_rect = tip_surf.get_rect(center=(self.SCREEN_WIDTH // 2, 50))
            self.screen.blit(tip_surf, tip_rect)


            # state = self._get_obs()
            # state_text = f"状态向量：[冰箱开关={state[0]}, 冰箱x={state[1]}, 冰箱y={state[2]}, 大象x={state[3]}, 大象y={state[4]}]"
            # state_surf = self.font.render(state_text, True, self.colors["black"])
            # self.screen.blit(state_surf, (10, 10))

        # 更新屏幕
        pygame.display.flip()

    # 补全 close 方法
    def close(self):
        """关闭环境，清理资源"""
        pygame.quit()


# 新增：测试运行代码
if __name__ == "__main__":
    env = FridgeGameEnv(render_mode="human")
    env.reset()
    clock = pygame.time.Clock()

    while True:
        clock.tick(env.metadata["render_fps"])
        env.render()

        # 处理事件（点击开始按钮切换到游戏阶段1）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if env.game_phase == 0 and env.start_button_rect.collidepoint(event.pos):
                    env.game_phase = 1
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
                elif env.game_phase > 0 and env.game_phase < 4:
                    action = int(pygame.key.name(event.key))
                    env.step(action)


class FridgeGameEnv:
    pass