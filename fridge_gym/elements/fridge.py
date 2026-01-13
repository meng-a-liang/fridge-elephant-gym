import pygame


class Fridge:
    """冰箱元素类，封装位置、尺寸和开关状态"""

    def __init__(self, x, y, width=200, height=180):
        # 接收并保存参数
        self.x = x  # 冰箱中心x坐标
        self.y = y  # 冰箱中心y坐标
        self.width = width  # 冰箱宽度
        self.height = height  # 冰箱高度
        self.is_open = False  # 冰箱开关状态

        # 初始化矩形（用于碰撞检测/定位）
        self.rect = pygame.Rect(0, 0, width, height)
        self.rect.center = (x, y)

    def update_pos(self, x, y):
        """更新冰箱位置（窗口缩放时调用）"""
        self.x = x
        self.y = y
        self.rect.center = (x, y)