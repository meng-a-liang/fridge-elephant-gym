import pygame


class Elephant:
    """大象元素类，封装位置和尺寸"""

    def __init__(self, x, y, width=100, height=100):
        # 接收并保存参数
        self.x = x  # 大象中心x坐标
        self.y = y  # 大象中心y坐标
        self.width = width  # 大象宽度
        self.height = height  # 大象高度

        # 初始化矩形（用于碰撞检测/定位）
        self.rect = pygame.Rect(0, 0, width, height)
        self.rect.center = (x, y)

    def update_pos(self, x, y):
        """更新大象位置（窗口缩放时调用）"""
        self.x = x
        self.y = y
        self.rect.center = (x, y)