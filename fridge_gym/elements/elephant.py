
class Elephant:
    """大象元素类（仅存储位置和更新方法）"""
    def __init__(self, x, y, width, height):
        # 中心点坐标
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def update_pos(self, x, y):
        """更新大象位置"""
        self.x = x
        self.y = y