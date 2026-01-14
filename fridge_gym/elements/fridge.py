
class Fridge:
    """冰箱元素类（存储位置和开关状态）"""
    def __init__(self, x, y, width, height):
        # 中心点坐标
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.is_open = False  # 冰箱门状态

    def update_pos(self, x, y):
        """更新冰箱位置"""
        self.x = x
        self.y = y

    def toggle_door(self):
        """切换冰箱门状态"""
        self.is_open = not self.is_open