
import pygame
import sys

from fridge_gym import FridgeGameEnv

def main():
    """游戏运行入口"""
    # 初始化环境
    env = FridgeGameEnv(render_mode="human")
    obs, info = env.reset()  # Gymnasium的reset返回(obs, info)
    clock = pygame.time.Clock()

    # 键盘→动作向量映射
    key_to_action = {
        pygame.K_UP: 0,    # 上 → 动作0
        pygame.K_DOWN: 1,  # 下 → 动作1
        pygame.K_LEFT: 2,  # 左 → 动作2
        pygame.K_RIGHT: 3, # 右 → 动作3
        pygame.K_o: 4,     # O键 → 开冰箱（动作4）
        pygame.K_c: 5      # C键 → 关冰箱（动作5）
    }

    # 主循环
    while True:
        clock.tick(env.metadata["render_fps"])
        env.render()

        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()
            # 键盘按下事件
            if event.type == pygame.KEYDOWN:
                # R键重置游戏
                if event.key == pygame.K_r:
                    env.reset()
                # 执行对应动作
                elif event.key in key_to_action:
                    action = key_to_action[event.key]
                    env.step(action)

        # 持续按键（按住方向键连续移动）
        keys = pygame.key.get_pressed()
        for key, action in key_to_action.items():
            if keys[key]:
                env.step(action)
                pygame.time.delay(10)  # 控制移动速度

if __name__ == "__main__":
    main()