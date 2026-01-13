from fridge_gym import FridgeGameEnv
import pygame

def main():
    # 初始化环境
    env = FridgeGameEnv(render_mode="human")
    obs, info = env.reset()
    print("="*50)
    print("初始状态向量：", obs)
    print("初始游戏信息：", info)
    print("="*50)

    running = True
    while running:
        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 窗口缩放
            elif event.type == pygame.VIDEORESIZE:
                env.SCREEN_WIDTH, env.SCREEN_HEIGHT = event.w, event.h
                env.screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT), pygame.RESIZABLE)
                env.fridge.update_pos(env.SCREEN_WIDTH//2, env.SCREEN_HEIGHT-100)
                env.elephant.update_pos(150, env.SCREEN_HEIGHT-100)
                env.start_button_rect.center = (env.SCREEN_WIDTH//2, env.SCREEN_HEIGHT//2)
            # 键盘交互
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("\n重置后状态向量：", obs)
                elif env.game_phase == 0:
                    env.game_phase = 1
                    obs = env._get_obs()
                    print("\n进入游戏，状态向量：", obs)
                elif event.key == pygame.K_1:
                    obs, reward, done, info = env.step(1)
                    print(f"\n执行动作1（开冰箱）- 奖励：{reward}, 状态向量：{obs}, 完成：{done}")
                elif event.key == pygame.K_2:
                    obs, reward, done, info = env.step(2)
                    print(f"\n执行动作2（放大象）- 奖励：{reward}, 状态向量：{obs}, 完成：{done}")
                elif event.key == pygame.K_3:
                    obs, reward, done, info = env.step(3)
                    print(f"\n执行动作3（关冰箱）- 奖励：{reward}, 状态向量：{obs}, 完成：{done}")
            # 鼠标交互
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if env.game_phase == 0 and env.start_button_rect.collidepoint(mouse_pos):
                    env.game_phase = 1
                    obs = env._get_obs()
                    print("\n点击开始，状态向量：", obs)
                elif env.game_phase == 1 and env.fridge.rect.collidepoint(mouse_pos):
                    obs, reward, done, info = env.step(1)
                    print(f"\n点击开冰箱 - 奖励：{reward}, 状态向量：{obs}")
                elif env.game_phase == 2 and env.elephant.rect.collidepoint(mouse_pos):
                    obs, reward, done, info = env.step(2)
                    print(f"\n点击放大象 - 奖励：{reward}, 状态向量：{obs}")
                elif env.game_phase == 3 and env.fridge.rect.collidepoint(mouse_pos):
                    obs, reward, done, info = env.step(3)
                    print(f"\n点击关冰箱 - 奖励：{reward}, 状态向量：{obs}, 完成：{done}")

        # 渲染界面
        env.render()

    env.close()

if __name__ == '__main__':
    main()