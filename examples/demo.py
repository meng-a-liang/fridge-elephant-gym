
import pygame
import sys

from fridge_gym import FridgeGameEnv
from fridge_gym.agents import RuleBasedAgent

def main():
    """游戏运行入口"""
    # 初始化环境
    env = FridgeGameEnv(render_mode="human")
    obs, info = env.reset()  # Gymnasium的reset返回(obs, info)
    clock = pygame.time.Clock()

    agent = RuleBasedAgent()
    auto_mode = False

    # 键盘→动作索引映射（按需求：open/close/up/forward/put/down）
    # 0=open,1=close,2=up,3=forward,4=put,5=down
    key_to_action_idx = {
        pygame.K_o: 0,  # O: open
        pygame.K_c: 1,  # C: close
        pygame.K_w: 2,  # W: up
        pygame.K_d: 3,  # D: forward
        pygame.K_p: 4,  # P: put
        pygame.K_s: 5,  # S: down
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
                    obs, info = env.reset()
                    print("重置环境 |", env.format_state_text())
                # A键切换自动模式
                elif event.key == pygame.K_a:
                    auto_mode = not auto_mode
                    print("切换模式 |", "自动模式" if auto_mode else "手动模式")
                # 执行对应动作
                elif (not auto_mode) and (event.key in key_to_action_idx):
                    action_idx = key_to_action_idx[event.key]
                    action_onehot = agent.onehot(action_idx)
                    obs, reward, terminated, truncated, info = env.step(action_onehot)
                    print(f"执行动作：{env._action_name(action_idx)} | 当前状态：{env.format_state_text()} | 奖励：{reward:.2f}")
                    if terminated or truncated:
                        print("Episode结束 |", "完成" if info.get("task_complete") else "终止")

        # 自动模式：每帧由智能体输出动作
        if auto_mode:
            out = agent.act(obs, info)
            action_idx = int(out.action_onehot.argmax())
            obs, reward, terminated, truncated, info = env.step(out.action_onehot)
            why = ""
            if out.debug and "why" in out.debug:
                why = f" | 规则：{out.debug['why']}"
            print(f"执行动作：{env._action_name(action_idx)} | 当前状态：{env.format_state_text()} | 奖励：{reward:.2f}{why}")
            pygame.time.delay(60)  # 防止刷屏过快
            if terminated or truncated:
                auto_mode = False
                print("Episode结束 | 自动模式停止。按 R 重置后可再次运行。")

if __name__ == "__main__":
    main()