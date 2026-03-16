
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
    # 模式说明：
    # - 手动模式：你用键盘直接控制（适合验证环境是否正常）
    # - 自动模式（规则基）：让规则智能体按“开门→靠近→放入→关门”跑
    mode = "manual"  # manual | rule_auto
    pygame.display.set_caption("把大象放进冰箱 | 手动模式(↑↓←→移动, O开门, P放入, C关门) | 2切规则自动")

    # 键盘→动作索引映射（按需求：open/close/up/forward/put/down）
    # 0=open,1=close,2=up,3=forward,4=put,5=down
    key_to_action_idx = {
        pygame.K_o: 0,  # O: open
        pygame.K_c: 1,  # C: close
        pygame.K_d: 3,  # D: forward
        pygame.K_p: 4,  # P: put
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
                # 数字键切换模式（按你的要求：不再用A键）
                elif event.key == pygame.K_1:
                    mode = "manual"
                    pygame.display.set_caption("把大象放进冰箱 | 手动模式(↑↓←→移动, O开门, P放入, C关门) | 2切规则自动")
                    print("切换模式 | 手动模式")
                elif event.key == pygame.K_2:
                    mode = "rule_auto"
                    pygame.display.set_caption("把大象放进冰箱 | 自动模式(规则基) | 1切手动")
                    print("切换模式 | 自动模式（规则基）")
                # 执行对应动作
                elif (mode == "manual") and (event.key in key_to_action_idx):
                    action_idx = key_to_action_idx[event.key]
                    action_onehot = agent.onehot(action_idx)
                    obs, reward, terminated, truncated, info = env.step(action_onehot)
                    print(f"[手动模式] 执行动作：{env._action_name(action_idx)} | 状态：{env.format_state_text()} | 奖励：{reward:.2f}")
                    if terminated or truncated:
                        print("Episode结束 |", "完成" if info.get("task_complete") else "终止")

        # 手动模式：支持↑↓←→自由移动（你提出的“上下左右自行移动”）
        if mode == "manual":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                env.manual_move_xy(-env.move_step_px, 0.0)
            if keys[pygame.K_RIGHT]:
                env.manual_move_xy(env.move_step_px, 0.0)
            if keys[pygame.K_UP]:
                env.manual_move_xy(0.0, -env.move_step_px)
            if keys[pygame.K_DOWN]:
                env.manual_move_xy(0.0, env.move_step_px)

        # 自动模式（规则基）：每帧由智能体输出动作
        if mode == "rule_auto":
            out = agent.act(obs, info)
            action_idx = int(out.action_onehot.argmax())
            obs, reward, terminated, truncated, info = env.step(out.action_onehot)
            why = ""
            if out.debug and "why" in out.debug:
                why = f" | 规则：{out.debug['why']}"
            print(f"[自动模式-规则基] 执行动作：{env._action_name(action_idx)} | 状态：{env.format_state_text()} | 奖励：{reward:.2f}{why}")
            pygame.time.delay(60)  # 防止刷屏过快
            if terminated or truncated:
                mode = "manual"
                pygame.display.set_caption("把大象放进冰箱 | 手动模式(↑↓←→移动, O开门, P放入, C关门) | 2切规则自动")
                print("Episode结束 | 自动模式停止，已切回手动模式。按 R 重置可再次运行。")

if __name__ == "__main__":
    main()