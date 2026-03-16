
"""
demo.py
=================
这个脚本提供三种模式，帮助你直观理解“规则基”和“强化学习(DQN)学习模式”的区别：
1. 手动模式：键盘直接控制（↑↓←→移动，O开门，P放入，C关门）
2. 自动模式（规则基）：调用 RuleBasedAgent，按人写好的 if-else 执行最优路径
3. 学习模式（DQN）：按 3 键启动训练（有探索/试错），训练后按 4 键用“学到的策略”自动执行

强化学习的“探索-利用”在这里体现在：
- 训练阶段：DQN 使用 epsilon-greedy 策略，会先大量随机走错路（探索），再逐步偏向高奖励动作（利用）；
- 执行阶段：只用 argmax Q(s,a) 的贪心策略，用的是“学到的最优路径”，不再随机。
"""

import pygame
import sys
import numpy as np

from fridge_gym import FridgeGameEnv
from fridge_gym.agents import RuleBasedAgent, DQNAgent


def train_dqn(
    num_episodes: int = 100,
    max_steps_per_ep: int = 200,
    *,
    elephant_init_distance_m: float | None = None,
    move_step_m: float | None = None,
    start_options: dict | None = None,
    start_noise_m: float = 0.2,
    agent: DQNAgent | None = None,
):
    """
    学习模式（DQN）训练过程。

    - 使用 epsilon-greedy：前期大量随机探索，后期逐步转为利用学到的Q值。
    - 使用经验回放 + 目标网络，保证训练稳定。
    - 控制台会打印每10个episode的平均回报和最近一次loss，方便你观察“从乱走到变聪明”的过程。
    """
    print("\n========== 启动 DQN 学习模式（训练） ==========")
    print("说明：训练阶段不会使用规则基，也不会手动干预，完全靠试错+奖励学习。")
    print(f"训练设置 | 起点扰动半径：±{start_noise_m:.2f}m | 每局最大步数：{max_steps_per_ep}")

    # 训练时不需要渲染窗口，用 render_mode='none' 节省资源
    env = FridgeGameEnv(render_mode="none", elephant_init_distance_m=elephant_init_distance_m, move_step_m=move_step_m)
    obs_dim = env.observation_space.shape[0]
    n_actions = 6
    # 关键：如果传入了 agent，就在原模型上继续训练（经验/epsilon/网络参数都会累积）
    if agent is None:
        agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions)

    reward_history = []
    last_loss = None
    success_history = []  # 记录每局是否真正完成（关门且大象在冰箱内）

    # 如果你把起点放得很远（比如 dx=6m），固定200步有时不够“走完一局”，训练会变得很不稳定。
    # 这里做个自适应：按“水平距离/步长”估算需要多少步，给一个上限更合理。
    adaptive_max_steps = int(max_steps_per_ep)
    if start_options and start_options.get("elephant_pos") is not None:
        x0, y0 = start_options["elephant_pos"]
        fridge_x = float(env.SCREEN_WIDTH * 0.7)
        dx_m_est = abs((fridge_x - float(x0)) / env.PIXELS_PER_METER)
        step_m = max(1e-6, float(env.move_step_m))
        adaptive_max_steps = max(adaptive_max_steps, int(dx_m_est / step_m) + 120)

    for ep in range(1, num_episodes + 1):
        # 关键改动：训练起点做“随机扰动”（domain randomization）
        # 原因：如果只在一个固定起点训练，DQN 很容易“记住这一个起点的最优动作序列”，
        # 一旦你手动改了初始位置，就会出现“进不去冰箱”的现象（泛化失败）。
        #
        # 做法：围绕你保存的起点 (elephant_pos) 加一个小的随机偏移，让智能体学会“在一片区域内”都能完成任务。
        # start_noise_m 越大，泛化越强，但学习难度也会变大；建议 0.3~1.0 之间。
        ep_options = dict(start_options or {})
        if "elephant_pos" in ep_options and ep_options["elephant_pos"] is not None and start_noise_m > 0:
            x0, y0 = ep_options["elephant_pos"]
            noise_px = float(start_noise_m * env.PIXELS_PER_METER)
            # 均匀扰动：[-noise, +noise]
            nx = float(x0) + float(np.random.uniform(-noise_px, noise_px))
            ny = float(y0) + float(np.random.uniform(-noise_px, noise_px))
            ep_options["elephant_pos"] = (nx, ny)

        obs, info = env.reset(options=ep_options)
        ep_reward = 0.0

        success = False
        for t in range(adaptive_max_steps):
            # 1) epsilon-greedy 选择动作（有随机探索）
            action_idx = agent.act_index(obs, explore=True)
            action_onehot = agent.onehot(action_idx)

            # 2) 与环境交互，获得“试错”结果
            next_obs, reward, terminated, truncated, info = env.step(action_onehot)
            done = bool(terminated or truncated)

            # 3) 将这一步的经验存入回放池
            # 训练稳定性：奖励截断（但不要把“关门成功”的关键大奖励截得太小）
            # 之前用[-5,5]会把 put/close 的关键奖励压扁，DQN容易学到“会进但不关门”。
            clipped_reward = float(max(-20.0, min(20.0, float(reward))))
            agent.push_transition(obs, action_idx, clipped_reward, next_obs, done)

            # 4) 从回放池随机采样，执行一次参数更新（可能返回None，表示buffer还不够大，暂不更新）
            loss = agent.train_one_step()
            if loss is not None:
                last_loss = loss

            obs = next_obs
            ep_reward += float(reward)

            if done:
                success = bool(info.get("task_complete", False))
                break

        reward_history.append(ep_reward)
        success_history.append(1 if success else 0)

        if ep % 10 == 0:
            avg_r = sum(reward_history[-10:]) / min(10, len(reward_history))
            succ10 = sum(success_history[-10:])
            print(
                f"[DQN训练] Episode {ep}/{num_episodes} | 最近10局平均回报：{avg_r:.2f} | 最近10局成功：{succ10}/10"
                + (f" | 最近一次loss：{last_loss:.4f}" if last_loss is not None else " | loss暂不可用（经验不足）")
            )

    env.close()
    print("========== DQN 训练结束，按 4 键可在主窗口使用“学习后的自动执行”模式 ==========\n")
    return agent


def main():
    """游戏运行入口（支持手动 / 规则基自动 / DQN学习模式）"""
    # 初始化可视化环境
    env = FridgeGameEnv(render_mode="human")
    obs, info = env.reset()  # Gymnasium的reset返回(obs, info)
    clock = pygame.time.Clock()

    rule_agent = RuleBasedAgent()
    dqn_agent = None  # 训练完成后会被赋值
    dqn_success_count = 0  # 统计“学习执行模式”下成功次数
    dqn_start_options = None  # 记录“你在手动模式下调整后的起点”，供训练/执行使用
    dqn_eval_steps = 0  # 当前DQN执行这一局走了多少步（防止卡死）
    dqn_eval_max_steps = 400

    # 模式说明：
    # - manual     ：你用键盘直接控制（适合理解任务/验证环境）
    # - rule_auto  ：规则基自动执行（人写if-else，上来就最优，但不是“学”的）
    # - dqn_eval   ：DQN学习后的执行（先按3训练，再按4切换到这里）
    mode = "manual"  # manual | rule_auto | dqn_eval
    pygame.display.set_caption(
        "把大象放进冰箱 | 1手动(↑↓←→,O开门,P放入,C关门) | 2规则自动 | 3学习训练 | 4学习执行"
    )

    # 键盘→动作索引映射（强化学习动作空间：open/close/up/forward/put/down）
    # 0=open,1=close,2=up,3=forward,4=put,5=down
    key_to_action_idx = {
        pygame.K_o: 0,  # O: open
        pygame.K_c: 1,  # C: close
        pygame.K_d: 3,  # D: forward
        pygame.K_p: 4,  # P: put
    }

    # 主循环（只负责显示 & 模式切换 & 执行当前模式一步）
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
                # 数字键切换/触发模式
                elif event.key == pygame.K_1:
                    mode = "manual"
                    pygame.display.set_caption(
                        "把大象放进冰箱 | 手动模式(↑↓←→移动, O开门, P放入, C关门) | 2规则自动 | 3学习训练 | 4学习执行"
                    )
                    print("切换模式 | 手动模式")
                # K 键：清空DQN学习进度（从零开始学）
                elif event.key == pygame.K_k:
                    dqn_agent = None
                    dqn_success_count = 0
                    # 保留 dqn_start_options（学习起点）不动：你通常希望“从同一起点重新学”
                    mode = "manual"
                    pygame.display.set_caption(
                        "已清空DQN进度 | 1手动调位置 | H保存起点 | 3重新训练(从零) | 4执行"
                    )
                    print("已清空DQN学习进度(K) | 模型参数/经验回放/epsilon进度已重置为初始状态。")
                # H 键：把当前手动调整的位置，保存为“学习起点”
                elif event.key == pygame.K_h:
                    dqn_start_options = {"elephant_pos": (float(env.elephant.x), float(env.elephant.y)), "fridge_open": False}
                    # 打印一下当前离冰箱多远（米），避免不小心把起点放得太远导致训练难度暴涨
                    fx, fy = info.get("fridge_pos", (env.fridge.x, env.fridge.y))
                    dx_m = abs((float(fx) - float(env.elephant.x)) / env.PIXELS_PER_METER)
                    dy_m = abs((float(fy) - float(env.elephant.y)) / env.PIXELS_PER_METER)
                    print("已保存学习起点(H) | elephant_pos =", dqn_start_options["elephant_pos"], f"| 距离冰箱：dx={dx_m:.2f}m dy={dy_m:.2f}m")
                elif event.key == pygame.K_2:
                    mode = "rule_auto"
                    obs, info = env.reset()
                    pygame.display.set_caption(
                        "把大象放进冰箱 | 自动模式(规则基) | 1手动 | 3学习训练 | 4学习执行"
                    )
                    print("切换模式 | 自动模式（规则基）")
                elif event.key == pygame.K_3:
                    # 启动 DQN 训练（阻塞当前循环，训练结束后返回）
                    print("切换模式 | 启动 DQN 学习模式（训练），训练过程请看控制台日志...")
                    # 如果你没按 H 保存学习起点，则默认使用当前手动位置作为起点（更符合你的直觉）
                    if dqn_start_options is None and mode == "manual":
                        dqn_start_options = {"elephant_pos": (float(env.elephant.x), float(env.elephant.y)), "fridge_open": False}
                        print("未设置学习起点(H)，已自动使用当前手动位置作为学习起点。")

                    dqn_agent = train_dqn(
                        elephant_init_distance_m=env.elephant_init_distance_m,
                        move_step_m=env.move_step_m,
                        start_options=dqn_start_options,
                        # 默认扰动别太大，先保证学会；想更泛化再手动调大
                        start_noise_m=0.2,
                        agent=dqn_agent,
                    )
                    # 训练结束后：把可视化环境也reset到“同一个学习起点”，保证按4看到的就是你设定的起点
                    obs, info = env.reset(options=dqn_start_options)
                    mode = "manual"
                    pygame.display.set_caption(
                        "把大象放进冰箱 | 手动模式(↑↓←→,O,P,C,H保存起点) | 2规则自动 | 3重新训练 | 4学习执行"
                    )
                elif event.key == pygame.K_4:
                    if dqn_agent is None:
                        print("提示：当前还没有训练好的DQN模型，请先按 3 启动训练。")
                    else:
                        mode = "dqn_eval"
                        obs, info = env.reset(options=dqn_start_options)
                        dqn_success_count = 0
                        dqn_eval_steps = 0
                        pygame.display.set_caption(
                            "把大象放进冰箱 | 学习模式(DQN执行) | 1手动 | 2规则自动 | 3重新训练"
                        )
                        print("切换模式 | 学习后的自动执行（DQN贪心策略，不再随机探索）")
                # 手动模式下，按O/P/C走“RL动作空间”的那条分支
                elif (mode == "manual") and (event.key in key_to_action_idx):
                    action_idx = key_to_action_idx[event.key]
                    action_onehot = rule_agent.onehot(action_idx)
                    obs, reward, terminated, truncated, info = env.step(action_onehot)
                    print(
                        f"[手动模式] 执行动作：{env._action_name(action_idx)} | 状态：{env.format_state_text()} | 奖励：{reward:.2f}"
                    )
                    if terminated or truncated:
                        print("Episode结束 |", "完成" if info.get("task_complete") else "终止")

        # 手动模式：支持↑↓←→自由移动（不计入RL动作空间，只用于人类测试）
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

        # 自动模式（规则基）：每帧由智能体输出动作（完全不学习）
        if mode == "rule_auto":
            out = rule_agent.act(obs, info)
            action_idx = int(out.action_onehot.argmax())
            obs, reward, terminated, truncated, info = env.step(out.action_onehot)
            why = ""
            if out.debug and "why" in out.debug:
                why = f" | 规则：{out.debug['why']}"
            print(
                f"[自动模式-规则基] 执行动作：{env._action_name(action_idx)} | 状态：{env.format_state_text()} | 奖励：{reward:.2f}{why}"
            )
            pygame.time.delay(60)  # 防止刷屏过快
            if terminated or truncated:
                mode = "manual"
                pygame.display.set_caption(
                    "把大象放进冰箱 | 手动模式(↑↓←→,O,P,C) | 2规则自动 | 3学习训练 | 4学习执行"
                )
                print("Episode结束 | 自动模式停止，已切回手动模式。按 R 重置可再次运行。")

        # 学习后的自动执行模式（DQN）：每帧按学到的Q值贪心选择动作（不再随机）
        if mode == "dqn_eval" and dqn_agent is not None:
            out = dqn_agent.act(obs, info)
            action_idx = int(out.action_onehot.argmax())
            obs, reward, terminated, truncated, info = env.step(out.action_onehot)
            dqn_eval_steps += 1
            print(
                f"[学习模式-DQN执行] 执行动作：{env._action_name(action_idx)} | 状态：{env.format_state_text()} | 奖励：{reward:.2f}"
            )
            pygame.time.delay(60)
            if dqn_eval_steps >= dqn_eval_max_steps and (not info.get("task_complete")):
                print(f"本局执行超过 {dqn_eval_max_steps} 步仍未完成，判定卡住，自动重置（建议按 3 继续累积训练）。")
                obs, info = env.reset(options=dqn_start_options)
                dqn_eval_steps = 0
            if terminated or truncated:
                # 如果这一局是“真正完成任务”（成功放入并关门）
                if info.get("task_complete"):
                    dqn_success_count += 1
                    print(f"Episode结束 | DQN 成功完成一次完整任务！累计成功次数：{dqn_success_count}")
                    # 你希望“最终找到位置结束，转换到重新开始界面”
                    # 所以成功一次就停止DQN执行，切回手动模式，等待你按 R 或重新设置起点再训练/执行。
                    mode = "manual"
                    pygame.display.set_caption(
                        "任务完成！| 按 R 重新开始 | 1手动调位置 | H保存起点 | 3继续训练(累积经验) | 4执行"
                    )
                    print("已完成任务 | 已切回手动模式：按 R 重置，或按 1/H/3/4 继续。")
                else:
                    print("Episode结束 | DQN 本局未完成任务（可按 3 继续累积训练经验）。")
                    # 未完成时，仍然从同一学习起点重置，方便你观察“再训练→再执行”的改进
                    obs, info = env.reset(options=dqn_start_options)
                    dqn_eval_steps = 0
                    print("自动重置环境 | 学习执行模式继续运行（按 3 可继续累积训练经验）")


if __name__ == "__main__":
    main()