from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from fridge_gym.agents.base import AgentOutput, BaseAgent


class RuleBasedAgent(BaseAgent):
    """
    规则基智能体（按“开门→坐标对齐→关门”逻辑输出动作）。

    约定动作索引(与环境一致)：
    0=open, 1=close, 2=up, 3=down, 4=left, 5=right
    """

    def __init__(self, align_dx_threshold: float = 0.8, align_dy_threshold: float = 0.8):
        """
        align_dx_threshold / align_dy_threshold 单位：米

        因为环境的 obs[1]/obs[2] 已经改成“米”，规则智能体也需要用“米”来判断是否对齐。
        """
        self.align_dx_threshold = float(align_dx_threshold)
        self.align_dy_threshold = float(align_dy_threshold)

    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> AgentOutput:
        """
        输入：环境观测obs（由环境返回）
        输出：6维独热动作向量（one-hot）

        obs含义（与环境一致）：
        - obs[0]：冰箱门是否打开（1=开，0=关）
        - obs[1], obs[2]：大象坐标 (x, y)，单位：米
        - obs[3], obs[4]：冰箱坐标 (x, y)，单位：米

        规则基智能体的核心思想：
        - 不“学习”，只按人类逻辑做决策：开门→移动到冰箱区域→关门
        - 优点：不用训练，立刻能跑通；缺点：遇到复杂情况不会自我改进
        """
        door_open = bool(obs[0] > 0.5)
        ex, ey = float(obs[1]), float(obs[2])
        fx, fy = float(obs[3]), float(obs[4])
        dx = fx - ex
        dy = fy - ey

        inside = (abs(dx) <= self.align_dx_threshold) and (abs(dy) <= self.align_dy_threshold)

        if not door_open:
            action = 0  # open
            why = "门关：先开门"
        elif inside:
            action = 1  # close
            why = "已在冰箱区域：关门完成"
        else:
            # 先调整高度，再调整水平位置
            if abs(dy) > self.align_dy_threshold:
                # dy > 0: 冰箱在大象下方 -> 向下(3)
                # dy < 0: 冰箱在大象上方 -> 向上(2)
                action = 3 if dy > 0 else 2
                why = "高度未对齐：先上下对齐"
            elif abs(dx) > self.align_dx_threshold:
                # dx > 0: 冰箱在大象右侧 -> 向右(5)
                # dx < 0: 冰箱在大象左侧 -> 向左(4)
                action = 5 if dx > 0 else 4
                why = "水平未对齐：左右移动靠近"
            else:
                action = 1
                why = "已对齐：关门完成"

        return AgentOutput(action_onehot=self.onehot(action), debug={"why": why})

