from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from fridge_gym.agents.base import AgentOutput, BaseAgent


class RuleBasedAgent(BaseAgent):
    """
    规则基智能体（按“开门→靠近→放入→关门”逻辑输出动作）。

    约定动作索引(与环境一致)：
    0=open, 1=close, 2=up, 3=forward, 4=put, 5=down
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
        - obs[1]：dx = fridge_x - elephant_x（水平相对距离，单位：像素）
        - obs[2]：dy = fridge_y - elephant_y（垂直相对距离，单位：像素）
        - obs[3]：大象是否已放入冰箱（1=是，0=否）

        规则基智能体的核心思想：
        - 不“学习”，只按人类逻辑做决策：开门→靠近→放入→关门
        - 优点：不用训练，立刻能跑通；缺点：遇到复杂情况不会自我改进
        """
        door_open = bool(obs[0] > 0.5)
        dx = float(obs[1])
        dy = float(obs[2])
        inside = bool(obs[3] > 0.5)

        aligned_for_put = (abs(dx) <= self.align_dx_threshold) and (abs(dy) <= self.align_dy_threshold)

        if (not door_open) and (not inside):
            action = 0  # open
            why = "门关且未放入：先开门"
        elif (not inside) and (not aligned_for_put):
            # 先调整高度，再向前
            if abs(dy) > self.align_dy_threshold:
                # dy = fridge_y - elephant_y
                # - dy > 0：冰箱在大象“下方”，说明大象偏上 → 需要向下(5)
                # - dy < 0：冰箱在大象“上方”，说明大象偏下 → 需要向上(2)
                action = 5 if dy > 0 else 2
                why = "高度未对齐：先上下对齐"
            else:
                action = 3  # forward
                why = "未到冰箱附近：向前靠近"
        elif (not inside) and door_open and aligned_for_put:
            action = 4  # put
            why = "已对齐且门开：放入冰箱"
        elif inside and door_open:
            action = 1  # close
            why = "已放入且门开：关门完成"
        else:
            # 兜底：尽量不要做无效动作
            action = 3
            why = "兜底：继续向前"

        return AgentOutput(action_onehot=self.onehot(action), debug={"why": why})

