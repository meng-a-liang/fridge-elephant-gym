from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class AgentOutput:
    """统一智能体输出：one-hot动作 + 可选调试信息。"""

    action_onehot: np.ndarray  # shape=(6,)
    debug: Optional[Dict[str, Any]] = None


class BaseAgent:
    """智能体接口：输入obs/info，输出6维独热动作向量。"""

    def reset(self):
        """每个episode开始时调用（可选）。"""

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> AgentOutput:
        raise NotImplementedError

    @staticmethod
    def onehot(action_index: int) -> np.ndarray:
        a = np.zeros((6,), dtype=np.int8)
        a[int(action_index)] = 1
        return a

