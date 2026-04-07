"""
dqn_agent.py
=================
这个文件实现 **DQN（Deep Q-Network）强化学习智能体**，用于“学习模式”。

对零基础的理解方式：
- **规则基智能体**：像写 if-else 一样，直接告诉它“下一步做什么”，不需要训练。
- **DQN学习智能体**：不写固定规则，而是让它通过“试错+奖励”学会在每个状态下选哪个动作更好。

本实现包含：
- 经验回放 Replay Buffer
- Q网络与目标网络（target network）
- epsilon-greedy 探索策略
- 训练过程（输出 reward / loss）
- 测试过程（训练后用学到的策略自动执行）

依赖：需要安装 PyTorch（torch）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque
import random

import numpy as np

from fridge_gym.agents.base import AgentOutput, BaseAgent


@dataclass
class DQNConfig:
    """DQN超参数配置（可按论文/课程需求调整）。"""

    gamma: float = 0.99
    # 学习率稍微调小，避免你看到的“loss爆炸/策略崩掉”
    lr: float = 3e-4
    batch_size: int = 128
    buffer_size: int = 50_000
    min_buffer_size: int = 1_000
    target_update_interval: int = 500  # 每隔多少个训练step同步一次target网络
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    # 衰减放慢，让前期探索更充分，避免过早“陷入边界局部最优”
    epsilon_decay_steps: int = 20_000


class ReplayBuffer:
    """
    经验回放：把智能体在环境中看到的数据存起来，然后随机采样训练。

    为什么要随机采样？
    - 如果按时间顺序训练，数据相关性很强，神经网络容易不稳定。
    - 随机采样能打散相关性，提高训练稳定性。
    """

    def __init__(self, capacity: int):
        self._buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=int(capacity))

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool):
        self._buf.append((s.astype(np.float32), int(a), float(r), s2.astype(np.float32), bool(done)))

    def __len__(self) -> int:
        return len(self._buf)

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, int(batch_size))
        s, a, r, s2, done = zip(*batch)
        return (
            np.stack(s, axis=0),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.stack(s2, axis=0),
            np.asarray(done, dtype=np.float32),
        )


class DQNAgent(BaseAgent):
    """
    DQN智能体：输入obs，输出6维one-hot动作。

    训练时：
    - 使用 epsilon-greedy：有概率随机动作（探索），其余选择Q值最大的动作（利用）
    测试时：
    - 直接选Q值最大的动作（不探索）
    """

    def __init__(self, obs_dim: int = 5, n_actions: int = 6, cfg: Optional[DQNConfig] = None, device: str = "cpu"):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.cfg = cfg or DQNConfig()
        self.device = device

        # 延迟导入torch：避免用户只跑手动/规则模式时也必须安装torch
        try:
            import torch
            import torch.nn as nn
        except ModuleNotFoundError as e:  # pragma: no cover
            # 用纯ASCII/简体中文，避免某些Windows终端出现乱码
            raise ModuleNotFoundError("学习模式需要安装 PyTorch：pip install torch") from e

        self.torch = torch

        class QNet(nn.Module):
            def __init__(self, in_dim: int, out_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, out_dim),
                )

            def forward(self, x):
                return self.net(x)

        self.q = QNet(self.obs_dim, self.n_actions).to(self.device)
        self.q_target = QNet(self.obs_dim, self.n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.optim = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer = ReplayBuffer(self.cfg.buffer_size)
        self.train_steps = 0

    def _epsilon(self) -> float:
        # 线性衰减：从start逐步降到end
        t = min(self.train_steps, self.cfg.epsilon_decay_steps)
        frac = t / float(self.cfg.epsilon_decay_steps)
        return float(self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start))

    def act_index(self, obs: np.ndarray, explore: bool = True) -> int:
        """返回动作索引（0..5）。"""
        if explore and (random.random() < self._epsilon()):
            return random.randrange(self.n_actions)

        with self.torch.no_grad():
            x = self.torch.tensor(obs, dtype=self.torch.float32, device=self.device).view(1, -1)
            q_values = self.q(x)
            return int(self.torch.argmax(q_values, dim=1).item())

    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> AgentOutput:
        a = self.act_index(obs, explore=False)
        return AgentOutput(action_onehot=self.onehot(a), debug={"policy": "greedy"})

    def push_transition(self, s: np.ndarray, a_idx: int, r: float, s2: np.ndarray, done: bool):
        self.buffer.push(s, a_idx, r, s2, done)

    def train_one_step(self) -> Optional[float]:
        """
        执行一次梯度更新。
        返回：loss（如果buffer不够大，返回None）
        """
        if len(self.buffer) < self.cfg.min_buffer_size:
            return None

        s, a, r, s2, done = self.buffer.sample(self.cfg.batch_size)
        torch = self.torch

        s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.int64, device=self.device).view(-1, 1)
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device).view(-1, 1)
        s2_t = torch.tensor(s2, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)

        # 当前Q(s,a)
        q_sa = self.q(s_t).gather(1, a_t)

        with torch.no_grad():
            # 目标：r + gamma * max_a' Q_target(s', a') * (1-done)
            max_next_q = self.q_target(s2_t).max(dim=1, keepdim=True).values
            target = r_t + self.cfg.gamma * max_next_q * (1.0 - done_t)

        loss = self.loss_fn(q_sa, target)

        self.optim.zero_grad()
        loss.backward()
        # 梯度裁剪：避免梯度爆炸导致loss突然飙升、策略崩坏
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.optim.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_interval == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return float(loss.item())

