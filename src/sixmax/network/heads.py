"""Policy and Value heads for the actor-critic network."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PolicyHead(nn.Module):
    """
    策略头 (Actor)。

    输出 num_actions 维 logits，经过 masked softmax 得到动作概率。

    设计要点:
    - 最后一层使用较小的初始化增益（输出接近均匀分布）
    - 支持动作掩码，屏蔽非法动作
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_actions: int = 6,
    ):
        """
        初始化 PolicyHead。

        Args:
            input_dim: 输入维度（Backbone 输出维度）
            hidden_dim: 隐藏层维度
            num_actions: 动作数量
        """
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

        # 保存维度信息
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # 策略头使用较小的初始化（输出接近均匀分布）
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化权重。"""
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=0.01)  # 小增益
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor, legal_mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        前向传播。

        Args:
            x: (B, input_dim) backbone 输出
            legal_mask: (B, num_actions) 合法动作掩码 (True=合法)

        Returns:
            action_probs: (B, num_actions) 动作概率分布
            logits: (B, num_actions) 原始 logits
        """
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)  # (B, num_actions)

        # 屏蔽非法动作
        masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
        action_probs = F.softmax(masked_logits, dim=-1)

        return action_probs, logits


class ValueHead(nn.Module):
    """
    价值头 (Critic)。

    输出标量状态价值，范围约 [-100, +100] BB。
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
    ):
        """
        初始化 ValueHead。

        Args:
            input_dim: 输入维度（Backbone 输出维度）
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # 保存维度信息
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self._init_weights()

    def _init_weights(self) -> None:
        """初始化权重。"""
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播。

        Args:
            x: (B, input_dim) backbone 输出

        Returns:
            value: (B, 1) 状态价值
        """
        x = F.relu(self.fc1(x))
        value = self.fc2(x)  # (B, 1)

        return value
