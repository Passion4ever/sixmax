"""Backbone network for policy-value network."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Backbone(nn.Module):
    """
    共享主干网络。

    结构: input_dim → hidden_dim → hidden_dim(残差) → output_dim

    设计原则:
    - 使用 LayerNorm 而非 BatchNorm（RL 训练更稳定）
    - 残差连接防止梯度消失
    - 正交初始化（PPO 最佳实践）
    """

    def __init__(
        self,
        input_dim: int = 261,
        hidden_dim: int = 512,
        output_dim: int = 256,
    ):
        """
        初始化 Backbone 网络。

        Args:
            input_dim: 输入维度（StateEncoder 输出维度）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        super().__init__()

        # Layer 1: 升维
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        # Residual Block
        self.res_fc = nn.Linear(hidden_dim, hidden_dim)
        self.res_ln = nn.LayerNorm(hidden_dim)

        # Layer 3: 降维
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

        # 保存维度信息
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 正交初始化
        self._init_weights()

    def _init_weights(self) -> None:
        """正交初始化权重。"""
        for module in [self.fc1, self.res_fc, self.fc2]:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播。

        Args:
            x: (B, input_dim) 输入特征

        Returns:
            (B, output_dim) 输出特征
        """
        # Layer 1
        x = F.relu(self.ln1(self.fc1(x)))

        # Residual Block
        residual = x
        x = F.relu(self.res_ln(self.res_fc(x)))
        x = x + residual  # 残差连接

        # Layer 3
        x = F.relu(self.ln2(self.fc2(x)))

        return x

    def get_output_dim(self) -> int:
        """返回输出维度。"""
        return self.output_dim
