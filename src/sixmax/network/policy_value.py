"""Policy-Value Network for PPO training."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..encoding import StateEncoder
from .backbone import Backbone
from .heads import PolicyHead, ValueHead


class PolicyValueNetwork(nn.Module):
    """
    Actor-Critic 网络。

    结合 StateEncoder、Backbone、PolicyHead、ValueHead。

    输入: 游戏状态 batch
    输出: 动作概率、状态价值

    参数量统计:
    - StateEncoder: ~51K
    - Backbone: ~531K
    - PolicyHead: ~33K
    - ValueHead: ~33K
    - 总计: ~648K
    """

    def __init__(
        self,
        state_encoder: StateEncoder | None = None,
        backbone_hidden: int = 512,
        backbone_output: int = 256,
        head_hidden: int = 128,
        num_actions: int = 6,
    ):
        """
        初始化 PolicyValueNetwork。

        Args:
            state_encoder: 状态编码器（可选，默认创建新的）
            backbone_hidden: Backbone 隐藏层维度
            backbone_output: Backbone 输出维度
            head_hidden: Head 隐藏层维度
            num_actions: 动作数量
        """
        super().__init__()

        # 状态编码器（可选，也可以外部传入已编码特征）
        self.state_encoder = state_encoder if state_encoder is not None else StateEncoder()
        input_dim = self.state_encoder.get_output_dim()  # 261

        # 主干网络
        self.backbone = Backbone(input_dim, backbone_hidden, backbone_output)

        # 输出头
        self.policy_head = PolicyHead(backbone_output, head_hidden, num_actions)
        self.value_head = ValueHead(backbone_output, head_hidden)

        # 保存配置
        self.num_actions = num_actions

    def forward(
        self,
        batch: dict[str, Tensor],
        legal_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        前向传播。

        Args:
            batch: 游戏状态 batch (来自 StateBatchBuilder)
            legal_mask: (B, num_actions) 合法动作掩码

        Returns:
            action_probs: (B, num_actions) 动作概率
            value: (B, 1) 状态价值
            logits: (B, num_actions) 原始 logits (用于计算损失)
        """
        # 1. 状态编码
        state_features = self.state_encoder(batch)  # (B, 261)

        # 2. Backbone
        backbone_out = self.backbone(state_features)  # (B, 256)

        # 3. 输出头
        action_probs, logits = self.policy_head(backbone_out, legal_mask)
        value = self.value_head(backbone_out)

        return action_probs, value, logits

    def forward_from_features(
        self,
        state_features: Tensor,
        legal_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        从已编码的特征进行前向传播（跳过 StateEncoder）。

        Args:
            state_features: (B, 261) 已编码的状态特征
            legal_mask: (B, num_actions) 合法动作掩码

        Returns:
            action_probs: (B, num_actions) 动作概率
            value: (B, 1) 状态价值
            logits: (B, num_actions) 原始 logits
        """
        # Backbone
        backbone_out = self.backbone(state_features)

        # 输出头
        action_probs, logits = self.policy_head(backbone_out, legal_mask)
        value = self.value_head(backbone_out)

        return action_probs, value, logits

    def get_action(
        self,
        batch: dict[str, Tensor],
        legal_mask: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        采样动作（用于推理/收集数据）。

        Args:
            batch: 游戏状态 batch
            legal_mask: (B, num_actions) 合法动作掩码
            deterministic: 是否选择最高概率动作

        Returns:
            action: (B,) 采样的动作
            log_prob: (B,) 动作的对数概率
            value: (B, 1) 状态价值
        """
        action_probs, value, _ = self.forward(batch, legal_mask)

        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

        # 计算 log probability
        log_prob = torch.log(
            action_probs.gather(1, action.unsqueeze(-1)) + 1e-8
        ).squeeze(-1)

        return action, log_prob, value

    def evaluate_actions(
        self,
        batch: dict[str, Tensor],
        legal_mask: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        评估给定动作（用于 PPO 更新）。

        Args:
            batch: 游戏状态 batch
            legal_mask: (B, num_actions) 合法动作掩码
            actions: (B,) 实际执行的动作

        Returns:
            log_prob: (B,) 动作的对数概率
            value: (B, 1) 状态价值
            entropy: (B,) 策略熵
        """
        action_probs, value, _ = self.forward(batch, legal_mask)

        dist = torch.distributions.Categorical(action_probs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, value, entropy

    def get_value(self, batch: dict[str, Tensor]) -> Tensor:
        """
        仅获取状态价值（用于 GAE 计算）。

        Args:
            batch: 游戏状态 batch

        Returns:
            value: (B, 1) 状态价值
        """
        state_features = self.state_encoder(batch)
        backbone_out = self.backbone(state_features)
        value = self.value_head(backbone_out)
        return value
