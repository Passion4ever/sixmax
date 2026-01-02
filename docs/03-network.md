# 模块 3: 神经网络详细设计

> Actor-Critic 网络架构，用于 PPO 训练
>
> **状态: ✅ 已完成** | 模块顺序: 3/3 | 依赖: encoding | 参数量: ~648K

---

## 1. 模块概述

### 1.1 设计目标

- 输入 StateEncoder 的 261 维特征向量
- 输出策略分布（6 维）和状态价值（1 维）
- 支持 GPU 高效推理（A100 优化）
- 参数量控制在 ~650K（轻量级）

### 1.2 在系统中的位置

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  游戏引擎   │────▶│ 状态表示层  │────▶│  神经网络   │
│  (Engine)   │     │ (Encoding)  │     │ (Network)   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                         ┌─────────────────────┴─────────────────────┐
                         │                                           │
                         ▼                                           ▼
                  ┌─────────────┐                             ┌─────────────┐
                  │ Policy Head │                             │ Value Head  │
                  │  (Actor)    │                             │  (Critic)   │
                  └──────┬──────┘                             └──────┬──────┘
                         │                                           │
                         ▼                                           ▼
                    6维动作概率                                  1维状态价值
```

---

## 2. 整体架构

### 2.1 网络结构图

```
输入: (B, 261) 状态特征
         │
         ▼
┌─────────────────────────────────────┐
│           Backbone Network          │
├─────────────────────────────────────┤
│  Linear(261, 512) + LayerNorm + ReLU│  ← Layer 1
│              │                      │
│              ▼                      │
│  ┌─────────────────────┐            │
│  │   Residual Block    │            │
│  │  Linear(512, 512)   │            │
│  │  LayerNorm + ReLU   │            │
│  └──────────┬──────────┘            │
│             │ (+)  ←── 残差连接      │
│             ▼                       │
│  Linear(512, 256) + LayerNorm + ReLU│  ← Layer 3
└─────────────────┬───────────────────┘
                  │
                  │ (B, 256)
       ┌──────────┴──────────┐
       │                     │
       ▼                     ▼
┌─────────────┐       ┌─────────────┐
│ Policy Head │       │ Value Head  │
├─────────────┤       ├─────────────┤
│ Linear(256, │       │ Linear(256, │
│   128)+ReLU │       │   128)+ReLU │
│ Linear(128, │       │ Linear(128, │
│     6)      │       │     1)      │
└──────┬──────┘       └──────┬──────┘
       │                     │
       ▼                     ▼
  (B, 6) logits         (B, 1) value
       │
       ▼
  masked_softmax
       │
       ▼
  (B, 6) action_probs
```

### 2.2 维度流转

| 层 | 输入维度 | 输出维度 | 说明 |
|----|----------|----------|------|
| 输入 | (B, 261) | - | StateEncoder 输出 |
| Backbone Layer 1 | (B, 261) | (B, 512) | 升维 |
| Residual Block | (B, 512) | (B, 512) | 残差连接 |
| Backbone Layer 3 | (B, 512) | (B, 256) | 降维 |
| Policy Head | (B, 256) | (B, 6) | 动作 logits |
| Value Head | (B, 256) | (B, 1) | 状态价值 |

---

## 3. 模块详细设计

### 3.1 Backbone Network

**设计原则:**
- 使用 LayerNorm 而非 BatchNorm（RL 训练更稳定）
- 残差连接防止梯度消失
- 正交初始化（PPO 最佳实践）

```python
class Backbone(nn.Module):
    """
    共享主干网络。

    结构: 261 → 512 → 512(残差) → 256
    """

    def __init__(self, input_dim: int = 261, hidden_dim: int = 512, output_dim: int = 256):
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

        # 正交初始化
        self._init_weights()

    def _init_weights(self):
        for module in [self.fc1, self.res_fc, self.fc2]:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        # Layer 1
        x = F.relu(self.ln1(self.fc1(x)))

        # Residual Block
        residual = x
        x = F.relu(self.res_ln(self.res_fc(x)))
        x = x + residual  # 残差连接

        # Layer 3
        x = F.relu(self.ln2(self.fc2(x)))

        return x  # (B, 256)
```

**参数量:**
- fc1: 261 × 512 + 512 = 134,144
- ln1: 512 × 2 = 1,024
- res_fc: 512 × 512 + 512 = 262,656
- res_ln: 512 × 2 = 1,024
- fc2: 512 × 256 + 256 = 131,328
- ln2: 256 × 2 = 512
- **总计: ~531K**

### 3.2 Policy Head (Actor)

**职责:** 输出动作概率分布

```python
class PolicyHead(nn.Module):
    """
    策略头 (Actor)。

    输出 6 维 logits，经过 masked softmax 得到动作概率。
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, num_actions: int = 6):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

        # 策略头使用较小的初始化（输出接近均匀分布）
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=0.01)  # 小增益
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor, legal_mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, 256) backbone 输出
            legal_mask: (B, 6) 合法动作掩码 (True=合法)

        Returns:
            action_probs: (B, 6) 动作概率分布
            logits: (B, 6) 原始 logits
        """
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)  # (B, 6)

        # 屏蔽非法动作
        masked_logits = logits.masked_fill(~legal_mask, float('-inf'))
        action_probs = F.softmax(masked_logits, dim=-1)

        return action_probs, logits
```

**参数量:**
- fc1: 256 × 128 + 128 = 32,896
- fc2: 128 × 6 + 6 = 774
- **总计: ~33K**

### 3.3 Value Head (Critic)

**职责:** 估计状态价值

```python
class ValueHead(nn.Module):
    """
    价值头 (Critic)。

    输出标量状态价值，范围约 [-100, +100] BB。
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 256) backbone 输出

        Returns:
            value: (B, 1) 状态价值
        """
        x = F.relu(self.fc1(x))
        value = self.fc2(x)  # (B, 1)

        return value
```

**参数量:**
- fc1: 256 × 128 + 128 = 32,896
- fc2: 128 × 1 + 1 = 129
- **总计: ~33K**

---

## 4. 完整网络实现

### 4.1 PolicyValueNetwork

```python
class PolicyValueNetwork(nn.Module):
    """
    Actor-Critic 网络。

    结合 StateEncoder、Backbone、PolicyHead、ValueHead。

    输入: 游戏状态 batch
    输出: 动作概率、状态价值
    """

    def __init__(
        self,
        state_encoder: StateEncoder | None = None,
        backbone_hidden: int = 512,
        backbone_output: int = 256,
        head_hidden: int = 128,
        num_actions: int = 6,
    ):
        super().__init__()

        # 状态编码器（可选，也可以外部传入已编码特征）
        self.state_encoder = state_encoder or StateEncoder()
        input_dim = self.state_encoder.get_output_dim()  # 261

        # 主干网络
        self.backbone = Backbone(input_dim, backbone_hidden, backbone_output)

        # 输出头
        self.policy_head = PolicyHead(backbone_output, head_hidden, num_actions)
        self.value_head = ValueHead(backbone_output, head_hidden)

    def forward(
        self,
        batch: dict[str, Tensor],
        legal_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        前向传播。

        Args:
            batch: 游戏状态 batch (来自 StateBatchBuilder)
            legal_mask: (B, 6) 合法动作掩码

        Returns:
            action_probs: (B, 6) 动作概率
            value: (B, 1) 状态价值
            logits: (B, 6) 原始 logits (用于计算损失)
        """
        # 1. 状态编码
        state_features = self.state_encoder(batch)  # (B, 261)

        # 2. Backbone
        backbone_out = self.backbone(state_features)  # (B, 256)

        # 3. 输出头
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
            legal_mask: (B, 6) 合法动作掩码
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
        log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)

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
            legal_mask: (B, 6) 合法动作掩码
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
```

### 4.2 辅助函数

```python
def count_parameters(model: nn.Module) -> int:
    """统计模型参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_policy_value_network(device: str = "cpu") -> PolicyValueNetwork:
    """创建默认配置的网络。"""
    network = PolicyValueNetwork()
    network = network.to(device)
    return network
```

---

## 5. 参数统计

### 5.1 各模块参数量

| 模块 | 参数量 | 占比 |
|------|--------|------|
| StateEncoder (已实现) | ~51K | 8% |
| Backbone | ~531K | 82% |
| PolicyHead | ~33K | 5% |
| ValueHead | ~33K | 5% |
| **总计** | **~648K** | 100% |

### 5.2 与其他模型对比

| 模型 | 参数量 | 说明 |
|------|--------|------|
| 本项目 | 0.65M | 轻量级 |
| AlphaGo Policy | 13M | 20x |
| GPT-2 Small | 124M | 190x |
| BERT-base | 110M | 169x |

---

## 6. 性能预估

### 6.1 A100 GPU 性能

| 指标 | 预估值 | 说明 |
|------|--------|------|
| 单次前向 | < 1ms | batch_size=1 |
| Batch 4096 | ~10ms | 推荐 batch |
| Batch 16384 | ~40ms | 最大 batch |
| 显存占用 | ~50MB | 模型权重 |

### 6.2 推荐配置

```python
# 训练配置
train_config = {
    "batch_size": 4096,      # 推荐 batch 大小
    "learning_rate": 3e-4,   # PPO 标准学习率
    "max_grad_norm": 0.5,    # 梯度裁剪
    "num_epochs": 4,         # PPO 更新轮数
}

# 推理配置
infer_config = {
    "batch_size": 1024,      # 推理 batch
    "deterministic": False,  # 探索模式
}
```

---

## 7. 与其他模块的接口

### 7.1 输入接口

```python
# 从 StateBatchBuilder 获取输入
from sixmax.encoding import StateBatchBuilder

builder = StateBatchBuilder()
batch = builder.build_batch(games, player_ids)

# batch 包含:
# - hole_ranks: (B, 2)
# - hole_suits: (B, 2)
# - board_ranks: (B, 5)
# - board_suits: (B, 5)
# - self_info: (B, 14)
# - opponent_info: (B, 15)
# - actions: (B, 24, 17)
# - action_mask: (B, 24)
```

### 7.2 输出接口

```python
# 网络输出
action_probs, value, logits = network(batch, legal_mask)

# action_probs: (B, 6) - 用于采样动作
# value: (B, 1) - 用于 PPO 优势估计
# logits: (B, 6) - 用于计算损失
```

### 7.3 与游戏引擎的集成

```python
# 完整推理流程
game = PokerGame()
game.reset_hand()

# 1. 获取状态
state_dict = game.get_state_for_player(player_id=0)

# 2. 构建 batch
batch = builder.build_from_dict(state_dict)
batch = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}

# 3. 获取合法动作
legal_actions = game.get_legal_actions()
legal_mask = torch.tensor([legal_actions], device=device)

# 4. 网络推理
with torch.no_grad():
    action, log_prob, value = network.get_action(batch, legal_mask)

# 5. 执行动作
action_type = ActionType(action.item())
game.step(action_type)
```

---

## 8. 文件结构

```
src/sixmax/network/
├── __init__.py          # 导出公共接口
├── backbone.py          # Backbone 网络
├── heads.py             # PolicyHead, ValueHead
├── policy_value.py      # PolicyValueNetwork 完整网络
└── utils.py             # 辅助函数

tests/
└── test_network.py      # 网络测试
```

---

## 9. 实现清单

| 组件 | 优先级 | 状态 |
|------|--------|------|
| Backbone | P0 | ✅ 已完成 |
| PolicyHead | P0 | ✅ 已完成 |
| ValueHead | P0 | ✅ 已完成 |
| PolicyValueNetwork | P0 | ✅ 已完成 |
| 单元测试 | P0 | ✅ 已完成 (36 个) |
| GPU 优化 | P1 | 🔲 后续 |
| torch.compile | P1 | 🔲 后续 |

---

## 附录

### A. 初始化策略

| 模块 | 初始化方法 | gain |
|------|-----------|------|
| Backbone 线性层 | 正交初始化 | √2 |
| PolicyHead fc1 | 正交初始化 | √2 |
| PolicyHead fc2 | 正交初始化 | 0.01 |
| ValueHead fc1 | 正交初始化 | √2 |
| ValueHead fc2 | 正交初始化 | 1.0 |
| 所有 bias | 零初始化 | - |

### B. PPO 相关接口

```python
# PPO 训练需要的接口
class PolicyValueNetwork:
    def forward(batch, legal_mask) -> (probs, value, logits)
    def get_action(batch, legal_mask, deterministic) -> (action, log_prob, value)
    def evaluate_actions(batch, legal_mask, actions) -> (log_prob, value, entropy)
```

### C. 参考资料

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/)
