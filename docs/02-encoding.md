# 模块 2: 状态表示层详细设计

> 将游戏状态编码为神经网络可处理的张量
>
> **状态: ✅ 已完成** | 模块顺序: 2/3 | 依赖: engine

---

## 目录

1. [设计目标](#1-设计目标)
2. [整体架构](#2-整体架构)
3. [牌面编码](#3-牌面编码)
4. [游戏状态编码](#4-游戏状态编码)
5. [动作历史编码](#5-动作历史编码)
6. [StateEncoder实现](#6-stateencoder实现)
7. [批量处理支持](#7-批量处理支持)
8. [文件结构](#8-文件结构)

---

## 1. 设计目标

### 1.1 核心需求

- **输入**: 游戏引擎的 `GameState` 或 `get_state_for_player()` 输出
- **输出**: 261维特征向量，供神经网络使用
- **要求**: GPU友好，支持批量处理

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| **端到端学习** | 使用Embedding而非手工特征，让网络自己学习表示 |
| **GPU友好** | 所有操作可在GPU上完成，支持大batch |
| **位置不变性** | 手牌排序消除顺序影响 |
| **归一化** | 数值特征归一化到合理范围 |

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         StateEncoder 架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   游戏状态输入                                                               │
│       │                                                                     │
│       ├──────────────┬──────────────┬──────────────┐                       │
│       ▼              ▼              ▼              ▼                       │
│   ┌────────┐    ┌────────┐    ┌────────┐    ┌────────────┐                │
│   │ 手牌   │    │ 公共牌 │    │ 游戏   │    │ 动作历史   │                │
│   │ 2张    │    │ 0-5张  │    │ 状态   │    │ 0-24个     │                │
│   └───┬────┘    └───┬────┘    └───┬────┘    └─────┬──────┘                │
│       │              │              │              │                       │
│       ▼              ▼              │              ▼                       │
│   ┌────────┐    ┌────────┐         │         ┌────────────┐               │
│   │CardEmb │    │CardEmb │         │         │ Transformer│               │
│   │ 48维   │    │ 120维  │         │         │   64维     │               │
│   └───┬────┘    └───┬────┘         │         └─────┬──────┘               │
│       │              │              │              │                       │
│       └──────────────┴──────────────┴──────────────┘                       │
│                              │                                              │
│                              ▼                                              │
│                      ┌─────────────┐                                       │
│                      │   Concat    │                                       │
│                      │   261维     │                                       │
│                      └─────────────┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 维度汇总

| 模块 | 维度 | 说明 |
|------|------|------|
| 手牌特征 | 48 | 2张 × 24维/张 |
| 公共牌特征 | 120 | 5张 × 24维/张 (padding补齐) |
| 自身信息 | 14 | 位置+筹码+底池+街道 |
| 对手信息 | 15 | 5个对手 × 3维/人 |
| 动作历史 | 64 | Transformer编码 |
| **总计** | **261** | |

---

## 3. 牌面编码

### 3.1 单张牌编码

每张牌用两个Embedding表示：

```python
# 牌值编码
rank: int  # 0-12 (2,3,4,5,6,7,8,9,T,J,Q,K,A)
suit: int  # 0-3 (♠,♥,♦,♣)

# 特殊值 (padding)
RANK_PADDING = 13
SUIT_PADDING = 4
CARD_PADDING = 52  # rank=13, suit=0
```

### 3.2 CardEmbedding 类

```python
class CardEmbedding(nn.Module):
    def __init__(self, rank_dim: int = 16, suit_dim: int = 8):
        super().__init__()
        self.rank_embed = nn.Embedding(14, rank_dim)  # 13 ranks + padding
        self.suit_embed = nn.Embedding(5, suit_dim)   # 4 suits + padding
        # 每张牌输出: rank_dim + suit_dim = 24维

    def forward(self, ranks: Tensor, suits: Tensor) -> Tensor:
        """
        Args:
            ranks: (batch, num_cards) 牌值
            suits: (batch, num_cards) 花色

        Returns:
            (batch, num_cards, 24) 牌的embedding
        """
        rank_emb = self.rank_embed(ranks)  # (batch, num_cards, 16)
        suit_emb = self.suit_embed(suits)  # (batch, num_cards, 8)
        return torch.cat([rank_emb, suit_emb], dim=-1)
```

### 3.3 手牌处理

**排序规则**: 按rank降序排列，消除顺序影响

```python
def normalize_hole_cards(card1: int, card2: int) -> tuple[int, int]:
    """标准化手牌顺序 (大牌在前)"""
    rank1, rank2 = card1 // 4, card2 // 4
    if rank1 >= rank2:
        return (card1, card2)
    return (card2, card1)
```

### 3.4 公共牌处理

- Flop: 3张牌 + 2张padding
- Turn: 4张牌 + 1张padding
- River: 5张牌

```python
def pad_board(board: list[int], max_len: int = 5) -> list[int]:
    """将公共牌补齐到5张"""
    padded = board.copy()
    while len(padded) < max_len:
        padded.append(CARD_PADDING)  # 52 = padding值
    return padded
```

### 3.5 牌面特征维度

| 组件 | 数量 | 单张维度 | 总维度 |
|------|------|----------|--------|
| 手牌 | 2 | 24 | 48 |
| 公共牌 | 5 | 24 | 120 |
| **合计** | | | **168** |

---

## 4. 游戏状态编码

### 4.1 自身信息 (14维)

| 特征 | 维度 | 编码方式 | 范围 |
|------|------|----------|------|
| position | 6 | one-hot | UTG/HJ/CO/BTN/SB/BB |
| stack | 1 | 归一化 | stack / 100 |
| pot | 1 | 归一化 | pot / 100 |
| to_call | 1 | 归一化 | to_call / 100 |
| street | 4 | one-hot | PREFLOP/FLOP/TURN/RIVER |
| raise_count | 1 | 归一化 | count / 4 |

```python
def encode_self_info(state: dict) -> Tensor:
    """编码自身信息 → 14维"""
    features = []

    # 位置 one-hot (6维)
    position = torch.zeros(6)
    position[state['my_position']] = 1.0
    features.append(position)

    # 数值特征 (4维)
    features.append(torch.tensor([
        state['my_stack'] / 100.0,
        state['pot'] / 100.0,
        state['to_call'] / 100.0,
        state['raise_count'] / 4.0,
    ]))

    # 街道 one-hot (4维)
    street = torch.zeros(4)
    street[state['street']] = 1.0
    features.append(street)

    return torch.cat(features)  # 14维
```

### 4.2 对手信息 (15维)

每个对手3个特征，共5个对手：

| 特征 | 维度 | 编码方式 |
|------|------|----------|
| is_active | 1 | 0/1 |
| stack | 1 | stack / 100 |
| invested | 1 | bet_total / 100 |

```python
def encode_opponents(opponents: list[dict]) -> Tensor:
    """编码5个对手信息 → 15维"""
    features = []
    for opp in opponents:
        features.extend([
            1.0 if opp['is_active'] else 0.0,
            opp['stack'] / 100.0,
            opp['invested'] / 100.0,
        ])
    return torch.tensor(features)  # 15维
```

---

## 5. 动作历史编码

### 5.1 单个动作Token (17维)

| 字段 | 维度 | 编码方式 |
|------|------|----------|
| player_id | 6 | one-hot |
| action_type | 6 | one-hot |
| amount | 1 | amount / 100 |
| street | 4 | one-hot |

```python
def encode_single_action(
    player: int,
    action: ActionType,
    amount: float,
    street: Street,
) -> Tensor:
    """编码单个动作 → 17维"""
    features = []

    # 玩家ID one-hot (6维)
    player_vec = torch.zeros(6)
    player_vec[player] = 1.0
    features.append(player_vec)

    # 动作类型 one-hot (6维)
    action_vec = torch.zeros(6)
    action_vec[action] = 1.0
    features.append(action_vec)

    # 金额 (1维)
    features.append(torch.tensor([amount / 100.0]))

    # 街道 one-hot (4维)
    street_vec = torch.zeros(4)
    street_vec[street] = 1.0
    features.append(street_vec)

    return torch.cat(features)  # 17维
```

### 5.2 ActionHistoryEncoder 类

使用单层Transformer编码动作序列：

```python
class ActionHistoryEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 17,
        hidden_dim: int = 64,
        num_heads: int = 4,
        max_len: int = 24,
    ):
        super().__init__()
        self.max_len = max_len

        # 投影到hidden_dim
        self.proj = nn.Linear(input_dim, hidden_dim)

        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_len, hidden_dim) * 0.02
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, actions: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            actions: (batch, seq_len, 17) 动作序列
            mask: (batch, seq_len) padding mask (True=padding)

        Returns:
            (batch, 64) 动作历史编码
        """
        # 处理全部是 padding 的情况
        if mask is not None and mask.all():
            return torch.zeros(actions.size(0), 64, device=actions.device)

        # 投影
        x = self.proj(actions)

        # 加位置编码
        x = x + self.pos_embed[:, :x.size(1), :]

        # Transformer (带mask)
        x = self.transformer(x, src_key_padding_mask=mask)

        # Mean pooling (忽略padding)
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return x  # (batch, 64)
```

### 5.3 序列设计参数

| 参数 | 值 | 说明 |
|------|-----|------|
| max_len | 24 | 足够覆盖一手牌的所有动作 |
| input_dim | 17 | 单个动作token维度 |
| hidden_dim | 64 | Transformer隐藏维度 |
| num_heads | 4 | 注意力头数 |
| num_layers | 1 | Transformer层数 |
| 输出维度 | 64 | Mean pooling后 |

---

## 6. StateEncoder实现

### 6.1 完整实现

```python
class StateEncoder(nn.Module):
    """
    状态编码器

    输入: 游戏状态字典
    输出: 261维特征向量
    """

    def __init__(
        self,
        rank_embed_dim: int = 16,
        suit_embed_dim: int = 8,
        action_hidden_dim: int = 64,
        max_actions: int = 24,
    ):
        super().__init__()

        self.card_dim = rank_embed_dim + suit_embed_dim  # 24
        self.max_actions = max_actions

        # 牌面编码
        self.rank_embed = nn.Embedding(14, rank_embed_dim)
        self.suit_embed = nn.Embedding(5, suit_embed_dim)

        # 动作历史编码
        self.action_encoder = ActionHistoryEncoder(
            input_dim=17,
            hidden_dim=action_hidden_dim,
            max_len=max_actions,
        )

        # 输出维度计算
        self.output_dim = (
            2 * self.card_dim +      # 手牌: 48
            5 * self.card_dim +      # 公共牌: 120
            14 +                      # 自身信息
            15 +                      # 对手信息
            action_hidden_dim         # 动作历史: 64
        )  # 总计: 261

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Args:
            batch: 包含以下字段的字典
                - hole_ranks: (B, 2)
                - hole_suits: (B, 2)
                - board_ranks: (B, 5)
                - board_suits: (B, 5)
                - self_info: (B, 14)
                - opponent_info: (B, 15)
                - actions: (B, max_actions, 17)
                - action_mask: (B, max_actions)

        Returns:
            (B, 261) 编码后的状态
        """
        B = batch['hole_ranks'].size(0)
        features = []

        # 1. 手牌编码 (48维)
        hole_rank_emb = self.rank_embed(batch['hole_ranks'])
        hole_suit_emb = self.suit_embed(batch['hole_suits'])
        hole_emb = torch.cat([hole_rank_emb, hole_suit_emb], dim=-1)
        features.append(hole_emb.view(B, -1))

        # 2. 公共牌编码 (120维)
        board_rank_emb = self.rank_embed(batch['board_ranks'])
        board_suit_emb = self.suit_embed(batch['board_suits'])
        board_emb = torch.cat([board_rank_emb, board_suit_emb], dim=-1)
        features.append(board_emb.view(B, -1))

        # 3. 自身信息 (14维)
        features.append(batch['self_info'])

        # 4. 对手信息 (15维)
        features.append(batch['opponent_info'])

        # 5. 动作历史 (64维)
        action_encoding = self.action_encoder(
            batch['actions'],
            batch['action_mask'],
        )
        features.append(action_encoding)

        return torch.cat(features, dim=-1)  # (B, 261)

    def get_output_dim(self) -> int:
        return self.output_dim
```

### 6.2 参数统计

| 组件 | 参数量 |
|------|--------|
| rank_embed | 14 × 16 = 224 |
| suit_embed | 5 × 8 = 40 |
| action_proj | 17 × 64 = 1,088 |
| pos_embed | 24 × 64 = 1,536 |
| transformer | ~50K |
| **总计** | **~53K** |

---

## 7. 批量处理支持

### 7.1 StateBatchBuilder

用于从游戏状态构建神经网络输入：

```python
class StateBatchBuilder:
    """将游戏状态转换为神经网络输入"""

    def __init__(self, max_actions: int = 24):
        self.max_actions = max_actions

    def build_single(self, game: PokerGame, player_id: int) -> dict[str, Tensor]:
        """从游戏实例构建单个样本"""
        state = game.get_state_for_player(player_id)
        return self.build_from_dict(state)

    def build_from_dict(self, state: dict) -> dict[str, Tensor]:
        """从状态字典构建单个样本"""
        # ... 实现细节

    def build_batch(
        self,
        games: list[PokerGame],
        player_ids: list[int]
    ) -> dict[str, Tensor]:
        """构建批量样本"""
        samples = [
            self.build_single(game, pid)
            for game, pid in zip(games, player_ids)
        ]
        return self._collate(samples)
```

### 7.2 VectorizedEncoder

用于高效批量编码：

```python
class VectorizedEncoder:
    """向量化状态编码器"""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.encoder = StateEncoder().to(device)
        self.builder = StateBatchBuilder()

    @torch.no_grad()
    def encode_games(
        self,
        games: list[PokerGame],
        player_ids: list[int],
    ) -> Tensor:
        """批量编码游戏状态"""
        batch = self.builder.build_batch(games, player_ids)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return self.encoder(batch)

    @torch.no_grad()
    def encode_states(self, states: list[dict]) -> Tensor:
        """批量编码状态字典"""
        batch = self.builder.build_batch_from_dicts(states)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return self.encoder(batch)
```

### 7.3 预期性能

| Batch Size | 编码时间 | 说明 |
|------------|----------|------|
| 1 | ~0.5ms | 单次推理 |
| 256 | ~2ms | 小批量 |
| 4096 | ~10ms | 推荐训练batch |
| 16384 | ~35ms | 最大化GPU |

---

## 8. 文件结构

```
src/sixmax/encoding/
├── __init__.py          # 导出公共接口
├── state_encoder.py     # StateEncoder 主类
├── card_embedding.py    # CardEmbedding, 牌面处理函数
├── action_encoder.py    # ActionHistoryEncoder
└── batch_builder.py     # StateBatchBuilder, VectorizedEncoder

tests/
└── test_encoding.py     # 编码模块测试
```

### 导出接口

```python
# from sixmax.encoding import ...
__all__ = [
    # 主编码器
    "StateEncoder",
    "create_empty_batch",

    # 牌面编码
    "CardEmbedding",
    "card_to_rank_suit",
    "cards_to_tensors",
    "normalize_hole_cards",
    "pad_board",

    # 动作编码
    "ActionHistoryEncoder",
    "encode_single_action",
    "encode_action_sequence",

    # 批量构建
    "StateBatchBuilder",
    "VectorizedEncoder",

    # 常量
    "CARD_PADDING",
    "RANK_PADDING",
    "SUIT_PADDING",
    "ACTION_TOKEN_DIM",
    "MAX_ACTIONS",
]
```
