# Texas Hold'em AI 项目设计文档

> 6max 100BB 剥削型德州扑克AI

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [动作空间设计](#3-动作空间设计)
4. [状态表示层设计](#4-状态表示层设计)
5. [神经网络架构](#5-神经网络架构)
6. [训练系统 (PPO)](#6-训练系统-ppo)
7. [游戏引擎](#7-游戏引擎)
8. [对手建模](#8-对手建模)
9. [开发计划](#9-开发计划)

---

## 1. 项目概述

### 1.1 项目目标

构建一个能够在6max无限注德州扑克中击败业余和常客玩家的AI系统。

| 维度 | 目标 |
|------|------|
| 游戏类型 | 6-Max No-Limit Texas Hold'em |
| 筹码深度 | 100BB |
| 策略风格 | 剥削型 (非完美GTO) |
| 应用场景 | 本地实时辅助/策略研究 |

### 1.2 硬件环境

| 组件 | 配置 |
|------|------|
| GPU | 1-4张 NVIDIA A100 (80GB) |
| CPU | 普通服务器CPU (非重点) |
| 内存 | 足够 |

**设计原则**: 最大化GPU利用率，最小化CPU依赖。

### 1.3 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| 深度学习框架 | PyTorch 2.0+ | A100优化好，torch.compile支持 |
| 训练算法 | PPO | GPU友好，稳定，适合剥削 |
| 游戏引擎 | 自实现 | 定制动作空间，完全控制 |
| 状态编码 | Embedding + Transformer | 端到端学习，GPU密集 |

**为什么不用CFR?**
- CFR是CPU密集型（树遍历、递归）
- 我们的CPU弱、GPU强，CFR不匹配
- PPO/NFSP等纯深度学习方法GPU利用率可达90%+

### 1.4 设计原则

1. **GPU优先**: 所有计算尽量在GPU上完成
2. **端到端学习**: 不做人工抽象，让网络自己学习
3. **简洁高效**: 控制模型复杂度，快速迭代
4. **剥削导向**: 目标是击败弱玩家，不追求完美GTO

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           系统架构                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐             │
│   │  模块1      │     │  模块2      │     │  模块3      │             │
│   │  游戏引擎   │────▶│  状态表示层 │────▶│  神经网络   │             │
│   │  (裁判)     │     │  (翻译官)   │     │  (大脑)     │             │
│   └─────────────┘     └─────────────┘     └─────────────┘             │
│         │                   │                   │                      │
│         │                   │                   │                      │
│         ▼                   ▼                   ▼                      │
│   ┌─────────────────────────────────────────────────────┐             │
│   │                 模块4: 训练系统 (PPO)                │             │
│   └─────────────────────────────────────────────────────┘             │
│                             │                                          │
│                             ▼                                          │
│   ┌─────────────────────────────────────────────────────┐             │
│   │                 模块5: 对手建模 (剥削)               │             │
│   └─────────────────────────────────────────────────────┘             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块划分

| # | 模块 | 职责 | 状态 | 详细文档 |
|---|------|------|------|----------|
| 1 | 游戏引擎 | 规则、发牌、比牌、底池管理 | ✅ 已完成 | [01-engine.md](./01-engine.md) |
| 2 | 状态表示层 | 将游戏状态编码为张量 | ✅ 已完成 | [02-encoding.md](./02-encoding.md) |
| 3 | 神经网络 | 策略输出、价值评估 | ✅ 已完成 | [03-network.md](./03-network.md) |
| 4 | 训练系统 | PPO算法、自博弈 | 🔲 待实现 | [04-training.md](./04-training.md) |
| 5 | 对手建模 | 识别对手类型、调整策略 | 🔲 待实现 | - |

---

## 3. 动作空间设计

### 3.1 Preflop动作 (4种)

| ID | 动作 | 说明 |
|----|------|------|
| 0 | FOLD | 弃牌 |
| 1 | CHECK_CALL | BB位无人加注时Check；其他情况Call |
| 2 | RAISE | 加注，尺度根据raise_count自动确定 |
| 5 | ALLIN | 全压 (100BB) |

**注意**: Preflop 不允许 Limp（直接跟注大盲），必须加注或弃牌。

**RAISE尺度规则:**

| raise_count | 场景 | 尺度 |
|-------------|------|------|
| 0 | Open (第一个加注) | 2.5 BB |
| 1 | 3-Bet | 9 BB |
| 2 | 4-Bet | 22 BB |
| ≥3 | 5-Bet+ | All-in (仅可ALLIN) |

### 3.2 Postflop动作 (5种)

| ID | 动作 | 下注(Bet) | 加注(Raise) |
|----|------|-----------|-------------|
| 0 | FOLD | - | 弃牌 |
| 1 | CHECK_CALL | 过牌 | 跟注 |
| 3 | RAISE_33 | 33% pot | 50% pot |
| 4 | RAISE_75 | 75% pot | 75% pot |
| 5 | ALLIN | 全压 | 全压 |

**尺度计算公式:**
```
bet_amount = ratio × pot            # 无人下注时 (R33=33%, R75=75%)
raise_amount = effective_ratio × pot  # 有人下注时 (R33→50%, R75不变)
```

### 3.3 统一编码 (6维)

为了神经网络处理方便，统一编码为6维:

| ID | 动作 | 翻前可用 | 翻后可用 |
|----|------|----------|----------|
| 0 | FOLD | ✓ | ✓ |
| 1 | CHECK_CALL | ✓ | ✓ |
| 2 | RAISE | ✓ | ✗ |
| 3 | RAISE_33 | ✗ | ✓ |
| 4 | RAISE_75 | ✗ | ✓ |
| 5 | ALLIN | ✓ | ✓ |

### 3.4 动作合法性掩码

**Preflop掩码:**

| 场景 | 掩码 [F,C,R,R33,R75,AI] | 可用动作 |
|------|-------------------------|----------|
| 无人入池(非BB) | [1,0,1,0,0,1] | FOLD/RAISE/ALLIN |
| BB且无人加注 | [0,1,1,0,0,1] | CHECK/RAISE/ALLIN |
| 面对加注 | [1,1,1,0,0,1] | FOLD/CALL/RAISE/ALLIN |
| 面对4bet+ | [1,1,0,0,0,1] | FOLD/CALL/ALLIN |

**Postflop掩码:**

| 场景 | 掩码 [F,C,R,R33,R75,AI] | 可用动作 |
|------|-------------------------|----------|
| 无人下注 | [0,1,0,1,1,1] | CHECK/BET33/BET75/ALLIN |
| 面对下注 | [1,1,0,1,1,1] | FOLD/CALL/R33/R75/ALLIN |

### 3.5 边界处理

- 若计算尺度 < min_raise → 使用 min_raise
- 若计算尺度 > 剩余筹码 → 变成 All-in

---

## 4. 状态表示层设计

> 详细设计见 [02-encoding.md](./02-encoding.md)

### 4.1 设计目标

将游戏状态编码为神经网络可处理的 **261维** 特征向量。

### 4.2 编码组成

| 模块 | 维度 | 说明 |
|------|------|------|
| 手牌 | 48 | 2张 × 24维 (Embedding) |
| 公共牌 | 120 | 5张 × 24维 (Embedding) |
| 游戏状态 | 29 | 位置/筹码/底池等 |
| 动作历史 | 64 | Transformer编码 |
| **总计** | **261** | |

### 4.3 核心组件

```python
class StateEncoder(nn.Module):
    def forward(self, batch: dict) -> Tensor:
        """
        输入: 游戏状态字典
        输出: (batch, 261) 特征向量
        """
        ...
```

### 4.4 与其他模块的接口

```
游戏引擎 ──get_state_for_player()──▶ StateBatchBuilder ──▶ StateEncoder ──▶ 神经网络
                                         (构建batch)        (261维)
```

---

## 5. 神经网络架构

> 详细设计见 [03-network.md](./03-network.md)

### 5.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        神经网络架构                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   输入: 手牌 + 公共牌 + 游戏状态 + 动作历史                              │
│                          │                                              │
│                          ▼                                              │
│                   ┌─────────────┐                                       │
│                   │ StateEncoder│ → 261维                               │
│                   └──────┬──────┘                                       │
│                          │                                              │
│                          ▼                                              │
│                   ┌─────────────┐                                       │
│                   │  Backbone   │ → 256维                               │
│                   │ (MLP+残差)  │                                       │
│                   └──────┬──────┘                                       │
│                          │                                              │
│              ┌───────────┴───────────┐                                  │
│              │                       │                                  │
│              ▼                       ▼                                  │
│       ┌─────────────┐         ┌─────────────┐                          │
│       │ Policy Head │         │ Value Head  │                          │
│       │   (Actor)   │         │  (Critic)   │                          │
│       └──────┬──────┘         └──────┬──────┘                          │
│              │                       │                                  │
│              ▼                       ▼                                  │
│         6维概率分布              1维状态价值                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 网络参数

| 组件 | 结构 | 参数量 |
|------|------|--------|
| StateEncoder | Embedding + Transformer | ~51K |
| Backbone | 261→512→512(残差)→256 | ~531K |
| PolicyHead | 256→128→6 | ~33K |
| ValueHead | 256→128→1 | ~33K |
| **总计** | | **~648K** |

### 5.3 设计要点

- LayerNorm代替BatchNorm（RL更稳定）
- 残差连接（防止梯度消失）
- 正交初始化（PPO最佳实践）
- Masked Softmax（处理非法动作）

### 5.4 A100性能预估

| 指标 | 预估值 |
|------|--------|
| 单次前向 | < 1ms |
| Batch 4096 | ~10ms |
| 建议batch_size | 4096 ~ 16384 |

---

## 6. 训练系统 (PPO)

> 🔲 待设计

### 6.1 PPO接口 (已定义)

```python
# PolicyValueNetwork 提供的接口
network.forward(batch, legal_mask)      # → (probs, value, logits)
network.get_action(batch, legal_mask)   # → (action, log_prob, value)
network.evaluate_actions(batch, legal_mask, actions)  # → (log_prob, value, entropy)
```

### 6.2 待实现内容

- [ ] PPO 训练循环
- [ ] GAE 优势估计
- [ ] 经验回放缓冲区
- [ ] 自博弈对战池
- [ ] 训练日志和可视化

---

## 7. 游戏引擎

> 详细设计见 [01-engine.md](./01-engine.md)

### 7.1 设计目标

- 实现标准 6-Max 无限注德州扑克规则
- 支持 100BB 筹码深度
- 提供 Gym 风格的 API (`reset`, `step`, `get_legal_actions`)
- 支持批量并行模拟 (4096+ 游戏)

### 7.2 核心模块

| 模块 | 职责 | 状态 |
|------|------|------|
| `PokerGame` | 主控制器，管理游戏流程 | ✅ 已完成 |
| `GameState` | 游戏状态数据结构 | ✅ 已完成 |
| `ActionType` | 动作类型定义 | ✅ 已完成 |
| `HandEvaluator` | 手牌评估、比牌 | ✅ 已完成 |
| `PotManager` | 底池计算、边池处理 | ✅ 已完成 |

### 7.3 核心API

```python
class PokerGame:
    def reset_hand(self) -> GameState: ...
    def step(self, action: ActionType) -> tuple[GameState, float, bool]: ...
    def get_legal_actions(self) -> list[bool]: ...
    def get_state_for_player(self, player_id: int) -> dict: ...
```

---

## 8. 对手建模

> 🔲 待设计 (后期扩展)

---

## 9. 开发计划

### 9.1 已完成

| 阶段 | 任务 | 状态 |
|------|------|------|
| Phase 1 | 游戏引擎 | ✅ 完成 |
| Phase 2 | 状态表示层 | ✅ 完成 |
| Phase 3 | 神经网络 | ✅ 完成 |

### 9.2 进行中/待开始

| 阶段 | 任务 | 状态 |
|------|------|------|
| Phase 4 | PPO训练系统 | 🔲 待开始 |
| Phase 5 | 对手建模 | 🔲 待开始 |
| Phase 6 | 实战测试 | 🔲 待开始 |

### 9.3 测试覆盖

| 模块 | 测试文件 | 覆盖率 |
|------|----------|--------|
| engine | test_engine.py, test_rules.py | 93% |
| encoding | test_encoding.py | 98% |
| network | test_network.py | 100% |
| **总计** | 138 tests | **95%** |

---

## 附录

### A. 开发环境

**包管理**: 使用 [uv](https://github.com/astral-sh/uv) 管理 Python 环境

```bash
# 安装依赖
uv sync

# 安装开发依赖
uv sync --all-extras

# 运行测试
uv run pytest

# 运行测试（带覆盖率）
uv run pytest --cov=sixmax

# 运行代码检查
uv run ruff check src/
uv run mypy src/
```

### B. 项目结构

```
6max/
├── pyproject.toml          # 项目配置 (uv)
├── docs/
│   ├── DESIGN.md           # 整体设计文档 (本文件)
│   ├── 01-engine.md        # 模块1: 游戏引擎详细设计
│   ├── 02-encoding.md      # 模块2: 状态表示层详细设计
│   ├── 03-network.md       # 模块3: 神经网络详细设计
│   └── 04-training.md      # 模块4: PPO训练系统详细设计
├── src/
│   └── sixmax/
│       ├── __init__.py
│       ├── engine/         # 游戏引擎
│       │   ├── __init__.py
│       │   ├── game.py     # PokerGame 主类
│       │   ├── state.py    # 状态数据结构
│       │   ├── actions.py  # 动作处理
│       │   ├── pot.py      # 底池管理
│       │   └── evaluator.py # 手牌评估
│       ├── encoding/       # 状态表示层
│       │   ├── __init__.py
│       │   ├── state_encoder.py  # StateEncoder
│       │   ├── card_embedding.py # 牌面编码
│       │   ├── action_encoder.py # 动作历史编码
│       │   └── batch_builder.py  # 批量构建
│       └── network/        # 神经网络
│           ├── __init__.py
│           ├── backbone.py      # Backbone 网络
│           ├── heads.py         # PolicyHead, ValueHead
│           ├── policy_value.py  # PolicyValueNetwork
│           └── utils.py         # 辅助函数
└── tests/
    ├── test_engine.py      # 引擎测试
    ├── test_rules.py       # 规则验证测试
    ├── test_encoding.py    # 编码测试
    └── test_network.py     # 网络测试
```

### C. 文档索引

| 文档 | 内容 | 状态 |
|------|------|------|
| [DESIGN.md](./DESIGN.md) | 整体架构和设计思路 | ✅ 最新 |
| [01-engine.md](./01-engine.md) | 模块1: 游戏引擎详细设计 | ✅ 最新 |
| [02-encoding.md](./02-encoding.md) | 模块2: 状态表示层详细设计 | ✅ 最新 |
| [03-network.md](./03-network.md) | 模块3: 神经网络详细设计 | ✅ 最新 |
| [04-training.md](./04-training.md) | 模块4: PPO训练系统详细设计 | ✅ 最新 |

### D. 参考资料

- [AlphaHoldem (AAAI 2022)](https://ojs.aaai.org/index.php/AAAI/article/view/20394)
- [PokerRL GitHub](https://github.com/EricSteinberger/PokerRL)
- [RLCard Documentation](https://rlcard.org/)
- [DeepStack Paper](https://arxiv.org/pdf/1701.01724)
- [PettingZoo Texas Hold'em](https://pettingzoo.farama.org/environments/classic/texas_holdem_no_limit/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/)
