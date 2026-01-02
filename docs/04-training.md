# 模块 4: PPO 训练系统详细设计

> 自博弈 + PPO 强化学习训练框架
>
> **状态: 🔲 待实现** | 模块顺序: 4/5 | 依赖: engine, encoding, network

---

## 目录

1. [模块概述](#1-模块概述)
2. [训练架构](#2-训练架构)
3. [自博弈系统](#3-自博弈系统)
4. [PPO 算法](#4-ppo-算法)
5. [奖励设计](#5-奖励设计)
6. [模块结构](#6-模块结构)
7. [核心接口](#7-核心接口)
8. [运行环境](#8-运行环境)
9. [实现清单](#9-实现清单)

---

## 1. 模块概述

### 1.1 设计目标

- 实现 PPO 自博弈训练循环
- 支持单 GPU 同步训练（Phase 1）和双 GPU 异步训练（Phase 2）
- 集成 wandb 实验追踪
- 支持 SLURM 集群调度

### 1.2 在系统中的位置

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  游戏引擎   │────▶│ 状态表示层  │────▶│  神经网络   │
│  (Engine)   │     │ (Encoding)  │     │ (Network)   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌──────────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │    训练系统 (PPO)   │
         ├─────────────────────┤
         │  • 自博弈数据收集   │
         │  • PPO 策略更新     │
         │  • 经验缓冲区       │
         │  • wandb 日志       │
         └─────────────────────┘
```

### 1.3 渐进式开发路线

| 阶段 | 目标 | 特性 |
|------|------|------|
| Phase 1 | 验证算法 | 同步自博弈、稀疏奖励、单 GPU |
| Phase 2 | 性能优化 | 异步架构、历史对手池、双 GPU |
| Phase 3 | 高级特性 | TD(λ) 奖励、对手建模、EV 估计 |

---

## 2. 训练架构

### 2.1 Phase 1: 同步自博弈

```
┌────────────────────────────────────────────────────────┐
│                    单 GPU 同步训练                      │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│  │ 收集数据 │───▶│ 计算优势 │───▶│ PPO更新  │        │
│  │(16384局) │    │  (GAE)   │    │ (4epochs)│        │
│  └──────────┘    └──────────┘    └──────────┘        │
│       │                               │               │
│       └───────────── 循环 ────────────┘               │
│                                                        │
└────────────────────────────────────────────────────────┘
```

**特点**:
- 单进程，易调试
- 收集和训练交替进行
- GPU 利用率约 50%

### 2.2 Phase 2: 异步 Actor-Learner

```
┌────────────────────────────────────────────────────────┐
│                   双 GPU 异步训练                       │
├────────────────────────────────────────────────────────┤
│                                                        │
│  GPU 1 (推理)              GPU 2 (训练)               │
│  ┌──────────────┐          ┌──────────────┐           │
│  │   Actor 1    │          │              │           │
│  │   Actor 2    │──Queue──▶│   Learner    │           │
│  │   Actor N    │          │              │           │
│  └──────────────┘          └──────────────┘           │
│        │                          │                   │
│        └────── 定期同步权重 ──────┘                   │
│                                                        │
└────────────────────────────────────────────────────────┘
```

**特点**:
- 收集和训练并行
- GPU 利用率 > 90%
- 需要处理权重同步

---

## 3. 自博弈系统

### 3.1 Phase 1: 纯自博弈

```python
# 6 个玩家全部使用当前网络
players = [current_network] * 6
```

**优点**: 简单
**缺点**: 可能陷入循环策略

### 3.2 Phase 2: 混合对手池

```python
# 70% 使用当前网络，30% 使用历史版本
def select_opponents():
    opponents = []
    for _ in range(5):  # 5 个对手
        if random.random() < 0.7:
            opponents.append(current_network)
        else:
            opponents.append(random.choice(history_pool))
    return opponents
```

**对手池管理**:
- 每 10000 手保存一个版本
- 保留最近 10 个版本
- 可选: 基于 Elo 评分筛选

### 3.3 数据收集流程

```python
def collect_rollout(games, network, n_hands=2000):
    """
    收集自博弈数据。

    Args:
        games: 并行游戏实例 (16384 个)
        network: 当前策略网络
        n_hands: 收集手数

    Returns:
        buffer: 经验缓冲区
    """
    buffer = RolloutBuffer()
    hands_played = 0

    while hands_played < n_hands:
        # 1. 获取所有游戏的当前状态
        states = [g.get_state() for g in games]
        batch = build_batch(states)
        legal_masks = get_legal_masks(games)

        # 2. 批量推理选动作
        with torch.no_grad():
            actions, log_probs, values = network.get_action(batch, legal_masks)

        # 3. 执行动作
        for i, game in enumerate(games):
            reward = game.step(actions[i])
            done = game.is_hand_over()

            buffer.add(
                state=states[i],
                action=actions[i],
                reward=reward,
                value=values[i],
                log_prob=log_probs[i],
                done=done
            )

            if done:
                game.reset_hand()
                hands_played += 1

    return buffer
```

---

## 4. PPO 算法

### 4.1 损失函数

```python
def compute_ppo_loss(batch, network, clip_epsilon=0.2):
    """
    计算 PPO 损失。

    L = L_policy + c1 * L_value - c2 * L_entropy
    """
    # 重新评估动作
    new_log_probs, new_values, entropy = network.evaluate_actions(
        batch.states, batch.legal_masks, batch.actions
    )

    # 策略损失 (PPO-Clip)
    ratio = torch.exp(new_log_probs - batch.old_log_probs)
    surr1 = ratio * batch.advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch.advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 价值损失
    value_loss = F.mse_loss(new_values.squeeze(), batch.returns)

    # 熵正则化
    entropy_loss = -entropy.mean()

    # 总损失
    total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

    return total_loss, {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.mean().item(),
    }
```

### 4.2 GAE 优势估计

```python
def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """
    计算 Generalized Advantage Estimation。

    A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    δ_t = r_t + γV(s_{t+1}) - V(s_t)
    """
    advantages = torch.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae

    returns = advantages + values
    return advantages, returns
```

### 4.3 超参数配置

| 参数 | 初始值 | 说明 |
|------|--------|------|
| learning_rate | 3e-4 | Adam 学习率 |
| clip_epsilon | 0.2 | PPO 裁剪范围 |
| value_coef | 0.5 | 价值损失权重 |
| entropy_coef | 0.01 | 熵正则化系数 |
| gamma | 0.99 | 折扣因子 |
| gae_lambda | 0.95 | GAE 参数 |
| n_epochs | 4 | 每批数据训练轮数 |
| batch_size | 4096 | 小批量大小 |
| max_grad_norm | 0.5 | 梯度裁剪 |
| n_games | 16384 | 并行游戏数 |
| n_hands_per_update | 2000 | 每次更新的手数 |

---

## 5. 奖励设计

### 5.1 Phase 1: 稀疏奖励

```python
def compute_reward(game, player_id):
    """
    手牌结束时的筹码变化 (BB 单位)。
    """
    if not game.is_hand_over():
        return 0.0

    initial_stack = 100.0  # 100 BB
    final_stack = game.get_player_stack(player_id)
    return final_stack - initial_stack
```

**特点**:
- 简单，无人为偏差
- 方差大（好决策可能因运气得负奖励）
- 依赖 GAE 和大样本量

### 5.2 Phase 2: TD Bootstrap

```python
def compute_td_reward(game, player_id, value_network):
    """
    使用价值网络降低方差。
    """
    if game.is_hand_over():
        return game.get_player_stack(player_id) - 100.0
    else:
        # 用 V(s') 估计未来收益
        next_state = game.get_state_for_player(player_id)
        with torch.no_grad():
            next_value = value_network.get_value(next_state)
        return gamma * next_value
```

### 5.3 Phase 3: EV 估计

配合对手建模，更精确估计动作的期望值。

---

## 6. 模块结构

```
src/sixmax/training/
├── __init__.py
├── buffer.py           # 经验缓冲区
├── ppo.py              # PPO 算法核心
├── rollout.py          # 自博弈数据收集
├── trainer.py          # 训练主循环
├── config.py           # 超参数配置
└── utils.py            # wandb 日志、检查点

scripts/
├── train.py            # 训练入口
├── slurm_train.sh      # SLURM 提交脚本
└── eval.py             # 模型评估
```

### 6.1 各模块职责

| 模块 | 职责 |
|------|------|
| buffer.py | 存储轨迹 (s,a,r,v,log_p)，计算 GAE/Returns |
| ppo.py | PPO 损失计算，网络更新 |
| rollout.py | 批量运行游戏，收集经验 |
| trainer.py | 主循环：收集 → GAE → 更新 → 日志 |
| config.py | 所有超参数集中管理 |
| utils.py | wandb 初始化、模型保存/加载 |

---

## 7. 核心接口

### 7.1 RolloutBuffer

```python
class RolloutBuffer:
    """经验缓冲区。"""

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        """清空缓冲区。"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, state, action, reward, value, log_prob, done):
        """添加一条经验。"""
        ...

    def compute_returns_and_advantages(self, last_value: float):
        """计算 GAE 优势和回报。"""
        ...

    def get_batches(self, batch_size: int):
        """生成训练小批量。"""
        ...
```

### 7.2 PPOTrainer

```python
class PPOTrainer:
    """PPO 训练器。"""

    def __init__(
        self,
        network: PolicyValueNetwork,
        config: TrainingConfig,
    ):
        self.network = network
        self.config = config
        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.learning_rate
        )

    def collect_rollouts(self, games: list, n_hands: int) -> RolloutBuffer:
        """收集自博弈数据。"""
        ...

    def update(self, buffer: RolloutBuffer) -> dict:
        """执行 PPO 更新，返回日志指标。"""
        ...

    def train(self, total_hands: int):
        """训练主循环。"""
        ...

    def save_checkpoint(self, path: str):
        """保存检查点。"""
        ...

    def load_checkpoint(self, path: str):
        """加载检查点。"""
        ...
```

### 7.3 TrainingConfig

```python
@dataclass
class TrainingConfig:
    """训练超参数配置。"""

    # PPO 参数
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_epochs: int = 4
    batch_size: int = 4096
    max_grad_norm: float = 0.5

    # 训练配置
    n_games: int = 16384
    n_hands_per_update: int = 2000
    total_hands: int = 10_000_000

    # 日志和检查点
    log_interval: int = 1000
    save_interval: int = 10000
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "6max-poker"
```

---

## 8. 运行环境

### 8.1 依赖

```toml
# pyproject.toml 新增依赖
[project.optional-dependencies]
training = [
    "wandb>=0.15.0",
    "optuna>=3.0.0",  # Phase 2: 超参数优化
]
```

### 8.2 SLURM 提交脚本

```bash
#!/bin/bash
#SBATCH --job-name=6max-train
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# 激活环境
source /path/to/venv/bin/activate

# 设置 GPU 可见性 (根据分配自动设置)
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

# wandb 离线模式 (可选)
# export WANDB_MODE=offline

# 启动训练
python scripts/train.py \
    --config configs/phase1.yaml \
    --wandb-project 6max-poker \
    --wandb-name "phase1-run-${SLURM_JOB_ID}"
```

### 8.3 训练启动

```bash
# 本地测试
python scripts/train.py --config configs/phase1.yaml

# SLURM 提交
sbatch scripts/slurm_train.sh

# 查看 wandb 面板
# https://wandb.ai/<username>/6max-poker
```

---

## 9. 实现清单

| 组件 | 优先级 | 状态 |
|------|--------|------|
| TrainingConfig | P0 | 🔲 待实现 |
| RolloutBuffer | P0 | 🔲 待实现 |
| PPO 损失计算 | P0 | 🔲 待实现 |
| GAE 计算 | P0 | 🔲 待实现 |
| 自博弈收集 | P0 | 🔲 待实现 |
| PPOTrainer | P0 | 🔲 待实现 |
| wandb 集成 | P0 | 🔲 待实现 |
| 检查点保存/加载 | P0 | 🔲 待实现 |
| train.py 入口 | P0 | 🔲 待实现 |
| slurm_train.sh | P0 | 🔲 待实现 |
| 单元测试 | P0 | 🔲 待实现 |
| 历史对手池 | P1 | 🔲 后续 |
| 异步 Actor-Learner | P1 | 🔲 后续 |
| Optuna 超参数优化 | P1 | 🔲 后续 |
| TD(λ) 奖励 | P2 | 🔲 后续 |

---

## 附录

### A. wandb 日志指标

| 指标 | 说明 |
|------|------|
| train/policy_loss | 策略损失 |
| train/value_loss | 价值损失 |
| train/entropy | 策略熵 |
| train/clip_fraction | PPO 裁剪比例 |
| train/learning_rate | 当前学习率 |
| rollout/hands | 已收集手数 |
| rollout/reward_mean | 平均每手奖励 |
| rollout/reward_std | 奖励标准差 |
| eval/win_rate | 评估胜率 |
| eval/bb_per_100 | BB/100 手 |

### B. 检查点格式

```python
checkpoint = {
    'step': current_step,
    'hands': total_hands_played,
    'network_state_dict': network.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config.__dict__,
    'wandb_run_id': wandb.run.id,
}
torch.save(checkpoint, path)
```

### C. 参考资料

- [PPO Paper (Schulman et al.)](https://arxiv.org/abs/1707.06347)
- [GAE Paper](https://arxiv.org/abs/1506.02438)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/)
