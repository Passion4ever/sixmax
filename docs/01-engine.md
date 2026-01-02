# 模块 1: 游戏引擎详细设计

> 6max 100BB 德州扑克游戏引擎
>
> **状态: ✅ 已完成** | 模块顺序: 1/3 | 依赖: 无

---

## 目录

1. [整体架构](#1-整体架构)
2. [核心数据结构](#2-核心数据结构)
3. [游戏流程](#3-游戏流程)
4. [动作处理](#4-动作处理)
5. [底池管理](#5-底池管理)
6. [手牌评估](#6-手牌评估)
7. [核心API](#7-核心api)
8. [文件结构](#8-文件结构)

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           游戏引擎架构                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         PokerGame (主控)                             │  │
│   │  - 管理游戏流程                                                      │  │
│   │  - 协调各子模块                                                      │  │
│   │  - 提供对外API                                                       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│         │              │              │              │                      │
│         ▼              ▼              ▼              ▼                      │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│   │   Deck   │  │  State   │  │  PotMgr  │  │  Hand    │                   │
│   │  发牌器  │  │  状态    │  │ 底池管理 │  │  评估器  │                   │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 模块职责

| 模块 | 文件 | 职责 | 状态 |
|------|------|------|------|
| PokerGame | `game.py` | 主控制器，管理游戏流程，提供对外API | ✅ |
| GameState | `state.py` | 游戏状态数据结构 | ✅ |
| Actions | `actions.py` | 动作处理、合法性验证 | ✅ |
| PotMgr | `pot.py` | 底池计算、边池处理 | ✅ |
| HandEvaluator | `evaluator.py` | 手牌评估、比牌 | ✅ |

---

## 2. 核心数据结构

### 2.1 卡牌表示

```python
# 单张牌: 整数 0-51
# card = rank * 4 + suit
# rank: 0-12 (2,3,4,5,6,7,8,9,T,J,Q,K,A)
# suit: 0-3 (♠,♥,♦,♣)

# 示例
# A♠ = 12 * 4 + 0 = 48
# 2♣ = 0 * 4 + 3 = 3
```

### 2.2 玩家状态

```python
@dataclass
class PlayerState:
    """单个玩家状态"""
    seat: int                           # 座位号 0-5
    stack: float                        # 当前筹码 (BB单位)
    bet_this_street: float              # 本街已下注金额
    bet_total: float                    # 本手已投入总金额
    is_active: bool                     # 是否还在本手牌中 (未fold)
    is_allin: bool                      # 是否已all-in
    hole_cards: tuple[int, int] | None  # 手牌 (2张)
```

### 2.3 游戏状态

```python
class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

class Position(IntEnum):
    """6max位置 (相对于按钮位)"""
    UTG = 0  # Under the Gun
    HJ = 1   # Hijack
    CO = 2   # Cutoff
    BTN = 3  # Button
    SB = 4   # Small Blind
    BB = 5   # Big Blind

@dataclass
class GameState:
    """完整游戏状态"""
    players: list[PlayerState]  # 6个玩家
    street: Street              # 当前街道
    current_player: int         # 当前行动玩家座位号
    button_seat: int            # 按钮位座位号
    pot: float                  # 主池
    side_pots: list[float]      # 边池列表
    current_bet: float          # 本街当前最大下注
    min_raise: float            # 最小加注额
    board: list[int]            # 公共牌 (0-5张)
    actions: list[ActionRecord] # 本手所有动作
    raise_count: int            # 本街加注次数
    deck: list[int]             # 牌堆 (仅内部使用)
    hand_over: bool             # 本手是否结束
```

### 2.4 动作记录

```python
class ActionType(IntEnum):
    """统一6维动作编码"""
    FOLD = 0
    CHECK_CALL = 1
    RAISE = 2       # 仅preflop
    RAISE_33 = 3    # 仅postflop
    RAISE_75 = 4    # 仅postflop
    ALLIN = 5

@dataclass
class ActionRecord:
    """动作记录"""
    player: int          # 玩家座位号
    action: ActionType   # 动作类型
    amount: float        # 实际金额 (BB单位)
    street: Street       # 发生在哪条街
```

---

## 3. 游戏流程

### 3.1 一手牌流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        一手牌流程                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. reset_hand()                                                │
│     └─ 收集盲注、洗牌、发手牌                                    │
│                                                                 │
│  2. PREFLOP                                                     │
│     └─ UTG先行动 → HJ → CO → BTN → SB → BB                      │
│     └─ 直到所有玩家行动完毕或只剩1人                             │
│                                                                 │
│  3. FLOP (如果≥2人)                                             │
│     └─ 发3张公共牌                                              │
│     └─ SB先行动 (或存活的第一个位置)                            │
│                                                                 │
│  4. TURN (如果≥2人)                                             │
│     └─ 发1张公共牌                                              │
│     └─ 同上                                                     │
│                                                                 │
│  5. RIVER (如果≥2人)                                            │
│     └─ 发1张公共牌                                              │
│     └─ 同上                                                     │
│                                                                 │
│  6. showdown()                                                  │
│     └─ 比牌、分配底池                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 行动顺序规则

```python
def get_action_order(street: Street, button: int, active_players: list[int]) -> list[int]:
    """获取行动顺序"""
    if street == Street.PREFLOP:
        # Preflop: UTG → HJ → CO → BTN → SB → BB
        order = [
            (button + 3) % 6,  # UTG
            (button + 4) % 6,  # HJ
            (button + 5) % 6,  # CO
            button,            # BTN
            (button + 1) % 6,  # SB
            (button + 2) % 6,  # BB
        ]
    else:
        # Postflop: SB → BB → UTG → HJ → CO → BTN
        order = [
            (button + 1) % 6,  # SB
            (button + 2) % 6,  # BB
            (button + 3) % 6,  # UTG
            (button + 4) % 6,  # HJ
            (button + 5) % 6,  # CO
            button,            # BTN
        ]

    # 过滤出仍在游戏中的玩家
    return [p for p in order if p in active_players]
```

---

## 4. 动作处理

### 4.1 动作合法性验证

```python
def get_legal_actions(state: GameState) -> list[bool]:
    """返回6维合法动作掩码 [FOLD, CHECK_CALL, RAISE, R33, R75, ALLIN]"""
    mask = [False] * 6
    player = state.players[state.current_player]
    to_call = state.current_bet - player.bet_this_street

    if state.street == Street.PREFLOP:
        # === Preflop 规则 ===
        if state.raise_count > 0:
            # 面对加注
            mask[ActionType.FOLD] = True
            mask[ActionType.CHECK_CALL] = True  # Call
            if state.raise_count < 3:
                mask[ActionType.RAISE] = True   # 可以re-raise
            mask[ActionType.ALLIN] = True
        else:
            # 无人加注
            if player.seat == (state.button_seat + 2) % 6:  # BB
                mask[ActionType.CHECK_CALL] = True  # Check
                mask[ActionType.RAISE] = True
                mask[ActionType.ALLIN] = True
            else:
                # 第一个入池 (open) - 不允许 Limp
                mask[ActionType.FOLD] = True
                mask[ActionType.RAISE] = True
                mask[ActionType.ALLIN] = True
    else:
        # === Postflop 规则 ===
        if to_call > 0:
            # 面对下注
            mask[ActionType.FOLD] = True
            mask[ActionType.CHECK_CALL] = True  # Call
            mask[ActionType.RAISE_33] = True
            mask[ActionType.RAISE_75] = True
            mask[ActionType.ALLIN] = True
        else:
            # 无人下注
            mask[ActionType.CHECK_CALL] = True  # Check
            mask[ActionType.RAISE_33] = True    # Bet 33%
            mask[ActionType.RAISE_75] = True    # Bet 75%
            mask[ActionType.ALLIN] = True

    # 筹码不足处理
    if player.stack <= to_call:
        mask = [False] * 6
        mask[ActionType.FOLD] = True
        mask[ActionType.ALLIN] = True

    return mask
```

### 4.2 加注尺度

**Preflop:**

```python
PREFLOP_RAISE_SIZES = [2.5, 9.0, 22.0, 100.0]  # Open, 3bet, 4bet, 5bet+

def get_preflop_raise_size(raise_count: int) -> float:
    return PREFLOP_RAISE_SIZES[min(raise_count, 3)]
```

**Postflop:**

```python
def calculate_postflop_bet_size(state: GameState, ratio: float) -> float:
    player = state.players[state.current_player]
    to_call = state.current_bet - player.bet_this_street

    if to_call == 0:
        # Bet: ratio × pot
        return ratio * state.pot
    else:
        # Raise: ratio × (pot + to_call)
        return ratio * (state.pot + to_call)
```

---

## 5. 底池管理

### 5.1 边池计算

```python
def calculate_side_pots(players: list[PlayerState]) -> list[tuple[float, list[int]]]:
    """
    计算边池

    Returns:
        [(pot_size, [eligible_players]), ...]
    """
    # 收集所有不同的all-in金额
    allin_amounts = sorted(set(
        p.bet_total for p in players if p.is_allin
    ))

    side_pots = []
    prev_amount = 0

    for amount in allin_amounts + [max(p.bet_total for p in players)]:
        layer = amount - prev_amount
        eligible = [p.seat for p in players if p.bet_total >= amount and p.is_active]
        pot_size = layer * len([p for p in players if p.bet_total >= amount])

        if pot_size > 0 and eligible:
            side_pots.append((pot_size, eligible))

        prev_amount = amount

    return side_pots
```

### 5.2 底池分配

```python
def distribute_pot(pot_amount: float, winners: list[int]) -> dict[int, float]:
    """
    分配底池给获胜者

    Returns:
        {player_seat: winnings}
    """
    share = pot_amount / len(winners)
    return {seat: share for seat in winners}
```

---

## 6. 手牌评估

### 6.1 牌型定义

```python
class HandRank(IntEnum):
    """牌型排名 (越大越强)"""
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
```

### 6.2 手牌评估器

```python
class HandEvaluator:
    """
    手牌评估器

    评估7张牌中最佳5张组合的牌力
    """

    def evaluate(self, hole_cards: tuple[int, int], board: list[int]) -> int:
        """
        评估最佳5张牌组合

        Args:
            hole_cards: 手牌 (2张)
            board: 公共牌 (3-5张)

        Returns:
            牌力值 (越大越强)
        """
        all_cards = list(hole_cards) + board
        best_rank = 0

        # 遍历所有5张牌组合
        for combo in combinations(all_cards, 5):
            rank = self._evaluate_5_cards(combo)
            best_rank = max(best_rank, rank)

        return best_rank

    def compare_hands(
        self,
        hands: list[tuple[int, int]],
        board: list[int]
    ) -> list[int]:
        """
        比较多手牌，返回获胜者座位列表
        """
        ...
```

---

## 7. 核心API

### 7.1 PokerGame 类

```python
class PokerGame:
    """
    德州扑克游戏引擎

    主要API:
    - reset_hand(): 开始新一手
    - step(action): 执行动作
    - get_state(): 获取当前状态
    - get_legal_actions(): 获取合法动作
    - get_state_for_player(): 获取玩家视角状态
    """

    def __init__(
        self,
        num_players: int = 6,
        starting_stack: float = 100.0,
        button_seat: int = 0,
    ):
        """
        初始化游戏

        Args:
            num_players: 玩家数量 (默认6)
            starting_stack: 起始筹码 (默认100BB)
            button_seat: 初始按钮位置 (默认0)
        """
        ...

    def seed(self, seed: int | None = None) -> None:
        """设置随机种子"""
        ...

    def reset_hand(self) -> GameState:
        """
        开始新一手牌

        Returns:
            初始游戏状态
        """
        ...

    def step(self, action: ActionType) -> tuple[GameState, float, bool]:
        """
        执行一个动作

        Args:
            action: 要执行的动作

        Returns:
            state: 新状态
            reward: 奖励 (本手结束时为收益)
            done: 是否结束
        """
        ...

    def get_state(self) -> GameState:
        """获取当前游戏状态"""
        ...

    def get_legal_actions(self) -> list[bool]:
        """获取当前玩家的合法动作掩码"""
        ...

    def get_state_for_player(self, player_id: int) -> dict:
        """
        获取特定玩家视角的状态 (隐藏其他玩家手牌)

        用于提供给神经网络

        Returns:
            {
                'hole_cards': (card1, card2),
                'board': [...],
                'my_position': int,
                'my_stack': float,
                'pot': float,
                'to_call': float,
                'street': int,
                'raise_count': int,
                'opponents': [{...}, ...],
                'actions': [...]
            }
        """
        ...
```

### 7.2 使用示例

```python
from sixmax.engine import PokerGame, ActionType

# 创建游戏
game = PokerGame()
game.seed(42)  # 可选：设置随机种子

# 开始新一手
game.reset_hand()

# 游戏循环
while not game.get_state().hand_over:
    # 获取合法动作
    legal = game.get_legal_actions()

    # 选择动作 (这里简单选择第一个合法动作)
    for action in ActionType:
        if legal[action]:
            break

    # 执行动作
    state, reward, done = game.step(action)

    if done:
        print(f"Hand over! Reward: {reward}")
```

---

## 8. 文件结构

```
src/sixmax/engine/
├── __init__.py      # 导出公共接口
├── game.py          # PokerGame 主类
├── state.py         # GameState, PlayerState, ActionType 等
├── actions.py       # 动作处理逻辑
├── pot.py           # 底池管理
└── evaluator.py     # HandEvaluator 手牌评估

tests/
├── test_engine.py   # 引擎功能测试
└── test_rules.py    # 规则验证测试
```

### 导出接口

```python
# from sixmax.engine import ...
__all__ = [
    "PokerGame",
    "GameState",
    "PlayerState",
    "ActionType",
    "ActionRecord",
    "Street",
    "Position",
]
```
