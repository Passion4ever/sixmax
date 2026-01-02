# 6max

6-Max 100BB Exploitative Texas Hold'em AI

## 项目概述

构建一个能够在 6max 无限注德州扑克中击败业余和常客玩家的 AI 系统。

- **游戏类型**: 6-Max No-Limit Texas Hold'em
- **筹码深度**: 100BB
- **策略风格**: 剥削型 (非完美 GTO)
- **技术栈**: PyTorch 2.0+ / PPO

## 快速开始

```bash
# 安装依赖
uv sync

# 开发模式 (含测试工具)
uv sync --all-extras

# 运行测试
uv run pytest
```

## 项目结构

```
6max/
├── docs/
│   ├── DESIGN.md      # 整体设计
│   └── engine.md      # 游戏引擎详细设计
├── src/sixmax/
│   ├── engine/        # 游戏引擎
│   ├── encoding/      # 状态表示层
│   └── network/       # 神经网络
└── tests/
```

## 文档

- [设计文档](docs/DESIGN.md)
- [游戏引擎设计](docs/engine.md)
