"""Utility functions for the network module."""

from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """
    统计模型可训练参数量。

    Args:
        model: PyTorch 模型

    Returns:
        可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """
    统计模型所有参数量（包括不可训练的）。

    Args:
        model: PyTorch 模型

    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters())


def create_policy_value_network(
    device: str | torch.device = "cpu",
    compile_model: bool = False,
) -> nn.Module:
    """
    创建默认配置的 PolicyValueNetwork。

    Args:
        device: 目标设备 ("cpu", "cuda", "cuda:0" 等)
        compile_model: 是否使用 torch.compile 优化（需要 PyTorch 2.0+）

    Returns:
        PolicyValueNetwork 实例
    """
    from .policy_value import PolicyValueNetwork

    network = PolicyValueNetwork()
    network = network.to(device)

    if compile_model:
        network = torch.compile(network)

    return network


def get_model_size_mb(model: nn.Module) -> float:
    """
    获取模型大小（MB）。

    Args:
        model: PyTorch 模型

    Returns:
        模型大小（MB）
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_model_summary(model: nn.Module, name: str = "Model") -> None:
    """
    打印模型摘要信息。

    Args:
        model: PyTorch 模型
        name: 模型名称
    """
    total_params = count_parameters(model)
    size_mb = get_model_size_mb(model)

    print(f"\n{'=' * 50}")
    print(f"{name} Summary")
    print(f"{'=' * 50}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Size: {size_mb:.2f} MB")
    print(f"{'=' * 50}\n")


def freeze_module(module: nn.Module) -> None:
    """
    冻结模块参数（不参与梯度计算）。

    Args:
        module: 要冻结的模块
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """
    解冻模块参数。

    Args:
        module: 要解冻的模块
    """
    for param in module.parameters():
        param.requires_grad = True


def get_device(model: nn.Module) -> torch.device:
    """
    获取模型所在设备。

    Args:
        model: PyTorch 模型

    Returns:
        模型所在的设备
    """
    return next(model.parameters()).device
