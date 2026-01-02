"""Training utilities."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For full reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_linear_schedule(
    start_value: float,
    end_value: float,
    total_steps: int,
) -> callable:
    """Create a linear schedule function.

    Args:
        start_value: Initial value.
        end_value: Final value.
        total_steps: Total number of steps.

    Returns:
        Function that takes step and returns scheduled value.
    """

    def schedule(step: int) -> float:
        progress = min(1.0, step / max(1, total_steps))
        return start_value + (end_value - start_value) * progress

    return schedule


def get_exponential_schedule(
    start_value: float,
    end_value: float,
    decay_rate: float = 0.99,
) -> callable:
    """Create an exponential decay schedule function.

    Args:
        start_value: Initial value.
        end_value: Minimum value.
        decay_rate: Decay rate per step.

    Returns:
        Function that takes step and returns scheduled value.
    """

    def schedule(step: int) -> float:
        return max(end_value, start_value * (decay_rate**step))

    return schedule


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute explained variance.

    Explained variance measures how well predictions explain the variance
    in targets. Returns 1.0 for perfect predictions, 0.0 for mean prediction,
    negative for worse than mean.

    Args:
        y_pred: Predicted values.
        y_true: True values.

    Returns:
        Explained variance ratio.
    """
    var_y = torch.var(y_true)
    if var_y == 0:
        return float("nan")
    return float(1 - torch.var(y_true - y_pred) / var_y)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_number(n: int) -> str:
    """Format large numbers with K/M suffix.

    Args:
        n: Number to format.

    Returns:
        Formatted string.
    """
    if n < 1000:
        return str(n)
    elif n < 1_000_000:
        return f"{n / 1000:.1f}K"
    else:
        return f"{n / 1_000_000:.1f}M"


class RunningMeanStd:
    """Running mean and standard deviation.

    Useful for reward/observation normalization.
    """

    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        """Initialize running statistics.

        Args:
            epsilon: Small constant for numerical stability.
            shape: Shape of the values being tracked.
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new batch.

        Args:
            x: Batch of values (first dim is batch).
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        """Update from batch statistics."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize values using running statistics.

        Args:
            x: Values to normalize.

        Returns:
            Normalized values.
        """
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class MetricsTracker:
    """Track and aggregate training metrics."""

    def __init__(self, window_size: int = 100):
        """Initialize metrics tracker.

        Args:
            window_size: Size of moving window for averaging.
        """
        self.window_size = window_size
        self.metrics: dict[str, list[float]] = {}

    def update(self, metrics: dict[str, float]) -> None:
        """Update with new metrics.

        Args:
            metrics: Dictionary of metric name -> value.
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

            # Keep only window_size recent values
            if len(self.metrics[key]) > self.window_size:
                self.metrics[key] = self.metrics[key][-self.window_size :]

    def get_mean(self, key: str) -> float:
        """Get mean of a metric over the window.

        Args:
            key: Metric name.

        Returns:
            Mean value or 0 if not found.
        """
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return sum(self.metrics[key]) / len(self.metrics[key])

    def get_all_means(self) -> dict[str, float]:
        """Get means of all tracked metrics.

        Returns:
            Dictionary of metric name -> mean value.
        """
        return {key: self.get_mean(key) for key in self.metrics}

    def reset(self) -> None:
        """Clear all tracked metrics."""
        self.metrics.clear()
