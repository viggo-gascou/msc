"""Base class for loss functions."""

import typing as t

import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    """Abstract base class for loss functions."""

    def __init__(self, weight: t.Optional[torch.Tensor] = None) -> None:
        """Initialize the loss function with an optional weight tensor."""
        super().__init__()
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss between the true and predicted values."""
        raise NotImplementedError("Implement this method")
