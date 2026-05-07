from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CNNConfig:
    in_channels: int = 3
    input_length: int = 1251
    block_filters: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] = (
        (32, 32, 32),
        (64, 64, 64),
        (64, 64, 64),
    )
    block_kernels: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] = (
        (7, 7, 7),
        (7, 7, 7),
        (7, 7, 7),
    )
    strides: tuple[int, int] = (1, 2)
    dropout: float = 0.2
    dense: int = 128
    se_after_conv: tuple[bool, bool, bool] = (False, False, True)
    se_reduction: int = 16


class SEBlock1D(nn.Module):
    """Squeeze-and-excitation block for Conv1d feature maps."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(2, int(channels) // max(1, int(reduction)))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1)
        weights = self.fc(weights).unsqueeze(-1)
        return x * weights


class _RegionBranch(nn.Module):
    """One region-specific branch with three Conv1d blocks."""

    def __init__(
        self,
        *,
        input_length: int,
        filters: tuple[int, int, int],
        kernels: tuple[int, int, int],
        strides: tuple[int, int],
        se_after_conv: tuple[bool, bool, bool],
        se_reduction: int,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        length = int(input_length)
        self.out_channels = int(filters[-1])
        for block_idx in range(3):
            out_ch = int(filters[block_idx])
            kernel = int(kernels[block_idx])
            stride = int(strides[block_idx]) if block_idx < 2 else 1
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=True))
            layers.append(nn.GELU())
            length = _conv_out_length(length, kernel, stride)
            if bool(se_after_conv[block_idx]):
                layers.append(SEBlock1D(out_ch, reduction=se_reduction))
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)
        self.out_length = int(length)
        self.out_features = int(self.out_channels * self.out_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)


def _conv_out_length(length: int, kernel: int, stride: int) -> int:
    padding = int(kernel) // 2
    return int((int(length) + 2 * padding - (int(kernel) - 1) - 1) // int(stride) + 1)


class SREMCNN(nn.Module):
    """SREM-CNN with region-specific convolutional branches.

    The input is the manuscript tensor converted to PyTorch shape M x P x N.
    Each spectral-region channel is passed through three Conv1d blocks using
    the target-specific filter/kernel triples listed in Table S1; branch
    embeddings are concatenated before the fully connected regression head.
    """

    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.cfg = cfg
        if int(cfg.in_channels) != 3:
            raise ValueError("SREM-CNN expects three spectral-region channels")
        branches = []
        for region_idx in range(3):
            filters = tuple(int(cfg.block_filters[block][region_idx]) for block in range(3))
            kernels = tuple(int(cfg.block_kernels[block][region_idx]) for block in range(3))
            branches.append(
                _RegionBranch(
                    input_length=int(cfg.input_length),
                    filters=filters,
                    kernels=kernels,
                    strides=tuple(int(s) for s in cfg.strides),
                    se_after_conv=tuple(bool(s) for s in cfg.se_after_conv),
                    se_reduction=int(cfg.se_reduction),
                )
            )
        self.branches = nn.ModuleList(branches)
        flat = sum(branch.out_features for branch in self.branches)
        self.regressor = nn.Sequential(
            nn.Linear(flat, cfg.dense),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dense, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_features = [branch(x[:, i : i + 1, :]) for i, branch in enumerate(self.branches)]
        fused = torch.cat(branch_features, dim=1)
        return self.regressor(fused).squeeze(-1)


class PlainCNN1D(nn.Module):
    """Plain 1D-CNN baseline for undifferentiated continuous spectra."""

    def __init__(
        self,
        input_length: int,
        filters: tuple[int, int, int] = (64, 64, 64),
        kernels: tuple[int, int, int] = (11, 9, 7),
        strides: tuple[int, int] = (1, 2),
        se_after_conv: tuple[bool, bool, bool] = (False, False, True),
        se_reduction: int = 16,
        dense: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for block_idx in range(3):
            out_ch = int(filters[block_idx])
            kernel = int(kernels[block_idx])
            stride = int(strides[block_idx]) if block_idx < 2 else 1
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2))
            layers.append(nn.GELU())
            if bool(se_after_conv[block_idx]):
                layers.append(SEBlock1D(out_ch, reduction=se_reduction))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            flat = int(self.features(torch.zeros(1, 1, input_length)).flatten(1).shape[1])
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, dense),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(x)).squeeze(-1)
