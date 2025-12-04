import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=padding,
                               dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               stride=stride, padding=padding,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        # residual mapping
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)

        # ensure length matches (TCN standard trick)
        out = out[:, :, :res.size(2)]

        return F.relu(out + res)


class TCNEncoder(nn.Module):
    """Encoder: IMU (B, T, 9) → (B, T, D)"""
    def __init__(self, input_ch=9, hidden_ch=[64, 128, 256],
                 kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = input_ch
        dilation = 1

        for out_ch in hidden_ch:
            layers.append(
                TemporalBlock(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    dropout=dropout
                )
            )
            in_ch = out_ch
            dilation *= 2  # exponential growth (TCN)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # expect x: (B, T, C)
        x = x.permute(0, 2, 1)   # -> (B, C, T)
        out = self.network(x)
        out = out.permute(0, 2, 1)  # -> (B, T, D)
        return out

