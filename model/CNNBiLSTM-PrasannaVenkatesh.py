from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperCNNBiLSTM(nn.Module):
    

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        target_seq_len: int = 232,
        lstm_hidden_size: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.target_seq_len = target_seq_len

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=151, stride=1, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=101, stride=1, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=51, stride=1, padding=0)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(target_seq_len * (2 * lstm_hidden_size), num_classes)

    def forward(self, x: torch.Tensor, return_probs: bool = False):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B,L) or (B,C,L), got {tuple(x.shape)}")

        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        if x.size(-1) != self.target_seq_len:
            x = F.adaptive_max_pool1d(x, self.target_seq_len)
        x = x.transpose(1, 2).contiguous()
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        x = x.flatten(1)
        logits = self.fc(x)

        if return_probs:
            probs = torch.softmax(logits, dim=1)
            return logits, probs
        return logits


class CategoricalCrossEntropyLoss(nn.Module):
    

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 2:
            targets = targets.argmax(dim=1)
        return self.ce(logits, targets.long())


def build_cnn_bilstm_model(
    in_channels: int = 1,
    num_classes: int = 5,
    target_seq_len: int = 232,
    lstm_hidden_size: int = 128,
    dropout: float = 0.5,
) -> PaperCNNBiLSTM:
    return PaperCNNBiLSTM(
        in_channels=in_channels,
        num_classes=num_classes,
        target_seq_len=target_seq_len,
        lstm_hidden_size=lstm_hidden_size,
        dropout=dropout,
    )


def build_cnn_bilstm_loss(label_smoothing: float = 0.0) -> CategoricalCrossEntropyLoss:
    return CategoricalCrossEntropyLoss(label_smoothing=label_smoothing)


def create_training_components(num_classes: int, labels=None):
    model = PaperCNNBiLSTM(in_channels=12, num_classes=num_classes, target_seq_len=232, lstm_hidden_size=128, dropout=0.5)
    return model, CategoricalCrossEntropyLoss()


def prepare_input(x: torch.Tensor) -> torch.Tensor:
    return x


def forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model(prepare_input(x))


def forward_and_loss(model, criterion, x: torch.Tensor, y: torch.Tensor):
    logits = forward_logits(model, x)
    return logits, criterion(logits, y)

TRAINING_CONFIG = {"batch_size": 100, "learning_rate": 0.005, "optimizer": "Adam", "scheduler": {"type": "step", "step_size": 10, "gamma": 0.8}}
