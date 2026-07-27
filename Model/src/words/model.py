"""Packed BiLSTM khusus kata; padding tidak ikut membentuk representasi."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class WordMotionBiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        safe_lengths = lengths.detach().to("cpu").clamp(min=1, max=inputs.size(1))
        packed = pack_padded_sequence(
            inputs,
            safe_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)

        # Layer terakhir: forward berada di -2, backward berada di -1.
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(representation))

    def summary(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        return (
            "Word Motion BiLSTM\n"
            f"  Input size : {self.input_size}\n"
            f"  Hidden     : {self.hidden_size}\n"
            f"  Layers     : {self.num_layers}\n"
            f"  Parameters : {total:,}"
        )
