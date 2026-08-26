"""
Arsitektur Model BiLSTM untuk Klasifikasi Gabungan 36 Kelas (Huruf & Kata).
Menggunakan Temporal Mean Pooling berbobot panjang sequence agar menangkap
bentuk pose statis (huruf) maupun dinamika ayunan temporal (kata).
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CombinedBiLSTM(nn.Module):
    """BiLSTM temporal-aware untuk klasifikasi 36 kelas (26 huruf + 10 kata)."""

    def __init__(
        self,
        input_size: int = 140,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 36,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (batch_size, seq_len, input_size)
            lengths: (batch_size,) panjang frame valid
        Returns:
            logits: (batch_size, num_classes)
        """
        if lengths is None:
            lengths = torch.full(
                (x.size(0),),
                x.size(1),
                dtype=torch.long,
                device=x.device,
            )

        lengths_cpu = lengths.to(dtype=torch.long).clamp(min=1, max=x.size(1)).cpu()
        packed = pack_padded_sequence(
            x,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        out_packed, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)

        batch_size, max_seq_len, _ = out.shape
        seq_range = torch.arange(max_seq_len, device=x.device).unsqueeze(0)
        mask = (seq_range < lengths.unsqueeze(1)).unsqueeze(-1).float()

        # Mean pooling sepanjang frame aktif
        sum_out = torch.sum(out * mask, dim=1)
        valid_lengths = lengths.unsqueeze(1).clamp(min=1).float()
        representation = sum_out / valid_lengths

        logits = self.classifier(representation)
        return logits

