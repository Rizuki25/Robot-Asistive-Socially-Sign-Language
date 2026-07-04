"""
=============================================================
Definisi Arsitektur Model BiLSTM
=============================================================
Model Bidirectional LSTM untuk klasifikasi bahasa isyarat.

Arsitektur:
    Input (batch, seq_len, input_size=63)
        → BiLSTM Layer(s)
        → Dropout
        → Fully Connected Layer
        → Output (num_classes)
=============================================================
"""

import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    """
    Model Bidirectional LSTM untuk klasifikasi sequence bahasa isyarat.

    Args:
        input_size (int): Ukuran input per timestep (default: 63 = 21 landmarks × 3 koordinat)
        hidden_size (int): Ukuran hidden state LSTM (default: 128)
        num_layers (int): Jumlah stacked LSTM layers (default: 2)
        num_classes (int): Jumlah kelas output
        dropout (float): Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        input_size: int = 63,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3
    ):
        super(BiLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Fully Connected layer
        # bidirectional=True → output size = hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor dengan shape (batch_size, seq_len, input_size)

        Returns:
            torch.Tensor: Output logits dengan shape (batch_size, num_classes)
        """
        # LSTM output: (batch_size, seq_len, hidden_size * 2)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Ambil output dari timestep terakhir
        # lstm_out[:, -1, :] → (batch_size, hidden_size * 2)
        last_output = lstm_out[:, -1, :]

        # Dropout
        out = self.dropout(last_output)

        # Fully connected
        out = self.fc(out)

        return out

    def get_model_summary(self) -> str:
        """Mengembalikan ringkasan model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        summary = (
            f"BiLSTM Model Summary:\n"
            f"  Input size    : {self.lstm.input_size}\n"
            f"  Hidden size   : {self.hidden_size}\n"
            f"  Num layers    : {self.num_layers}\n"
            f"  Bidirectional : True\n"
            f"  Total params  : {total_params:,}\n"
            f"  Trainable     : {trainable_params:,}\n"
        )
        return summary
