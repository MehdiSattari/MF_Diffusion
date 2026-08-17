"""ConvLSTM building blocks for the DiU temporal encoder.

Implements the convolutional LSTM described in the paper's Appendix A. Each cell
keeps a hidden state Z and cell state S of shape [B, hidden, H, W] and updates
them with a single 3x3 convolution over the concatenation of the input frame and
the previous hidden state:

    [i, f, o, g] = Conv2D([X_n, Z_{n-1}])
    i, f, o = sigmoid(.)            g = tanh(.)
    S_n = f * S_{n-1} + i * g
    Z_n = o * tanh(S_n)
"""

from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        # One conv produces all four gates at once (i, f, o, g).
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels,
                              kernel_size, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor,
                state: Tuple[torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        gates = self.conv(torch.cat([x, h_prev], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch_size: int, spatial: Tuple[int, int],
                   device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = spatial
        zeros = torch.zeros(batch_size, self.hidden_channels, H, W,
                            device=device, dtype=dtype)
        return zeros, zeros.clone()


class ConvLSTM(nn.Module):
    """Stack of ConvLSTM cells processing a sequence [B, T, C, H, W].

    Returns the top-layer hidden state at the final time step, plus the final
    (h, c) state of every layer.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 kernel_size: int = 3, num_layers: int = 1):
        super().__init__()
        self.num_layers = num_layers
        cells = []
        for layer in range(num_layers):
            cin = in_channels if layer == 0 else hidden_channels
            cells.append(ConvLSTMCell(cin, hidden_channels, kernel_size))
        self.cells = nn.ModuleList(cells)

    def step(self, x_t: torch.Tensor, states):
        """Advance ONE time step. x_t: [B, C, H, W]; states: per-layer (h, c) list.
        Returns (top_hidden, new_states). This is the O(1)-per-step recurrent update
        used for autoregressive inference -- no reprocessing of the history."""
        inp = x_t
        new_states = []
        for layer, cell in enumerate(self.cells):
            h, c = cell(inp, states[layer])
            new_states.append((h, c))
            inp = h
        return inp, new_states

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C, H, W = x.shape
        states = [cell.init_state(B, (H, W), x.device, x.dtype) for cell in self.cells]
        top_hidden = None
        for t in range(T):
            top_hidden, states = self.step(x[:, t], states)
        return top_hidden, states
