"""
brain.py — the shared "Brain" network (Step 4 of the SentiSpeak project
plan).

Per the project plan: the Brain takes (barcode, time position, pitch-
encoder output) and "guesses what the sound picture should look like at
that point" — i.e. predicts one log-mel-spectrogram frame. One Brain is
shared across every clip; each clip gets its own private, learnable
128-number barcode (src/models/barcodes.py, not this file). This is the
"auto-decoder" design the project plan explicitly traces to DeepSDF-style
3D shape representations: concatenate a per-item code with a query
coordinate, feed both through one shared network.

This file defines ONLY the Brain's architecture — no training loop, no
loss function, no optimizer, no data loading, no CLI — per the README's
rule for src/models/: "Architecture definitions only — no training loops."
The actual joint training loop (Brain + barcodes + the ongoing pitch
puzzle from Step 3, all nudged together) is train_joint.py, a separate
file not included here.

IMPORTANT SEPARATION (see the project plan's own note that the pitch
puzzle must keep running "alongside, not instead of" joint training): this
file does NOT import or reference PitchEncoder at all. The Brain only
needs to know the *dimension* of the pitch signal it receives as one of
its three inputs (PitchEncoder's output_dim, default 1) — PitchEncoder
itself stays a separate, independently-trainable nn.Module, instantiated
and updated on its own in train_joint.py. Absorbing it into or combining
it with the Brain here would break that intentional separation.

Two genuinely new design choices made here (not traceable to existing code
or the project plan beyond its general framing — reasoning below):

- Time representation: a sinusoidal positional encoding (NeRF/SIREN-style
  — see the project plan's own "auto-decoder... first popularized for 3D
  shapes" framing), not a raw scalar frame index. A raw scalar gives a
  plain ReLU MLP a hard time representing high-frequency variation over
  time (the well-known "spectral bias" of coordinate-input MLPs, which is
  exactly why NeRF and similar networks use this encoding); expanding one
  scalar into several sin/cos pairs at different frequencies makes that
  variation much easier for a plain ReLU MLP to represent. Defaults to
  time_encoding_bands=6 (12 encoded numbers): at the finest band this
  still resolves fairly small fractions of a clip's ~285 typical frames,
  which seems like a reasonable starting resolution — if reconstructed
  mel-frames come out over-smoothed in time, raising this (e.g. to 8-10)
  is the first thing worth trying.

- Architecture size: 4 hidden layers at 512 units each — the upper end of
  the agreed 3-4 layers / 256-512 units ranges, not the minimum. Chosen
  because, unlike pitch_encoder.py's small MLP (one clip's single mel-bin
  vector in, a near-arbitrary scalar out), this network's weights are
  shared across every clip in the dataset — only the barcode differs per
  clip — so it needs enough raw capacity to hold reconstruction detail for
  the whole dataset at once, not just one relationship. This mirrors why
  DeepSDF/NeRF-style auto-decoders typically lean toward wider/deeper MLPs
  than a single-example network would need.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

try:
    from ..utils.config import BARCODE_DIM, N_MELS
except ImportError:  # allows importing/running this file outside the `src` package too
    import sys

    # brain.py lives at src/models/brain.py, and config.py lives at
    # src/utils/config.py — utils/ is a *sibling* of models/, not a parent,
    # so the directory to add is src/ (two levels up from this file) — the
    # exact same situation and fix pitch_encoder.py's own fallback already
    # uses, reused here rather than rederived.
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.config import BARCODE_DIM, N_MELS


class Brain(nn.Module):
    """
    Shared "Brain" network (Step 4): predicts one log-mel-spectrogram frame
    from (barcode, time, pitch). See this file's module docstring for the
    auto-decoder framing and the two new design choices (time encoding,
    architecture size) made here.
    """

    def __init__(
        self,
        barcode_dim: int = BARCODE_DIM,
        pitch_dim: int = 1,
        n_mels: int = N_MELS,
        time_encoding_bands: int = 6,
        hidden_dim: int = 512,
    ) -> None:
        """
        Args:
            barcode_dim: size of the per-clip barcode vector. Defaults to
                config.BARCODE_DIM (128) rather than a hardcoded literal —
                same reasoning pitch_encoder.py already applies to its own
                n_mels default.
            pitch_dim: size of the pitch signal fed in per frame. Defaults
                to 1, matching PitchEncoder's own default output_dim=1.
                This is a coupling between the two files that has to be
                kept in sync by hand if PitchEncoder's output_dim is ever
                changed — the same kind of hand-kept coupling
                pitch_encoder.py's own docstring already flags for its
                window_size vs. train_pitch_encoder.py.
            n_mels: dimension of the predicted mel-frame output. Defaults
                to config.N_MELS (128).
            time_encoding_bands: number of frequency bands L in the
                sinusoidal time encoding (_encode_time below); produces
                2*L numbers from a single scalar time value. Default 6 —
                see the module docstring for reasoning.
            hidden_dim: width of each hidden linear layer. Default 512 —
                see the module docstring for reasoning.
        """
        super().__init__()

        for name, value in (
            ("barcode_dim", barcode_dim),
            ("pitch_dim", pitch_dim),
            ("n_mels", n_mels),
            ("time_encoding_bands", time_encoding_bands),
            ("hidden_dim", hidden_dim),
        ):
            if value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}")

        self.barcode_dim = barcode_dim
        self.pitch_dim = pitch_dim
        self.n_mels = n_mels
        self.time_encoding_bands = time_encoding_bands
        self.hidden_dim = hidden_dim

        # Concatenated input = barcode + sinusoidally-encoded time + raw
        # pitch signal (decision #3: plain concatenation, DeepSDF-style —
        # no FiLM, no attention). Encoded time contributes
        # 2 * time_encoding_bands numbers (one sin and one cos per
        # frequency band — see _encode_time below).
        input_dim = barcode_dim + (2 * time_encoding_bands) + pitch_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_mels),
            # No activation after the final layer: a log-mel-spectrogram
            # frame is an unbounded real-valued target, not something
            # squashed into a fixed range — same reasoning pitch_encoder.py
            # already used for its own final layer.
        )

    def _encode_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal positional encoding of a normalized time value
        (NeRF/SIREN-style — see the module docstring for why).

        Args:
            t: shape (batch_size, 1), already normalized to [0, 1] by the
                caller (frame_index / that clip's total frame count) — this
                method does not do that normalization itself.

        Returns:
            shape (batch_size, 2 * time_encoding_bands):
            [sin(2^0*pi*t), cos(2^0*pi*t), sin(2^1*pi*t), cos(2^1*pi*t),
             ..., sin(2^(L-1)*pi*t), cos(2^(L-1)*pi*t)]
        """
        # (time_encoding_bands,): [2^0, 2^1, ..., 2^(L-1)]
        freq_bands = 2.0 ** torch.arange(
            self.time_encoding_bands, dtype=t.dtype, device=t.device
        )
        angles = t * freq_bands.unsqueeze(0) * torch.pi  # (batch_size, L)
        # Stack sin/cos per band, then flatten, so the last dimension comes
        # out interleaved as [sin_0, cos_0, sin_1, cos_1, ...] — matching
        # the order specified in the project decisions above.
        encoded = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)  # (batch_size, L, 2)
        return encoded.reshape(t.shape[0], -1)  # (batch_size, 2 * L)

    def forward(
        self, barcode: torch.Tensor, time: torch.Tensor, pitch: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            barcode: shape (batch_size, barcode_dim) — that clip's
                learnable per-item code (owned/optimized by
                src/models/barcodes.py, not this file).
            time: shape (batch_size,) or (batch_size, 1) — normalized time
                value in [0, 1] per example (frame_index / that clip's
                total frame count). Either shape is accepted: (batch_size,)
                is reshaped to (batch_size, 1) here, since _encode_time()
                expects the trailing dimension explicitly; any other shape
                raises ValueError.
            pitch: shape (batch_size, pitch_dim) — the pitch encoder's raw,
                unnormalized output for this exact frame, computed by a
                separately-instantiated PitchEncoder (see this file's
                module docstring for why brain.py itself never imports or
                calls PitchEncoder). Used as-is here, with no rescaling.

        Returns:
            shape (batch_size, n_mels) — the predicted log-mel-spectrogram
            frame at this (barcode, time, pitch).
        """
        if time.dim() == 1:
            time = time.unsqueeze(-1)  # (batch_size,) -> (batch_size, 1)
        elif time.dim() != 2 or time.shape[1] != 1:
            raise ValueError(
                f"time must have shape (batch_size,) or (batch_size, 1), "
                f"got {tuple(time.shape)}"
            )

        encoded_time = self._encode_time(time)  # (batch_size, 2 * time_encoding_bands)
        x = torch.cat([barcode, encoded_time, pitch], dim=-1)  # (batch_size, input_dim)
        return self.net(x)