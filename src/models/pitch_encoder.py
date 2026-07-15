"""
pitch_encoder.py — the SPICE-style self-supervised pitch-discovery encoder
(Step 3 of the SentiSpeak project plan).

Per the project plan: "a small encoder network whose only job is to look at
a short slice of sound and output a single number (or a small handful of
numbers)." It's trained via the shift-matching puzzle, which will live in
src/training/train_pitch_encoder.py (not yet built) — that script feeds
this network (original_frame, shifted_frame) pairs from
audio_utils.get_pitch_pair_mels() / slice_frame() and pushes
encoder(shifted_frame) - encoder(original_frame) to be proportional to the
known shift_semitones applied.

This file defines ONLY the network architecture — no training loop, no loss
function, no optimizer, no data loading, no CLI — per the README's rule for
src/models/: "Architecture definitions only — no training loops."

Architecture choice (a genuine design decision, not something traceable to
the project plan or existing code — the plan only specifies the network's
job, not its layer types): a small fully-connected (MLP) network, not
convolutional. With the default window_size=1, each input is a single
time-frame — a flat vector of n_mels values with no meaningful 2D spatial
structure for convolutions to exploit. A small MLP operating directly on
the mel values is the simpler, appropriate choice here. If window_size is
later widened to explore multi-frame context, this may be worth revisiting.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

try:
    from ..utils.config import N_MELS
except ImportError:  # allows importing/running this file outside the `src` package too
    import sys

    # pitch_encoder.py lives at src/models/pitch_encoder.py, and config.py
    # lives at src/utils/config.py — utils/ is a *sibling* of models/, not a
    # parent, so the directory to add is src/ (two levels up from this
    # file), not this file's own folder like the data_prep fallbacks do.
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.config import N_MELS


class PitchEncoder(nn.Module):
    """
    SPICE-style self-supervised pitch-discovery encoder (Step 3).

    Takes a batch of mel-spectrogram frame slices — exactly the shape
    produced by audio_utils.slice_frame(), stacked into a batch — and
    outputs a small vector per example (1 number by default). Nothing
    about "pitch" is ever told to this network directly: the shift-
    matching puzzle in train_pitch_encoder.py is what gives its output
    meaning, by training encoder(shifted) - encoder(original) to track the
    known, applied shift_semitones. See the project plan, Step 3, for why
    this is the only signal the network can use to solve that puzzle.
    """

    def __init__(
        self,
        output_dim: int = 1,
        n_mels: int = N_MELS,
        window_size: int = 1,
    ) -> None:
        """
        Args:
            output_dim: size of the encoder's output vector. Default 1,
                matching the project plan's "a single number" as the
                primary case; pass a larger value to try "a small handful."
            n_mels: number of mel bins per input frame. Defaults to
                config.N_MELS (128) rather than a hardcoded literal, so
                this can't silently drift from extract_mel.py's actual
                settings.
            window_size: number of consecutive time-frames per input
                example, matching audio_utils.slice_frame()'s window_size
                parameter. Defaults to 1 (single-frame granularity). NOTE:
                config.py has no WINDOW_SIZE constant, so this has to be
                passed in and kept in sync by hand with whatever
                train_pitch_encoder.py passes to slice_frame() — see my
                closing note, flagging this as something to confirm.
        """
        super().__init__()

        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")

        self.output_dim = output_dim
        self.n_mels = n_mels
        self.window_size = window_size

        input_dim = n_mels * window_size  # flattened (n_mels, window_size) per example

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            # No activation after the final layer: the output must be an
            # unbounded real number (or small vector of them), not squashed
            # into a fixed range — pitch differences need to be able to
            # grow proportionally with larger shift amounts.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch_size, n_mels, window_size) — a batch of
                mel-spectrogram frame slices, e.g. formed by stacking
                multiple audio_utils.slice_frame() outputs along a new
                batch dimension. NOT expected to be a single, unbatched
                (n_mels, window_size) example.

        Returns:
            shape (batch_size, output_dim).
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # (batch_size, n_mels, window_size) -> (batch_size, n_mels * window_size)
        return self.net(x)