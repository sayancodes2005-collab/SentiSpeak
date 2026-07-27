"""
barcodes.py — the learnable per-clip barcode table (Step 4 of the
SentiSpeak project plan).

Per the project plan: "one shared Brain, one private barcode per item, both
updated together to match real data" — this file owns that second half,
"one private barcode per item." Each of the 1,440 original RAVDESS clips
gets its own 128-number vector, jointly optimized alongside the Brain
(brain.py) during train_joint.py's training loop, not computed by any
formula. This is the other side of the same DeepSDF-style "auto-decoder"
lineage brain.py's own module docstring already describes: a per-item
learnable code, looked up by ID and fed into a shared network.

This file defines ONLY the barcode storage/lookup mechanism — no training
loop, no loss function, no optimizer, no data loading, no CLI — per the
README's rule for src/models/: "Architecture definitions only — no
training loops." The actual joint training loop is train_joint.py, a
separate file not included here.

WHY num_clips DEFAULTS TO 1,440, NOT 10,080 (1,440 originals + 8,640
pitch-shifted copies): per the project plan's Step 6 design, a
pitch-shifted copy is reused later as an *augmentation* example that
should still map back to its *original* clip's emotion identity — "Reuse
your pitch-shifted training copies... as extra training examples with the
same emotion label, so the barcode learns that pitch is irrelevant to
emotion." That only makes sense if a shifted copy looks up the SAME
barcode row as its original clip, not a fresh row of its own. Giving
shifted copies their own barcode rows would defeat that entire mechanism
— it would let the barcode memorize pitch-specific content per shifted
copy instead of being forced to represent emotion independent of pitch.
So this table is sized for exactly the 1,440 original clips confirmed in
data/metadata.csv, and the actual clip_id -> integer-row remapping (for
both original and shifted-copy paths) is train_joint.py's job, not this
file's — see decision #3 below.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

try:
    from ..utils.config import BARCODE_DIM
except ImportError:  # allows importing/running this file outside the `src` package too
    import sys

    # barcodes.py lives at src/models/barcodes.py, and config.py lives at
    # src/utils/config.py — utils/ is a *sibling* of models/, not a parent,
    # so the directory to add is src/ (two levels up from this file) — the
    # exact same situation and fix pitch_encoder.py's and brain.py's own
    # fallbacks already use, reused here rather than rederived.
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.config import BARCODE_DIM


class BarcodeTable(nn.Module):
    """
    A learnable, per-clip lookup table of barcode vectors (Step 4).

    Wraps a plain nn.Embedding rather than a hand-rolled nn.Parameter
    matrix with manual indexing — nn.Embedding is the standard, idiomatic
    PyTorch tool for exactly this "one learnable vector per discrete
    integer ID" use case, and already handles indexing, gradients, and
    device placement correctly.

    This class is dataset-agnostic: it takes only `num_clips` and
    `barcode_dim`, and does no clip_id parsing or CSV reading of its own.
    The mapping from a real clip identifier (e.g. data/metadata.csv's
    clip_id column, like "clip_00000") to an integer row-index — and the
    remapping of a pitch-shifted copy's path back to its *original* clip's
    row-index (see this file's module docstring) — both belong to
    train_joint.py, not here.

    Regularizing barcode values (e.g. L2 / weight decay) is also a
    training-time concern, not an architectural one — deferred entirely to
    train_joint.py (e.g. via the optimizer's weight_decay argument, or an
    explicit loss term), not handled anywhere in this file.
    """

    def __init__(self, num_clips: int = 1440, barcode_dim: int = BARCODE_DIM) -> None:
        """
        Args:
            num_clips: number of rows in the table — one per *original*
                clip, NOT one per (original + pitch-shifted-copy) example.
                Defaults to 1440, the confirmed row count of
                data/metadata.csv (24 actors x 60 clips each). See this
                file's module docstring for why this is 1440 and not
                10080.
            barcode_dim: length of each barcode vector. Defaults to
                config.BARCODE_DIM (128) rather than a hardcoded literal —
                same reasoning brain.py already applies to its own
                barcode_dim default.
        """
        super().__init__()

        for name, value in (("num_clips", num_clips), ("barcode_dim", barcode_dim)):
            if value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}")

        self.num_clips = num_clips
        self.barcode_dim = barcode_dim

        self.embedding = nn.Embedding(num_embeddings=num_clips, embedding_dim=barcode_dim)

        # Small random noise, not all-zeros (decision #2): matches common
        # DeepSDF-style auto-decoder practice — starting every barcode at
        # exactly zero risks symmetric, uninformative early gradients
        # (every clip would start identical and receive identical initial
        # updates), whereas small random noise gives each clip's barcode a
        # distinct starting point to diverge from.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, clip_indices: torch.Tensor) -> torch.Tensor:
        """
        Looks up a batch of barcodes by integer row-index.

        Args:
            clip_indices: shape (batch_size,), dtype must be an integer
                type (e.g. torch.long) — nn.Embedding requires integer/long
                indices. Passing a float tensor is an easy mistake when
                indices come from general-purpose data-loading code, so
                that case is caught explicitly below and raises a clear
                error, rather than letting nn.Embedding's own (less
                obvious) internal error surface instead.

        Returns:
            shape (batch_size, barcode_dim) — that batch's looked-up,
            currently-learned barcodes.
        """
        if torch.is_floating_point(clip_indices):
            raise TypeError(
                f"clip_indices must be an integer tensor (e.g. torch.long), "
                f"got dtype {clip_indices.dtype}. nn.Embedding requires "
                f"integer indices — cast with clip_indices.long() before "
                f"calling this."
            )

        return self.embedding(clip_indices)