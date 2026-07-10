"""
audio_utils.py — shared mel-loading and framing helpers for SentiSpeak.

Used by src/models/pitch_encoder.py and src/training/train_pitch_encoder.py
(Step 3: the self-supervised pitch-discovery puzzle) to load mel-spectrogram
tensors and pull matched (original, pitch-shifted) frame pairs, with
consistent shapes and dtypes across every caller.

This file only defines shared helper functions — no training loop, no model
class, no CLI. All paths are imported from src/utils/config.py rather than
redeclared here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .config import MEL_METADATA_CSV, MEL_PITCH_SHIFTED_METADATA_CSV
except ImportError:  # allows running/importing this file outside the package too
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from config import MEL_METADATA_CSV, MEL_PITCH_SHIFTED_METADATA_CSV


def load_mel(mel_path: Path) -> torch.Tensor:
    """
    Load a single mel-spectrogram .npy file (as saved by extract_mel.py)
    and return it as a torch.Tensor, dtype float32, shape (n_mels, n_frames).

    Raises FileNotFoundError with a clear message if mel_path doesn't exist,
    rather than letting a raw numpy/torch error surface — same
    explicit-error-message style as find_wav_files() in data_prep/utils.py.
    """
    mel_path = Path(mel_path)
    if not mel_path.exists():
        raise FileNotFoundError(
            f"Mel-spectrogram file does not exist: {mel_path}\n"
            f"(was extract_mel.py run on the folder this file lives in? "
            f"check mel_metadata.csv / mel_pitch_shifted_metadata.csv for the "
            f"expected mel_path values)"
        )

    mel_array = np.load(mel_path)  # (n_mels, n_frames), already float32 (extract_mel.py
                                    # saves via log_mel.astype(np.float32) before np.save)
    return torch.from_numpy(mel_array).float()  # numpy -> torch.Tensor conversion happens here


def get_pitch_pair_mels(
    pitch_pairs_row: dict | pd.Series,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Given one row from pitch_pairs_metadata.csv (a dict or pandas Series
    with at least `original_path`, `shifted_path`, `shift_semitones`),
    return the matched pair of mel-spectrograms needed for Step 3's
    shift-matching puzzle: (original_mel, shifted_mel, shift_semitones).

    original_mel is looked up in mel_metadata.csv (matched on
    wav_path == original_path); shifted_mel is looked up in
    mel_pitch_shifted_metadata.csv (matched on wav_path == shifted_path).
    Neither manifest already contains the other's mel_path, and there's no
    single CSV that joins all three files together — this function does
    that join itself, at call time.
    """

    def _lookup_mel_path(manifest_csv: Path, wav_path: str) -> Path:
        # PERFORMANCE NOTE (flagging, not fixing, per your instructions):
        # this reads the full manifest CSV from disk and does a pandas
        # filter on every single call. Fine for occasional/interactive use,
        # but if get_pitch_pair_mels() ends up called once per training
        # example inside train_pitch_encoder.py's training loop (likely,
        # given Step 3 samples frames per clip across many epochs), that's
        # thousands of redundant CSV parses. I have deliberately NOT added
        # any caching (e.g. loading each manifest once into a dict keyed by
        # wav_path, or moving this into a PyTorch Dataset's __init__) —
        # see my closing note, this needs your go-ahead first.
        if not manifest_csv.exists():
            raise FileNotFoundError(f"Manifest CSV does not exist: {manifest_csv}")

        manifest = pd.read_csv(manifest_csv)
        matches = manifest.loc[manifest["wav_path"] == wav_path, "mel_path"]

        if matches.empty:
            raise ValueError(
                f"No row with wav_path == '{wav_path}' found in {manifest_csv}. "
                f"Was extract_mel.py run on the folder this file lives in?"
            )
        # Assumes wav_path is unique within each manifest CSV — true as long as
        # each manifest was built by a single extract_mel.py run, since RAVDESS
        # filenames are unique within their Actor_XX folder. Flagging this as
        # an assumption, not something this function verifies.
        return Path(matches.iloc[0])

    original_path = pitch_pairs_row["original_path"]
    shifted_path = pitch_pairs_row["shifted_path"]
    shift_semitones = float(pitch_pairs_row["shift_semitones"])

    original_mel_path = _lookup_mel_path(MEL_METADATA_CSV, original_path)
    shifted_mel_path = _lookup_mel_path(MEL_PITCH_SHIFTED_METADATA_CSV, shifted_path)

    original_mel = load_mel(original_mel_path)
    shifted_mel = load_mel(shifted_mel_path)

    return original_mel, shifted_mel, shift_semitones


def slice_frame(mel: torch.Tensor, frame_index: int, window_size: int = 1) -> torch.Tensor:
    """
    Extract `window_size` consecutive time-frames from `mel`, starting at
    `frame_index`. Returns shape (n_mels, window_size).

    Default window_size=1 matches the project plan's "a single number, or
    a small handful" per-frame granularity for the pitch encoder, but is
    parameterized so wider windows can be tried later.

    Raises ValueError if frame_index is negative, window_size < 1, or
    frame_index + window_size exceeds the clip's total frame count — rather
    than silently returning a truncated or wrong-shaped tensor, since
    Step 3's puzzle depends on every sampled frame having the same shape.
    """
    n_frames = mel.shape[1]

    if frame_index < 0:
        raise ValueError(f"frame_index must be >= 0, got {frame_index}")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    end_index = frame_index + window_size
    if end_index > n_frames:
        raise ValueError(
            f"Requested frames [{frame_index}:{end_index}) exceed this mel's "
            f"total frame count ({n_frames}). Keep frame_index + window_size "
            f"<= n_frames."
        )

    return mel[:, frame_index:end_index]