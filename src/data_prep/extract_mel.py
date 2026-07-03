#!/usr/bin/env python3
"""
extract_mel.py — Step 2 (part 1) of the SentiSpeak pipeline.

Converts every .wav file under an input directory into a log-mel-spectrogram
and saves it as a .npy array under an output directory, mirroring the
Actor_XX folder structure. Also writes a manifest CSV mapping each clip to
its mel array path and shape.

Works on the raw RAVDESS folder by default, but can also be pointed at
data/processed/pitch_shifted/ (the output of make_pitch_pairs.py) to turn
the pitch-shifted copies into mel-spectrograms using the exact same
settings.

Usage:
    python -m src.data_prep.extract_mel
    python -m src.data_prep.extract_mel \
        --input-dir data/raw/Audio_Speech_Actors_01-24 \
        --output-dir data/processed/mel \
        --sr 22050 --n-mels 128 --n-fft 1024 --hop-length 256

    # Reuse on the pitch-shifted copies with identical settings:
    python -m src.data_prep.extract_mel \
        --input-dir data/processed/pitch_shifted \
        --output-dir data/processed/mel_pitch_shifted \
        --manifest data/processed/mel_pitch_shifted_metadata.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # tqdm is a "nice to have" progress bar, not a hard requirement
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from .utils import find_wav_files, relative_output_path, setup_logging
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from utils import find_wav_files, relative_output_path, setup_logging

DEFAULT_INPUT_DIR = Path("data/raw/Audio_Speech_Actors_01-24")
DEFAULT_OUTPUT_DIR = Path("data/processed/mel")
DEFAULT_MANIFEST = Path("data/processed/mel_metadata.csv")

DEFAULT_SR = 22050
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 256
DEFAULT_N_MELS = 128


def compute_log_mel(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Return a (n_mels, n_frames) float32 log-scaled mel-spectrogram."""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax or sr / 2,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def extract_mel(
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    verbose: bool = False,
) -> pd.DataFrame:
    try:
        import librosa  # noqa: F401  (fail fast with a clear error if missing)
    except ImportError as exc:
        raise SystemExit(
            "librosa is required for extract_mel.py. Install it with:\n"
            "    pip install librosa soundfile"
        ) from exc

    log = setup_logging(verbose)
    wav_files = find_wav_files(input_dir)
    log.info(f"Found {len(wav_files)} .wav files under {input_dir}")
    if not wav_files:
        raise SystemExit(f"No .wav files found under {input_dir}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n_failed = 0

    for wav_path in tqdm(wav_files, desc="Extracting mel-spectrograms"):
        out_path = relative_output_path(input_dir, wav_path, output_dir, ".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            y, loaded_sr = librosa.load(wav_path, sr=sr, mono=True)
            if y.size == 0:
                raise ValueError("loaded empty audio array")
            log_mel = compute_log_mel(y, loaded_sr, n_fft, hop_length, n_mels)
        except Exception as exc:  # keep going even if one clip is corrupt/unreadable
            log.warning(f"Failed on {wav_path}: {exc}")
            n_failed += 1
            continue

        np.save(out_path, log_mel)
        rows.append(
            {
                "wav_path": str(wav_path),
                "mel_path": str(out_path),
                "sr": loaded_sr,
                "n_mels": log_mel.shape[0],
                "n_frames": log_mel.shape[1],
                "duration_sec": round(float(y.size) / loaded_sr, 3),
            }
        )

    if n_failed:
        log.warning(f"{n_failed} file(s) failed to process; see warnings above.")

    df = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)

    log.info(f"Saved {len(df)} mel-spectrograms to {output_dir}")
    log.info(f"Wrote manifest to {manifest_path}")
    if not df.empty:
        log.info(
            f"Mel shape example: {df.iloc[0]['n_mels']} mel bins x "
            f"{df.iloc[0]['n_frames']} frames (n_fft={n_fft}, hop_length={hop_length}, sr={sr})"
        )

    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert audio clips to log-mel-spectrograms.")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--sr", type=int, default=DEFAULT_SR, help="Target sample rate (Hz)")
    p.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    p.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    p.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    extract_mel(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()