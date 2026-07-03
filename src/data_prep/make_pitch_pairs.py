#!/usr/bin/env python3
"""
make_pitch_pairs.py — Step 2 (part 2) of the SentiSpeak pipeline.

For every raw RAVDESS clip, creates one or more pitch-shifted copies at
known, chosen semitone shifts (via librosa.effects.pitch_shift) and saves
them as new .wav files, mirroring the Actor_XX folder structure under
data/processed/pitch_shifted/.

Writes data/processed/pitch_pairs_metadata.csv with one row per
(original, shifted) pair, recording the *exact* shift amount applied.
This manifest is reused twice later in the project:
  - Step 3: the self-supervised pitch-discovery puzzle (shift-matching) —
    the model is trained to make its output for the shifted clip differ
    from its output for the original by an amount proportional to
    `shift_semitones`.
  - Step 6: Voice Changer augmentation — the shifted copies are re-used
    as extra training examples that keep the *same* emotion label as
    their original, teaching the barcode that pitch is irrelevant to mood.

Usage:
    python -m src.data_prep.make_pitch_pairs
    python -m src.data_prep.make_pitch_pairs --shifts -3 -2 -1 1 2 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # tqdm is a "nice to have" progress bar, not a hard requirement
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from .utils import (
        RavdessFilenameError,
        find_wav_files,
        parse_ravdess_filename,
        setup_logging,
    )
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from utils import (
        RavdessFilenameError,
        find_wav_files,
        parse_ravdess_filename,
        setup_logging,
    )

DEFAULT_INPUT_DIR = Path("data/raw/Audio_Speech_Actors_01-24")
DEFAULT_OUTPUT_DIR = Path("data/processed/pitch_shifted")
DEFAULT_MANIFEST = Path("data/processed/pitch_pairs_metadata.csv")

# Known, controlled shift amounts (semitones). Deliberately excludes 0,
# and mixes up/down shifts so the pitch-discovery puzzle in Step 3 sees
# both directions.
DEFAULT_SHIFTS = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]


def shift_suffix(shift_semitones: float) -> str:
    """'+2.0' -> 'up2', '-3.5' -> 'down3p5' — filesystem/CSV-safe tag."""
    sign = "up" if shift_semitones >= 0 else "down"
    magnitude = f"{abs(shift_semitones):.2f}".rstrip("0").rstrip(".")
    return f"{sign}{magnitude.replace('.', 'p')}"


def make_pitch_pairs(
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    shifts: list[float],
    sr: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    try:
        import librosa
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit(
            "librosa and soundfile are required for make_pitch_pairs.py. Install with:\n"
            "    pip install librosa soundfile"
        ) from exc

    log = setup_logging(verbose)

    shifts = list(shifts)
    if not shifts:
        raise ValueError("`shifts` must contain at least one non-zero semitone value.")
    if any(s == 0 for s in shifts):
        log.warning("A shift of 0 semitones produces an identical copy; consider removing it.")

    wav_files = find_wav_files(input_dir)
    log.info(f"Found {len(wav_files)} .wav files under {input_dir}")
    log.info(f"Will generate {len(shifts)} shifted copies per clip: {shifts}")
    if not wav_files:
        raise SystemExit(f"No .wav files found under {input_dir}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n_failed = 0

    for wav_path in tqdm(wav_files, desc="Creating pitch-shifted pairs"):
        try:
            meta = parse_ravdess_filename(wav_path)
            actor_folder = f"Actor_{meta.actor:02d}"
        except RavdessFilenameError:
            # Still process the file, just without RAVDESS-derived metadata columns.
            meta = None
            actor_folder = wav_path.parent.name

        try:
            y, loaded_sr = librosa.load(wav_path, sr=sr, mono=True)
            if y.size == 0:
                raise ValueError("loaded empty audio array")
        except Exception as exc:
            log.warning(f"Failed to load {wav_path}: {exc}")
            n_failed += 1
            continue

        out_dir = output_dir / actor_folder
        out_dir.mkdir(parents=True, exist_ok=True)

        for shift in shifts:
            try:
                y_shifted = librosa.effects.pitch_shift(y=y, sr=loaded_sr, n_steps=shift)
            except Exception as exc:
                log.warning(f"Pitch-shift failed on {wav_path} at {shift} semitones: {exc}")
                n_failed += 1
                continue

            out_name = f"{wav_path.stem}__shift_{shift_suffix(shift)}.wav"
            out_path = out_dir / out_name
            sf.write(out_path, y_shifted, loaded_sr)

            row = {
                "original_path": str(wav_path),
                "shifted_path": str(out_path),
                "shift_semitones": shift,
                "sr": loaded_sr,
            }
            if meta is not None:
                row.update(
                    {
                        "actor": meta.actor,
                        "gender": meta.gender,
                        "emotion": meta.emotion,
                        "intensity": meta.intensity,
                        "statement": meta.statement,
                        "repetition": meta.repetition,
                    }
                )
            rows.append(row)

    if n_failed:
        log.warning(f"{n_failed} operation(s) failed; see warnings above.")

    df = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)

    log.info(f"Wrote {len(df)} pitch-shifted pairs to {output_dir}")
    log.info(f"Wrote manifest to {manifest_path}")

    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create pitch-shifted copies of RAVDESS clips at known semitone shifts."
    )
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument(
        "--shifts",
        type=float,
        nargs="+",
        default=DEFAULT_SHIFTS,
        help=f"Semitone shifts to generate per clip (default: {DEFAULT_SHIFTS})",
    )
    p.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Resample to this rate before shifting; default keeps each clip's native sample rate",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    make_pitch_pairs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        shifts=args.shifts,
        sr=args.sr,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()