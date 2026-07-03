#!/usr/bin/env python3
"""
build_metadata.py — Step 1 of the SentiSpeak pipeline.

Walks the raw RAVDESS folder structure:

    data/raw/Audio_Speech_Actors_01-24/Actor_01/03-01-06-01-02-01-01.wav
    data/raw/Audio_Speech_Actors_01-24/Actor_02/...
    ...

parses every filename according to the RAVDESS naming convention, and
writes one row per clip to data/metadata.csv.

Usage:
    python -m src.data_prep.build_metadata
    python -m src.data_prep.build_metadata --raw-dir data/raw/Audio_Speech_Actors_01-24 --out data/metadata.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .utils import (
        RavdessFilenameError,
        find_wav_files,
        parse_ravdess_filename,
        setup_logging,
    )
except ImportError:  # allows `python build_metadata.py` as a plain script too
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from utils import (
        RavdessFilenameError,
        find_wav_files,
        parse_ravdess_filename,
        setup_logging,
    )

DEFAULT_RAW_DIR = Path("data/raw/Audio_Speech_Actors_01-24")
DEFAULT_OUT_CSV = Path("data/metadata.csv")


def build_metadata(raw_dir: Path, out_csv: Path, verbose: bool = False) -> pd.DataFrame:
    log = setup_logging(verbose)

    wav_files = find_wav_files(raw_dir)
    log.info(f"Found {len(wav_files)} .wav files under {raw_dir}")
    if not wav_files:
        raise SystemExit(
            f"No .wav files found under {raw_dir}. "
            f"Check that --raw-dir matches your actual folder structure."
        )

    rows = []
    n_skipped = 0
    for wav_path in wav_files:
        try:
            meta = parse_ravdess_filename(wav_path)
        except RavdessFilenameError as exc:
            log.warning(f"Skipping unparsable file: {exc}")
            n_skipped += 1
            continue
        rows.append(meta.as_dict())

    if n_skipped:
        log.warning(
            f"Skipped {n_skipped} file(s) that didn't match the RAVDESS naming convention."
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(
            "No files could be parsed. Nothing was written. "
            "Double-check --raw-dir points at the folder that directly contains Actor_01, Actor_02, ..."
        )

    df = df.sort_values(["actor", "emotion", "statement_code", "repetition"]).reset_index(
        drop=True
    )
    df.insert(0, "clip_id", [f"clip_{i:05d}" for i in range(len(df))])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    log.info(f"Wrote {len(df)} rows to {out_csv}")
    log.info("Emotion counts:\n" + df["emotion"].value_counts().to_string())
    log.info(
        f"Actors found: {df['actor'].nunique()} (min={df['actor'].min()}, max={df['actor'].max()})"
    )
    log.info(f"Gender split:\n" + df["gender"].value_counts().to_string())

    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build data/metadata.csv from raw RAVDESS audio.")
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Folder that directly contains Actor_01..Actor_24 (default: {DEFAULT_RAW_DIR})",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUT_CSV})",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_metadata(args.raw_dir, args.out, args.verbose)


if __name__ == "__main__":
    main()