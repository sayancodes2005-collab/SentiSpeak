"""
config.py — central hyperparameters and paths for the SentiSpeak pipeline.

Referenced by every stage of the project (data_prep, models, training,
evaluation, inference) so that constants like sample rate, mel settings,
pitch-shift amounts, and folder locations are defined exactly once and stay
consistent everywhere they're used.

This file is pure constants and path definitions only — no functions, no
classes, no argparse CLI, no training logic. Anything that *does* work
belongs in the scripts that import from here, not in this file.

Pipeline step reference (see the project plan / README):
    Step 1  -> METADATA_CSV
    Step 2  -> MEL_DIR, MEL_METADATA_CSV, PITCH_SHIFTED_DIR,
               PITCH_PAIRS_METADATA_CSV, MEL_PITCH_SHIFTED_DIR,
               MEL_PITCH_SHIFTED_METADATA_CSV
    Step 3  -> PITCH_ENCODER_CHECKPOINT_DIR, PITCH_ENCODER_* placeholders below
    Step 4  -> BARCODE_DIM, BRAIN_CHECKPOINT_DIR
    Step 6  -> ADVERSARIAL_CHECKPOINT_DIR (Sabotage Method, optional)
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Audio / mel-spectrogram constants
# ---------------------------------------------------------------------------
# These match the values already used as defaults in
# src/data_prep/extract_mel.py (DEFAULT_SR, DEFAULT_N_FFT, DEFAULT_HOP_LENGTH,
# DEFAULT_N_MELS) and confirmed from the actual pipeline run. Centralised
# here so later scripts (train_pitch_encoder.py, train_joint.py, ...) don't
# have to redeclare them, or risk drifting from them.
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

# ---------------------------------------------------------------------------
# Pitch-shift augmentation (Step 2 / Step 3)
# ---------------------------------------------------------------------------
# Matches DEFAULT_SHIFTS in src/data_prep/make_pitch_pairs.py exactly.
# Kept here so train_pitch_encoder.py (Step 3) can reference the same known,
# controlled shift amounts without re-declaring them. A tuple, not a list,
# since this should never be mutated at runtime.
PITCH_SHIFTS_SEMITONES = (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)

# ---------------------------------------------------------------------------
# Barcode size (Step 4)
# ---------------------------------------------------------------------------
# Per the project plan: each clip is compressed into a 128-number "barcode."
BARCODE_DIM = 128

# ---------------------------------------------------------------------------
# Raw data (Step 1)
# ---------------------------------------------------------------------------
# Matches DEFAULT_RAW_DIR / DEFAULT_INPUT_DIR already used in
# build_metadata.py, extract_mel.py and make_pitch_pairs.py. All paths below
# are relative, matching those existing scripts, and assume the process is
# run from the project root (SentiSpeak/), same as those scripts already do.
RAW_AUDIO_DIR = Path("data/raw/Audio_Speech_Actors_01-24")

# ---------------------------------------------------------------------------
# Step 1: parsed ground-truth metadata
# ---------------------------------------------------------------------------
METADATA_CSV = Path("data/metadata.csv")

# ---------------------------------------------------------------------------
# Step 2: mel-spectrograms of the original (unshifted) clips
# ---------------------------------------------------------------------------
MEL_DIR = Path("data/processed/mel")
MEL_METADATA_CSV = Path("data/processed/mel_metadata.csv")

# ---------------------------------------------------------------------------
# Step 2: pitch-shifted audio copies, at the known shifts in
# PITCH_SHIFTS_SEMITONES above
# ---------------------------------------------------------------------------
PITCH_SHIFTED_DIR = Path("data/processed/pitch_shifted")
PITCH_PAIRS_METADATA_CSV = Path("data/processed/pitch_pairs_metadata.csv")

# ---------------------------------------------------------------------------
# Step 2/3: mel-spectrograms of the pitch-shifted copies (same SAMPLE_RATE /
# N_FFT / HOP_LENGTH / N_MELS settings above, produced by running
# extract_mel.py a second time against PITCH_SHIFTED_DIR)
# ---------------------------------------------------------------------------
MEL_PITCH_SHIFTED_DIR = Path("data/processed/mel_pitch_shifted")
MEL_PITCH_SHIFTED_METADATA_CSV = Path("data/processed/mel_pitch_shifted_metadata.csv")

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
# Currently empty except for .gitkeep placeholders, per the README.
CHECKPOINTS_DIR = Path("checkpoints")
PITCH_ENCODER_CHECKPOINT_DIR = CHECKPOINTS_DIR / "pitch_encoder"  # Step 3
BRAIN_CHECKPOINT_DIR = CHECKPOINTS_DIR / "brain"                  # Step 4
ADVERSARIAL_CHECKPOINT_DIR = CHECKPOINTS_DIR / "adversarial"      # Step 6, optional

# ---------------------------------------------------------------------------
# Step 3 (train_pitch_encoder.py) hyperparameters — PLACEHOLDER, NOT YET USED
# ---------------------------------------------------------------------------
# Nothing in the codebase reads these yet. They're stubbed in here only so
# train_pitch_encoder.py has an obvious place to pull values from once we've
# actually decided them together. Don't expect any functional behavior from
# these until train_pitch_encoder.py exists and imports them.
PITCH_ENCODER_LEARNING_RATE = None  # TODO: decide, e.g. 1e-3 / 1e-4
PITCH_ENCODER_BATCH_SIZE = None  # TODO: decide
PITCH_ENCODER_NUM_EPOCHS = None  # TODO: decide
PITCH_ENCODER_OUTPUT_DIM = None  # TODO: decide — plan says "a single number, or a small handful"