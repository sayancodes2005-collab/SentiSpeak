#!/usr/bin/env python3
"""
train_pitch_encoder.py — Step 3 of the SentiSpeak pipeline.

Trains the SPICE-style pitch-discovery encoder (src/models/pitch_encoder.py)
on the self-supervised shift-matching puzzle described in the project plan:
for a clip and a pitch-shifted copy of it (a known, applied shift, in
semitones), push encoder(shifted_frame) - encoder(original_frame) to match
shift_semitones. Nothing in this file ever reads an emotion label.

One self-contained script — dataset logic, model instantiation, training
loop, validation, and checkpoint saving — matching the same single-file
pattern already used by build_metadata.py / extract_mel.py /
make_pitch_pairs.py.

Usage:
    python -m src.training.train_pitch_encoder
    python -m src.training.train_pitch_encoder --batch-size 128 --lr 5e-4 --epochs 30
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from ..data_prep.utils import setup_logging
    from ..models.pitch_encoder import PitchEncoder
    from ..utils.audio_utils import load_mel, slice_frame
    from ..utils.config import (
        MEL_METADATA_CSV,
        MEL_PITCH_SHIFTED_METADATA_CSV,
        N_MELS,
        PITCH_ENCODER_CHECKPOINT_DIR,
        PITCH_PAIRS_METADATA_CSV,
    )
except ImportError:  # allows `python src/training/train_pitch_encoder.py` as a plain script too
    import sys

    # train_pitch_encoder.py lives at src/training/train_pitch_encoder.py;
    # data_prep/, models/, and utils/ are all siblings of training/, not
    # parents of it — same situation as pitch_encoder.py's own fallback, so
    # the directory to add is src/ (two levels up from this file), not this
    # file's own folder like the data_prep scripts' simpler fallbacks do.
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from data_prep.utils import setup_logging
    from models.pitch_encoder import PitchEncoder
    from utils.audio_utils import load_mel, slice_frame
    from utils.config import (
        MEL_METADATA_CSV,
        MEL_PITCH_SHIFTED_METADATA_CSV,
        N_MELS,
        PITCH_ENCODER_CHECKPOINT_DIR,
        PITCH_PAIRS_METADATA_CSV,
    )

# ---------------------------------------------------------------------------
# Held-out actors for validation (decision: hold out actors 21-24 entirely,
# all other actors' pairs go to training). A constant near the top so it's
# easy to find and change later.
# ---------------------------------------------------------------------------
HELD_OUT_ACTORS: tuple[int, ...] = (21, 22, 23, 24)

# ---------------------------------------------------------------------------
# Named hyperparameter constants — not buried only in argparse defaults, so
# they're easy to spot and tune. argparse below uses these as its defaults
# and exposes CLI overrides for each.
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
WINDOW_SIZE = 1  # matches PitchEncoder's and audio_utils.slice_frame()'s own defaults


class PitchPairDataset(Dataset):
    """
    One dataset example = one row of pitch_pairs_metadata.csv, i.e. one
    (original_clip, pitch-shifted_clip, known shift_semitones) pair.

    Loads pitch_pairs_metadata.csv, mel_metadata.csv, and
    mel_pitch_shifted_metadata.csv exactly ONCE, in __init__, and builds
    in-memory wav_path -> mel_path lookup dicts from the two mel manifests.
    audio_utils.get_pitch_pair_mels() re-reads both manifest CSVs on every
    call, which its own docstring flags as fine for occasional use but not
    for a training loop calling it thousands of times — this class exists
    specifically to fix that, which is why it does NOT use
    get_pitch_pair_mels() (it does reuse audio_utils.load_mel() and
    audio_utils.slice_frame(), which don't have that problem).

    Each __getitem__ call samples ONE random frame_index — freshly
    re-sampled every call, respecting that specific pair's real frame count
    (not a hardcoded/assumed constant) — and returns
    (original_frame, shifted_frame, shift_semitones_tensor) for that frame.
    """

    def __init__(
        self,
        split: str,
        pitch_pairs_csv: Path,
        mel_metadata_csv: Path,
        mel_pitch_shifted_metadata_csv: Path,
        window_size: int = WINDOW_SIZE,
        held_out_actors: tuple[int, ...] = HELD_OUT_ACTORS,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.window_size = window_size
        self.mel_metadata_csv = mel_metadata_csv
        self.mel_pitch_shifted_metadata_csv = mel_pitch_shifted_metadata_csv

        # --- Load all three manifests exactly once ----------------------
        pairs_df = pd.read_csv(pitch_pairs_csv)
        mel_manifest = pd.read_csv(mel_metadata_csv)
        mel_shifted_manifest = pd.read_csv(mel_pitch_shifted_metadata_csv)

        # In-memory lookup dicts, built once here — no CSV reads happen
        # inside __getitem__. Assumes wav_path is unique within each
        # manifest (same assumption already flagged in
        # audio_utils.get_pitch_pair_mels()'s docstring).
        self._original_lookup: dict[str, str] = dict(
            zip(mel_manifest["wav_path"], mel_manifest["mel_path"])
        )
        self._shifted_lookup: dict[str, str] = dict(
            zip(mel_shifted_manifest["wav_path"], mel_shifted_manifest["mel_path"])
        )

        # --- Actor-based train/val split ---------------------------------
        if "actor" not in pairs_df.columns:
            raise ValueError(
                f"'actor' column not found in {pitch_pairs_csv} — required for "
                f"the actor-based train/val split."
            )
        # A handful of rows can be missing `actor` if make_pitch_pairs.py hit a
        # filename it couldn't parse (see RavdessFilenameError handling there).
        # Such rows can't be reliably assigned to either split, so they're
        # dropped here rather than silently defaulting into "train".
        pairs_df = pairs_df.dropna(subset=["actor"]).copy()
        pairs_df["actor"] = pairs_df["actor"].astype(int)

        if split == "train":
            pairs_df = pairs_df[~pairs_df["actor"].isin(held_out_actors)]
        else:  # split == "val"
            pairs_df = pairs_df[pairs_df["actor"].isin(held_out_actors)]

        self.pairs_df = pairs_df.reset_index(drop=True)
        self.split = split

    def __len__(self) -> int:
        return len(self.pairs_df)

    @staticmethod
    def _lookup_mel_path(lookup: dict[str, str], wav_path: str, manifest_csv: Path) -> Path:
        mel_path = lookup.get(wav_path)
        if mel_path is None:
            raise ValueError(
                f"No row with wav_path == '{wav_path}' found in {manifest_csv}. "
                f"Was extract_mel.py run on the folder this file lives in?"
            )
        return Path(mel_path)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.pairs_df.iloc[idx]

        original_path = row["original_path"]
        shifted_path = row["shifted_path"]
        shift_semitones = float(row["shift_semitones"])

        original_mel_path = self._lookup_mel_path(
            self._original_lookup, original_path, self.mel_metadata_csv
        )
        shifted_mel_path = self._lookup_mel_path(
            self._shifted_lookup, shifted_path, self.mel_pitch_shifted_metadata_csv
        )

        original_mel = load_mel(original_mel_path)
        shifted_mel = load_mel(shifted_mel_path)

        # Clips and their pitch-shifted twins are expected to have identical
        # n_frames (confirmed on real pipeline output — pitch-shifting never
        # changes clip length). We still take the min defensively rather
        # than assuming a constant, since frame_index must land inside BOTH
        # mels (slice_frame() is called on each with the same frame_index).
        n_frames = min(original_mel.shape[1], shifted_mel.shape[1])
        max_start = n_frames - self.window_size

        if max_start < 0:
            raise ValueError(
                f"Clip is shorter than window_size: n_frames={n_frames}, "
                f"window_size={self.window_size} (original_path={original_path!r}). "
                f"Reduce --window-size, or exclude this clip."
            )

        # Freshly re-sampled every __getitem__ call — a different random
        # frame per pair each time the DataLoader pulls this index.
        frame_index = random.randint(0, max_start)

        original_frame = slice_frame(original_mel, frame_index, self.window_size)
        shifted_frame = slice_frame(shifted_mel, frame_index, self.window_size)
        # Shape (1,): matches PitchEncoder's default output_dim=1, so
        # predicted_diff (batch_size, 1) and this target, once batched by
        # the DataLoader into (batch_size, 1), line up directly for MSELoss
        # with no reshaping in the training loop. NOTE: if output_dim is
        # ever changed to >1, this direct scalar-vs-vector comparison would
        # need revisiting — flagged in my closing note.
        shift_target = torch.tensor([shift_semitones], dtype=torch.float32)

        return original_frame, shifted_frame, shift_target


def train_pitch_encoder(
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    window_size: int,
    pitch_pairs_csv: Path,
    mel_metadata_csv: Path,
    mel_pitch_shifted_metadata_csv: Path,
    checkpoint_dir: Path,
    device: torch.device,
    verbose: bool = False,
) -> None:
    log = setup_logging(verbose)

    train_dataset = PitchPairDataset(
        split="train",
        pitch_pairs_csv=pitch_pairs_csv,
        mel_metadata_csv=mel_metadata_csv,
        mel_pitch_shifted_metadata_csv=mel_pitch_shifted_metadata_csv,
        window_size=window_size,
    )
    val_dataset = PitchPairDataset(
        split="val",
        pitch_pairs_csv=pitch_pairs_csv,
        mel_metadata_csv=mel_metadata_csv,
        mel_pitch_shifted_metadata_csv=mel_pitch_shifted_metadata_csv,
        window_size=window_size,
    )

    log.info(
        f"Train pairs: {len(train_dataset)} | "
        f"Val pairs (held-out actors {HELD_OUT_ACTORS}): {len(val_dataset)}"
    )
    if len(train_dataset) == 0:
        raise SystemExit(
            f"No training pairs found in {pitch_pairs_csv} after excluding held-out "
            f"actors {HELD_OUT_ACTORS}. Was make_pitch_pairs.py run?"
        )
    if len(val_dataset) == 0:
        log.warning(
            f"Validation set is empty — no pitch pairs found for held-out actors "
            f"{HELD_OUT_ACTORS}. Val loss will be reported as NaN every epoch."
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # n_mels/window_size passed explicitly and consistently into both the
    # dataset (above) and the model (here) — see pitch_encoder.py's own
    # docstring note that this pairing has to be kept in sync by hand,
    # since config.py has no single WINDOW_SIZE constant to import instead.
    model = PitchEncoder(n_mels=N_MELS, window_size=window_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "last.pt"
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_examples = 0

        for original_frame, shifted_frame, shift_target in train_loader:
            original_frame = original_frame.to(device)
            shifted_frame = shifted_frame.to(device)
            shift_target = shift_target.to(device)

            optimizer.zero_grad()
            # The self-supervised puzzle itself: the difference between the
            # encoder's output on the shifted vs. original frame should
            # match the known, applied shift_semitones (project plan, Step 3).
            predicted_diff = model(shifted_frame) - model(original_frame)
            loss = criterion(predicted_diff, shift_target)
            loss.backward()
            optimizer.step()

            batch_n = original_frame.size(0)
            train_loss_sum += loss.item() * batch_n
            train_examples += batch_n

        train_loss = train_loss_sum / train_examples if train_examples else float("nan")

        model.eval()
        val_loss_sum = 0.0
        val_examples = 0

        with torch.no_grad():
            for original_frame, shifted_frame, shift_target in val_loader:
                original_frame = original_frame.to(device)
                shifted_frame = shifted_frame.to(device)
                shift_target = shift_target.to(device)

                predicted_diff = model(shifted_frame) - model(original_frame)
                loss = criterion(predicted_diff, shift_target)

                batch_n = original_frame.size(0)
                val_loss_sum += loss.item() * batch_n
                val_examples += batch_n

        val_loss = val_loss_sum / val_examples if val_examples else float("nan")

        log.info(
            f"Epoch {epoch}/{num_epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_examples and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            log.info(
                f"New best checkpoint at epoch {epoch} (val_loss={val_loss:.6f}) "
                f"saved to {best_checkpoint_path}"
            )

    torch.save(model.state_dict(), last_checkpoint_path)
    log.info(f"Saved final-epoch checkpoint to {last_checkpoint_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the SPICE-style pitch-discovery encoder (Step 3) "
        "on the self-supervised shift-matching puzzle."
    )
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help="Frames per training example; must match what PitchEncoder is built with.",
    )
    p.add_argument("--pitch-pairs-csv", type=Path, default=PITCH_PAIRS_METADATA_CSV)
    p.add_argument("--mel-metadata-csv", type=Path, default=MEL_METADATA_CSV)
    p.add_argument(
        "--mel-pitch-shifted-metadata-csv",
        type=Path,
        default=MEL_PITCH_SHIFTED_METADATA_CSV,
    )
    p.add_argument("--checkpoint-dir", type=Path, default=PITCH_ENCODER_CHECKPOINT_DIR)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log = setup_logging(args.verbose)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    train_pitch_encoder(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        window_size=args.window_size,
        pitch_pairs_csv=args.pitch_pairs_csv,
        mel_metadata_csv=args.mel_metadata_csv,
        mel_pitch_shifted_metadata_csv=args.mel_pitch_shifted_metadata_csv,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()