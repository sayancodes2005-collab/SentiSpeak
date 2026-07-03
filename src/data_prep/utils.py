"""
Shared utilities for the SentiSpeak data-prep pipeline.

Centralises:
- RAVDESS filename parsing (modality/vocal-channel/emotion/intensity/
  statement/repetition/actor are all encoded in the filename itself)
- Label lookup tables
- Small filesystem helpers reused by build_metadata.py, extract_mel.py and
  make_pitch_pairs.py

Kept dependency-free (stdlib only) so it can be imported even in
environments where librosa/soundfile aren't installed yet.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# RAVDESS filename convention
# ---------------------------------------------------------------------------
# Filenames look like: 03-01-06-01-02-01-12.wav
#   1 Modality             01=full-AV 02=video-only 03=audio-only
#   2 Vocal channel        01=speech  02=song
#   3 Emotion               01..08 (see EMOTION_MAP)
#   4 Emotional intensity   01=normal 02=strong (no "strong" version for neutral)
#   5 Statement             01="Kids are talking by the door" 02="Dogs are sitting by the door"
#   6 Repetition             01=1st rep, 02=2nd rep
#   7 Actor                  01-24 (odd=male, even=female)

MODALITY_MAP = {"01": "full_av", "02": "video_only", "03": "audio_only"}
VOCAL_CHANNEL_MAP = {"01": "speech", "02": "song"}
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
INTENSITY_MAP = {"01": "normal", "02": "strong"}
STATEMENT_MAP = {
    "01": "kids_are_talking_by_the_door",
    "02": "dogs_are_sitting_by_the_door",
}

RAVDESS_N_FIELDS = 7


class RavdessFilenameError(ValueError):
    """Raised when a filename doesn't follow the RAVDESS naming convention."""


@dataclass
class RavdessMeta:
    file_path: str
    file_name: str
    modality_code: str
    modality: str
    vocal_channel_code: str
    vocal_channel: str
    emotion_code: str
    emotion: str
    intensity_code: str
    intensity: str
    statement_code: str
    statement: str
    repetition: int
    actor: int
    gender: str

    def as_dict(self) -> dict:
        return asdict(self)


def parse_ravdess_filename(path: Path) -> RavdessMeta:
    """
    Parse a single RAVDESS .wav path into its metadata fields.

    Raises RavdessFilenameError if the filename doesn't match the
    7-field RAVDESS convention, or contains a code that isn't in the
    known lookup tables above.
    """
    stem = path.stem  # filename without the .wav extension
    parts = stem.split("-")

    if len(parts) != RAVDESS_N_FIELDS:
        raise RavdessFilenameError(
            f"Expected {RAVDESS_N_FIELDS} dash-separated fields in '{path.name}', "
            f"got {len(parts)}: {parts}"
        )

    (
        modality_code,
        vocal_code,
        emotion_code,
        intensity_code,
        statement_code,
        repetition_code,
        actor_code,
    ) = parts

    for code, table, label in (
        (modality_code, MODALITY_MAP, "modality"),
        (vocal_code, VOCAL_CHANNEL_MAP, "vocal channel"),
        (emotion_code, EMOTION_MAP, "emotion"),
        (intensity_code, INTENSITY_MAP, "intensity"),
        (statement_code, STATEMENT_MAP, "statement"),
    ):
        if code not in table:
            raise RavdessFilenameError(
                f"Unrecognised {label} code '{code}' in '{path.name}'"
            )

    try:
        repetition = int(repetition_code)
        actor = int(actor_code)
    except ValueError as exc:
        raise RavdessFilenameError(
            f"Repetition/actor codes must be numeric in '{path.name}'"
        ) from exc

    gender = "male" if actor % 2 == 1 else "female"

    return RavdessMeta(
        file_path=str(path),
        file_name=path.name,
        modality_code=modality_code,
        modality=MODALITY_MAP[modality_code],
        vocal_channel_code=vocal_code,
        vocal_channel=VOCAL_CHANNEL_MAP[vocal_code],
        emotion_code=emotion_code,
        emotion=EMOTION_MAP[emotion_code],
        intensity_code=intensity_code,
        intensity=INTENSITY_MAP[intensity_code],
        statement_code=statement_code,
        statement=STATEMENT_MAP[statement_code],
        repetition=repetition,
        actor=actor,
        gender=gender,
    )


def find_wav_files(root: Path) -> list[Path]:
    """Recursively find every .wav file under `root`, sorted for determinism."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {root}\n"
            f"(check the path against your actual folder structure, e.g. "
            f"data/raw/Audio_Speech_Actors_01-24/Actor_01/...)"
        )
    return sorted(root.rglob("*.wav"))


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("sentispeak")


def relative_output_path(
    input_root: Path, file_path: Path, output_root: Path, new_suffix: str
) -> Path:
    """
    Mirror a file's location under `input_root` into `output_root`,
    keeping the Actor_XX subfolder, but swapping the file extension.

    e.g. data/raw/Audio_Speech_Actors_01-24/Actor_01/03-01-06-01-02-01-01.wav
      -> data/processed/mel/Actor_01/03-01-06-01-02-01-01.npy
    """
    rel = Path(file_path).relative_to(Path(input_root))
    return (Path(output_root) / rel).with_suffix(new_suffix)