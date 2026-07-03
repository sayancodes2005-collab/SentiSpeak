SentiSpeak

Self-supervised speech emotion recognition using an INR auto-decoder ("Shared Brain") framework, with self-discovered pitch disentanglement (SPICE-style).

SentiSpeak learns to sort voice clips by emotional feeling without ever using emotion labels during training. Each clip is compressed into a 128-number "barcode." Where that barcode lands relative to others reveals the speaker's mood — emotion labels are used only at the very end, to grade results, never to train the model.

A key design choice: pitch is not hand-extracted from a formula. A small encoder network discovers pitch entirely on its own, through a self-supervised puzzle (the SPICE technique), before the main model is trained. This gives the system a genuine, well-anchored notion of "pitch" to separate from "emotion" — without ever touching a label.

Project Structure

SentiSpeak
checkpoints (Trained model checkpoints)
data (Dataset raw and processed)
notebooks (Research and exploratory notebooks)
src (Main project source code)
README.md (Project overview and usage guide)
requirements.txt (Required Python packages)

Below is what each top-level folder is for, and what lives inside it including files that don't exist yet but will be added as the project progresses through its nine build steps.

checkpoints

Stores trained model weights so nothing has to be retrained from scratch to be reused later.

pitch_encoder: Weights for the small SPICE-style pitch-discovery encoder (Step 3). Trained first, on its own, and later frozen for use at test time (Step 8).
brain: Weights for the main auto-decoder "Brain" (Step 4), plus the trained barcode table. Also frozen for test-time inference (Step 8).
adversarial (future): Weights for the optional "adversary" network used in the Sabotage Method (Step 6, stretch goal). Kept separate since it's optional and not part of the core pipeline.

All checkpoint files are gitignored (binary, regenerable) only empty folder placeholders (.gitkeep) are tracked in git.

data

Everything related to the dataset: raw, in-progress, and fully processed.

raw/Audio_Speech_Actors_01-24: Original, untouched RAVDESS audio files (Step 1). Actors repeat identical sentences under different emotions, which makes it easy to confirm the system reacts to feeling, not wording. Never modified directly every other file in data can be regenerated from here.
processed/mel_spectrograms: Mel-spectrogram "pictures" of each clip (Step 2), generated with librosa. This is the reconstruction target the Brain learns to predict throughout Step 4 training.
processed/pitch_pairs: Pitch-shifted copies of each clip, paired with the exact known shift amount applied (Step 2). Used twice: to train the pitch encoder's self-supervised puzzle (Step 3), and later reused as emotion-preserving augmentation to strip pitch out of the barcode (Step 6).
processed/barcodes (future): The actual learned 128-number barcode per clip, produced during Step 4 training. Plotted in Steps 5 and 7, and reused as a starting point for evaluation in Step 9.
metadata.csv: Parsed ground-truth labels (actor ID, emotion, intensity, statement, repetition) decoded from RAVDESS filenames. Used only for grading and plotting (Steps 5, 7, 9) never fed into training, in line with the project's self-supervised design.

Raw audio and processed tensors are gitignored due to size; metadata.csv is small and tracked directly for reproducibility.

notebooks

Exploratory, interactive work for looking at things before or alongside the "real" pipeline scripts. Not part of the reproducible pipeline itself; nothing here should be required to reproduce final results.

01_explore_ravdess.ipynb: Initial look at the raw dataset sample clips, label balance across actors/emotions, early spectrogram previews (Step 1).
02_check_pitch_encoder.ipynb (future): Sanity-checks the trained pitch encoder's output against a standard pitch-detection formula, purely as a diagnostic never used as a training signal (Step 9, optional check).
03_visualize_barcodes.ipynb (future): Interactive UMAP/t-SNE exploration of barcodes, colored by true emotion label, for the before/after pitch-disentanglement comparison (Steps 5 and 7).

src

All reproducible pipeline code, organized by stage of work. This is the core of the project.

src/data_prep
Scripts that turn raw audio into everything under data/processed.
build_metadata.py: Parses RAVDESS filenames into data/metadata.csv (Step 1).
extract_mel.py: Converts raw .wav files into mel-spectrograms (Step 2).
make_pitch_pairs.py: Generates pitch-shifted clip pairs with known, recorded shift amounts (Step 2).

src/models
Architecture definitions only no training loops.
pitch_encoder.py: The small encoder network that discovers pitch via the SPICE self-supervised puzzle (Step 3).
brain.py (future): The shared "Brain" network takes (barcode, time, pitch signal) and predicts a spectrogram slice (Step 4).
barcodes.py (future): The learnable per-clip barcode table (128 numbers per clip), optimized jointly with the Brain (Step 4).
adversary.py (future): Optional small network that tries to guess pitch from the barcode, used in the Sabotage Method (Step 6, stretch goal).

src/training
Training loops one per distinct training procedure.
train_pitch_encoder.py: Pretrains the pitch encoder on the shift-matching puzzle (Step 3). Kept alive as a side-goal during later joint training to prevent pitch-representation drift.
train_joint.py (future): The core auto-decoder training loop: Brain plus barcodes plus ongoing pitch puzzle, trained together (Step 4). Also handles the "Voice Changer" augmentation retraining via pitch-shifted pairs (Step 6), since it's the same loop with extra training examples.
train_contrastive.py (future, optional): The "Math Magnet" supervised contrastive method uses true emotion labels to pull same-emotion barcodes together. Kept strictly as a labeled comparison "ceiling," never the main method (Step 6).
train_adversarial.py (future, optional): The "Sabotage Method" trains the Brain/barcodes to defeat the adversary network from adversary.py. Fully label-free (Step 6, stretch goal).

src/evaluation (future)
Measuring and visualizing results without training on labels.
plot_barcodes.py: Produces the 2D UMAP/t-SNE maps of barcodes, colored by true emotion label for inspection only the before (Step 5) and after (Step 7) pictures.
metrics.py: Computes Silhouette Score, Cluster Purity, Adjusted Rand Index, and reconstruction-quality comparisons (Step 9). Emotion labels are used here only for grading.

src/inference (future)
Running the trained system on brand-new, unseen audio.
fit_new_clip.py: Freezes the trained Brain and pitch encoder, generates a fresh random barcode for a new clip, and optimizes only that barcode until it reconstructs the new sound well (Step 8).

src/utils
Shared code used across the rest of src.
audio_utils.py: Common audio loading, framing, and preprocessing helpers, used consistently across mel extraction, pitch-pair generation, and all training scripts. Consistency here matters Step 3's puzzle depends on exact frame alignment between a clip and its pitch-shifted twin.
config.py: Central hyperparameters and paths (sample rate, mel bins, barcode size, pitch-shift ranges, learning rates) so values stay consistent across every script that depends on them.

Build Order

The codebase is meant to be built up in this order, matching the project's nine steps:

Data prep: data_prep scripts populate data/processed (Steps 1 and 2).

Pitch encoder: models/pitch_encoder.py and training/train_pitch_encoder.py (Step 3).

Joint training: models/brain.py, models/barcodes.py, training/train_joint.py (Step 4).

First evaluation: evaluation/plot_barcodes.py to confirm pitch is still leaking into the barcode (Step 5).

Disentanglement: re-run train_joint.py with pitch-pair augmentation enabled (Step 6, primary method).

Re-evaluation: evaluation/plot_barcodes.py and evaluation/metrics.py again, to confirm disentanglement worked (Step 7).

Inference: inference/fit_new_clip.py for testing on unseen audio (Step 8).

Optional comparisons: training/train_contrastive.py and training/train_adversarial.py, evaluated the same way (Step 6 extras).

Generalization: repeat the best setup on CREMA-D.

Key Design Principles

Emotion labels are never used to train anything. They exist only in data/metadata.csv and are used exclusively inside src/evaluation for coloring plots and computing grading metrics.

Pitch is self-discovered, not hand-extracted. The pitch encoder learns from a controlled pitch-shift puzzle (SPICE), not a formula, giving it a real, well-anchored meaning before it ever feeds into the Brain.

The pitch puzzle stays active throughout joint training. It is never pretrained once and abandoned this prevents the encoder's representation from drifting away from true pitch.

The Voice Changer augmentation method is the primary way pitch is stripped from the barcode. The contrastive and adversarial methods exist only as optional comparisons, not the headline approach.

Tools & Dependencies

Python, PyTorch: Brain network, barcodes, and pitch encoder
librosa: mel-spectrogram extraction and controlled pitch-shifting
umap-learn / scikit-learn (t-SNE): 2D barcode visualization
scikit-learn: Silhouette Score and Adjusted Rand Index
GPU recommended: many barcodes are optimized alongside the Brain over many training rounds

See requirements.txt for pinned versions.

Setup

git clone repo-url
cd SentiSpeak
pip install -r requirements.txt

Download the RAVDESS dataset and place it under data/raw/Audio_Speech_Actors_01-24 (not included in this repository due to size). Then run the data prep scripts in order:

python src/data_prep/build_metadata.py
python src/data_prep/extract_mel.py
python src/data_prep/make_pitch_pairs.py