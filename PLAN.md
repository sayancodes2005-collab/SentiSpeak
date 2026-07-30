# Project Plan v3: Sorting Voices by Feeling Using Tiny "Barcodes"
### (Pitch is now discovered by the model itself, not hand-extracted)

## The Big Picture

You want a system that listens to a voice clip and squeezes it into a short list of 128 numbers (the "barcode"). By looking at where this barcode lands compared to others, you should be able to tell the speaker's mood — without ever directly teaching the computer "this is anger" during training. Mood labels are only used at the very end, to grade your work, never to train it.

This version fixes a flaw you correctly caught: simply letting a "pitch line" be optimized purely to help reconstruction, with no anchor, gives the network no actual reason to make that slot mean "pitch" specifically — it could end up meaning almost anything. The fix below lets the model genuinely discover pitch on its own, through a self-supervised puzzle, instead of either (a) being handed pitch numbers from a formula, or (b) being given a free, meaningless slot with no anchor at all.

---

## Step 1: Pick the Right Starting Dataset

Don't jump into a huge, messy dataset right away.

**Start with RAVDESS**
- Actors say the exact same sentences in different moods (happy, sad, angry, calm, fearful, disgusted, surprised, neutral).
- Same words, different feeling — so it's easy to confirm your system reacts to emotion, not wording.
- Small enough that your computer won't choke on it.

**Move to CREMA-D next** — bigger, more speakers, more natural emotional speech.

**Save IEMOCAP for later (optional, advanced)** — longer, natural conversations; harder, but more impressive if it works.

Get the full pipeline working end-to-end on RAVDESS before moving on.

---

## Step 2: Prepare Your Audio, Including Pitch-Shifted Copies

Convert every clip into a mel-spectrogram picture (a picture of which pitches are loud at which moments) using `librosa`.

**Also create pitch-shifted copies of every clip, by a known, controlled amount** (e.g., shift some copies up by 2 semitones, some down by 3, and so on — `librosa.effects.pitch_shift()` does this in one line, and you choose and record the exact shift amount yourself). You were already planning to do this for the Voice Changer augmentation step — now it does double duty, because you'll also use these same shifted pairs to teach the model to discover pitch on its own in the next step.

---

## Step 3: Teach the Model to Discover Pitch Itself, Before Touching the Main Brain

This is a separate, smaller training task, done first, on its own.

**What to build:** a small encoder network whose only job is to look at a short slice of sound and output a single number (or a small handful of numbers).

**The self-supervised puzzle:**
1. Take one clip and one pitch-shifted copy of it, where you know the exact shift amount you applied (say, "+2 semitones").
2. Feed both versions, frame by frame, through the same small encoder.
3. Require that the difference between the encoder's output for the shifted version and the original version be proportional to the known shift amount you applied.
4. Nudge the encoder's settings to make this hold true across many clips and many different shift amounts.

Why this works and actually solves your concern: the *only* thing that reliably changes between a clip and its pitch-shifted twin is pitch. The words, the timing, the emotion — none of that changes. So the one and only way the encoder can succeed at this puzzle is by genuinely learning to track pitch, on its own, with nobody ever telling it "this is pitch" directly. This is a real, published technique (it's called SPICE in the research world), and it has been shown to estimate pitch about as well as fully label-trained systems, without ever using hand-labeled pitch data.

**Important strict rule going forward:** keep this pitch-discovery puzzle active even later, during the main joint training in Step 4 — don't just pretrain it once and then let it drift freely. If you let this encoder later be updated using only the reconstruction goal, with no further pressure to keep tracking real pitch shifts, it can slowly drift back into absorbing other information, the same problem you originally worried about. Keeping the puzzle task alive as a small ongoing side-goal throughout training prevents that drift.

---

## Step 4: Build "The Brain" and "The Barcodes," Fed by the Learned Pitch Encoder

**Brain's inputs:** barcode (128 numbers) + time position + the learned pitch encoder's output at that exact moment (from Step 3 — not a raw extracted formula value, a value the model discovered itself).

**Training loop, in plain words:**
1. Pick one clip.
2. Feed its barcode, a time position, and the pitch-encoder's output at that time into the Brain.
3. The Brain guesses what the sound picture should look like at that point.
4. Compare the guess to the real picture.
5. Nudge the Brain's settings and that clip's barcode to close the gap. Also keep nudging the pitch-encoder slightly, but always together with its self-supervised pitch puzzle from Step 3 running alongside, not instead of it.
6. Repeat across all clips, many times.

This overall design — one shared Brain, one private barcode per item, both updated together to match real data — is called an "auto-decoder," first popularized for 3D shapes. You're applying the same idea to sound, with a self-discovered pitch signal feeding in alongside it. This lineage is worth stating clearly in your report.

**Goal for this step:** confirm reconstruction works well. Don't worry about emotion sorting yet.

---

## Step 5: Confirm the Barcode Still Picked Up Pitch By Accident

Even with pitch fed in separately, the barcode can still end up carrying pitch information too, since nothing yet stops it from doing so. Plot all barcodes in 2D (UMAP or t-SNE), color by true emotion label (only for checking, never for training), and confirm clips are grouping by vocal depth rather than mood. This is your "before" picture.

---

## Step 6: Strip Pitch Out of the Barcode — Now Safe to Do Aggressively

Because the Brain now gets pitch information from a separate, dedicated, self-discovered source, you can push hard to remove pitch from the barcode without breaking reconstruction.

**Easiest, validated by real published systems: Voice Changer Method (Augmentation)**
- Reuse your pitch-shifted training copies (from Step 2) as extra training examples with the same emotion label, so the barcode learns that pitch is irrelevant to emotion.

**Second option, only as a labeled comparison, never your main method: The Math Magnet (Supervised Contrastive Loss)**
- Uses true emotion labels to pull same-emotion barcodes together. This makes that specific run supervised — keep it only as a "ceiling" comparison in your report, not your headline method.

**Third option, optional, for extra credit: The Sabotage Method (Adversarial Disentanglement)**
- A small second network tries to guess pitch from the barcode; you actively defeat it. Fully label-free, since it only needs the self-discovered pitch signal from Step 3, never emotion labels. Attempt only once Steps 3–6 (augmentation version) are solid.

---

## Step 7: Re-check the Map, and Re-check Sound Quality

Redo the 2D map and compare before/after — emotions should now cluster regardless of vocal depth.

Also compare reconstruction quality before and after stripping pitch from the barcode. It should barely move, since pitch never depended on the barcode in the first place. Showing this side by side is strong proof your fix worked.

---

## Step 8: Test on Brand-New, Never-Seen Audio

1. Freeze the Brain and the pitch encoder completely.
2. Run the new clip through the frozen pitch encoder to get its pitch signal (no training happening here, just using what was already learned).
3. Create a fresh, random barcode for the new clip.
4. Nudge only the new barcode, using (barcode, time, pitch signal) fed to the frozen Brain, until it can recreate the new sound well.
5. Plot the new barcode on your map. Landing inside the right emotion group means your system correctly read the mood of a never-seen clip.

---

## Step 9: How to Measure Success (Without Cheating Into Supervised Learning)

Use true emotion labels only to grade, never to train:

- **Silhouette Score** — how tight and separated your clusters are, no labels needed.
- **Cluster Purity or Adjusted Rand Index** — compares your unsupervised groupings to true labels, only for grading.
- **Reconstruction quality** — confirm it stays stable before/after stripping pitch from the barcode.
- **Pitch-discovery quality check (optional, technical)** — you can verify your self-discovered pitch encoder is sensible by checking that its output correlates well with a standard pitch-detection formula's output, purely as a sanity check, never as a training signal.

---

## Suggested Order of Work (Timeline-Style)

1. Download RAVDESS, convert clips to mel-spectrogram pictures, create pitch-shifted copies with known shift amounts.
2. Pretrain the small pitch-discovery encoder using the self-supervised shift-matching puzzle.
3. Build the Brain (barcode + time + learned pitch signal) and the barcode list; confirm reconstruction works well, keeping the pitch puzzle alive as a side-goal.
4. Plot barcodes, confirm pitch-confusion still shows up inside the barcode itself.
5. Add pitch-shifted training copies as emotion-preserving augmentation (Voice Changer Method); retrain.
6. Re-plot barcodes, recheck reconstruction stayed stable, compute silhouette score and purity.
7. Implement the test-time process for brand-new clips.
8. (If time allows) Try the Math Magnet method as a labeled comparison baseline only.
9. (Stretch goal) Try the Sabotage Method, fully label-free.
10. Move to CREMA-D and repeat your best setup to show it generalizes.

---

## Tools You'll Actually Need

- Python, PyTorch (Brain network, barcodes, and the small pitch-discovery encoder)
- `librosa` (mel-spectrogram pictures and pitch-shifting with known amounts)
- `umap-learn` or `scikit-learn`'s t-SNE (2D maps)
- `scikit-learn` (silhouette score and Adjusted Rand Index)
- A GPU helps a lot, since you're training many barcodes alongside the Brain over many rounds.

---

## One Important Mindset Note

Keep a clear line between different uses of information:
- **Never** let emotion labels directly tell the network what to output.
- **Never** let emotion labels pull or push barcodes during training if you want the project fully self-supervised (the contrastive method does this — keep it as a labeled comparison only).
- **Always feel free** to use emotion labels afterward, just to color plots and compute scores.
- **Pitch shift amounts you apply yourself during augmentation are not emotion labels** — they're a number you chose and control (like "+2 semitones"), used only to teach the model what pitch is. This keeps the whole system self-supervised, while still giving the model a real, well-defined anchor to discover pitch on its own, exactly as you wanted.