# Voice Age Estimation

The `/voice` page shows an **estimated age group from voice** next to (and separate from) voice
emotion and speaker gender. It estimates an age **group** (a decade band), never an exact age, and
the labels it learned from are **self-reported** Common Voice metadata.

```
Single microphone stream (emotion_detector.py) ── Silero VAD ──┬─► HuBERT emotion model (unchanged) ─► emotion
                                                               ├─► gender tracker (unchanged)        ─► gender
                                                               └─► LiveAgeTracker (voice_age/)       ─► age group
```

- **Independent of emotion:** the emotion model `superb/hubert-base-superb-er`, its labels and its
  preprocessing are untouched.
- **Separate from gender:** age has its own frozen encoder instance, classifier and worker thread.
- **One microphone:** there is no second stream. The existing VAD chunk callback forwards each
  chunk to both the gender and the age tracker.
- **Failures stay contained:** an age failure, or a model that was never trained, is shown as a
  state and never affects emotion or gender.

## Commands

```bash
python scripts/inspect_voice_age_dataset.py          # verify access + schema, scan metadata, print statistics
python scripts/train_voice_age.py                    # subset, download selected audio, train, evaluate, save
#   --language en --max-samples-per-class 1400 --max-clips-per-speaker 3 --epochs 60 --batch-size 64
#   --seed 42 --encoder facebook/hubert-base-ls960 --head auto --output-dir models/voice_age_model
python -m unittest discover tests
```

Nothing has to be downloaded by hand. Both scripts read Common Voice directly from the Hugging Face
Hub, and re-running resumes after network interruptions.

## 1. Dataset: Common Voice

- **What was published officially:** Mozilla withdrew Common Voice from the Hugging Face Hub in
  October 2025. `mozilla-foundation/common_voice_17_0` now contains only a README pointing to the
  Mozilla Data Collective.
- **What is used:** `fixie-ai/common_voice_17_0`, a public, ungated Parquet re-upload of **Common
  Voice 17.0** with one folder per language. Common Voice is released under **CC0**, so
  redistribution is permitted. It is not an official Mozilla distribution.
- **Language and split:** English (`en`), split `validated`: every community-validated clip, a
  superset of Common Voice's own train/dev/test.
- **Schema, verified programmatically, not assumed:** `client_id`, `path`, `audio`
  (struct of encoded MP3 `bytes` and `path`, declared 48 kHz), `sentence`, `up_votes`,
  `down_votes`, `age`, `gender`, `accent`, `locale`, `segment`, `variant`, `continuation`.
  `check_schema()` fails loudly if `audio`, `age` or `client_id` is missing.
- **If the source becomes gated:** `dataset_access()` raises with the exact steps (accept the
  terms on the dataset page, then `hf auth login`). `VOICE_AGE_DATASET` can point at another
  mirror with the same schema.

## 2. How the data is acquired (without downloading 68 GB)

The English `validated` split is **138 Parquet shards, 67.7 GB**, almost all of it MP3 audio. The
pipeline never downloads a whole shard.

1. **Footer:** for each shard it reads the Parquet footer with HTTP range requests
   (`voice_age/remote_parquet.py`).
2. **Metadata only:** it fetches just the column chunks of `client_id`, `age`, `gender` and `path`
   for every row group. That is about 4 MB per shard, roughly 0.5 GB for all 138, cached as
   `data/voice_age/metadata_en_validated.parquet`.
3. **Subset:** it chooses the subset from that metadata (sections 3–5).
4. **Selected audio:** it downloads the `audio` column chunk only of the row groups (100 clips
   each, about 4 MB) that contain selected clips. It keeps only the selected clips' original MP3
   bytes, one small file per row group in `data/voice_age/audio/`. Nothing is stored twice; the
   decoded audio is never written to disk.

`datasets` is not used, because its streaming mode reads every column of every shard. Access uses
`huggingface_hub` plus `pyarrow`, and `soundfile` decodes the MP3.

## 3. Age labels (self-reported, categorical)

Common Voice contributors may optionally state an age **decade**; nobody verifies it. In English
`validated`:

| Raw label | Clips | Speakers |
|---|---|---|
| (blank) | 632,146 | – |
| teens | 106,604 | 2,827 |
| twenties | 436,832 | 9,719 |
| thirties | 256,285 | 4,718 |
| fourties *(Common Voice's spelling)* | 168,775 | 2,349 |
| fifties | 101,783 | 1,423 |
| sixties | 80,894 | 823 |
| seventies | 13,943 | 323 |
| eighties | 1,836 | 44 |
| nineties | 190 | 7 |

Overall: 1,799,288 clips from 80,104 speakers; 1,167,142 clips (64.9 %) from 22,193 speakers carry
an age.

## 4. Age-group mapping

`config.AGE_GROUP_MAPPING` (configurable):

| Age group | Common Voice labels |
|---|---|
| Teens | teens |
| Twenties | twenties |
| Thirties | thirties |
| Forties | fourties (and forties) |
| Fifties | fifties |
| Sixties | sixties |
| **Seventies+** | seventies + eighties + nineties |

Eighties (44 speakers) and nineties (7) are far too sparse to be classes, so they are merged with
seventies (323) into **Seventies+** (374 speakers). Training refuses any group with fewer than
`MIN_SPEAKERS_PER_GROUP = 60` speakers.

**One label per speaker:** 40 speakers carry more than one age label (they recorded years apart, or
were inconsistent). Each speaker gets their majority label and minority-label clips are dropped
(16,391 clips). A tied speaker would be dropped; there were none.

## 5. Speaker-wise split and subset

- **Split by speaker:** speakers, not clips, are split **70 / 15 / 15**
  (train / validation / test), stratified by age group, with seed 42.
- **Leakage check:** `verify_disjoint()` checks train ∩ validation, train ∩ test and
  validation ∩ test after the split, after subsetting and after preprocessing. It raises
  `SpeakerLeakageError` on any overlap. All three intersections are empty.
- **Subset size:** at most `AGE_MAX_SAMPLES_PER_CLASS = 1400` clips per age group in total
  (980 train, 210 validation, 210 test).
- **Per-speaker cap:** at most `AGE_MAX_CLIPS_PER_SPEAKER = 3` clips per speaker. The first pass
  takes one clip per speaker, for as many different voices as possible; the second tops up to 3.
  Without a cap, one speaker would contribute up to 50,669 clips.
- **Download cost:** row groups rich in rare age groups are visited first, so fewer audio row
  groups need downloading.

Used after preprocessing (clips / speakers). Speaker overlap between the splits is 0, 0 and 0:

| Age group | Train | Validation | Test |
|---|---|---|---|
| Teens | 955 / 955 | 204 / 204 | 206 / 206 |
| Twenties | 961 / 961 | 207 / 207 | 209 / 209 |
| Thirties | 956 / 956 | 207 / 207 | 207 / 207 |
| Forties | 956 / 956 | 206 / 206 | 210 / 210 |
| Fifties | 959 / 959 | 203 / 203 | 204 / 204 |
| Sixties | 949 / 556 | 204 / 122 | 204 / 121 |
| Seventies+ | 726 / 261 | 160 / 56 | 163 / 56 |
| **Total** | **6,462 / 5,604** | **1,391 / 1,205** | **1,403 / 1,213** |

- **Clips:** 9,474 selected and 9,256 usable. 218 were skipped: 157 shorter than 1 s after
  trimming and 61 silent. None failed to decode.
- **Download:** the 9,474 clips sat in 2,185 row groups, about 9.2 GB transferred. About 0.4 GB
  of MP3 was kept.

Sixties and Seventies+ have fewer speakers than the quota needs, so those groups have fewer
clips and speakers. Class weighting compensates during training.

**What is never a model input:** speaker identity, file names and metadata. `client_id` is used
only to split, and the model sees audio only. No augmentation is used.

## 6. Audio preprocessing

`voice_age/audio.py`, identical for training and live use:

- decode the MP3 (or WAV/FLAC/OGG for uploads), convert to **mono**, resample to **16 kHz**,
  as **float32**;
- reject the clip if it contains NaN/inf, is silent (RMS < −50 dBFS), or cannot be decoded;
- trim leading and trailing silence (frames more than 40 dB below the loudest frame);
- require at least **1.0 s** after trimming; cut clips over **6.0 s** to their centre 6 s
  (live speech is split into ≤ 6 s pieces instead).

A clip that fails any check is skipped and its reason counted (`skipped_clips` in the metadata).
One malformed clip never stops training.

## 7. Model architecture

```
16 kHz speech → facebook/hubert-base-ls960 (frozen, 94.4 M params, never trained)
              → hidden states of ONE layer, mean-pooled over time (padding masked)
              → StandardScaler → lightweight classifier → 7 age-group probabilities
```

- **Frozen self-supervised encoder plus a light head.** This is a standard, CPU-friendly baseline
  for speaker traits (as in SUPERB-style probing). Fine-tuning a 94 M-parameter model on about 10k
  clips would be slow on a Mac and prone to overfitting.
- **Why this encoder:** `hubert-base-ls960` was already in the local Hugging Face cache, used by
  the gender model, so no new large download was needed with 4 GB of free disk. The age pipeline
  loads its **own** instance, truncated to the selected layer.
- **Selection on validation speakers only:**
  - the encoder layer, compared with class-balanced logistic regression;
  - then the head: logistic regression (C ∈ {0.03, 0.1, 0.3, 1}) or a one-hidden-layer MLP (256 or
    512 units, dropout, class-weighted loss, AdamW, early stopping on validation macro F1 with
    patience 8, best checkpoint kept).
- **Configurable:** the encoder (`--encoder` / `VOICE_AGE_ENCODER`), head (`--head`) and device
  (`VOICE_AGE_DEVICE`; MPS or CUDA when available, otherwise CPU).

**Selected on validation speakers:**

- **Encoder layer:** 5. Validation macro F1 by layer, from 0 to 12: 0.283, 0.285, 0.315, 0.321,
  0.331, **0.351**, 0.334, 0.333, 0.298, 0.294, 0.304, 0.309, 0.314. The middle layers carry the
  most speaker-trait information.
- **Head:** an MLP with 768 inputs, 512 hidden units (ReLU, dropout 0.5) and 7 outputs, giving
  **397,319 trainable parameters**. Early stopping chose epoch 3 of 11.
- **Head comparison (validation macro F1):** logistic regression 0.332–0.362; MLP-256 0.373;
  MLP-512 **0.385**.
- **Frozen encoder:** 94.4 M parameters, never trained. At inference it is truncated to its first
  5 layers.

## 8. Evaluation

Test speakers are never used for any choice; the test set is scored once. Reported metrics:

- accuracy, balanced accuracy, macro and weighted F1, per-class precision, recall and F1;
- the confusion matrix (`voice_age/results/confusion_matrix.png`);
- a majority-class baseline;
- "within one group" accuracy (age groups are ordered);
- a **speaker-level** score: clip probabilities averaged per speaker, closer to the live
  behaviour of averaging several windows.

### Results on 1,403 clips from 1,213 unseen test speakers (evaluated once)

| | Model | Majority baseline (always "Twenties") | Chance (7 groups) |
|---|---|---|---|
| Accuracy | **0.376** | 0.149 | 0.143 |
| Balanced accuracy | **0.382** | 0.143 | 0.143 |
| Macro F1 | **0.377** | 0.037 | |
| Weighted F1 | **0.372** | | |
| Macro precision / recall | 0.375 / 0.382 | | |
| Within one age group | **0.742** | | |
| Speaker-level (probabilities averaged per speaker) | acc 0.361, bal-acc 0.393, macro F1 0.361 | | |

Validation (used for selection): accuracy 0.385, balanced accuracy 0.392, macro F1 0.385. It is
close to test, so selection did not overfit the validation speakers noticeably.

| Age group | Precision | Recall | F1 | Test clips / speakers |
|---|---|---|---|---|
| Teens | 0.51 | 0.49 | 0.50 | 206 / 206 |
| Twenties | 0.38 | 0.41 | 0.40 | 209 / 209 |
| Thirties | 0.31 | 0.34 | 0.32 | 207 / 207 |
| Forties | 0.30 | **0.20** | 0.24 | 210 / 210 |
| Fifties | 0.28 | 0.28 | 0.28 | 204 / 204 |
| Sixties | 0.37 | 0.38 | 0.38 | 204 / 121 |
| Seventies+ | 0.47 | 0.57 | 0.52 | 163 / 56 |

Confusion matrix (rows = self-reported group, columns = prediction, order as above):

```
[[101, 55, 22,  7,  9,  8,  4],
 [ 47, 86, 43, 16, 12,  4,  1],
 [ 19, 43, 70, 34, 28,  7,  6],
 [ 14, 21, 53, 43, 38, 27, 14],
 [ 11, 15, 21, 25, 57, 51, 24],
 [  4,  4, 14, 14, 36, 78, 54],
 [  2,  2,  4,  5, 23, 34, 93]]
```

**Honest reading:**

- **Better than chance, but modest.** The model is clearly better than chance and the majority
  baseline (macro F1 0.377 vs 0.037), yet it gets the exact decade right for only about
  **3 in 8** clips.
- **Mostly near misses.** Errors are mostly neighbouring groups: 74 % of predictions are within
  one group.
- **Middle ages are hardest.** Forties (recall 0.20) and fifties are hard to tell apart. The
  extremes (teens, seventies and over) are easiest.
- **Seventies+ is uncertain.** It has only 56 test speakers, so its scores are the least precise.
- **Per speaker is not better here.** Most test speakers have a single clip, so averaging per
  speaker adds little.

## 9. Real-time inference

- **Input:** `LiveAgeTracker` receives every 32 ms VAD chunk of the existing microphone stream.
- **Windowing:** speech (with about 100 ms of padding) is collected until
  `AGE_INFERENCE_WINDOW_SECONDS = 3.0` s of speech is buffered, or a phrase of at least 2.0 s ends
  with 0.7 s of silence. A shorter phrase gives the state **"Insufficient speech"**; the model is
  never forced to predict from a fraction of a second.
- **Inference:** each window is classified on the tracker's own `VoiceAgeWorker` thread. The
  emotion worker only appends chunks to a list.
- **Smoothing:** the shown estimate is the average of the last `AGE_SMOOTHING_WINDOWS = 5` window
  probabilities. 8 s of silence resets it, because the next speaker may be someone else.
- **States:** `model_not_trained` ("Voice age model not trained yet."), `loading_model`,
  `listening`, `insufficient_speech`, `estimated`, `unavailable`, `stopped`.

API:

| Endpoint | Purpose |
|---|---|
| `GET /voice/status` | the existing response (emotion, gender, …) plus `"age": {...}` (state, age_group, confidence, probabilities, windows, label, disclaimer) |
| `GET /voice/age/status` | model available / path / version / classes / dataset / inference ready |
| `POST /voice/age` | multipart `audio` file → `{"success", "age_group", "confidence", "probabilities", "speech_seconds", "label", "disclaimer"}`; `422` for insufficient speech, `503` without a model |

## 10. Limitations and scientific caveats

- **Self-reported, categorical labels.** Common Voice age is an optional decade chosen by the
  contributor. It is never verified, sometimes inconsistent, and not biological age.
- **Age group, not age.** The model outputs a decade band, and neighbouring bands overlap
  acoustically: a 29-year-old and a 31-year-old sound alike. Confusion between adjacent groups is
  expected.
- **Approximate.** Voice-based age estimation is inherently uncertain (see the metrics above).
- **Recording conditions matter.** Microphone, distance, room, noise, accent, language, speaking
  style, emotion and health all change the voice. The model learned from Common Voice read speech,
  mostly through consumer microphones.
- **Demographics.** Common Voice English contributors are not representative: about 74 % of
  gender-labelled clips are male, twenties dominate, and older age groups are much sparser.
  Accuracy can differ between groups.
- **Not for decisions.** Do not use it for identity or age verification, access control, or any
  high-stakes decision.
