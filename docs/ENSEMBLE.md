# Multi-dataset HuBERT ensemble

The app's acoustic emotion classifier is now an ensemble of per-corpus
fine-tuned HuBERT models instead of a single pretrained one. Everything
downstream of the classifier — VAD, windowing, temporal smoothing, the
manual > voice > face fusion, the robot trigger, the UI — is unchanged.

**Nothing here is active until you build it.** With no
`models/ensemble/manifest.json` on disk the app runs exactly as before, on
`superb/hubert-base-superb-er`. Confirm either way without starting the app:

```
python -m ser check
```

---

## 1. Label space

Four canonical classes, in the order the app indexes them:
`happy, angry, sad, neutral` (`ser/labels.py`, `CANONICAL_EMOTIONS`).

Each corpus maps into that space explicitly, label by label. An emotion code
that the map does not mention raises `LabelError` rather than being silently
dropped, so a differently-packaged release fails loudly.

| Corpus | → happy | → angry | → sad | → neutral | dropped |
|---|---|---|---|---|---|
| IEMOCAP | `hap`, `exc` | `ang` | `sad` | `neu` | `fru`, `fea`, `sur`, `dis`, `oth`, `xxx` |
| CREMA-D | `HAP` | `ANG` | `SAD` | `NEU` | `DIS`, `FEA` |
| RAVDESS | `03` | `05` | `04` | `01` | `02` calm, `06` fear, `07` disgust, `08` surprise |
| MSP-IMPROV | `H` | `A` | `S` | `N` | `O`, `X` |
| MSP-Podcast | `H` | `A` | `S` | `N` | `U`, `F`, `D`, `C`, `O`, `X` |

Two judgement calls worth knowing about, both reversible from the CLI:

- **IEMOCAP `exc` → happy.** The standard 4-class IEMOCAP protocol merges
  excitement into happiness, and nearly all published 4-class numbers assume
  it. `--no-merge-excitement` drops it instead — but then your numbers are not
  comparable to the literature's.
- **RAVDESS `calm` dropped.** It is acted low-arousal, neutral-valence speech
  with no counterpart in the other four corpora; folding it into neutral
  inflates neutral recall with a voice quality nothing else has seen.
  `--ravdess-calm` maps it to neutral instead.

## 2. Speaker separation

Splits are over **speakers**, never utterances (`ser/splits.py`). Each corpus
gives every speaker dozens of clips, so a random utterance split lets the model
recognise the voice rather than the emotion, and validation accuracy stops
meaning anything.

- IEMOCAP: **by session** by default (1–3 train, 4 val, 5 test) — stronger than
  speaker-disjoint, because the two actors of a session also share recording
  conditions and a dialogue partner. `--iemocap-speaker-split` to override.
  Note the speaker is read from the *turn* id (`..._F000`), not the dialog name
  — the dialog name names the session's lead actor, so using it would put both
  actors of a session under one id.
- MSP-Podcast: its **official** Train/Development/Test1 split, which is already
  speaker-disjoint and is what published numbers use. `--msp-resplit` to
  override. Clips with `SpkrID = Unknown` are dropped, because they cannot be
  held out of each other's splits.
- Everything else: greedy speaker-disjoint 70/15/15, balanced by utterance
  count. Greedy-on-deficit matters here — RAVDESS has 24 speakers and
  MSP-IMPROV only 12, so a single large speaker can otherwise swallow a whole
  validation set.

`assert_no_speaker_leakage()` runs on **every** split map that is written,
including the pooled ones the ensemble is fitted on, and it raises rather than
warns. It also catches the same audio file landing in two splits.

## 3. Windows — and why VAD stays out of the classifier

One window spec (`ser/audio.py`, `WindowSpec`) is shared by training and by
both runtime paths: **3.0 s windows, 1.5 s hop, 2.0 s minimum, ≥50 % speech.**

VAD's only output is *which samples are speech*. Its 32 ms frames are never fed
to HuBERT; they decide which spans become a window, and the classifier only
ever sees the 2–4 s result. The two stay in separate modules and separate
pipeline stages.

Training on whole utterances and classifying 3 s live windows would be a silent
train/test mismatch — HuBERT's pooled representation of 12 s is not its
representation of 3 s, and each member's calibration (which the ensemble
depends on) is fitted to whatever length it saw. Hence one spec everywhere.

## 4. Per-corpus models, not a concatenation

`ser/train.py` fine-tunes one model per corpus (or per `--group`). Concatenating
the five would let the largest corpus set the decision boundary for all of them,
and would produce members that agree — which is the one thing that makes an
ensemble pointless.

Each member starts from `superb/hubert-base-superb-er`, gets a fresh 4-class
head in canonical order, freezes the CNN feature encoder (a generic waveform
front-end; fine-tuning it on a few thousand utterances mostly memorises the
recording channel), and trains with inverse-frequency class weights. The
checkpoint kept is the one with the best **validation macro-F1**, scored at
utterance level.

## 5. Calibration

Members are combined by probability, so their probabilities have to mean the
same thing. A fine-tuned head is over-confident; averaging an over-confident
member with a well-behaved one lets the over-confident one win every
disagreement regardless of who is right.

`ser/calibrate.py` fits **temperature scaling** and **vector scaling** on each
member's own validation logits by L-BFGS, and keeps whichever gives the lower
validation NLL (ties to temperature, which cannot change the model's ranking).
Before/after NLL and ECE for every candidate are written to
`runs/<run>/calibration.json`, so a calibrator that did not help is visible
rather than silently applied.

## 6. Combining — all three, compared

`ser/ensemble.py` fits all three and `select_best` picks the winner:

| Combiner | Parameters | What it can learn |
|---|---|---|
| `average` | none | nothing — the baseline the others must beat |
| `weighted` | one weight per member, on the simplex | "trust MSP-Podcast more than RAVDESS" |
| `stacking` | multinomial LR over every member's log-probs | "believe CREMA-D about anger, IEMOCAP about sadness" |

No weight is hardcoded. The weighted average is parameterised as
`softmax(theta)` so weights stay non-negative and sum to one by construction;
the stacker works on **log**-probabilities so it strictly contains the
averaging family, and is L2-regularised.

**Selection is guarded against itself.** The pooled validation set is halved
*by speaker*: combiners are fitted on one half and the winner is chosen on the
other. Fitting and selecting on the same rows would hand the win to stacking by
construction, every time. Once chosen, the winner is refitted on the whole
validation pool. Test data is touched exactly once, at report time.

A stacker can output a class no member's argmax supports — that is legitimate
(it learns per-class bias corrections), and it is exactly when the
`agreement: 0.0` readout in the logs earns its keep.

## 7. What gets reported

`runs/<run>/comparison.txt` and `.json`, printed at the end of a run:

- A pooled-test table of **every** model on the same utterances: the original
  IEMOCAP-only `superb/hubert-base-superb-er` baseline, every individual
  dataset model, and all three combiners, with the selected one marked.
- For each: accuracy, macro-F1, UAR, NLL, ECE, per-class precision/recall/F1
  and the confusion matrix.
- A per-dataset macro-F1 matrix (rows = model, cols = test corpus). The
  diagonal is in-corpus skill; the off-diagonal is transfer. A member that only
  wins on its own corpus is still useful — the disagreement is what the
  combiner is fitted to exploit.

Macro-F1 is the headline rather than accuracy: MSP-Podcast is dominated by
neutral, and accuracy alone rewards a model that rarely predicts sad.

## 8. Running it

```bash
pip install -r requirements_ser_training.txt

# everything, end to end
python -m ser all --run r1 \
  --iemocap     /data/IEMOCAP_full_release \
  --cremad      /data/CREMA-D \
  --ravdess     /data/RAVDESS \
  --msp-improv  /data/MSP-IMPROV \
  --msp-podcast /data/MSP-PODCAST

# or stage by stage, resuming where you stopped
python -m ser prepare   --run r1 --cremad ... --ravdess ...
python -m ser train     --run r1 --epochs 4
python -m ser infer     --run r1
python -m ser calibrate --run r1
python -m ser ensemble  --run r1
python -m ser report    --run r1
```

Use whichever corpora you actually have — two is enough for an ensemble. Only
CREMA-D and RAVDESS are open downloads; IEMOCAP, MSP-IMPROV and MSP-Podcast are
licence-gated, so the scanners are written against each release's own layout and
will tell you plainly if a root does not look right.

Useful flags: `--group msp=msp_improv+msp_podcast` (train one model on two
corpora), `--max-train-windows` (cap MSP-Podcast), `--vad rms` (faster indexing),
`--no-baseline` (skip the original-model comparison).

Artefacts: `runs/<run>/` holds the splits, window index, cached logits,
calibration and comparison; `models/<member>/` holds each fine-tuned model;
`models/ensemble/manifest.json` is what the app loads.

## 9. Runtime and debugging

`ser/runtime.py` exposes `predict(audio) -> {emotion: prob}` — the exact
signature the single model had, which is the whole integration.
`emotion_detector.build_emotion_model()` decides which object the app gets and
falls back to the single model, loudly in the log, if the ensemble is missing,
half-built, or fails to load.

`predict_detailed()` additionally returns every member's own distribution.
That surfaces in three places:

- **Console**, per window: `[VOICE MEMBERS]` and `[VOICE DISAGREE]` live,
  `[UPLOAD …]` per-window and a per-member summary for uploaded files.
- **API**: `/voice/status` carries `member_probabilities`, `member_top`,
  `member_ms`, `member_agreement`, `member_dissenting`, `combiner`, `ensemble`;
  `/status` carries the same for the uploaded file under `upload_analysis`.
- **UI**: a collapsed "Ensemble" disclosure in the voice panel and a
  "Per-model breakdown" one in the upload panel. Both hide themselves entirely
  when a single model is running.

Environment variables:

| Variable | Default | Effect |
|---|---|---|
| `VOICE_USE_ENSEMBLE` | `1` | `0` forces the single pretrained model (A/B against the old system) |
| `VOICE_ENSEMBLE_MANIFEST` | `models/ensemble/manifest.json` | alternative manifest |
| `SER_ENSEMBLE_MEMBERS` | all | comma-separated subset, e.g. `iemocap,cremad` |
| `VOICE_LOG_MEMBERS` | `1` | `0` silences the per-window member log |
| `VOICE_UPLOAD_HOP_SECONDS` | `1.5` | hop between uploaded-file windows |

**Cost.** N members means N forward passes per window. On CPU that is the one
real risk to live latency — measure with `python -m ser check`, which prints
per-member milliseconds. If the live path gets tight, `SER_ENSEMBLE_MEMBERS`
restricts the ensemble without retraining anything (it degrades to an
unweighted average of the members that loaded, and says so, rather than reusing
weights fitted for a different member set).

## 10. What changed in the existing app

| File | Change |
|---|---|
| `emotion_detector.py` | `build_emotion_model()` factory; `predict_detailed()` on the single model so both kinds share one interface; `_analyse` records and logs members; new status fields; `UPLOAD_HOP_SECONDS` |
| `app.py` | uploaded-file windows now **overlap** (were non-overlapping) and are speech-weighted; per-member aggregation, logging and state |
| `templates/index.html` | two self-hiding disclosure panels |

Untouched: microphone capture, resampling, VAD, the live smoothing/streak
logic, the manual > voice > face fusion, upload pinning, the robot trigger.
