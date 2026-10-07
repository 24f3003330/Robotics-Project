# Handwriting analysis: data, training and models

The handwriting page runs **two separate experiments**. They share the preprocessing code and the
Transformer architecture, but nothing else: each has its own dataset, labels, checkpoint and
API route. **Neither experiment's labels are ever converted into the other's.**

| Experiment | Dataset | Labels | Status |
|---|---|---|---|
| **A. EMOTHAW** | EMOTHAW (research dataset, DASS-42 questionnaire scores) | **Anxiety, Depression, Stress**: each normal / elevated | research-backed labels |
| **B. 4-class** | `seed_4class` now, `real_4class` later | **Happy, Neutral, Sad, Angry** | **PROTOTYPE**: seed data from graphology references, *not* scientifically validated |

| | Experiment A: EMOTHAW | Experiment B: 4-class |
|---|---|---|
| Import / build data | `python -m handwriting.import_emothaw` | `python -m handwriting.generate_seed_data` (seed) · `python -m handwriting.import_4class` (real) |
| Train | `python -m handwriting.train` | `python -m handwriting.train --dataset seed_4class` (later `real_4class`) |
| Checkpoint | `models/handwriting_transformer.pt` | `models/handwriting_4class.pt` |
| API | `POST /handwriting/analyze` | `POST /handwriting/4class/analyze` |

The handwriting modality is also separate from camera and voice:

| Modality | Model | Output |
|---|---|---|
| Camera | YuNet + FER+ | the existing camera emotions |
| Voice | HuBERT | happy / angry / sad / neutral (+ separate gender) |
| Handwriting A | pen-trajectory Transformer | **Anxiety, Depression, Stress: normal / elevated** |
| Handwriting B | the same Transformer, its own checkpoint | **Happy / Neutral / Sad / Angry: PROTOTYPE** |

The modalities share no model, no data and no fusion. The `/handwriting` page loads neither the
camera model nor the voice model. Handwriting B's Happy / Neutral / Sad / Angry come from its
own dataset; they are not the voice model's labels and are not fused with them.

Nothing in `data/` is committed (`data/*` is git-ignored). No trained handwriting checkpoint
ships with the app. Until one is trained, the page reports "not trained/installed" for that
model and makes no prediction.

---

# Part A: EMOTHAW (Anxiety / Depression / Stress)

## 1. Dataset: EMOTHAW

> L. Likforman-Sulem, A. Esposito, M. Faundez-Zanuy, S. Clémençon, G. Cordasco,
> "EMOTHAW: A Novel Database for Emotional State Recognition From Handwriting and Drawing",
> *IEEE Transactions on Human-Machine Systems* 47(2), 2017.

- **Access:** request it via <https://forms.gle/3KXifbvuqSvvrd757> (Università della Campania,
  <https://www.psicologia.unicampania.it/the-lab/our-activities>). It is licensed for
  non-commercial research only. Never commit it.
- **Recordings:** each participant did seven tasks on a Wacom tablet. Every recording is one
  `.svc` file named `uXXXXXsYYYYY_hwZZZZZ.svc` (writer, session, task). Its rows are
  `x y timestamp(ms) pen_status azimuth altitude pressure`.

| Task | Content |
|---|---|
| 1 | copy of two pentagons |
| 2 | copy of a house |
| 3 | four Italian words in capital letters |
| 4, 5 | loops (left hand, right hand) |
| 6 | clock drawing |
| 7 | a sentence in cursive |

- **Labels:** each participant filled in the DASS-42 questionnaire. The dataset gives a
  depression, an anxiety and a stress score per participant. As in the EMOTHAW paper, each
  score is split into **normal** and **elevated** (above the normal band):

| Target | Normal | Elevated |
|---|---|---|
| Anxiety | 0-7 | > 7 |
| Depression | 0-9 | > 9 |
| Stress | 0-14 | > 14 |

  These bounds come from `config.EMOTHAW_HEADS` (`dass_cutoff`). The labels belong to the
  **participant**, so every recording of a participant carries the same three labels. The three
  targets co-occur, so they are three independent binary outputs, not one class.

**Not verified here:** the dataset was not available on the machine this code was written on,
so the label table's exact file name, sheet and column names were not seen. The importer
therefore recognises columns by name and stops rather than guessing (see below).

---

## 2. Layout and import

```
data/emothaw/
    raw/                      <- the dataset exactly as received (never modified)
    processed/                <- written by the importer
        samples/uXXXXX/uXXXXXsYYYYY_hwZZZZZ.csv
        manifest.csv          one row per recording: writer, session, task, source, sha256, ...
        labels.csv            one row per writer: anxiety, depression, stress + the raw values
        import_report.json    counts, pen value ranges, skipped files
        preview/              optional PNG renders (--preview)
```

```bash
.venv/bin/python -m handwriting.import_emothaw --preview 2
```

**Recordings**
- The importer finds `.svc` files at any depth under `raw/`.
- The **writer id comes from the file name** (`uXXXXX`), never from the folder. In the public
  EMOTHAW sample, 7 of 64 folders are named after a different writer than the files inside them.
- Files with another name pattern are listed and skipped.
- Identical duplicate copies are imported once. A *different* file with the same
  writer/session/task is skipped and reported.
- **Orientation:** the tablet was used rotated by 90°. The importer converts to screen
  orientation (`x = -tablet_y`, `y = -tablet_x`). Check the `--preview` PNGs: the writing must
  be upright and not mirrored.
- **Pen channels:** pen channels are scaled as follows.
  - pressure: raw / 1023 → 0..1
  - altitude: raw × 90 / 1023 → degrees
  - azimuth: raw × 360 / 4095 → degrees
  - If any file exceeds those full-scale values, the import **stops** instead of mis-scaling.
    Pass `--pressure-max` / `--altitude-max` / `--azimuth-max` once you know the device's range.
- **Output rows:** one per raw tablet sample. Nothing is added or dropped, and in-air samples
  (pen up) are kept.

**Label table**
- The DASS table that comes with the dataset is found automatically under `raw/` (`.csv`,
  `.tsv`, `.txt`, or `.xlsx` with `openpyxl` installed). Otherwise, pass it with `--labels`.
- Columns are recognised by name:
  - writer: `writer_id`, `user`, `subject`, `participant`, `id`, ...
  - `anxiety`, `depression`, `stress`, plus a few variants, including the Italian ones
  - Name them explicitly with `--writer-col`, `--anxiety-col`, `--depression-col` and
    `--stress-col` if they are named differently or are ambiguous.
- `--label-encoding`:
  - `auto` (default): numbers above 1 are read as DASS-42 scores, words as severity labels.
  - `dass`: DASS-42 scale scores (0-42), split at the bounds above.
  - `severity`: `normal` → normal; `mild`, `moderate`, `severe` or `extremely severe` →
    elevated.
  - `binary`: already split (`0`/`1` or `normal`/`elevated`, with 1 = elevated).
  - A column holding **only 0 and 1** could be either scores or binary labels, so `auto`
    stops and asks you to choose.
- Scores outside 0..42 are rejected.
- A participant missing any of the three scores is left out and listed. Nothing is filled in.
- A participant listed twice with different labels stops the import.

The importer prints and saves:
- participants with recordings
- participants with labels, and those with both
- normal/elevated counts per target
- recordings per task
- raw pen maxima
- the median sample interval

---

## 3. Preprocessing and features

The same code runs for training files and for browser canvas input
(`handwriting/preprocessing.py`):

1. Drop non-finite rows and sort by time.
2. Trim to the span from the first to the last pen-down point. In-air movement between strokes
   is kept, because the tablet records hover.
3. Resample to at most 50 Hz, always keeping stroke starts and ends.
4. Compute per-point features:
   - position within the writing's bounding box (scale-free)
   - dx, dy
   - log time step
   - speed, signed acceleration
   - heading (sin, cos) and turn angle
   - pen state, stroke start, stroke length
   - **pressure** and **pen altitude**
5. Standardise with the **training-set** mean and std (stored in the checkpoint), and clip to ±6.
6. Split into windows of 512 steps (overlapping by half). Probabilities are averaged over the
   windows.

**Pen azimuth** is imported and kept in the processed CSVs, but it is not in the default feature
set. The tablet was rotated, and EMOTHAW does not document its azimuth zero direction, so it
cannot be aligned reliably with the browser's `azimuthAngle`. Use `--features pen_azimuth` to
train with it anyway.

**Missing pen channels** (a mouse or trackpad has no pressure; most devices other than a stylus
have no tilt) are filled with the training mean. The API response and the page say which
channels were filled in.

## 4. Model

`handwriting/model.py` (about 106k parameters, runs in milliseconds on CPU):

```
trajectory -> preprocessing -> per-point features
  -> Linear + LayerNorm (d=64) -> sinusoidal positions
  -> 3-layer pre-norm Transformer encoder (4 attention heads, padding masked)
  -> learned attention pooling -> three independent heads:
       Anxiety    Linear(64 -> 2)  normal / elevated
       Depression Linear(64 -> 2)  normal / elevated
       Stress     Linear(64 -> 2)  normal / elevated
```

Each head is trained with its own class-weighted cross-entropy, and the loss is averaged over
the three heads.

## 5. Training and evaluation

```bash
.venv/bin/python -m handwriting.import_emothaw      # once
.venv/bin/python -m handwriting.train               # tasks 3 + 7 (the writing tasks), features "pen"
.venv/bin/python -m handwriting.train --tasks all   # all seven tasks
.venv/bin/python -m handwriting.train --features kinematic
```

By default the model is trained on the two **writing** tasks (3 and 7), because the web page asks
people to write, not draw.

**Split**
- **Participant-independent**: 70% train, 15% validation, 15% test.
- Participants are stratified by their anxiety/depression/stress combination.
- Before training, the script checks that no participant, no file and no identical recording
  (sha256) appears in two splits. If one does, it stops.
- It prints each split's participant ids.

**Training**
- Normalisation statistics come from training participants only.
- Early stopping uses the validation loss.
- The test participants are evaluated **once**.

**What is printed and saved**
- number of participants and samples
- class distribution per target (samples and participants)
- train/validation/test sizes and class distributions
- for each target on the **held-out test set**, per recording and per participant (the average
  over a participant's recordings):
  - accuracy
  - balanced accuracy
  - precision, recall and F1 (elevated = positive)
  - macro F1
  - per-class precision, recall and F1
  - confusion matrix
  - the majority-class baseline

**Output files**
- `models/handwriting_transformer.pt`: the web app picks it up on the next analysis. The app
  refuses a checkpoint that was not trained on these three targets.
- `runs/handwriting/<timestamp>/results.json`: everything above, plus per-epoch losses and the
  sha256 of the manifest and labels used.

Training accuracy is logged for debugging only. **No accuracy is quoted here because no model has
been trained on the real data yet.**

## 6. Inference API

`POST /handwriting/analyze` with `{"points": [{"x", "y", "t" (ms), "pen_down", "pressure"?, "altitude"?, "azimuth"?}]}`

```json
{
  "ok": true,
  "predictions":   {"Anxiety": "Elevated", "Depression": "Normal", "Stress": "Elevated"},
  "probabilities": {"Anxiety": 0.82, "Depression": 0.31, "Stress": 0.76},
  "elevated": ["Anxiety", "Stress"],
  "heads": [{"name": "anxiety", "display": "Anxiety", "label": "elevated", "confidence": 0.82,
             "probabilities": {"normal": 0.18, "elevated": 0.82}, "classes": ["normal", "elevated"]}, ...],
  "input": {"points_received": 812, "strokes": 14, "defaulted_features": ["pressure", "altitude"], ...}
}
```

`probabilities` is the probability of **elevated** for each target.

**Status codes**
- **400**: invalid input (empty, malformed, or too short). The message is written for the user.
- **503**: no trained model is installed, or it is incompatible.
- **500**: inference error (logged).

`GET /handwriting/status` reports the model state, targets, split sizes and held-out test metrics.

## 7. Limitations

- **Size.** EMOTHAW has about 130 participants, so a 15% test split is about 20 participants,
  and some targets have few elevated cases. Metrics will be noisy. Compare them with the
  printed majority-class baseline and look at balanced accuracy, not accuracy.
- **Labels.** The labels are questionnaire scores split at a threshold. They are not clinical
  diagnoses, and they describe the participant, not any single recording.
- **Device mismatch.** The training data comes from a Wacom pen tablet. A mouse, trackpad or
  finger moves differently and reports no pressure or tilt, so predictions from those devices
  are less reliable than from a stylus.
- **Text.** The model has only seen EMOTHAW's tasks (Italian words in capitals and an Italian
  cursive sentence by default). Very different writing is outside what it was trained on.
- **Not a diagnosis.** This is a research demonstration.

---

# Part B: 4-class prototype (Happy / Neutral / Sad / Angry)

> **The seed dataset is not scientifically validated.** It exists only to establish and test the
> 4-class pipeline, and it will be replaced by a real emotion-labelled handwriting dataset when
> one becomes available.
>
> These samples are being used as heuristic seed data based on graphology references and will
> be replaced with real emotion-labelled handwriting data.
>
> These samples are graphology-based reference/seed examples and are not scientifically validated
> emotion labels. They are used only to establish and test the prototype pipeline and are intended
> to be replaced by a properly labelled research dataset.

## B1. Why seed data

No publicly available, properly labelled handwriting dataset for Happy / Neutral / Sad / Angry
was found. To build and test the whole 4-class pipeline now, we use a **seed dataset**: a small
set of handwriting examples that graphology references associate with each emotion. The whole
pipeline is covered: data layout, preprocessing, training, evaluation, checkpoint, API and page.

**What the seed data is not.** Graphology is not a scientifically validated way to infer
emotions from handwriting. So the seed labels are:
- **not** ground truth
- **not** clinically or scientifically validated
- **not** measurements of anyone's emotional state

**The trained model** is a **Prototype / Seed Model**. Its accuracy on held-out seed samples only
shows that the pipeline runs end to end. It is **not** evidence that the model can recognise
human emotions.

**Where this is marked:**
- Every processed file and the checkpoint carry `dataset_type: seed`,
  `source_type: graphology_reference`, `scientifically_validated: false`.
- The API response carries `"prototype": true` and a notice.
- The page shows: *"Prototype trained on graphology-based seed data. Performance is not
  representative of scientifically validated emotion recognition."*

## B2. Layout

```
data/handwriting/
    seed_4class/
        graphology_pages.csv      <- the public pages the collector may read (you list them; B9)
        candidates/               <- collected, NOT yet reviewed - never used for training
            happy/  neutral/  sad/  angry/
        candidates.csv            <- one row per collected image: evidence, URLs, status, hashes
        collection_log.csv        <- every image seen, and why it was / was not collected
        manual_review.csv         <- pages / images the collector was not allowed to fetch
        rejected/<label>/         <- rejected by the reviewer - never used for training
        review/                   <- contact sheets
        sources/                  <- APPROVED graphology examples (and any you add yourself)
            happy/  neutral/  sad/  angry/
            sources.csv           file, label, writer_id, reference (image URL), source_page,
                                  candidate_id, notes - written on approval
        processed/                <- written by generate_seed_data
            samples/<label>/<source>__orig.csv, __aug01.csv, ...
            manifest.csv          one row per sample, with provenance
            dataset.json          dataset_type, source_type, scientifically_validated, statement, ...
            preview/              optional PNGs (--preview): the traced strokes, blue -> red = order
    real_4class/                  <- later: same layout, real labelled data
```

(EMOTHAW keeps its existing location, `data/emothaw/`, so nothing about Part A changes.)

**Labels** come from the folder name, or from `sources.csv`. They must be exactly `Happy`,
`Neutral`, `Sad` or `Angry` (case-insensitive). Anything else, e.g. `calm` or `positive`, is
skipped and listed, not mapped. Decide yourself which folder a reference example belongs in.

**Writer IDs** are optional for seed data. Without them each source example is its own split
group. If two examples come from the same writer, give them the same `writer_id` in
`sources.csv`; otherwise they can land in different splits.

## B3. How seed examples enter the trajectory pipeline

The Transformer reads **pen trajectories**. Graphology examples are usually **static images**,
which record where the ink is and nothing else.

| Information | From a trajectory file (.csv / .json / .svc) | From an image |
|---|---|---|
| x / y shape of the writing | recorded | **real** (from the ink) |
| stroke order, writing direction | recorded | **estimated** |
| pen-up / pen-down, stroke boundaries | recorded | **estimated** |
| timestamps, speed, acceleration | recorded | **unavailable**: not invented |
| pressure, altitude, azimuth | recorded if the device has them | **unavailable**: not invented |
| hover (in-air) movement | recorded if the device has it | **unavailable** |

**Image → trajectory reconstruction** (`handwriting/image_trajectory.py`):
1. Grayscale, then Otsu binarisation (ink = dark), then remove specks.
2. Zhang-Suen thinning to a 1-pixel skeleton.
3. Build the skeleton graph (segments between end points and junctions) and prune spurs.
4. Trace strokes: within each connected piece of ink, follow the smoothest continuation at
   junctions, and lift the pen where no continuation is left.
5. Order the pieces line by line, top to bottom, and left to right within a line.

In the output, `x`, `y` are real pixel geometry. `pen_down` and stroke boundaries are estimates.
`t_ms` is only a **sequence index**, not a time, and the manifest marks it with `t_is_time = 0`.
Pressure, altitude and azimuth are left empty. Check the `--preview` PNGs to see the traced order.

**The seed model uses only what an image supports.** The seed profile
(`config.FOUR_CLASS_DATASETS["seed_4class"]`) sets three things:
- **Feature set `geometric`:** position, dx/dy, heading, turn angle, pen state, stroke start and
  stroke length. No timing-derived features (dt, speed, acceleration) and no pen channels.
- **Arc-length resampling:** each stroke is resampled at 1/8 of the median stroke height, so
  point density depends on the writing's shape, not its pixel size or speed.
- **In-air movement collapsed** to the lift point, because images contain none.

The browser canvas input is processed the same way at inference. Its real timing and pressure
are simply not used by this model.

## B4. Seed generation: `python -m handwriting.generate_seed_data`

```bash
.venv/bin/python -m handwriting.generate_seed_data --preview 3          # 20 augmented copies per source
.venv/bin/python -m handwriting.generate_seed_data --augment 0 --force  # originals only
```

- **Sources are never modified.** The output folder must be outside `sources/`.
- **Reproducible:** a fixed `--seed` (default 0). Each augmented copy's random stream depends
  only on the seed, the **source file's content hash** and the copy number.
- **Recorded per sample** (`manifest.csv`):
  - `sample_id`, `label`, `source_sample`, `source_file`, `source_sha256`
  - `input_kind` (image / trajectory), `trajectory_origin` (reconstructed_from_image / recorded)
  - `t_is_time`
  - `synthetic` (0 = original, 1 = augmented) and the exact `augmentation` parameters
  - `dataset_type=seed`, `source_type=graphology_reference`, `validated=false`
- **Identical files are imported once.** The same image under two labels would otherwise leak
  across splits.

**Class-blind augmentation.** The augmentation function never receives the label. Every source
gets the same number of copies, and every class draws its parameters from the **same**
distributions:

| Transformation | Range | Applied to |
|---|---|---|
| rotation | ±2° | all |
| slant (shear) | ±0.05 (about ±3°) | all |
| aspect ratio | 0.95-1.05 | all |
| smooth stroke-level displacement | 0-3% of the median stroke height | all |
| speed variation | ×0.9-1.1 | **recorded** trajectories only (images have no timing to vary) |

- **No class-specific rule**, such as "Happy = faster" or "Angry = larger", can arise. A test
  checks that relabelling a file gives byte-identical copies.
- **Global scale and translation** are omitted on purpose: the features are scale- and
  position-free, so they would change nothing.
- **Rotation and slant are kept small,** because baseline direction and slant are exactly the
  cues graphology associates with emotion. Large random values would wash them out.

## B5. Training and evaluation

```bash
.venv/bin/python -m handwriting.train --dataset seed_4class      # -> models/handwriting_4class.pt
```

The command uses the same `train.py` and the same Transformer (one head, 4 classes).

**Split:**
- 70 / 15 / 15 by **group**: the writer where `writer_id` is known, otherwise the source example.
- Every augmented copy stays in its source's group.
- **Validation and test use original samples only.** Augmented copies are for training only.
- The leakage check (group, file, source content hash) runs before training.

**Normalisation** uses training data only.

**Printed and saved** (to `runs/handwriting/<timestamp>-seed_4class/results.json`):
- the seed metadata
- class distribution
- split sizes
- test accuracy, balanced accuracy, macro precision / recall / F1
- per-class precision / recall / F1 and the confusion matrix

All of these are explicitly labelled as a **pipeline check, not emotion-recognition accuracy**.
With a handful of seed examples per class, the test set has a few samples at most. The script
warns when a split has no example of a class.

## B6. API

`POST /handwriting/4class/analyze` takes the same `{"points": [...]}` input as the EMOTHAW route:

```json
{
  "ok": true,
  "emotion": "Happy",
  "confidence": 0.82,
  "probabilities": {"Happy": 0.82, "Neutral": 0.08, "Sad": 0.05, "Angry": 0.05},
  "prototype": true,
  "notice": "Prototype trained on graphology-based seed data. Performance is not representative of scientifically validated emotion recognition.",
  "heads": [...], "input": {...}, "model": {..., "dataset_type": "seed", "scientifically_validated": false}
}
```

This is a **prototype prediction**, not a validated emotion measurement.

- `GET /handwriting/4class/status` reports the model state.
- The status codes are the same as the EMOTHAW route: 400 input, 503 no or incompatible model,
  500 error.
- Each route refuses the other experiment's checkpoint.

### Uploading a handwriting image (4-class tab)

On `/handwriting`, the **Happy · Neutral · Sad · Angry** tab has two inputs: write on the canvas,
**or** press **Upload Handwriting Image** and choose a PNG, JPG or WEBP file (max 10 MB). A
preview replaces the canvas; **Remove Image** returns to the canvas.

`POST /handwriting/4class/analyze-image` takes a multipart upload in form field `image`.
- **Processing:** the image goes through the same `image_trajectory.py` reconstruction as the
  seed images, then the checkpoint's own preprocessing and model.
- **Response:** the same structure as `/handwriting/4class/analyze`, i.e. `emotion`,
  `confidence`, `probabilities`, `prototype`, `notice`, with `input.source_type =
  "uploaded_image"`, `trajectory_source = "reconstructed_from_image"`,
  `timing_available = false`, `pressure_available = false` and
  `stroke_order_available = "estimated"`.
- **Refused (400 / 413, with a user-facing message):**
  - empty, missing, corrupt, unsupported or oversized files
  - blank images
  - images that don't look like dark ink on light, evenly lit paper, such as photos with
    shadows or coloured backgrounds. The reconstruction would trace those as thousands of junk
    strokes.
  - any upload when the installed 4-class checkpoint uses timing features, which a static image
    cannot provide
- **EMOTHAW has no image pathway.** Switching tabs discards an uploaded image.

## B7. Replacing the seed data with a real dataset

```
CURRENT                                   FUTURE
graphology references                     real research dataset (verified labels)
  -> seed_4class (generate_seed_data)       -> real_4class (import_4class)
  -> same preprocessing                     -> same preprocessing
  -> same Transformer                       -> same Transformer
  -> models/handwriting_4class.pt           -> retrain, writer-independent evaluation
  -> pipeline validated (prototype)         -> replace models/handwriting_4class.pt
```

1. **Add the data** to `data/handwriting/real_4class/sources/`, with a `sources.csv` giving
   `file, label, writer_id` for every file. Writer IDs are **required**.
2. **Import** it, mapping the dataset's verified labels explicitly:
   ```bash
   .venv/bin/python -m handwriting.import_4class --source "Dataset name, citation" \
       --label-map "joy=Happy,neutral=Neutral,sadness=Sad,anger=Angry" --labels-verified
   ```
   Labels missing from the map are skipped and listed, never guessed. No augmentation is applied
   by default. Images are refused unless you pass `--allow-images`, and are then marked
   reconstructed.
3. **Train:** `python -m handwriting.train --dataset real_4class`. Its default profile uses the
   full pen features (`pen`) and time resampling, for online pen data. Use
   `--features geometric --resample arclength` if the real data is also images.
4. The new checkpoint **replaces** `models/handwriting_4class.pt`.

The route, the response format and the page panel stay the same. The prototype label and notice
disappear automatically, because they come from the checkpoint's `dataset_type`.

## B9. Collecting graphology references: collect -> review -> sources

```
public graphology reference pages (listed by you in graphology_pages.csv)
        |  python -m handwriting.collect_graphology
        v
candidates/<label>/ + candidates.csv       status=candidate, scientifically_validated=false
        |  python -m handwriting.review_graphology      (a human approves / rejects / relabels)
        v
sources/<label>/ + sources/sources.csv     status=approved (still not scientifically validated)
        |  python -m handwriting.generate_seed_data --force
        v
processed seed data   ->   python -m handwriting.train --dataset seed_4class   (run it yourself)
```

**Nothing is trained automatically.** Collection and review never start training.

### Collecting: `python -m handwriting.collect_graphology`

1. **List the pages.** Put one page per row in `data/handwriting/seed_4class/graphology_pages.csv`
   (`url, license_or_access_note, notes`; rows starting with `#` are ignored).
   - The collector reads **only those pages**. It does not follow links and does not use search
     engines. Finding suitable pages stays a human decision.
   - Record what you know about each site's terms in `license_or_access_note`. If a page links a
     licence (e.g. Creative Commons), that link is added automatically. Otherwise the note says
     "unknown - check the site's terms".
2. **Run it:** `python -m handwriting.collect_graphology` (add `--dry-run` to see what it would
   collect without saving anything).

**Access rules (never bypassed):**
- Every URL is checked against the site's `robots.txt`, including its `Crawl-delay`. If
  `robots.txt` answers 401/403, the whole site counts as disallowed.
- Requests to the same host are at least `--delay` seconds apart (default 3).
- A host that answers 429 is left alone for the rest of the run.
- Pages or images answering 401/402/403/451, and pages that look like a CAPTCHA, login or paywall,
  are **not** retried or worked around. They are written to `manual_review.csv`, so you can look
  at them yourself if you choose.
- The collector identifies itself with a descriptive User-Agent.

**How an image gets a candidate label.** The source must explicitly associate the image with the
emotion. Only the image's own context counts: its `alt` text, its `title`, its `<figcaption>`,
or a caption element around or right after it.
- It must name exactly one class, using the terms in `LABEL_TERMS` (`collect_graphology.py`).
  For example, "happy / optimistic / cheerful" → Happy; "calm / emotionally stable" → Neutral;
  "sad / melancholy / pessimistic" → Sad; "angry / aggressive / hostile" → Angry.
- **Page text never labels an image.** A page *about* happy handwriting does not make its images
  Happy.
- The nearest section heading counts only with `--allow-section-headings`, and is then recorded
  as `evidence_source=section_heading`.
- Negated mentions ("not a happy writer") are ignored. A context naming two classes is
  *ambiguous* and not collected.
- Clinical terms such as "depression" are deliberately **not** mapped to Sad. Those constructs
  belong to the EMOTHAW experiment.
- The matched text is stored in `label_evidence`, so you can see exactly why an image was
  labelled.

**What is kept.**
- Images are saved **unchanged**, as their original bytes. Unusable ones are logged in
  `collection_log.csv` and not saved: undecodable, unsupported format, or too small
  (icons / decorations).
- **Duplicates** of any candidate, approved, rejected or `sources/` image are skipped:
  - identical bytes (sha256), or
  - the same picture re-encoded, resized or re-cropped. This uses a perceptual hash of the ink
    region as a prefilter, confirmed by correlation ≥ 0.975.
  - Measured on rendered text: re-encoded copies score ≥ 0.988; different samples score ≤ 0.963,
    even the same words in a near-identical font.
  - Rejected images therefore never come back on a re-run.

**`candidates.csv`**:
- id, filename, candidate_label, original_candidate_label, label_evidence, evidence_source
- source_page, image_url, page_title, description
- status, scientifically_validated (always `false`), license_or_access_note
- sha256, phash, width, height
- trajectory_source=`reconstructed_from_image`, timing_available=`false`,
  pressure_available=`false`, stroke_order_available=`estimated`
- collected_at, reviewed_at, review_note, approved_file

### Reviewing: `python -m handwriting.review_graphology`

```
[1/23] g3f9a1c07b2de  candidate label: Happy  (812x240)
  evidence (figcaption): Handwriting of a happy, optimistic writer
  page: ...  image: ...  access: ...
[y] approve  [n] reject  [s] skip  [r] relabel  [q] quit >
```

- `n` asks for a reason: 1 poor quality, 2 ambiguous, 3 duplicate, 4 irrelevant, 5 not
  handwriting, or free text.
- `r` offers 1 Happy, 2 Neutral, 3 Sad, 4 Angry. The item stays a candidate and is shown again,
  and the original label is kept in `original_candidate_label`.
- `--open` shows each image in the system viewer.
- `--label sad` reviews one class only.
- `--contact-sheet` writes `review/contact_candidate_<label>.png`: numbered thumbnails with ids,
  for a quick visual pass first.
- `--stats` prints:
  ```
  Happy:     23 candidates |  15 approved |   4 rejected
  ```
- Batch actions: `--approve ID ...`, `--reject ID ... --reason "..."`, `--relabel ID Sad`.

**What each decision does:**
- **Approve:** moves the file to `sources/<label>/` and sets `status=approved`. It writes a row to
  `sources/sources.csv` with `reference` = image URL, `source_page` and `candidate_id`, so every
  approved image traces back to its original page. That reference is carried into the processed
  seed manifest. `scientifically_validated` stays `false`: approval means "usable as a
  graphology seed example", nothing more.
- **Reject:** moves the file to `rejected/<label>/` and sets `status=rejected`.

### Prototype mode: `--auto-approve` (no human review)

```bash
.venv/bin/python -m handwriting.collect_graphology --auto-approve
```

- Every candidate that passes the collector's filters goes straight to `sources/` with
  `status=auto_approved`, `prototype_only=true`, `scientifically_validated=false`. The filters
  are: explicit image-level evidence, handwriting-likeness (ink on paper, not a photo or
  cartoon), duplicates, and a valid image.
- This is how the current seed dataset was built. `generate_seed_data` accepts `approved` and
  `auto_approved` sources and still refuses pending or rejected images.
- **Same-writer grouping is the one manual step.** When several collected images are
  evidently by one person, give them a shared `writer_id` in `sources/sources.csv`, so the
  split keeps them together. For example, all apdaga.com samples reuse the same sentence in
  the same hand.

### Guarantees that unreviewed images never reach training

- `generate_seed_data` reads only `sources/`. It refuses `--sources` pointing into `candidates/`
  or `rejected/`.
- It also skips any file in `sources/` whose content matches a collected image that is not
  approved, e.g. one copied there by hand, and lists it.
- Collection and review never run seed generation or training.

### Static images

Collected images go through the same image → trajectory reconstruction as any other seed image
(B3):
- The ink geometry is real, and stroke order and pen lifts are **estimated**.
- Timing, velocity, pressure and tilt are **unavailable** and are not invented.
- The processed manifest states this per sample: `trajectory_origin=reconstructed_from_image`,
  `timing_available=false`, `pressure_available=false`, `stroke_order_available=estimated`.

### Replacing the seed data later

None of this touches the `real_4class` path (B7). A real dataset is imported with
`import_4class` and trained with `--dataset real_4class`, using the same preprocessing, model
and API. The graphology collection is only a way to fill the temporary seed dataset.

## B10. Participant canvas samples (Collect tab): the first real trajectories

Graphology pages yielded too little (see B9), so the 4-class dataset is now grown from
people writing on the app's own canvas.

**Session workflow**
1. Open `/handwriting` and choose **Collect samples**.
2. Enter an anonymous **Writer ID** (`writer_001`, `writer_002`, …), or press **New ID** for
   the next unused one. Names are refused: the ID must match `writer_` + 3-6 digits. Keep the
   same ID for all of one person's samples; the browser remembers it.
3. Select **Happy / Neutral / Sad / Angry**, write, and press **Save Sample**. The page shows
   "Sample saved: Happy #3" and deselects the label, so every sample needs a deliberate choice.
4. Repeat. The counts table shows totals for all writers and for this writer.

**What participants are told.** Only "Write naturally for this sample." and "Please do not
write names or other personal information." They are **never** told how a label should look:
no faster/slower/larger rules. The label is the one the participant chose. It is not a measured
or validated emotional state, and is stored as `scientifically_validated: false`. Get each
participant's consent before collecting.

**Target.** About 10-20 writers × 4 labels × 5 samples, i.e. 200-400 samples. Nothing
requires this number; training works with whatever exists, but the evaluation is only as
meaningful as the number of writers.

**Storage** (never overwritten; the graphology `sources/` are untouched):
```
data/handwriting/seed_4class/human/<label>/<writer_id>_<label>_<NNN>.json
{"schema": "human_canvas/1", "writer_id": "writer_003", "label": "Happy",
 "points": [{"x", "y", "t", "pen_down", "pressure"?, "altitude"?, "azimuth"?}, ...],
 "timestamp": "...Z", "source_type": "human_canvas", "scientifically_validated": false,
 "pointer_types": ["pen"], "n_points": 812}
```
Optional pen channels are stored only when the device reports them (stylus), never as nulls.

**Building and training.** Origins are never mixed invisibly. `generate_seed_data` reads
`sources/` (graphology) and `human/` (canvas), and every processed row records
`sample_source` = `graphology` | `human_canvas` and `synthetic` = 0 | 1.
```bash
.venv/bin/python -m handwriting.generate_seed_data --force                  # both origins (default)
.venv/bin/python -m handwriting.train --dataset seed_4class --sample-source human_canvas   # recommended first
.venv/bin/python -m handwriting.train --dataset seed_4class                # graphology + canvas together
```
- **Split by writer.** Each canvas sample's group is its `writer_id`: 70 / 15 / 15 % of writers
  go to train / validation / test. No writer ever appears in two splits, and the leakage
  check verifies this. Validation and test use original samples only; augmented copies are
  for training only.
- **Reported:** writers and samples per class and per split, and on the test writers:
  accuracy, balanced accuracy, macro F1, per-class precision / recall / F1 and the confusion
  matrix.
- **Don't pick `--seed` for good numbers.** With few writers, a different seed can move the
  metrics a lot. Report the default split.
- **Features:** the `seed_4class` profile uses shape-only features, so canvas and
  graphology-image samples are comparable. For a canvas-only model, which has real timing and,
  from a stylus, pressure, `--sample-source human_canvas --features pen --resample time` uses
  the full EMOTHAW-style features. The checkpoint records the choice, and inference follows it.
- **Checkpoint:** the result replaces `models/handwriting_4class.pt`, and
  `/handwriting/4class/analyze` is unchanged. A canvas-only model shows its own notice
  ("participant-labelled canvas samples").

## B8. Limitations

- **Seed labels are heuristic.** Seed data teaches the model graphology's associations at best.
  Collected labels are only what a web page's caption claims; human approval checks that the
  image is a usable handwriting example with that stated association, not that the association
  is true.
  It says nothing validated about emotions.
- **Very little data.** A few examples per class cannot support any claim about generalisation.
- **Reconstruction is an estimate.** Stroke order and pen lifts traced from an image can differ
  from how the text was actually written. Ruled lines, stamps or printed text in an image are
  traced as ink, so crop the examples to the handwriting alone.
- **Different input.** Canvas writing (real, online) and reconstructed image writing are made
  comparable by using geometry only. That also means the seed model ignores timing and pressure
  entirely.
