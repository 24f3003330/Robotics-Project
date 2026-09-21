# Existing voice pipeline — audit before the ensemble upgrade

Recorded before any code was changed, so the ensemble work can be checked against it.

| Concern | Where it lives today | Verdict |
|---|---|---|
| Mic capture | `emotion_detector.open_input_stream` + `_Session.callback` (copies samples only) | unchanged |
| Resampling to 16 kHz | `emotion_detector._StreamingResampler` (live), `app._extract_audio_wav` → `_read_wav_16k` (upload) | unchanged |
| VAD | `_SileroVAD` / `_EnergyVAD`, built by `_make_vad`; 512-sample (32 ms) chunks | unchanged, stays separate from classification |
| Live window construction | `LiveEmotionDetector._process_audio` keeps a 3.0 s ring of 32 ms chunks; `_analyse` runs every 0.5 s, crops leading silence, enforces `MIN_ANALYSIS_SECONDS=2.0` | already 2–4 s with overlap — unchanged |
| Upload window construction | `app._analyze_uploaded_video_audio`, `hop = win` (**non-overlapping**) | **changed**: now overlapping |
| HuBERT inference | `_EmotionModel.predict(audio) -> {emotion: prob}` — the single choke point both paths call | **extended**: pluggable, `predict_detailed()` added |
| Live temporal smoothing | `_analyse`: mean of last 6 probability vectors + confidence/margin gate + 3-window streak | unchanged |
| Upload aggregation | mean over speech windows in `_analyze_uploaded_video_audio` | **changed**: speech-weighted mean over overlapping windows |
| App-side fusion | `_apply_emotion` → `_recompute_final_locked` (manual > voice > face) | unchanged |
| Upload result pinning | `_recompute_final_locked` sets `ttl = None` for `kind == "upload"`; cleared by `reset_detection_state` / `video_use` | unchanged — already pinned to the file, not a timer |
| UI | `/voice/status`, `/status`, `templates/index.html` | additive fields only |

## The one integration seam

Every prediction in the app goes through `_EmotionModel.predict`. The live loop calls it from
`_analyse`; the upload job calls `voice_detector._model.predict` directly. Swapping what that
object *is* upgrades both paths at once without touching capture, VAD, smoothing, fusion or the UI.
That is the seam the ensemble plugs into.
