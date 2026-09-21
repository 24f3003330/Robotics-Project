# models/ensemble/

`manifest.json` lands here when you build an ensemble:

    python -m ser all --run r1 --cremad /path/CREMA-D --ravdess /path/RAVDESS

Until then this directory is empty and the app runs on the single pretrained
`superb/hubert-base-superb-er`, exactly as it did before. That fallback is
deliberate: a missing or half-built ensemble must never take the app down.

Check what the app would load, without starting the app:

    python -m ser check
