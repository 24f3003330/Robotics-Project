"""Handwriting modality: pen trajectories -> Transformer -> EMOTHAW's three targets,
Anxiety / Depression / Stress, each normal vs elevated.

    config.py            label heads, features, preprocessing constants, paths
    import_emothaw.py    raw EMOTHAW (.svc + DASS label table) -> standard per-sample CSVs,
                         manifest.csv, labels.csv (raw data never modified)
    preprocessing.py     .svc / standard CSV / browser points -> cleaned, resampled, per-point features
    dataset.py           processed EMOTHAW loader, writer-independent split, leakage check
    model.py             HandwritingTransformer (encoder + attention pooling + one head per target)
    train.py             python -m handwriting.train
    inference.py         HandwritingAnalyzer used by app.py (kind "emothaw" or "4class")

Separate 4-class experiment (Happy / Neutral / Sad / Angry), same preprocessing and model:
    four_class.py        sources -> processed layout (class-blind augmentation), loader
    image_trajectory.py  static image -> RECONSTRUCTED trajectory (geometry only; order estimated)
    generate_seed_data.py  python -m handwriting.generate_seed_data - graphology SEED data
                         (prototype only - not scientifically validated emotion labels)
    import_4class.py     python -m handwriting.import_4class - a real labelled dataset later
    collect_graphology.py  python -m handwriting.collect_graphology - CANDIDATE graphology images
                         from listed pages (robots.txt, rate limits; explicit image-level evidence only)
    human_collection.py  participant canvas samples (Collect tab) -> seed_4class/human/<label>/*.json
    review_graphology.py python -m handwriting.review_graphology - human approve / reject / relabel;
                         only approved images move to sources/
"""
