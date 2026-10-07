"""
Body Vitals modality - physiological state from ECG (heart rate, HRV), EDA (GSR)
and respiration, classified by a model trained on WESAD into baseline / stress /
amusement. The trained model is the only prediction mechanism.

Independent of the camera, voice and handwriting modalities. See docs/BODY_VITALS.md.

    python -m body_vitals.prepare_wesad --data /path/to/WESAD
    python -m body_vitals.train
    python -m body_vitals.evaluate
"""
