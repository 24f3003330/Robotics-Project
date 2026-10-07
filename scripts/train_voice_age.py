"""
python scripts/train_voice_age.py [--language en] [--max-samples-per-class 1400] [--epochs 60]
                                  [--batch-size 64] [--seed 42] [--output-dir models/voice_age_model]

Trains the voice AGE-GROUP estimation model on Common Voice. The dataset is read
programmatically from the Hugging Face Hub (only metadata columns and the audio of
the selected clips are transferred; nothing has to be downloaded by hand).
See voice_age/train.py and docs/VOICE_AGE.md.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_age.train import parse_args, run  # noqa: E402

if __name__ == "__main__":
    run(parse_args())
