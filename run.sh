#!/bin/bash
# Run training and prediction for SynthesEyes.
python scripts/train.py --config configs/syntheseyes.yaml
python scripts/predict.py --config configs/syntheseyes.yaml

# Run training and prediction for UE2 with combined views.
python scripts/train.py --config configs/ue2.yaml
python scripts/predict.py --config configs/ue2.yaml

# Run training and prediction for UE2 with separate views.
python scripts/train.py --config configs/ue2_separate.yaml
python scripts/predict.py --config configs/ue2_separate.yaml