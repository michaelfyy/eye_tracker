rem Run training and prediction for SynthesEyes.
@REM python scripts/train.py --config configs/syntheseyes.yaml
@REM python scripts/predict.py --config configs/syntheseyes.yaml

@REM rem Run training and prediction for UE2 with combined views.
@REM python scripts/train.py --config configs/ue2.yaml
python scripts/predict.py --config configs/ue2.yaml

rem Run training and prediction for UE2 with separate views.
python scripts/train.py --config configs/ue2_separate.yaml
python scripts/predict.py --config configs/ue2_separate.yaml