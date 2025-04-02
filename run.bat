rem Run training and prediction for UE2 with multiview.
python scripts/train_multiview.py --config configs/ue2_multiview.yaml
python scripts/predict_multiview.py --config configs/ue2_multiview.yaml

@REM rem Run training and prediction for SynthesEyes.
@REM python scripts/train_video_eval.py --config configs/syntheseyes.yaml
@REM python scripts/predict.py --config configs/syntheseyes.yaml

@REM rem Run training and prediction for UE2 with combined views.
@REM python scripts/train_video_eval.py --config configs/ue2.yaml
@REM python scripts/predict.py --config configs/ue2.yaml

@REM rem Run training and prediction for UE2 with separate views.
@REM python scripts/train_video_eval.py --config configs/ue2_separate.yaml
@REM python scripts/predict.py --config configs/ue2_separate.yaml