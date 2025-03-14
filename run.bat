rem Run training and prediction for SynthesEyes.
python scripts/train_video_eval.py --config configs/syntheseyes.yaml
python scripts/predict.py --config configs/syntheseyes.yaml

rem Run training and prediction for UE2 with combined views.
python scripts/train_video_eval.py --config configs/ue2.yaml
python scripts/predict.py --config configs/ue2.yaml

rem Run training and prediction for UE2 with separate views.
python scripts/train_video_eval.py --config configs/ue2_separate.yaml
python scripts/predict.py --config configs/ue2_separate.yaml