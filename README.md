# eye_tracker_2d_eval

2D evaluation for eye tracking model on SynthesEyes and UnityEyes 2
## Usage notes:
On windows: ```./run.bat```
On MacOS/Linux: ```./run.sh```

Edit configs/syntheseyes.yaml, configs/ue2.yaml, configs/ue2_separate.yaml for paths to eye videos and dataset.
Expected directory structure:
eye_videos_folder/
├── e1.mp4
├── e2.mp4
├── e3.mp4
├── e4.mp4
└── annotations/
    ├── e1_annotations.xml
    ├── e2_annotations.xml
    ├── e3_annotations.xml
    └── e4_annotations.xml
