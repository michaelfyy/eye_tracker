# eye_tracker_2d_eval

**2D evaluation for eye tracking models on SynthesEyes and UnityEyes 2.**

## 🚀 Usage

### 🔧 Configuration

Edit the following configuration files to specify paths to eye videos and datasets:
- `configs/syntheseyes.yaml`
- `configs/ue2.yaml`
- `configs/ue2_separate.yaml`

#### Windows
Run the batch script:
```sh
./run.bat
```

#### MacOS/Linux
Run the shell script:
```sh
./run.sh
```

### 📂 Expected Directory Structure

Your dataset should be organized as follows:

```
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
```
