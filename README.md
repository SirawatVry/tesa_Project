# 🚁 Drone Detection & Localization Pipeline

Real-time drone detection, tracking, and GPS localization from video streams using YOLO and machine learning.

---

## 🚀 What Can This Do?

| Feature | Capability | Metrics |
|---------|-----------|---------|
| 🎯 **Detection** | High-accuracy drone detection | 99.1% rate, mAP: 81% |
| 📍 **Tracking** | Multi-drone tracking with ByteTrack | 2+ drones, ~99% ID accuracy |
| 🗺️ **Localization** | Real-time GPS coordinate prediction | lat, lon, alt (3 models) |
| 🎬 **Processing** | CPU & GPU support | 12.9 FPS (CPU), ~30 FPS (GPU) |
| 📊 **Output** | Annotated video + metadata | 69.42 MB, 1920x1080, 25 FPS |

---

## ⚡ Quick Start (Choose One)

### **Option 1: Using Python (Local)**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pipeline
python src/problem_3_pipeline.py

# 3. View output
outputs/problem_3/final/P3_OUTPUT_FINAL.mp4
```

### **Option 2: Using Docker (Recommended)**

```bash
# CPU version
docker-compose up

# GPU version (faster)
docker-compose -f docker-compose.gpu.yml up
```

### **Option 3: Interactive Analysis**

```bash
# Analyze tracking patterns
python scripts/05_evaluation/analyze_track_patterns.py

# Check detected drone IDs
python scripts/05_evaluation/check_actual_track_ids.py
```

---

## 📋 What's Inside?

```
📦 drone-detection-pipeline/
├── 💻 src/                    # Main pipeline code
│   ├── problem_3_pipeline.py  # ← Start here!
│   ├── detector.py            # YOLO detection
│   ├── tracker.py             # Multi-object tracking
│   ├── localizer.py           # GPS prediction
│   └── visualizer.py          # Video annotation
│
├── ⚙️ configs/                # Configuration files
├── 🤖 models/                 # Pre-trained models
├── 📊 data/                   # Training data & metadata
├── 🔬 scripts/                # Analysis & utility scripts
├── 📁 docs/                   # Full documentation
├── 📤 outputs/                # Results go here
├── 🐳 Dockerfile              # Container files
├── docker-compose.yml         # Docker orchestration
└── requirements.txt           # Dependencies
```

---

## 🎯 Core Capabilities

### 1. **YOLO Detection**
- Detects drones from video frames
- 99.1% accuracy on test data
- Pre-trained model included

### 2. **ByteTrack Tracking**
- Maintains unique ID for each drone
- Handles drone occlusion and re-appearance
- Configurable tracking sensitivity

### 3. **GPS Localization**
- Predicts latitude, longitude, altitude
- Uses bbox features from YOLO detections
- Multiple model approaches available

### 4. **Video Annotation**
- Draws bounding boxes on detected drones
- Shows track IDs and coordinates
- Saves annotated video output

---

## 🔧 Setup Requirements

### Minimum (Python)
- Python 3.8+
- 4GB RAM
- 10GB disk space

### Recommended (GPU)
- NVIDIA GPU with CUDA 11.8+
- 8GB VRAM
- 15GB disk space

### Installation
```bash
pip install -r requirements.txt
```

---

## 📊 Performance Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Detection Rate** | 99.1% | ✅ Excellent |
| **Detection mAP** | 81% | ✅ High |
| **Tracking Accuracy** | ~99% | ✅ Excellent |
| **Processing Speed** | 12.9 FPS (CPU) | ✅ Good |
| **GPU Speed** | ~30 FPS | ✅ Very Good |
| **Output Bitrate** | 1920x1080@25FPS | ✅ HD Quality |
| **Total Detections** | 3,530+ | ✅ Comprehensive |
| **Max Track IDs** | 2+ drones | ✅ Multi-object |
| **Total Processing Time** | 2.5 min (CPU), 1.5 min (GPU) | ✅ Fast |

---

## 🐳 Docker Support

Run in isolated containers (no dependency conflicts):

```bash
# CPU version
docker-compose up

# GPU version (NVIDIA required)
docker-compose -f docker-compose.gpu.yml up

# Interactive shell
docker-compose run --rm tesa-drone-detection /bin/bash
```

**See [DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md) for details**

---

## 📚 Full Documentation

| Document | Purpose |
|----------|---------|
| [OVERVIEW.md](docs/OVERVIEW.md) | Complete project details |
| [QUICK_START.md](docs/QUICK_START.md) | Step-by-step setup |
| [DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md) | Docker usage guide |
| [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | Directory organization |
| [SUMMARY.md](docs/SUMMARY.md) | Results & metrics |

---

## 🎮 Usage Examples

### Run Main Pipeline
```bash
python src/problem_3_pipeline.py
```

### Analyze Results
```bash
# View track patterns
python scripts/05_evaluation/analyze_track_patterns.py

# Validate drone detection
python scripts/05_evaluation/check_actual_track_ids.py

# Frame-level analysis
python scripts/05_evaluation/analyze_specific_frames.py
```

### Test Environment
```bash
python scripts/08_utilities/test_environment.py
```

---

## 📍 Input/Output

**Input:**
- Video file (1920x1080, 25 FPS, any duration)

**Output:**
- Annotated video with drone boxes, IDs, and coordinates
- CSV predictions (optional)

---

## ✅ Project Status

- ✅ Detection: Complete (High accuracy)
- ✅ Tracking: Complete (Multi-object)
- ✅ Localization: Complete (GPS models)
- ✅ Visualization: Complete
- ✅ Docker: Complete (CPU & GPU)

**Last Updated:** April 2026
- **v20**: Merged additional labels
- **v21**: Maximum dataset (train + valid merged)
- **tomorbest.pt**: Best performing model

### XGBoost Models
- **v1**: Basic features
- **v2**: Enhanced features
- **v3**: Feature selection applied
- **v16**: Aligned with YOLO v16
- **v21**: Aligned with YOLO v21

### Ensemble Approaches
- **Ensemble v1 + v21**: Weighted average
- **Stacking**: Meta-learner approach
- **Approximation**: Geometric + NN correction
- **Residual Learning**: Base + residual correction

---


### Requirements
1. ✅ วิดีโอแสดง:
   - Bounding box รอบโดรน
   - Track ID
   - Predicted coordinates (lat, lon, alt)

2. ✅ Source code (.zip)
   - ไม่รวมรูปภาพ/วิดีโอ
   - สามารถรันได้จริง

---
