# Drone Detection & Localization Pipeline

Complete documentation and technical specifications for the drone detection, tracking, and localization system.

---

## 📁 Project Structure

```
drone-detection-pipeline/
├── src/                       # Main source code
│   ├── problem_3_pipeline.py  # Main pipeline entry point
│   ├── detector.py            # YOLO detection module
│   ├── tracker.py             # Multi-object tracking
│   ├── localizer.py           # GPS coordinate prediction
│   └── visualizer.py          # Video annotation
│
├── configs/                   # Configuration files
│   ├── botsort_custom.yaml    # Tracking configuration
│   └── feature_columns.json   # Feature definitions
│
├── models/                    # Pre-trained models
│   ├── best.pt               # Best YOLO model
│   ├── models_approximation/ # Localization models
│   └── models_stacking/      # Ensemble models
│
├── data/                      # Training data & metadata
│   ├── metadata.csv
│   ├── samples.csv
│   └── samples.json
│
├── scripts/                   # Utility scripts
│   ├── 01_data_exploration/  # Data analysis
│   ├── 02_yolo_preparation/  # Dataset preparation
│   ├── 03_yolo_training/     # Training scripts
│   ├── 04_xgboost_training/  # XGBoost models
│   ├── 05_evaluation/        # Evaluation scripts
│   ├── 06_prediction/        # Prediction scripts
│   ├── 07_ensemble/          # Ensemble methods
│   └── 08_utilities/         # Utility tools
│
├── outputs/                   # Results directory
│   ├── predictions/          # CSV predictions
│   ├── visualizations/       # Generated plots
│   └── reports/              # Analysis reports
│
├── docs/                      # Documentation
│   ├── OVERVIEW.md           # This file
│   ├── QUICK_START.md        # Quick start guide
│   ├── DOCKER_GUIDE.md       # Docker setup
│   ├── PROJECT_STRUCTURE.md  # Detailed structure
│   └── SUMMARY.md            # Project summary
│
├── Dockerfile                # CPU container
├── Dockerfile.gpu            # GPU container
├── docker-compose.yml        # CPU orchestration
├── docker-compose.gpu.yml    # GPU orchestration
├── requirements.txt          # Python dependencies
└── README.md                 # Main readme
```

---

## 🎯 Core Architecture

### Detection Pipeline
```
Video Input → YOLO Detection → Bounding Boxes
```

### Tracking Pipeline
```
Bounding Boxes → ByteTrack → Track IDs
```

### Localization Pipeline
```
YOLO Features → ML Models → GPS Coordinates (lat, lon, alt)
```

### Visualization Pipeline
```
Tracked Results → Video Annotation → Output Video
```

---

## 📊 Component Details

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **detector.py** | YOLO detection | Video frame | Bounding boxes + confidence |
| **tracker.py** | ByteTrack tracking | Bounding boxes | Track IDs |
| **localizer.py** | GPS prediction | YOLO features | Lat, Lon, Alt |
| **visualizer.py** | Video annotation | Detections + coords | Annotated frame |

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. ตรวจสอบสภาพแวดล้อม
```bash
python scripts/08_utilities/check_gpu.py
python scripts/08_utilities/test_environment.py
```

### 3. Pipeline สำหรับ Problem 3 (Drone Tracking & Localization)
```bash
python src/problem_3_pipeline.py
```

---

## 📊 Model Versions

### YOLO Models
- **v1**: Base YOLOv8n-OBB
- **v2**: Tuned hyperparameters
- **v16**: With augmented data
- **v21**: Latest version with max data

### XGBoost Models
- Multiple versions for latitude, longitude, and altitude prediction
- Enhanced feature engineering versions available
