# Detailed Project Structure

Complete directory organization and component descriptions.

---

## 📂 Directory Tree

```
drone-detection-pipeline/
│
├── 📄 Root Files
│   ├── README.md                 # Main project documentation
│   ├── requirements.txt          # Python dependencies
│   ├── Dockerfile               # CPU container
│   ├── Dockerfile.gpu           # GPU container
│   ├── docker-compose.yml       # Docker orchestration
│   ├── docker-compose.gpu.yml   # GPU orchestration
│   ├── .gitignore               # Git exclusions
│   └── .dockerignore            # Docker exclusions
│
├── 📂 src/                      # Main pipeline code
│   ├── problem_3_pipeline.py   # Main entry point
│   ├── detector.py              # YOLO detection
│   ├── tracker.py               # ByteTrack tracking
│   ├── localizer.py             # GPS prediction
│   └── visualizer.py            # Video annotation
│
├── ⚙️ configs/                  # Configuration files
│   ├── botsort_custom.yaml     # Tracking config
│   ├── data.yaml               # Dataset config
│   ├── feature_columns.json    # Feature definitions
│   └── ensemble_config.json    # Ensemble settings
│
├── 📊 data/                     # Training data & metadata
│   ├── metadata.csv             # Training metadata
│   ├── samples.csv              # Sample data
│   └── samples.json             # Sample metadata
│
├── 🖼️ datasets/                # Raw dataset storage
│   ├── DATA_TRAIN/             # Training images
│   ├── DATA_TEST/              # Test images
│   └── train_data/             # Processed data
│
├── 🤖 models/                   # Pre-trained models
│   ├── best.pt                 # YOLO model
│   ├── models_approximation/   # Localization models
│   │   ├── nn_best.pth
│   │   ├── bbox_features.json
│   │   └── correction_params.json
│   └── models_stacking/        # Ensemble models
│
├── 🏃 runs/                     # Training runs
│   ├── detect/                 # Detection training
│   │   └── [training versions]
│   └── obb/                    # OBB training
│
├── 🔬 scripts/                  # Utility scripts
│   ├── 01_data_exploration/   # Data analysis
│   ├── 02_yolo_preparation/   # Dataset prep
│   ├── 03_yolo_training/      # Training
│   ├── 04_xgboost_training/   # XGBoost
│   ├── 05_evaluation/         # Evaluation
│   ├── 06_prediction/         # Predictions
│   ├── 07_ensemble/           # Ensemble
│   └── 08_utilities/          # Utilities
│
├── 📤 outputs/                  # Results directory
│   ├── problem_3/              # Main results
│   │   ├── final/              # Final deliverables
│   │   ├── analysis/           # Analysis data
│   │   └── experiments/        # Experiment outputs
│   ├── predictions/            # CSV predictions
│   ├── visualizations/         # Generated plots
│   └── reports/                # Analysis reports
│
└── 📚 docs/                     # Documentation
    ├── README.md               # Overview (in root)
    ├── OVERVIEW.md             # This file
    ├── QUICK_START.md          # Quick start guide
    ├── DOCKER_GUIDE.md         # Docker setup
    ├── PROJECT_STRUCTURE.md    # Directory info
    └── SUMMARY.md              # Project summary
```

---

## 🎯 Component Breakdown

### Source Code (src/)

| File | Purpose |
|------|---------|
| **problem_3_pipeline.py** | Main orchestration |
| **detector.py** | YOLO detection implementation |
| **tracker.py** | ByteTrack tracking |
| **localizer.py** | GPS coordinate prediction |
| **visualizer.py** | Video frame annotation |

### Scripts (scripts/)

| Folder | Purpose | Count |
|--------|---------|-------|
| 01_data_exploration | Data analysis & EDA | 2 |
| 02_yolo_preparation | Dataset preparation | 6 |
| 03_yolo_training | Training pipelines | 5 |
| 04_xgboost_training | XGBoost models | 8 |
| 05_evaluation | Evaluation metrics | 7 |
| 06_prediction | Inference scripts | 7 |
| 07_ensemble | Ensemble methods | 7 |
| 08_utilities | Helper utilities | 15 |

### Models (models/)

| Type | Purpose | Count |
|------|---------|-------|
| YOLO | Object detection | 2+ |
| Approximation | Localization | 3 |
| XGBoost | Coordinate prediction | Multiple |
| Stacking | Ensemble | Multiple |
