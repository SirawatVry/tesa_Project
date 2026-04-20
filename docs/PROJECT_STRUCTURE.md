# Project Structure - TESA Problem 2

## 📁 โครงสร้างโปรเจค

```
tesa_problem_2/
│
├── 📹 P3_VIDEO.mp4                    # Input video (75.7s, 1920x1080, 25 FPS)
│
├── 📋 Documentation
│   ├── README.md                      # Project overview
│   ├── PROBLEM_3_TASKS.md            # Task tracking
│   └── PROJECT_STRUCTURE.md          # This file
│
├── ⚙️ configs/                        # Configuration files
│   ├── botsort_custom.yaml           # Tracker config (track_buffer: 180)
│   ├── data.yaml                     # Dataset config
│   ├── ensemble_config.json          # Ensemble settings
│   └── feature_columns_*.json        # Feature definitions
│
├── 📊 data/                           # Processed data
│   ├── gcp_*.csv                     # GCP reference data
│   ├── train_metadata_*.csv          # Engineered features
│   └── train_metadata_with_yolo_*.csv # YOLO predictions
│
├── 🖼️ datasets/                       # Raw datasets
│   ├── DATA_TRAIN/                   # Training data
│   │   ├── csv/                      # Metadata
│   │   ├── image/                    # Original images
│   │   ├── labels/                   # YOLO labels
│   │   └── train/valid/              # Split datasets
│   └── DATA_TEST/                    # Test data
│
├── 🤖 models/                         # Trained models
│   ├── yolo11n.pt                    # YOLO11n pretrained
│   ├── yolov8n.pt                    # YOLOv8n pretrained
│   ├── tomorbest.pt                  # Custom model
│   ├── models_approximation/         # Localization models
│   │   ├── nn_best.pth               # Neural network
│   │   ├── bbox_features.json        # Feature stats
│   │   └── correction_params.json    # Calibration
│   └── models_stacking/              # Ensemble models
│
├── 🏃 runs/                           # Training runs
│   ├── detect/                       # Detection training
│   │   ├── drone_detect_v21_max_data/  # Best model (mAP: 81%)
│   │   │   └── weights/best.pt
│   │   └── [other versions]/
│   └── obb/                          # OBB training
│
├── 📤 outputs/                        # Results
│   ├── problem_3/
│   │   ├── final/                    # ✅ Final outputs
│   │   │   └── P3_OUTPUT_FINAL.mp4   # Final video (< 200 MB)
│   │   ├── analysis/                 # 📊 Analysis results
│   │   │   ├── track_patterns/
│   │   │   └── frame_analysis/
│   │   └── experiments/              # 🧪 Experimental outputs
│   ├── predictions/
│   ├── visualization_results/
│   └── reports/
│
├── 🔬 scripts/                        # Analysis & utilities
│   ├── 01_data_exploration/
│   ├── 02_yolo_preparation/
│   ├── 03_yolo_training/
│   ├── 04_xgboost_training/
│   ├── 05_evaluation/                # Analysis scripts
│   │   ├── analyze_track_patterns.py
│   │   ├── check_actual_track_ids.py
│   │   └── analyze_specific_frames.py
│   ├── 06_prediction/
│   ├── 07_ensemble/
│   └── 08_utilities/
│       └── merge_tracks.py
│
├── 💻 src/                            # Main source code
│   ├── problem_3_pipeline.py         # 🎯 Main pipeline
│   ├── detector.py                   # YOLO detection
│   ├── tracker.py                    # Multi-object tracking
│   ├── localizer.py                  # GPS prediction
│   └── visualizer.py                 # Visualization
│
└── 📚 notebooks/                      # Jupyter notebooks

```

---

## 🎯 Main Pipeline: `src/problem_3_pipeline.py`

### **Input:**
- Video: `P3_VIDEO.mp4`
- Model: `runs/detect/drone_detect_v21_max_data/weights/best.pt`
- Config: `configs/botsort_custom.yaml`

### **Output:**
- Video: `outputs/problem_3/final/P3_OUTPUT_FINAL.mp4`
- Stats: Console output
