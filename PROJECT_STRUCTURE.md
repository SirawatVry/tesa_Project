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

### **Processing:**
```
Video → Detection → ByteTrack → Weighted NMS → Track Merging → Localization → Visualization
```

---

## 📊 Key Files

### **Models:**
| File | Type | Performance | Usage |
|------|------|-------------|-------|
| `runs/detect/drone_detect_v21_max_data/weights/best.pt` | YOLOv8n | mAP: 81%, Recall: 90% | Main detector |
| `models/models_approximation/nn_best.pth` | NN | - | GPS prediction |

### **Configs:**
| File | Purpose | Key Settings |
|------|---------|--------------|
| `configs/botsort_custom.yaml` | Tracker | track_buffer: 180 frames |
| `configs/feature_columns_v16.json` | Features | 9 bbox features |

### **Outputs:**
| File | Type | Description |
|------|------|-------------|
| `outputs/problem_3/final/P3_OUTPUT_FINAL.mp4` | Video | Final output (69.42 MB) |
| `outputs/problem_3/analysis/track_data.npy` | Data | Track patterns |

---

## 🔧 Current Configuration (Optimized)

```python
# Detection
model: drone_detect_v21_max_data/best.pt
conf_threshold: 0.10
iou_threshold: 0.3

# Tracking
tracker: ByteTrack
track_buffer: 180 frames
persist: True

# Track Merging
Track 1 → Drone 1 (right, stable)
Tracks 8,38,48,62 → Drone 2 (left, fragments)

# Weighted NMS
iou_threshold: 0.3
merge_overlapping: True

# Visualization
tracking_path: 50 points
info_panel: Top-left (transparent)
frame_info: Bottom
```

---

## 📈 Results Summary

| Metric | Value |
|--------|-------|
| Detection Rate | 99.1% (1859/1875 frames) |
| Total Detections | 3,530 |
| Unique Track IDs | 2 [1, 2] |
| Processing Speed | 14.2 FPS (CPU) |
| Output File Size | 69.42 MB (< 200 MB limit) |

---

## 🚀 Quick Start

### **Run Main Pipeline:**
```bash
python src/problem_3_pipeline.py
```

### **Analyze Track Patterns:**
```bash
python scripts/05_evaluation/analyze_track_patterns.py
```

### **Check Track IDs:**
```bash
python scripts/05_evaluation/check_actual_track_ids.py
```

---

## 📝 Version History

### **v1.0 - Current (IOU=0.3, Weighted NMS)**
- ✅ 2 Track IDs (correct)
- ✅ Weighted NMS (IOU=0.3)
- ✅ ByteTrack tracker
- ✅ Tracking path visualization
- ✅ Info panel (compact, transparent)
- ✅ 99.1% detection rate

### **Previous Versions:**
- v0.9: IOU=0.6, 7 track IDs → 2 after merging
- v0.8: BoT-SORT tracker testing
- v0.7: CLAHE enhancement (rejected)
- v0.6: Track buffer optimization (30→180)

---

## 🗂️ File Organization Rules

### **Outputs:**
```
outputs/problem_3/
├── final/              # Final deliverables only
├── analysis/           # Analysis results, visualizations
└── experiments/        # Experimental/testing outputs
```

### **Naming Convention:**
```
Final output:     P3_OUTPUT_FINAL.mp4
Analysis files:   track_analysis_*.npy, frame_stats_*.csv
Experiments:      test_iou_*.mp4, experiment_*.jpg
```

---

## 🔗 Dependencies

See individual source files for specific requirements:
- `ultralytics` (YOLO)
- `opencv-python`
- `numpy`
- `xgboost`
- `scikit-learn`
- `tqdm`

---

**Last Updated:** November 13, 2025  
**Status:** ✅ Production Ready
