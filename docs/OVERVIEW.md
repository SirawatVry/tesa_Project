# TESA Problem 2 - Drone Detection & Localization

## 📋 โครงสร้างโปรเจค

```
tesa_problem_2/
├── configs/                      # ไฟล์ Configuration
│   ├── data.yaml                 # YOLO dataset config
│   ├── data_augmented.yaml       # Augmented dataset config
│   ├── ensemble_config.json      # Ensemble model config
│   ├── feature_columns*.json     # Feature definitions (v1, v2, v16, v21)
│   └── selected_features_v3.json # Selected features
│
├── data/                         # ข้อมูล Training และ Metadata
│   ├── train_metadata*.csv       # Metadata ต่างๆ (original, engineered, enhanced)
│   ├── gcp_collection_targets.csv
│   ├── gcp_samples.csv
│   └── gcp_samples.json
│
├── datasets/                     # Dataset ต้นฉบับ
│   ├── DATA_TRAIN/              # Training images & labels
│   ├── DATA_TEST/               # Test images
│   └── train_data/              # Processed training data
│
├── models/                       # โมเดลที่ Train แล้ว
│   ├── tomorbest.pt             # Best YOLO model
│   ├── yolo*.pt                 # Pre-trained YOLO models
│   ├── xgb_model_*.pkl          # XGBoost models (lat, lon, alt) หลายเวอร์ชัน
│   ├── models_approximation/    # Approximation approach models
│   │   ├── nn_best.pth
│   │   ├── nn_custom_loss.pth
│   │   ├── bbox_features.json
│   │   └── correction_params.json
│   └── models_stacking/         # Stacking ensemble models
│
├── yolo_dataset/                # YOLO Training Dataset
│   ├── train/
│   └── valid/
│
├── yolo_dataset_augmented/      # Augmented YOLO Dataset
│   ├── train/
│   └── valid/
│
├── more_label_1/                # Additional labeled data
│   ├── labels/
│   └── train/
│
├── scripts/                     # Scripts แบ่งตามหมวดหมู่
│   ├── 01_data_exploration/     # 📊 Data Analysis & EDA
│   ├── 02_yolo_preparation/     # 🏷️ YOLO Dataset Preparation
│   ├── 03_yolo_training/        # 🎯 YOLO Model Training
│   ├── 04_xgboost_training/     # 🌲 XGBoost Model Training
│   ├── 05_evaluation/           # 📈 Model Evaluation
│   ├── 06_prediction/           # 🔮 Prediction Scripts
│   ├── 07_ensemble/             # 🎭 Ensemble Methods
│   └── 08_utilities/            # 🔧 Utilities & Analysis
│
├── outputs/                     # ผลลัพธ์การประมวลผล
│   ├── predictions/             # CSV predictions
│   ├── visualizations/          # รูปภาพและกราฟ
│   ├── reports/                 # Reports และ Documentation
│   ├── visualization_results/   # Additional visualization outputs
│   └── multi_drone_strategies/  # Multi-drone analysis results
│
├── runs/                        # YOLO Training Runs
│   ├── detect/
│   └── obb/
│
└── docs/                        # Documentation
    ├── OVERVIEW.md              # This file
    ├── PROJECT_STRUCTURE.md     # Detailed structure
    ├── QUICK_START.md           # Quick start guide
    ├── SUMMARY.md               # Project summary
    └── PROBLEM_3_TASKS.md       # Task checklist
```

---

## 🚀 Quick Start

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
