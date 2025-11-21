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
│   │   ├── 01_load_data.py
│   │   ├── 01_data_exploration.py
│   │   └── 02_eda_analysis.py
│   │
│   ├── 02_yolo_preparation/     # 🏷️ YOLO Dataset Preparation
│   │   ├── 03_prepare_yolo_dataset.py
│   │   ├── 03b_fix_label_class.py
│   │   ├── 03c_convert_to_standard_yolo.py
│   │   ├── 03d_fix_remaining_class1.py
│   │   ├── 20_merge_new_labels.py
│   │   └── 22_merge_valid_to_train.py
│   │
│   ├── 03_yolo_training/        # 🎯 YOLO Model Training
│   │   ├── 04_train_yolo_obb.py
│   │   ├── 11_tune_yolo_v2.py
│   │   ├── 17_train_yolo_augmented.py
│   │   ├── 21_train_yolo_v20.py
│   │   └── 23_train_yolo_v21_max.py
│   │
│   ├── 04_xgboost_training/     # 🌲 XGBoost Model Training
│   │   ├── 06_extract_yolo_features.py
│   │   ├── 07_feature_engineering.py
│   │   ├── 08_train_xgboost.py
│   │   ├── 12_enhanced_feature_engineering.py
│   │   ├── 13_retrain_xgboost_enhanced.py
│   │   ├── 14_feature_selection_train.py
│   │   ├── 18_extract_yolo_v16_features.py
│   │   ├── 19_feature_eng_train_xgb_v16.py
│   │   ├── 25_tune_xgboost_for_angle.py
│   │   ├── 26_train_with_geodetic_angles.py
│   │   ├── 27_xgboost_yolo_features_only.py
│   │   └── 28_tune_xgboost_baseline.py
│   │
│   ├── 05_evaluation/           # 📈 Model Evaluation
│   │   ├── 05_evaluate_yolo.py
│   │   ├── 09_evaluate_xgboost.py
│   │   ├── 15_evaluate_v3_models.py
│   │   ├── eval_v16_quick.py
│   │   ├── eval_v20_full.py
│   │   ├── eval_v21_performance.py
│   │   └── compare_yolo_vs_gt.py
│   │
│   ├── 06_prediction/           # 🔮 Prediction Scripts
│   │   ├── 10_predict_test_set.py
│   │   ├── 31_predict_test_ensemble.py
│   │   ├── 36_predict_test_approximation.py
│   │   ├── 39_predict_test_with_residual.py
│   │   ├── 45_predict_all_drones.py
│   │   ├── 46_predict_top2_drones.py
│   │   └── 47_predict_top2_confident.py
│   │
│   ├── 07_ensemble/             # 🎭 Ensemble Methods
│   │   ├── 16_generate_pseudo_labels.py
│   │   ├── 24_v21_complete_pipeline.py
│   │   ├── 29_baseline_with_yolo_v21.py
│   │   ├── 30_ensemble_v1_v21.py
│   │   ├── 35_approximation_approach.py
│   │   ├── 49_compare_advanced_methods.py
│   │   └── 50_stacking_ensemble.py
│   │
│   └── 08_utilities/            # 🔧 Utilities & Analysis
│       ├── check_gpu.py
│       ├── check_existing_models.py
│       ├── check_gcp_availability.py
│       ├── test_environment.py
│       ├── test_yolo_detection.py
│       ├── test_tomorbest_model.py
│       ├── inspect_tomorbest.py
│       ├── inspect_gcp_collection.py
│       ├── calculate_score.py
│       ├── analyze_error_sources.py
│       ├── 32_visualize_and_report.py
│       ├── 33_fix_no_leakage.py
│       ├── 34_fixed_bearing_distance.py
│       ├── 37_calculate_approximation_score.py
│       ├── 38_analyze_and_improve_predictions.py
│       ├── 40_compare_optimization_methods.py
│       ├── 41_test_neural_network.py
│       ├── 42_visualize_predictions.py
│       ├── 43_analyze_multi_drone.py
│       ├── 44_test_drone_selection_strategies.py
│       ├── 48_visualize_top2_confident.py
│       └── 51_visualize_stacking.py
│
├── outputs/                     # ผลลัพธ์การประมวลผล
│   ├── predictions/             # CSV predictions
│   │   ├── test_predictions*.csv
│   │   ├── validation_*.csv
│   │   ├── method_comparison_results.csv
│   │   └── optimization_comparison_results.json
│   │
│   ├── visualizations/          # รูปภาพและกราฟ
│   │   ├── *.png
│   │   └── *.html
│   │
│   ├── reports/                 # Reports และ Documentation
│   │   ├── problem_3.txt
│   │   ├── PROJECT_PLAN.md
│   │   ├── CAMERA_GEOMETRY_GUIDE.md
│   │   └── *_summary.txt
│   │
│   ├── visualization_results/   # Additional visualization outputs
│   └── multi_drone_strategies/  # Multi-drone analysis results
│
├── runs/                        # YOLO Training Runs
│   ├── detect/
│   └── obb/
│
├── notebooks/                   # Jupyter Notebooks (if any)
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
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

#### Option 1: ใช้ YOLO Built-in Tracking + Localization Model
```bash
# ถ้ามี trained localization model
python scripts/06_prediction/track_and_localize.py --video input.mp4 --output output.mp4
```

#### Option 2: ใช้ YOLO + ByteTrack
```bash
# ถ้าต้องการ customize tracking
python scripts/06_prediction/track_with_bytetrack.py --video input.mp4 --output output.mp4
```

---

## 📊 Model Versions

### YOLO Models
- **v1**: Base YOLOv8n-OBB
- **v2**: Tuned hyperparameters
- **v16**: With augmented data
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

## 🎯 Problem 3 - Drone Tracking Solutions

### วิธีที่ 1: YOLO.track() ⭐ (แนะนำ)
```python
from ultralytics import YOLO

model = YOLO('models/tomorbest.pt')
results = model.track(
    source='video.mp4',
    tracker='bytetrack.yaml',  # or 'botsort.yaml'
    persist=True,
    conf=0.3,
    save=True
)
```

**ข้อดี:**
- Setup ง่าย
- Stable tracking
- Optimized แล้ว

### วิธีที่ 2: YOLO + DeepSORT
```python
# ใช้ deep_sort_realtime
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=30)
# ... implementation
```

**ข้อดี:**
- Re-ID ดีกว่า
- Handle occlusion ดี

**ข้อเสีย:**
- Slower
- More complex

---

## 📝 สำหรับการส่งผลงาน

### Requirements
1. ✅ วิดีโอแสดง:
   - Bounding box รอบโดรน
   - Track ID
   - Predicted coordinates (lat, lon, alt)

2. ✅ Source code (.zip)
   - ไม่รวมรูปภาพ/วิดีโอ
   - สามารถรันได้จริง

3. ✅ ต้องสาธิตได้เมื่อกรรมการขอ

### Output Format
```
Frame แต่ละเฟรม:
┌─────────────────────┐
│ track_id: 1         │
│ lat: 14.30485       │
│ lon: 101.17280      │
│ alt: 40.52          │
│   ┌──────────┐      │
│   │  DRONE   │      │
│   │   (1)    │      │
│   └──────────┘      │
└─────────────────────┘
```

---

## 📞 Contact & Notes

- Video limit: 200MB
- Formats: MP4, AVI, MOV, MKV, MPEG4
- ❌ ห้ามใส่กรอบด้วยมือ - ต้องรันโค้ดอัตโนมัติ
- ⚠️ ต้องสาธิตได้จริง มิฉะนั้นอาจถูกตัดสิทธิ์

---

## 🏆 Scoring Criteria (7 คะแนน)

1. **ความถูกต้องของการติดตาม** (1-5 คะแนน)
2. **ความต่อเนื่อง/เสถียรภาพ** (1-5 คะแนน)
3. **ความชัดเจนของการนำเสนอ** (1-5 คะแนน)

แปลงเป็นสเกล 7 คะแนน

---

**Good luck! 🚁**
