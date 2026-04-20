# TESA Problem 2 - Drone Detection & Localization

A complete pipeline for drone detection, tracking, and localization from video streams using YOLO and machine learning models.

## 🎯 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the main pipeline
python src/problem_3_pipeline.py

# 3. Output will be saved to:
# outputs/problem_3/final/P3_OUTPUT_FINAL.mp4
```

## 📁 Project Structure

```
.
├── src/                    # Main source code
│   ├── problem_3_pipeline.py    # Main pipeline
│   ├── detector.py              # YOLO detection
│   ├── tracker.py               # Object tracking
│   ├── localizer.py             # GPS prediction
│   └── visualizer.py            # Visualization
│
├── configs/                # Configuration files
├── models/                 # Pre-trained models
├── data/                   # Processed data
├── scripts/                # Analysis & utility scripts
├── outputs/                # Results and outputs
├── docs/                   # Documentation
│   ├── OVERVIEW.md        # Project overview
│   ├── QUICK_START.md     # Quick start guide
│   ├── PROJECT_STRUCTURE.md # Detailed structure
│   ├── SUMMARY.md         # Project summary
│   └── PROBLEM_3_TASKS.md # Task checklist
│
└── requirements.txt        # Python dependencies
```

## 📖 Documentation

Full documentation is available in the `docs/` folder:
- **[OVERVIEW.md](docs/OVERVIEW.md)** - Complete project overview
- **[QUICK_START.md](docs/QUICK_START.md)** - Quick start guide
- **[PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Detailed structure
- **[SUMMARY.md](docs/SUMMARY.md)** - Project summary
- **[PROBLEM_3_TASKS.md](docs/PROBLEM_3_TASKS.md)** - Task checklist
- **[DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md)** - Docker setup and usage

## 🚀 Main Features

- **YOLO Detection**: High-accuracy drone detection (mAP: 81%)
- **Object Tracking**: Multi-object tracking with ByteTrack
- **Localization**: GPS coordinate prediction
- **Visualization**: Real-time video annotation
- **Ensemble Methods**: Multiple model approaches

## 🔧 Setup & Requirements

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)

### Installation
```bash
pip install -r requirements.txt
```

### Verify Setup
```bash
pyt

## 🐳 Docker Support

Run the project in Docker containers (CPU or GPU):

### Quick Start with Docker
```bash
# CPU version
docker-compose up

# GPU version (requires NVIDIA runtime)
docker-compose -f docker-compose.gpu.yml up
```

See [DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md) for detailed Docker instructions.hon scripts/08_utilities/test_environment.py
```

## 📊 Results

| Metric | Value |
|--------|-------|
| Detection Rate | 99.1% |
| Total Detections | 3,530 |
| Track IDs | 2 |
| Processing Speed | 12.9 FPS |
| Output Size | 69.42 MB |

## 🎮 Usage

### Basic Usage
```bash
python src/problem_3_pipeline.py
```

### Advanced Analysis
```bash
# Analyze track patterns
python scripts/05_evaluation/analyze_track_patterns.py

# Check actual track IDs
python scripts/05_evaluation/check_actual_track_ids.py

# Analyze specific frames
python scripts/05_evaluation/analyze_specific_frames.py
```

## 📚 Project Organization

This project uses a well-organized structure:
- **src/**: All source code modules
- **scripts/**: Analysis and utility scripts (numbered by phase)
- **configs/**: Configuration files
- **models/**: Pre-trained model files
- **data/**: Processed data and metadata
- **outputs/**: Results and generated files
- **docs/**: Documentation files

## 📝 Notes

- Input video: `P3_VIDEO.mp4` (1920x1080, 25 FPS)
- Output video: `outputs/problem_3/final/P3_OUTPUT_FINAL.mp4`
- Processing time: ~2.5 minutes (CPU) or ~1.5 minutes (GPU)

## 🔗 Related Files

- Configuration: [configs/botsort_custom.yaml](configs/botsort_custom.yaml)
- Main script: [src/problem_3_pipeline.py](src/problem_3_pipeline.py)
- Scripts directory: [scripts/](scripts/)

---

**Status**: ✅ Complete  
**Last Updated**: November 13, 2025
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
