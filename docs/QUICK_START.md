# 🚀 Quick Start Guide - TESA Problem 2

## ⚡ การใช้งานด่วน

### **1. รัน Pipeline (แนะนำ)**

```bash
# รัน pipeline หลัก - สร้าง output video
python src/problem_3_pipeline.py
```

**Output:**
- ไฟล์: `outputs/problem_3/final/P3_OUTPUT_FINAL.mp4`
- ขนาด: ~70 MB
- เวลา: ~2.5 นาที (CPU)

---

### **2. ตรวจสอบผลลัพธ์**

```bash
# ดู output video
outputs/problem_3/final/P3_OUTPUT_FINAL.mp4

# อ่าน documentation
outputs/problem_3/final/README.md
```

---

### **3. วิเคราะห์ Track Patterns (Optional)**

```bash
# วิเคราะห์รูปแบบการเคลื่อนที่
python scripts/05_evaluation/analyze_track_patterns.py

# ตรวจสอบ Track IDs
python scripts/05_evaluation/check_actual_track_ids.py
```

---

## 📁 โครงสร้างที่สำคัญ

```
tesa_problem_2/
├── 📹 P3_VIDEO.mp4                    # Input video
│
├── 💻 src/
│   └── problem_3_pipeline.py          # 🎯 Main script
│   ├── detector.py                    # Detection module
│   ├── tracker.py                     # Tracking module
│   ├── localizer.py                   # Localization module
│   └── visualizer.py                  # Visualization module
│
├── ⚙️ configs/
│   └── botsort_custom.yaml            # Tracker configuration
│
├── 🤖 models/
│   └── models_approximation/best.pt   # Pre-trained model
│
└── 📤 outputs/
    └── problem_3/
        └── final/
            └── P3_OUTPUT_FINAL.mp4    # Output video
```

---

## 🔧 Environment Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Test Environment
```bash
python scripts/08_utilities/test_environment.py
```

---

## 📊 Expected Output

| Metric | Value |
|--------|-------|
| **Detection Rate** | 99.1% |
| **Total Detections** | 3,530 |
| **Track IDs** | 2 |
| **Processing Speed** | 12.9 FPS |
| **File Size** | 69.42 MB |
