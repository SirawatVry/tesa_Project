# Problem 3: Drone Tracking - Task Checklist

## 📋 Overview
**Goal:** ตรวจจับและติดตามโดรนจากวิดีโอ + ทำนายพิกัด (lat, lon, alt) แบบเรียลไทม์

**Pipeline:** Video → YOLO Detection → Tracking → Localization → Visualization

**Status:** ✅ **COMPLETE** (November 13, 2025)

---

## ✅ Phase 1: Setup & Environment

### 1.1 Install Dependencies ✅
- ✅ Install required packages
  ```bash
  pip install ultralytics
  pip install opencv-python
  pip install torch torchvision
  pip install deep-sort-realtime
  pip install numpy pandas matplotlib xgboost
  pip install tqdm
  ```

### 1.2 Prepare Models ✅
- ✅ Copy YOLOv8 model from Problem 1
  - Path: `models/tomorbest.pt` ✅
  - Verified detection capability
  
- ✅ Use Approximation models from Problem 2
  - Path: `models/models_approximation/` ✅
  - 9 bbox features (simpler than XGBoost v21)
  - Models: distance, bearing_sin/cos, altitude

### 1.3 Prepare Data ✅
- ✅ Test video file
  - Located: `P3_VIDEO.mp4` ✅
  - Properties: 1920x1080, 25 FPS, 1899 frames, 76s
  - Contains 2 identical drones
  
- ✅ Analysis complete
  - Video analyzed and optimized
  - Detection threshold optimized to 0.15

---

## ✅ Phase 2: Core Development

### 2.1 YOLO Detection Integration ✅
- ✅ Load YOLO model
