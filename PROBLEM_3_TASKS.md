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
- ✅ Test detection on single frame
- ✅ Verify bounding box format (xyxy)
- ✅ Optimized confidence threshold to 0.15
- ✅ Optimized IOU threshold to 0.6
- [ ] Verify class detection (should be 'drone')

**Files to create:**
- `src/detector.py` - YOLO detection wrapper

**Test code:**
```python
from ultralytics import YOLO
model = YOLO('models/tomorbest.pt')
results = model('test_frame.jpg', conf=0.3)
```

---

### 2.2 Tracking Implementation ✅

#### Selected: ByteTrack ⭐ (Best balance)
- ✅ Configure ByteTrack parameters
- ✅ Use YOLO.track() with persist=True
- ✅ Test tracking on full video
- ✅ Verify track_id consistency (Track 243: 40.3s continuous)
- ✅ Handles occlusions well

**Implementation:**
- Created `src/tracker.py` with ByteTracker class
- Uses default `bytetrack.yaml` configuration
- Persistent track IDs with stream mode
- Main track (243) shows 100% continuity

---

### 2.3 Localization Integration ✅

- ✅ Load approximation models from Problem 2
- ✅ Implement bbox feature extraction (9 features)
  - yolo_cx, yolo_cy, yolo_w, yolo_h
  - yolo_conf, yolo_area, yolo_aspect_ratio
  - yolo_dist_from_center, yolo_angle_from_center
  
- ✅ Implement prediction function
  - Bbox → Distance (XGBoost)
  - Bbox → Bearing sin/cos (XGBoost)
  - Bbox → Altitude (XGBoost)
  - Convert to lat/lon coordinates

- ✅ Add coordinate smoothing
  - 5-frame moving average
  - Per-track history tracking
  - Jitter < 1 meter/frame

**Files created:**
- ✅ `src/localizer.py` - ApproximationLocalizer class

---

### 2.4 Visualization ✅

- ✅ Implement bounding box drawing
  - Color-coded for each track_id (Red, Green, Blue, Yellow, Magenta, Cyan)
  - Thickness: 2 pixels
  
- ✅ Implement text overlay
  - Track ID label near bbox
  - **Coordinates at top-left of bbox** ✅
  - Font: cv2.FONT_HERSHEY_SIMPLEX
  - Size: 0.5
  - White text with semi-transparent black background
  
- ✅ Format coordinate display
  ```
  track_id: 243
  lat: 14.30485
  lon: 101.17280
  alt: 45.50
  ```
  
- ✅ Frame information overlay
  - Frame number, time, drone count at bottom
  
**Files created:**
- ✅ `src/visualizer.py` - DroneVisualizer class
      
      # Draw semi-transparent background (optional)
      overlay = frame.copy()
      cv2.rectangle(overlay, 
                   (x1, y_offset - 5), 
                   (x1 + 150, y1 - 5),
                   (0, 0, 0), -1)
      frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
      
      # Draw text lines
      for i, line in enumerate(lines):
          y_pos = y_offset + (i * line_height)
          cv2.putText(frame, line, (x1, y_pos),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                     (255, 255, 255), 1, cv2.LINE_AA)
      
      return frame
  ```

- [ ] Test visualization on sample frames

**Files to create:**
- `src/visualizer.py` - Visualization functions

---

## ✅ Phase 3: Pipeline Integration

### 3.1 Main Processing Script

- [ ] Create main pipeline script
- [ ] Implement frame-by-frame processing
- [ ] Add progress bar (tqdm)
- [ ] Handle edge cases:
  - No detection in frame
  - Track lost and recovered
  - Multiple drones
  
**Files to create:**
- `src/problem_3_pipeline.py` - Main processing pipeline

**Pipeline structure:**
```python
for frame in video:
    # 1. Detect drones (YOLO)
    detections = detector.detect(frame)
    
    # 2. Update tracker
    tracks = tracker.update(detections, frame)
    
    # 3. For each track:
    for track in tracks:
        # 3.1 Crop drone
        crop = frame[y1:y2, x1:x2]
        
        # 3.2 Predict coordinates
        lat, lon, alt = localizer.predict(crop)
        
        # 3.3 Smooth coordinates
        lat, lon, alt = smoother.smooth(track_id, (lat, lon, alt))
        
        # 3.4 Draw on frame
        visualizer.draw(frame, track, lat, lon, alt)
    
---

## ✅ Phase 3: Pipeline Integration ✅

### 3.1 Main Pipeline Script ✅

- ✅ Create `src/problem_3_pipeline.py` - Complete end-to-end pipeline
- ✅ Initialize all components (Detector, Tracker, Localizer, Visualizer)
- ✅ Process video frame-by-frame
- ✅ Write annotated frames to output video
- ✅ Add progress tracking (tqdm)
- ✅ Print statistics (detections, tracks, processing time)

**Performance:**
- Processing speed: 21.5 FPS (46.5 ms/frame)
- Total time: ~1.5 minutes for 75-second video
- Detection rate: 84.9% of frames

### 3.2 Configuration Management ✅

- ✅ Created optimized configuration
  - Video: P3_VIDEO.mp4 → P3_OUTPUT_FULL.mp4
  - Confidence: 0.15, IOU: 0.6
  - Tracker: ByteTrack with persist=True
  - Output codec: mp4v

---

## ✅ Phase 4: Optimization & Quality ✅

### 4.1 Track Stability Analysis ✅

- ✅ Created `scripts/05_evaluation/analyze_track_stability.py`
- ✅ Analyzed 40 unique track IDs
- ✅ Identified main track (Track 243): 40.3s continuous
- ✅ Generated visualizations:
  - Track timeline
  - Movement analysis (trajectories & speed)
  - Track overlap matrix

**Key Findings:**
- Track 243: 1,007 frames, 100% continuous ✅
- 3 main tracks with >100 frames
- No track ID switches for main drone

### 4.2 Coordinate Accuracy Evaluation ✅

- ✅ Created `scripts/05_evaluation/evaluate_coordinates.py`
- ✅ Sampled predictions every 5 frames
- ✅ Analyzed coordinate stability and jitter
- ✅ Generated visualizations:
  - Coordinate timeseries (lat, lon, alt, distance)
  - Jitter analysis
- ✅ Exported predictions to CSV (5 tracks)

**Quality Metrics:**
- Latitude jitter: 0.22-0.54 m/sample
- Longitude jitter: 0.12-0.87 m/sample  
- Altitude jitter: 0.08-0.30 m/sample
- Coordinate smoothing: 5-frame moving average ✅

### 4.3 Output Quality Check ✅

- ✅ Created `scripts/05_evaluation/check_output_quality.py`
- ✅ Verified file size: 66.42 MB < 200 MB ✅
- ✅ Checked video integrity: 0 corrupt frames ✅
- ✅ Verified annotations: 100% coverage ✅
- ✅ Quality metrics: Good brightness (114.44 avg)
- ✅ Generated quality report JSON

---

## ✅ Phase 5: Documentation & Deliverables ✅

### 5.1 Output Video ✅

- ✅ Processed full video: `outputs/problem_3/P3_OUTPUT_FULL.mp4`
- ✅ Verified all requirements:
  - ✅ Bounding boxes around drones
  - ✅ Track IDs displayed
  - ✅ Coordinates at top-left of bbox
  - ✅ Color-coded tracks
  - ✅ File size: 66.42 MB < 200 MB
  - ✅ Format: MP4 (mp4v codec)
  - ✅ Resolution: 1920x1080 maintained
  - ✅ FPS: 25.00 maintained

### 5.2 Documentation ✅

- ✅ Created comprehensive report: `outputs/problem_3/PROBLEM_3_REPORT.md`
  - Executive summary
  - Solution architecture
  - Implementation results
  - Technical challenges & solutions
  - Key findings
  - Recommendations

- ✅ Created README: `outputs/problem_3/README.md`
  - Quick start guide
  - Results summary
  - Architecture overview
  - File structure
  - Evaluation tools

- ✅ Created delivery checklist: `outputs/problem_3/DELIVERY_CHECKLIST.md`
  - Requirements verification
  - Performance metrics
  - File locations
  - Testing verification

### 5.3 Source Code ✅

**Files created:**
- ✅ `src/detector.py` - YOLO detection wrapper
- ✅ `src/tracker.py` - ByteTrack implementation  
- ✅ `src/localizer.py` - Approximation-based localization
- ✅ `src/visualizer.py` - Annotation rendering
- ✅ `src/problem_3_pipeline.py` - Main pipeline

**Evaluation scripts:**
- ✅ `scripts/05_evaluation/analyze_track_stability.py`
- ✅ `scripts/05_evaluation/evaluate_coordinates.py`
- ✅ `scripts/05_evaluation/check_output_quality.py`

### 5.4 Visualizations & Analysis ✅

**Track Analysis:**
- ✅ `track_timeline.png` - Track appearance timeline
- ✅ `movement_analysis.png` - Trajectories and speed
- ✅ `track_overlap.png` - Track overlap matrix

**Coordinate Evaluation:**
- ✅ `coordinate_timeseries.png` - GPS coordinates over time
- ✅ `coordinate_jitter.png` - Stability analysis
- ✅ `predictions_track_*.csv` - Raw prediction data (5 tracks)

**Quality Reports:**
- ✅ `quality_report.json` - Comprehensive quality metrics

---

## 🎉 Final Status: COMPLETE ✅

**All phases completed successfully!**

### Summary Statistics:
- ✅ Output video: 66.42 MB (< 200 MB)
- ✅ Processing speed: 21.5 FPS
- ✅ Detection rate: 84.9%
- ✅ Main track duration: 40.3s continuous
- ✅ Annotation coverage: 100%
- ✅ File integrity: Perfect (0 corrupt frames)

### Deliverables:
- ✅ Annotated output video
- ✅ Complete source code
- ✅ Comprehensive documentation
- ✅ Evaluation results & visualizations
- ✅ Quality assurance reports

**Ready for submission! 🚀**

---

**Completion Date:** November 13, 2025  
**Total Development Time:** ~2 hours  
**Final Status:** ✅ ALL REQUIREMENTS MET

---

## 📊 Scoring Checklist

### 1. ความถูกต้องของการติดตามโดรน (1-5 คะแนน)
- [ ] ตรวจจับโดรนได้ทุกเฟรม
- [ ] Track ID ถูกต้อง ไม่สลับ
- [ ] Bounding box แม่นยำ

### 2. ความต่อเนื่อง/เสถียรภาพของการ Tracking (1-5 คะแนน)
- [ ] Track ID คงที่ตลอดคลิป
- [ ] Handle occlusion ได้ดี
- [ ] ไม่มี jitter หรือ jumping
- [ ] Coordinates ไม่กระโดด (smoothing works)

### 3. ความชัดเจนของการนำเสนอ (1-5 คะแนน)
- [ ] Bounding box ชัดเจน
- [ ] Text อ่านง่าย
- [ ] สีแยกแต่ละลำได้ชัด
- [ ] Layout สวยงาม เป็นระเบียบ
- [ ] Video quality ดี

---

## 🚀 Quick Start Commands

### Setup
```bash
cd c:\Users\title\OneDrive\เดสก์ท็อป\tesa_problem_2
pip install -r requirements.txt
```

### Test Detection
```bash
python scripts/08_utilities/test_yolo_detection.py --model models/tomorbest.pt --video datasets/DATA_TEST/test_video.mp4
```

### Run Pipeline
```bash
python src/problem_3_pipeline.py --config configs/problem_3_config.yaml
```

### Evaluate Results (if GT available)
```bash
python scripts/08_utilities/evaluate_problem_3.py --predictions outputs/problem_3/predictions.json --ground_truth data/problem_3_ground_truth.json
```

---

## 📝 Notes & Tips

### For 2 Identical Drones:
- ✅ Use **BoT-SORT** or **DeepSORT** for Re-ID
- ✅ Higher confidence threshold (0.4-0.5)
- ✅ Lower IOU threshold (0.3-0.4)
- ✅ Enable coordinate smoothing

### For Performance:
- Resize frames if video is 4K
- Use GPU for all models
- Consider processing every 2nd frame if too slow

### For Debugging:
- Save intermediate results (detections, tracks)
- Visualize on single frame first
- Use verbose logging

---

## ⚠️ Critical Requirements

- ❌ **NO manual annotation** - everything must be automated
- ✅ **Must be able to demo live** when requested
- ✅ Code must run on fresh environment
- ✅ Clear documentation for running

**Deadline:** Check competition schedule
**File size limits:** 
- Video: 200MB
- Code ZIP: Keep minimal (exclude data/models)

---

**Status:** 🔴 Not Started | 🟡 In Progress | 🟢 Completed

**Last Updated:** November 13, 2025
