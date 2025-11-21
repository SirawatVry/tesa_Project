# 📐 Camera Geometry Calibration Guide

## ✅ ข้อมูลที่มีอยู่แล้ว
- ✅ ขนาดโดรน (physical dimensions)
- ✅ GCP บางจุด (Ground Control Points = รู้ตำแหน่งจริง)
- ✅ Camera position: 14.305029°N, 101.173010°E (fixed)
- ✅ Training data: 438 images with GPS coordinates
- ✅ YOLO bbox: ตำแหน่งและขนาดโดรนในภาพ

---

## 🎯 แนวทาง: Hybrid Camera Geometry + Calibration

### Step 1: Camera Calibration (ใช้ GCP)
**สิ่งที่ต้องหา:**

1. **Focal Length (f)** - ความยาวโฟกัส
   ```
   f = image_width / (2 × tan(FOV_horizontal / 2))
   ```
   - ประมาณจาก GCP: วัด bbox_width ของโดรนที่ระยะทราบ
   - สูตร: `f = (bbox_width_pixels × distance) / drone_physical_width`

2. **Field of View (FOV)** - มุมมองกล้อง
   ```
   FOV_horizontal = 2 × atan(image_width / (2 × f))
   FOV_vertical = 2 × atan(image_height / (2 × f))
   ```
   - ทั่วไป: 50-70° horizontal (ลอง 60° ก่อน)
   - Verify ด้วย GCP ที่มี

3. **Camera Orientation** - ทิศกล้อง
   ```
   base_bearing = direction camera is pointing (0-360°)
   tilt_angle = camera pitch (มองขึ้น/ลง, degrees)
   ```
   - หาจาก GCP: เฉลี่ย bearing ของโดรนที่อยู่ตรงกลางภาพ
   - Tilt: ประมาณจากความสูงเฉลี่ยที่เห็นโดรน

4. **Camera Height** - ความสูงกล้องจากพื้น
   ```
   camera_altitude_m = ความสูงติดตั้งกล้อง (meters)
   ```
   - ถ้าไม่รู้: ประมาณจาก altitude ของโดรนที่ใกล้กล้องที่สุด

5. **Drone Physical Size** - ขนาดโดรนจริง
   ```
   drone_width_m = ความกว้างโดรน (meters)
   drone_height_m = ความสูงโดรน (meters)
   ```
   - ✅ คุณมีอยู่แล้ว!

---

### Step 2: Calibration Process

```python
# 1. เลือก GCP samples (10-20 ภาพที่มีข้อมูลครบ)
gcp_samples = training_data.sample(20)

# 2. Calculate focal length จากแต่ละ GCP
focal_lengths = []
for sample in gcp_samples:
    # รู้: distance (จาก GPS), bbox_width (จาก YOLO), drone_width (จริง)
    f = (bbox_width_pixels * distance_m) / drone_width_m
    focal_lengths.append(f)

focal_length_avg = np.mean(focal_lengths)

# 3. Calculate FOV
FOV_h = 2 * np.arctan(image_width / (2 * focal_length_avg))
FOV_v = 2 * np.arctan(image_height / (2 * focal_length_avg))

# 4. Find camera base bearing
bearings_at_center = []
for sample in gcp_samples:
    if 0.45 < bbox_center_x < 0.55:  # ใกล้กลางภาพ
        bearings_at_center.append(sample.true_bearing)
        
camera_base_bearing = np.mean(bearings_at_center)

# 5. Estimate camera tilt
# จาก geometry: tilt ≈ atan((drone_alt - camera_alt) / distance)
tilts = []
for sample in gcp_samples:
    tilt = np.arctan((sample.altitude - camera_height) / sample.distance)
    tilts.append(np.degrees(tilt))
    
camera_tilt_deg = np.mean(tilts)
```

---

### Step 3: Geometric Calculations

**A. Bearing Calculation** (ทิศทาง)
```python
def calculate_bearing_from_bbox(bbox_center_x, image_width, 
                                 focal_length, camera_base_bearing):
    """
    คำนวณ bearing จาก position ใน image
    """
    # Pixel offset from image center
    pixel_offset = (bbox_center_x * image_width) - (image_width / 2)
    
    # Angular offset
    angle_offset_rad = np.arctan(pixel_offset / focal_length)
    angle_offset_deg = np.degrees(angle_offset_rad)
    
    # Final bearing
    bearing = (camera_base_bearing + angle_offset_deg) % 360
    
    return bearing
```

**B. Distance Calculation** (ระยะทาง)
```python
def calculate_distance_from_bbox(bbox_height, image_height,
                                  focal_length, drone_physical_height,
                                  altitude_diff):
    """
    คำนวณระยะจาก bbox size และ altitude
    """
    # Method 1: From bbox height (pinhole camera model)
    distance_pixels = (drone_physical_height * focal_length) / (bbox_height * image_height)
    
    # Method 2: Adjust for altitude difference
    # distance_ground = sqrt(distance_slant^2 - altitude_diff^2)
    distance_slant = distance_pixels
    distance_ground = np.sqrt(max(0, distance_slant**2 - altitude_diff**2))
    
    return distance_ground
```

**C. Altitude Calculation** (ความสูง)
```python
def calculate_altitude_from_bbox(bbox_center_y, image_height,
                                  camera_tilt, camera_height,
                                  distance, focal_length):
    """
    คำนวณ altitude จาก vertical position
    """
    # Pixel offset from image center (vertical)
    pixel_offset_y = (image_height / 2) - (bbox_center_y * image_height)
    
    # Vertical angle
    angle_vertical = np.arctan(pixel_offset_y / focal_length)
    angle_total = camera_tilt + np.degrees(angle_vertical)
    
    # Altitude
    altitude = camera_height + distance * np.tan(np.radians(angle_total))
    
    return altitude
```

---

### Step 4: Refinement with ML

```python
# หลังจากคำนวณแบบ geometry แล้ว
# ใช้ ML เพื่อ fine-tune residuals

# Features for ML refinement
features = [
    'bearing_geometry',   # จาก Step 3A
    'distance_geometry',  # จาก Step 3B  
    'altitude_geometry',  # จาก Step 3C
    'bbox_x', 'bbox_y',
    'bbox_width', 'bbox_height',
    'bbox_area', 'confidence'
]

# Target: difference from ground truth
targets = [
    'bearing_residual',   # true_bearing - bearing_geometry
    'distance_residual',  # true_distance - distance_geometry
    'altitude_residual'   # true_altitude - altitude_geometry
]

# Train small XGBoost to learn residuals
model_residual = XGBRegressor(max_depth=3, n_estimators=50)
model_residual.fit(X_train[features], y_train[targets])

# Final prediction
bearing_final = bearing_geometry + model.predict(bearing_residual)
distance_final = distance_geometry + model.predict(distance_residual)
altitude_final = altitude_geometry + model.predict(altitude_residual)
```

---

## 📊 ข้อมูลที่ต้องเตรียม (Checklist)

### ข้อมูลที่รู้แล้ว ✅
- [x] Camera GPS: 14.305029°N, 101.173010°E
- [x] Drone physical size (width, height)
- [x] GCP samples (บางจุด)
- [x] Training images with GPS
- [x] YOLO bbox detections

### ข้อมูลที่ต้องหา/ประมาณ 🔍
- [ ] **Focal length** (pixels) - หาจาก GCP calibration
- [ ] **FOV** (degrees) - คำนวณจาก focal length
- [ ] **Camera base bearing** (degrees 0-360) - หาจาก GCP ที่กลางภาพ
- [ ] **Camera tilt angle** (degrees) - ประมาณจาก altitude differences
- [ ] **Camera height** (meters) - ประมาณจาก terrain + installation
- [ ] **Image resolution** - รู้แล้ว (1280×720 หรือตามจริง)

---

## 🎯 Implementation Plan

### Phase 1: Camera Calibration (1-2 hr)
```python
python 34_camera_calibration.py
# Output: camera_params.json
```

### Phase 2: Geometric Prediction (1 hr)
```python
python 35_geometric_prediction.py
# Output: predictions_geometry.csv
# Score: คาดว่า 7-9 (ยังไม่ fine-tune)
```

### Phase 3: ML Refinement (1 hr)
```python
python 36_ml_refinement.py
# Output: models_geometry_refined/
# Score: คาดว่า 5-7 (ดีขึ้น)
```

### Phase 4: Validation & Ensemble (30 min)
```python
python 37_geometry_ensemble.py
# Ensemble: geometry + original models
# Score: คาดว่า 4.5-5.5 (best)
```

---

## 💡 Expected Results

| Approach | Expected Score | Pros | Cons |
|----------|---------------|------|------|
| Pure Geometry | 7-9 | ไม่มี leakage | ต้อง calibrate ดี |
| Geometry + ML | 5-7 | ดีขึ้น, ไม่มี leakage | ซับซ้อนกว่า |
| Geometry Ensemble | 4.5-5.5 | Best of both | ต้องทำหลายขั้น |
| Current Ensemble | 5.28 | มี leakage แต่ score ดี | ไม่ใช้ได้จริง |

---

## 🚀 Quick Start

อยากให้ผมสร้าง scripts ตาม 4 phases เลยไหมครับ?
หรืออยากเริ่มที่ Phase 1 (Camera Calibration) ก่อน?
