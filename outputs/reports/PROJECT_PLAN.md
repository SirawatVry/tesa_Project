# 🚁 Drone Localization Project Plan

## 📋 ภาพรวมโปรเจค

**วัตถุประสงค์**: พัฒนาระบบทำนายตำแหน่งพิกัดของโดรน (Latitude, Longitude, Altitude) จากรูปภาพโดยอัตโนมัติ

**คะแนน**: 9 คะแนน (เต็ม)

**การประเมินผล**:
- Error มุมทิศ (Direction Angle): น้ำหนัก 70%
- Error ความสูง (Altitude): น้ำหนัก 30%
- สูตรคะแนน: `total_error = 0.7 × mean_angle_error + 0.3 × mean_height_error`

**สถานะปัจจุบัน**: 🟢 **พร้อมเริ่มพัฒนา - มีข้อมูล Bounding Box Labels แล้ว!**

---

## 📊 ข้อมูลที่มีอยู่

### 1. พิกัดกล้อง (Camera Position - Fixed)
- **Latitude**: 14.305029
- **Longitude**: 101.173010
- **ตำแหน่ง**: คงที่ตลอดการถ่ายภาพ

### 2. ชุดข้อมูล (Dataset Overview)

#### 🎯 Training Data (DATA_TRAIN/)
```
DATA_TRAIN/
├── image/                 # ✅ 438 รูปภาพต้นฉบับ (img_0001.jpg - img_0438.jpg)
├── csv/                   # ✅ 438 ไฟล์พิกัดจริง (Ground Truth)
├── train/                 # ✅ 50 ภาพที่ annotate ด้วย Roboflow (Train split)
├── valid/                 # ✅ 42 ภาพที่ annotate ด้วย Roboflow (Valid split)
├── labels/                # ✅ 50 YOLO OBB labels (Oriented Bounding Box)
├── labelsvalid/          # ✅ 42 YOLO OBB labels (Validation)
├── plotbox.py            # ✅ Script วาด bounding box
└── tranform.py           # ✅ Script แปลง COCO → YOLO OBB
```

**สรุป**:
- ✅ รูปภาพทั้งหมด: **438 ภาพ**
- ✅ Ground Truth Coordinates: **438 ไฟล์ CSV**
- ✅ Annotated Images (Roboflow): **92 ภาพ** (50 train + 42 valid)
- ✅ YOLO OBB Labels: **92 labels** พร้อมใช้งาน
- ⚠️ ภาพที่ยังไม่ annotate: **346 ภาพ** (อาจใช้ semi-supervised หรือ train ด้วย 92 ภาพก่อน)

#### 🧪 Test Data (DATA_TEST/)
```
DATA_TEST/
└── img_0001.jpg - img_0264.jpg  # ✅ 264 ภาพสำหรับทำนาย
```

### 3. รูปแบบข้อมูล

#### 📄 CSV Format (Ground Truth)
```csv
Latitude,Longitude,Altitude
14.3047389,101.1728838,44.34
```

#### 📦 YOLO OBB Label Format
```
class cx cy w h angle
```
- `class`: 1 (drone)
- `cx, cy`: Center coordinates (normalized 0-1)
- `w, h`: Width, Height (normalized 0-1)
- `angle`: Rotation angle (radians)

ตัวอย่าง:
```
1 0.057342 0.345745 0.030039 0.022804 0.213257
```

#### 📤 Submission Format (ที่ต้องส่ง)
```csv
ImageName, Latitude, Longitude, Altitude
img_0001.jpg, 14.3042744,101.1738384,35.99
img_0002.jpg, 14.3037139,101.1767931,31.79
```

---

## 🎯 กลยุทธ์การพัฒนา (Development Strategy)

### 🏆 แนวทางหลัก: YOLO Detection + Coordinate Regression

**เหตุผล**:
- ✅ มี YOLO OBB labels พร้อมแล้ว (92 ภาพ)
- ✅ ได้ตำแหน่ง pixel ของโดรนที่แม่นยำ
- ✅ สามารถใช้ geometry-based calculation ร่วมกับ ML
- ✅ เหมาะกับ scoring ที่เน้น direction angle

**ขั้นตอนการทำงาน**:

```
1. YOLO Object Detection
   ↓
2. Drone Position (pixel coordinates)
   ↓
3. Feature Engineering (pixel position + image features + metadata)
   ↓
4. ML Regression (XGBoost/Neural Network)
   ↓
5. Coordinate Prediction (lat, lon, alt)
```

---

## ✅ TASK LIST - พร้อมเริ่มทำงาน

### 📦 Phase 1: Data Exploration & Preparation (วันที่ 1)

#### ✅ Task 1.1: Setup Environment
- [ ] สร้าง virtual environment
- [ ] ติดตั้ง libraries: ultralytics, opencv-python, pandas, numpy, matplotlib, seaborn, xgboost, scikit-learn
- [ ] สร้างโครงสร้างโฟลเดอร์โปรเจค
- [ ] ตั้งค่า config files

#### ✅ Task 1.2: Data Loading & Validation
- [ ] โหลดข้อมูล CSV ทั้งหมด (438 ไฟล์)
- [ ] ตรวจสอบความสมบูรณ์ของข้อมูล (missing values, outliers)
- [ ] สร้าง DataFrame รวม: `image_name, lat, lon, alt`
- [ ] บันทึก `train_metadata.csv`

#### ✅ Task 1.3: Exploratory Data Analysis (EDA)
- [ ] วิเคราะห์ distribution ของ lat, lon, alt
- [ ] คำนวณ statistics: min, max, mean, std, median
- [ ] Plot histogram และ scatter plots
- [ ] วิเคราะห์ correlation matrix

#### ✅ Task 1.4: Geospatial Analysis
- [ ] คำนวณระยะทางจากกล้อง → โดรน (Haversine distance)
- [ ] คำนวณมุมทิศ (azimuth/bearing) จากกล้อง → โดรน
- [ ] Plot trajectory บนแผนที่ 2D (lat-lon plane)
- [ ] วิเคราะห์ flight pattern (circular, linear, random)
- [ ] บันทึกผล EDA เป็น Jupyter notebook

**Output**: 
- `01_data_exploration.ipynb` ✅
- `train_metadata.csv`
- Visualization plots

---

### � Phase 2: YOLO Model Training (วันที่ 2-3)

#### ✅ Task 2.1: Prepare YOLO Dataset
- [ ] จัดโครงสร้างข้อมูลตาม YOLOv8 format:
  ```
  dataset/
  ├── train/
  │   ├── images/  (50 ภาพ)
  │   └── labels/  (50 labels)
  └── valid/
      ├── images/  (42 ภาพ)
      └── labels/  (42 labels)
  ```
- [ ] สร้าง `data.yaml` config file
- [ ] ตรวจสอบ label format (class cx cy w h angle)

#### ✅ Task 2.2: Train YOLOv8-OBB Model
- [ ] เลือก pretrained model: `yolov8n-obb.pt` (nano - เร็ว)
- [ ] ตั้งค่า hyperparameters: epochs=100, imgsz=640, batch=16
- [ ] Train model ด้วย Roboflow annotations
- [ ] Monitor training: loss curves, mAP, precision, recall
- [ ] บันทึก best model checkpoint

#### ✅ Task 2.3: Model Evaluation
- [ ] Validate บน validation set (42 ภาพ)
- [ ] คำนวณ metrics: mAP50, mAP50-95, precision, recall
- [ ] Visualize predictions (plot bounding boxes)
- [ ] วิเคราะห์ false positives/negatives

#### ✅ Task 2.4: Inference on All Training Images
- [ ] Run inference บนภาพทั้งหมด (438 ภาพ)
- [ ] Extract drone positions (cx, cy) จาก bounding boxes
- [ ] บันทึก `yolo_predictions.csv`: image_name, cx, cy, conf, angle
- [ ] Handle cases ที่ไม่เจอโดรน (fallback strategy)

**Output**:
- `runs/obb/train/weights/best.pt` (YOLO model)
- `yolo_predictions.csv`
- Training report & visualizations

---

### 🔧 Phase 3: Feature Engineering (วันที่ 4)

#### ✅ Task 3.1: Extract YOLO Features
- [ ] Load `yolo_predictions.csv`
- [ ] คำนวณ pixel coordinates: `px = cx * img_width`, `py = cy * img_height`
- [ ] Normalized position: `norm_x = cx - 0.5`, `norm_y = cy - 0.5`
- [ ] Bounding box angle (rotation)
- [ ] Confidence score

#### ✅ Task 3.2: Image-based Features
- [ ] Image dimensions: `width, height, aspect_ratio`
- [ ] Color features: `mean_rgb, std_rgb`
- [ ] Brightness: `mean_hsv_v`
- [ ] Sky detection: `sky_ratio` (วิเคราะห์ส่วนบนของภาพ)
- [ ] Edge density: Canny edge detection count

#### ✅ Task 3.3: Metadata Features
- [ ] Image number: `img_id` (extracted from filename)
- [ ] Normalized sequence: `seq_norm = img_id / 438`
- [ ] Camera position: `cam_lat=14.305029, cam_lon=101.173010`

#### ✅ Task 3.4: Geometric Features
- [ ] Distance from image center: `dist_from_center = sqrt(norm_x² + norm_y²)`
- [ ] Angle from image center: `angle_from_center = atan2(norm_y, norm_x)`
- [ ] Quadrant position (1-4)

#### ✅ Task 3.5: Create Feature Matrix
- [ ] Combine all features into DataFrame
- [ ] Merge with ground truth (lat, lon, alt)
- [ ] Handle missing values (imputation or removal)
- [ ] บันทึก `features_train.csv`

**Output**:
- `features_train.csv` (438 rows × N features)
- Feature engineering notebook

---

### 🤖 Phase 4: ML Model Training (วันที่ 5-6)

#### ✅ Task 4.1: Data Preparation
- [ ] Load `features_train.csv`
- [ ] Split features (X) and targets (y_lat, y_lon, y_alt)
- [ ] Train/Validation split: 80/20 (350 train, 88 valid)
- [ ] Feature scaling: StandardScaler
- [ ] บันทึก scaler สำหรับใช้กับ test set

#### ✅ Task 4.2: Baseline Model
- [ ] Train 3 XGBoost models แยก: `model_lat`, `model_lon`, `model_alt`
- [ ] Use default parameters
- [ ] Evaluate: MAE, RMSE
- [ ] Calculate direction angle error
- [ ] Calculate altitude error
- [ ] Compute total score: `0.7 × angle_error + 0.3 × alt_error`

#### ✅ Task 4.3: Feature Importance Analysis
- [ ] Plot feature importance จาก XGBoost
- [ ] วิเคราะห์ features ที่มีผลมากที่สุด
- [ ] Remove low-importance features (threshold < 0.01)
- [ ] Re-train model ด้วย selected features

#### ✅ Task 4.4: Hyperparameter Tuning
- [ ] Define parameter grid:
  ```python
  params = {
      'max_depth': [3, 5, 7, 10],
      'learning_rate': [0.01, 0.05, 0.1],
      'n_estimators': [100, 200, 300],
      'subsample': [0.7, 0.8, 1.0],
      'colsample_bytree': [0.7, 0.8, 1.0]
  }
  ```
- [ ] RandomizedSearchCV (3-fold CV)
- [ ] Select best parameters
- [ ] Train final models

#### ✅ Task 4.5: Model Validation
- [ ] K-Fold Cross Validation (5 folds)
- [ ] Calculate CV scores
- [ ] Check for overfitting
- [ ] Error analysis: identify high-error samples

**Output**:
- `model_lat.pkl`, `model_lon.pkl`, `model_alt.pkl`
- `scaler.pkl`
- Training logs & metrics
- Feature importance plots

---

### 📊 Phase 5: Evaluation & Optimization (วันที่ 7)

#### ✅ Task 5.1: Direction Angle Calculation
- [ ] Implement function `calculate_bearing(lat1, lon1, lat2, lon2)`
- [ ] Calculate true angles: camera → ground truth
- [ ] Calculate predicted angles: camera → predictions
- [ ] Compute angle error (handle 0°/360° wraparound)

#### ✅ Task 5.2: Scoring Metrics
- [ ] Calculate `mean_angle_error` (degrees)
- [ ] Calculate `mean_height_error` (meters)
- [ ] Compute total score: `0.7 × angle + 0.3 × height`
- [ ] เป้าหมาย: total_error < 10° (คะแนน 8-9/9)

#### ✅ Task 5.3: Error Analysis
- [ ] Plot error distribution (histogram)
- [ ] Scatter plot: predicted vs actual
- [ ] Map visualization: overlay predictions on ground truth
- [ ] Identify systematic errors or biases
- [ ] Analyze high-error cases

#### ✅ Task 5.4: Model Refinement
- [ ] Try ensemble methods (Voting, Stacking)
- [ ] Experiment with other models: Random Forest, Neural Network
- [ ] Post-processing: clip values, smoothing
- [ ] Iteratively improve based on error analysis

**Output**:
- Evaluation report with metrics
- Error analysis visualizations
- Improved model versions

---

### 🧪 Phase 6: Test Set Prediction (วันที่ 8)

#### ✅ Task 6.1: Prepare Test Data
- [ ] Load test images (264 ภาพ)
- [ ] Run YOLO inference บน test images
- [ ] Extract drone positions (cx, cy, conf, angle)
- [ ] บันทึก `yolo_predictions_test.csv`

#### ✅ Task 6.2: Extract Test Features
- [ ] Apply same feature engineering pipeline
- [ ] Extract image features, geometric features, metadata
- [ ] Create `features_test.csv` (264 rows × N features)
- [ ] Apply saved scaler (StandardScaler)

#### ✅ Task 6.3: Generate Predictions
- [ ] Load trained models: `model_lat`, `model_lon`, `model_alt`
- [ ] Predict coordinates for test set
- [ ] Post-processing:
  - Clip lat/lon to reasonable range
  - Clip altitude: 20-60m (based on train statistics)
  - Apply smoothing if needed

#### ✅ Task 6.4: Create Submission File
- [ ] Format: `ImageName, Latitude, Longitude, Altitude`
- [ ] Add spaces after commas (ตามโจทย์)
- [ ] Validate:
  - 264 rows (no missing)
  - Image names match test files
  - Values in valid range
- [ ] บันทึก `submission.csv`

#### ✅ Task 6.5: Quality Check
- [ ] Visual inspection: plot predictions on map
- [ ] Check for outliers or anomalies
- [ ] Sanity check: coordinates within Thailand
- [ ] Compare with train distribution

**Output**:
- `submission.csv` (ready to submit!)
- Test prediction visualizations

---

### 📝 Phase 7: Documentation & Demo (วันที่ 9)

#### ✅ Task 7.1: Code Organization
- [ ] Clean up code
- [ ] Refactor into modules/functions
- [ ] Add docstrings and comments
- [ ] Create requirements.txt
- [ ] Test end-to-end pipeline

#### ✅ Task 7.2: Documentation
- [ ] Write README.md:
  - Project overview
  - Installation instructions
  - How to run training
  - How to generate predictions
- [ ] Document methodology and approach
- [ ] Include performance metrics
- [ ] Add visualizations

#### ✅ Task 7.3: Prepare Demo
- [ ] Create demo Jupyter notebook
- [ ] Show key results and visualizations
- [ ] Explain model decisions
- [ ] Prepare presentation slides (if needed)

#### ✅ Task 7.4: Final Testing
- [ ] Run entire pipeline from scratch
- [ ] Verify reproducibility
- [ ] Test on different machine (if possible)
- [ ] Fix any bugs or issues

#### ✅ Task 7.5: Package for Submission
- [ ] Create .zip file with:
  - Source code
  - README.md
  - requirements.txt
  - Trained models (optional)
  - NOT including: images/videos/data
- [ ] Verify submission.csv is final version
- [ ] Double-check all requirements

**Output**:
- Clean, documented codebase
- README.md
- Demo materials
- submission.zip (code)
- submission.csv (predictions)

---

## 📚 Helper Functions & Code Snippets

### Calculate Bearing (Azimuth)
```python
from math import radians, degrees, sin, cos, atan2

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point1 to point2 (0-360°)"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = atan2(x, y)
    bearing = degrees(bearing)
    return (bearing + 360) % 360
```

### Calculate Distance (Haversine)
```python
from geopy.distance import geodesic

# Simple way
distance_m = geodesic((lat1, lon1), (lat2, lon2)).meters
```

---

## 🎯 Success Criteria & Timeline

### Expected Results
| Version | Labels | Angle Error | Alt Error | Score |
|---------|--------|-------------|-----------|-------|
| v1 (Baseline) | 50 | 12-18° | 4-6m | 6-7/9 ⭐ |
| v2 (Better) | 100 | 8-12° | 3-4m | 7-8/9 ⭐⭐ |
| v3 (Optimal) | 150+ | 5-8° | 2-3m | 8-9/9 ⭐⭐⭐ |

### Timeline (9 วัน)
- **Day 1-3**: Data exploration + YOLO training (92 images)
- **Day 4-6**: Feature engineering + XGBoost training
- **Day 7-8**: Test prediction + Optimization
- **Day 9**: Documentation + Demo preparation

### Target Score
- 🎯 **Minimum**: 6/9 คะแนน (ผ่าน)
- 🎯 **Good**: 7-8/9 คะแนน (ดี)
- 🎯 **Excellent**: 8-9/9 คะแนน (ยอดเยี่ยม)

---

## 💡 Tips & Best Practices

1. **Start Simple**: เริ่มจาก baseline model ก่อน แล้วค่อยปรับปรุง
2. **Validate Early**: ตรวจสอบ performance บน validation set บ่อยๆ
3. **Feature Importance**: ดู feature importance เพื่อเข้าใจ model
4. **Error Analysis**: วิเคราะห์ samples ที่ error สูง
5. **Post-processing**: Clip values ให้อยู่ในช่วงที่สมเหตุสมผล
6. **Visualization**: Plot predictions บนแผนที่เพื่อ sanity check
7. **Incremental Improvement**: Label เพิ่ม → Re-train → Better score
8. **Documentation**: เขียน README ให้ชัดเจนสำหรับ demo

---

## � Required Libraries

```bash
pip install ultralytics xgboost scikit-learn pandas numpy opencv-python geopy folium matplotlib seaborn tqdm
```

---

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Exploration**
   ```bash
   python 01_data_exploration.py
   ```

3. **Train YOLO + XGBoost**
   ```bash
   python 02_train_yolo.py
   python 03_train_xgboost.py
   ```

4. **Generate Predictions**
   ```bash
   python 04_predict.py
   ```

5. **Submit!**
   - `submission.csv` (predictions)
   - `submission.zip` (source code)

---

## 📞 Project Status

**Current Status**: 🟢 **Ready to Start!**

✅ ข้อมูล: 438 train images + 264 test images  
✅ Labels: 92 YOLO OBB annotations (Roboflow)  
✅ Ground Truth: 438 CSV files with coordinates  
✅ Scripts: plotbox.py, transform.py  

**Next Action**: เริ่ม Task 1.1 - Setup Environment

**Good Luck! 🚀🎯**

### 🔧 Environment Setup
- [ ] ติดตั้ง Python packages: `pip install xgboost scikit-learn pandas numpy matplotlib seaborn`
- [ ] ติดตั้ง geospatial libraries: `pip install geopy geographiclib folium`
- [ ] ติดตั้ง image processing: `pip install opencv-python pillow`
- [ ] ทดสอบ import ทุก libraries
- [ ] ตั้งค่า Jupyter Notebook (ถ้าใช้)

### 📊 Data Exploration & Analysis
- [ ] โหลด CSV ทั้งหมด 438 ไฟล์จาก `DATA_TRAIN/csv/`
- [ ] สร้าง DataFrame รวม: lat, lon, alt
- [ ] คำนวณ statistics: mean, std, min, max
- [ ] Plot distributions: histogram, box plot
- [ ] คำนวณ azimuth & distance จากกล้อง → โดรน
- [ ] Plot trajectory บนแผนที่ (Folium/Matplotlib)
- [ ] วิเคราะห์ pattern: circular? linear? random?
- [ ] ตรวจสอบ outliers และ anomalies
- [ ] สร้าง EDA report/notebook

### 🎨 Feature Engineering
- [ ] สร้าง function `extract_features(image_path, csv_path=None)`
- [ ] Implement basic features: img_number, sequence
- [ ] Implement image features: size, colors, brightness, contrast
- [ ] Implement edge features: Canny edge detection
- [ ] Implement GCP features: distances & angles to 5 points
- [ ] (Optional) Implement advanced features: HOG, SIFT, histograms
- [ ] (Optional) Implement temporal features: prev/next, velocity
- [ ] สร้าง feature matrix สำหรับ train set (438 × N features)
- [ ] ตรวจสอบ missing values และจัดการ
- [ ] วิเคราะห์ feature correlation matrix
- [ ] Save feature matrix เป็น CSV/pickle

### 🤖 Model Development
- [ ] แยก data: X (features) และ y (lat, lon, alt)
- [ ] Train/validation split (80/20 หรือ 350/88)
- [ ] Scale features ด้วย StandardScaler (save scaler!)
- [ ] **Model 1**: เทรน XGBoost สำหรับ Latitude
  - [ ] ใช้ default parameters ก่อน
  - [ ] ประเมินผล validation set
  - [ ] บันทึก metrics: MAE, RMSE
- [ ] **Model 2**: เทรน XGBoost สำหรับ Longitude
  - [ ] ใช้ parameters เหมือน Model 1
  - [ ] ประเมินผล validation set
- [ ] **Model 3**: เทรน XGBoost สำหรับ Altitude
  - [ ] อาจใช้ parameters ต่างจาก Model 1-2
  - [ ] ประเมินผล validation set
- [ ] คำนวณ angle error และ altitude error
- [ ] คำนวณ total_score ตาม formula
- [ ] Save models: `model_lat.pkl`, `model_lon.pkl`, `model_alt.pkl`

### 🎯 Model Optimization
- [ ] วิเคราะห์ feature importance (top 10-20)
- [ ] ลบ features ที่ไม่มีประโยชน์
- [ ] Hyperparameter tuning:
  - [ ] `max_depth`: ทดลอง [4, 6, 8, 10]
  - [ ] `learning_rate`: ทดลอง [0.01, 0.05, 0.1, 0.2]
  - [ ] `n_estimators`: ทดลอง [100, 200, 300, 500]
  - [ ] `min_child_weight`: ทดลอง [1, 3, 5]
  - [ ] `subsample`: ทดลอง [0.7, 0.8, 0.9, 1.0]
  - [ ] `colsample_bytree`: ทดลอง [0.7, 0.8, 0.9, 1.0]
- [ ] ใช้ GridSearchCV หรือ RandomizedSearchCV
- [ ] ใช้ early_stopping_rounds (20-50)
- [ ] Re-train models ด้วย best parameters
- [ ] ประเมินผลใหม่และเปรียบเทียบ

### 📈 Evaluation & Validation
- [ ] K-Fold Cross-Validation (5 folds)
- [ ] คำนวณ CV score: mean ± std
- [ ] Error analysis: หา samples ที่ error สูง
- [ ] Plot predictions vs ground truth (scatter plot)
- [ ] Plot error distribution (histogram)
- [ ] Plot residuals
- [ ] Map visualization: actual vs predicted positions
- [ ] Calculate final metrics:
  - [ ] Mean Angle Error (°)
  - [ ] Mean Altitude Error (m)
  - [ ] Total Score
- [ ] เป้าหมาย: Angle Error < 15°, Alt Error < 5m

### 🧪 Test Set Prediction
- [ ] Extract features จาก 264 test images
- [ ] Load trained models และ scaler
- [ ] Predict lat, lon, alt สำหรับทุกภาพ
- [ ] Post-processing:
  - [ ] Clip lat ให้อยู่ในช่วง [14.29, 14.31]
  - [ ] Clip lon ให้อยู่ในช่วง [101.15, 101.18]
  - [ ] Clip alt ให้อยู่ในช่วง [20, 60]
  - [ ] (Optional) Apply smoothing filter
- [ ] สร้าง DataFrame: ImageName, Lat, Lon, Alt
- [ ] บันทึกเป็น `submission.csv`
- [ ] Validate submission:
  - [ ] มี 264 rows (ตรง?)
  - [ ] ไม่มี missing values
  - [ ] Format ถูกต้อง
  - [ ] ค่าอยู่ในช่วงที่สมเหตุสมผล

### 🗺️ Visualization & Quality Check
- [ ] Plot predicted positions บนแผนที่ (Folium)
- [ ] สร้าง interactive map
- [ ] Visual inspection: ดูว่า predictions สมเหตุสมผล
- [ ] เปรียบเทียบกับ train set trajectory
- [ ] สร้าง summary statistics table
- [ ] Export visualizations เป็น PNG/HTML

### 📄 Documentation & Code
- [ ] Clean และ organize code
- [ ] สร้าง main script: `train.py`, `predict.py`
- [ ] สร้าง utilities: `features.py`, `utils.py`
- [ ] เขียน docstrings สำหรับ functions
- [ ] เขียน README.md:
  - [ ] Project overview
  - [ ] Installation instructions
  - [ ] Usage guide
  - [ ] Methodology explanation
  - [ ] Results และ metrics
- [ ] สร้าง requirements.txt
- [ ] ทดสอบการรัน end-to-end จากศูนย์
- [ ] สร้าง demo notebook (Jupyter)

### 🎤 Demo Preparation
- [ ] เตรียม slides/presentation (ถ้าต้องการ)
- [ ] เตรียมคำอธิบาย approach
- [ ] เตรียมอธิบาย features ที่สำคัญ
- [ ] เตรียมแสดง results และ visualizations
- [ ] ทดลองสาธิตการรัน code
- [ ] เตรียมตอบคำถามที่อาจถูกถาม:
  - [ ] ทำไมเลือก XGBoost?
  - [ ] Features ไหนสำคัญที่สุด?
  - [ ] จัดการ overfitting อย่างไร?
  - [ ] คะแนนที่ได้ดีแค่ไหน?
  - [ ] มีแนวทางปรับปรุงอย่างไร?

### ⏰ Timeline Check
- [ ] Day 1: EDA + Feature Engineering (4-6 hrs)
- [ ] Day 2: Model Training + Tuning (4-6 hrs)
- [ ] Day 3: Optimization + Evaluation (3-4 hrs)
- [ ] Day 4: Test Prediction + Submission (2-3 hrs)
- [ ] Day 5: Documentation + Demo Prep (2-3 hrs)
- [ ] Total: ~15-22 hours

---

## ⚠️ สิ่งที่ขาดและต้องดำเนินการ

### 🔴 Critical (ต้องทำ)
1. **Feature Engineering Script** - ยังไม่มี code
2. **XGBoost Training Pipeline** - ยังไม่มี code
3. **Hyperparameter Tuning** - ยังไม่ได้ทำ
4. **Evaluation Metrics** - ยังไม่ได้คำนวณ
5. **Test Prediction Script** - ยังไม่มี code

### 🟡 Important (ควรทำ)
6. **Feature Importance Analysis** - ช่วยเข้าใจ model
7. **Cross-Validation** - ตรวจสอบ generalization
8. **Error Analysis** - หาจุดอ่อนของ model
9. **Visualization Suite** - แสดงผลลัพธ์
10. **Documentation** - อธิบาย methodology

### 🟢 Optional (ดีถ้ามี)
11. **Ensemble Methods** - เพิ่มความแม่นยำ
12. **Advanced Features** - HOG, SIFT, etc.
13. **Hybrid Approach** - รวม Geometry + ML
14. **Interactive Dashboard** - Streamlit/Dash
15. **Automated Pipeline** - MLflow/DVC

---

## 📊 เมตริกการประเมิน

### Angle Error
```python
# แปลง lat, lon → azimuth angle (0-360°)
# จาก camera position

error_angle = |angle_pred - angle_true|
mean_angle_error = average(error_angle)
```

### Height Error
```python
error_height = |alt_pred - alt_true|
mean_height_error = average(error_height)
```

### Total Score
```python
total_error = 0.7 × mean_angle_error + 0.3 × mean_height_error

# ยิ่ง error น้อย → คะแนนสูง
# ต้องดู scoring rubric ว่าแปลง error → คะแนนอย่างไร
```

---

## 🚀 Quick Start Commands

### 1. Environment Setup
```bash
pip install geopy geographiclib pandas numpy matplotlib folium
```

### 2. Data Exploration
```python
import pandas as pd
import glob

# อ่าน CSV ทั้งหมด
csv_files = glob.glob('datasets/DATA_TRAIN/*.csv')
dfs = [pd.read_csv(f) for f in csv_files]
data = pd.concat(dfs)

print(data.describe())
print(f"Lat range: {data.Latitude.min()} - {data.Latitude.max()}")
print(f"Lon range: {data.Longitude.min()} - {data.Longitude.max()}")
print(f"Alt range: {data.Altitude.min()} - {data.Altitude.max()}")
```

### 3. Calculate Azimuth
```python
from geopy.distance import geodesic
import math

def calculate_azimuth(lat1, lon1, lat2, lon2):
    """คำนวณมุมทิศ (0-360°) จาก point1 → point2"""
    dlon = math.radians(lon2 - lon1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - \
        math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    
    azimuth = math.degrees(math.atan2(x, y))
    return (azimuth + 360) % 360

# Example
cam_lat, cam_lon = 14.305029, 101.173010
drone_lat, drone_lon = 14.3047389, 101.1728838

angle = calculate_azimuth(cam_lat, cam_lon, drone_lat, drone_lon)
print(f"Azimuth: {angle:.2f}°")
```

---

## 🎓 แนวคิดการแก้โจทย์

### Insight 1: ระยะทางจากกล้อง
```python
# โดรนมักบินในระยะประมาณ 100-150m จากกล้อง
# ถ้าทราบ azimuth และ distance → คำนวณ lat, lon ได้

from geopy.distance import distance
from geopy import Point

def predict_position(cam_lat, cam_lon, azimuth, distance_m):
    """ทำนายพิกัดโดรนจาก azimuth และ distance"""
    origin = Point(cam_lat, cam_lon)
    destination = distance(meters=distance_m).destination(
        origin, bearing=azimuth
    )
    return destination.latitude, destination.longitude
```

### Insight 2: Altitude Pattern
```python
# ความสูงมักอยู่ในช่วง 30-50m
# อาจใช้ค่าเฉลี่ย หรือ sample จาก distribution
```

### Insight 3: Sequential Pattern
```python
# ถ้าโดรนบินตามเส้นทาง (path)
# อาจ interpolate ระหว่างจุดที่รู้ค่า
```

---

## ⚡ ข้อเสนอแนะ

### ลำดับความสำคัญ

1. **เร่งด่วน**: หาข้อมูลรูปภาพ DATA_TRAIN
2. **สำคัญ**: ทำความเข้าใจ pattern ของการบินโดรน
3. **ควรมี**: สร้าง YOLO labels (ถ้าต้องการใช้ ML)
4. **ดีถ้ามี**: หา camera parameters

### Minimum Viable Solution

ถ้าข้อมูลไม่ครบ สามารถทำ **baseline solution**:

```python
# 1. คำนวณ mean azimuth & altitude จาก train
# 2. ใช้ค่า mean ทำนาย test set ทั้งหมด
# 3. หรือ random sample ภายในช่วงที่เป็นไปได้

# จะได้คะแนนบ้าง แต่อาจไม่สูง
# แต่อย่างน้อยมี submission ที่ run ได้
```

---

## 📚 Resources

### Libraries
- **geopy**: Geographic calculations
- **geographiclib**: Geodesic computations
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib/folium**: Visualization
- **opencv-python**: Image processing (ถ้าใช้)
- **ultralytics**: YOLO (ถ้าเทรน)

### References
- [Geodesic calculations](https://geographiclib.sourceforge.io/)
- [YOLO Documentation](https://docs.ultralytics.com/)
- [Camera Pose Estimation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

---

## ✅ Next Steps & Timeline

### ⚡ **SPRINT MODE: 9 ชั่วโมง** (Hybrid: YOLO + XGBoost + Fine-tune)

**สถานะปัจจุบัน**: มี labels แล้ว ~50 รูป ✅

**กลยุทธ์ 3-Phase**:
1. **Phase 1** (Hour 0-3): YOLO Detection (50 labels) → bbox features
2. **Phase 2** (Hour 3-7): XGBoost Regression (bbox + image features)
3. **Phase 3** (Hour 7-9): Submission + Fine-tune loop

---

### 🎯 **Hour 0-1: Setup + Quick YOLO Prep**
```bash
# [0:00-0:15] Install
pip install ultralytics xgboost scikit-learn pandas numpy opencv-python geopy folium tqdm

# [0:15-0:30] Check Labels
# มี 50 labels ใน datasets/DATA_TRAIN/label/
# Format: class x_center y_center width height (normalized)

# [0:30-1:00] Create data.yaml
```

```yaml
# datasets/data.yaml
path: C:/Users/siriz/OneDrive/Desktop/coding/tesa_problem_2/datasets
train: DATA_TRAIN/image
val: DATA_TRAIN/image  # use same for now (small dataset)

nc: 1  # number of classes
names: ['drone']
```

**Output**: ✅ Environment ready + data.yaml

---

### 🚀 **Hour 1-3: Train YOLO (Quick) + Extract Bbox**
```python
# [1:00-2:00] Train YOLOv8 (fast - 50 labels only)
from ultralytics import YOLO

# Load pretrained YOLOv8n (nano - fastest)
model = YOLO('yolov8n.pt')

# Train (เร็วมาก เพราะมี label แค่ 50 รูป)
results = model.train(
    data='datasets/data.yaml',
    epochs=50,              # เพิ่มได้ถ้ามีเวลา
    imgsz=640,
    batch=16,
    patience=10,            # early stopping
    save=True,
    device=0,               # GPU ถ้ามี, ไม่มีใช้ 'cpu'
    name='drone_detector',
    exist_ok=True
)

# Save best model
best_model = YOLO('runs/detect/drone_detector/weights/best.pt')
print("✅ YOLO trained!")

# [2:00-3:00] Extract Bbox Features from ALL images (438 train + 264 test)
import cv2
import pandas as pd
from tqdm import tqdm

def extract_bbox_features(img_path, model):
    """Detect drone และดึง bbox features"""
    results = model(img_path, verbose=False)
    
    features = {}
    
    if len(results[0].boxes) > 0:
        # ถ้าเจอโดรน ใช้ bbox แรก (confidence สูงสุด)
        box = results[0].boxes[0]
        
        # Bbox features (normalized)
        x, y, w, h = box.xywhn[0].tolist()  # normalized coordinates
        features['bbox_x'] = x
        features['bbox_y'] = y
        features['bbox_w'] = w
        features['bbox_h'] = h
        features['bbox_conf'] = float(box.conf[0])
        
        # Absolute position in image
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        features['bbox_x_abs'] = x * img_w
        features['bbox_y_abs'] = y * img_h
        features['bbox_area'] = w * h  # normalized area
        
    else:
        # ถ้าไม่เจอ ใช้ค่า default (center)
        features['bbox_x'] = 0.5
        features['bbox_y'] = 0.5
        features['bbox_w'] = 0.1
        features['bbox_h'] = 0.1
        features['bbox_conf'] = 0.0
        features['bbox_x_abs'] = 320  # assume 640x480
        features['bbox_y_abs'] = 240
        features['bbox_area'] = 0.01
    
    return features

# Extract for all train images
print("Extracting bbox features from train set...")
train_features = []
for i in tqdm(range(1, 439)):
    img_path = f'datasets/DATA_TRAIN/image/img_{i:04d}.jpg'
    features = extract_bbox_features(img_path, best_model)
    features['img_num'] = i
    train_features.append(features)

train_bbox = pd.DataFrame(train_features)
train_bbox.to_csv('train_bbox_features.csv', index=False)
print(f"✅ Train bbox features: {train_bbox.shape}")
```

**Output**: ✅ YOLO model + bbox features (438 samples)

---

### 🎨 **Hour 3-4: Combine Features (Bbox + Image + GCP)**
```python
# [3:00-3:30] Image Features (quick)
import numpy as np
from geopy.distance import geodesic

def extract_combined_features(img_path, img_num, bbox_features):
    """Combine bbox + image + geospatial features"""
    features = bbox_features.copy()
    
    # Image basic
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    features['img_w'] = w
    features['img_h'] = h
    
    # Sequential
    features['img_seq'] = img_num / 438
    
    # Color (fast)
    features['mean_blue'] = np.mean(img[:,:,0])
    features['mean_green'] = np.mean(img[:,:,1])
    features['mean_red'] = np.mean(img[:,:,2])
    
    # Brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features['brightness'] = np.mean(gray)
    
    # GCP distances (3 points only)
    cam_lat, cam_lon = 14.305029, 101.173010
    gcp_points = {
        'center': (14.304742, 101.172577),
        'right': (14.304981, 101.172100),
        'left': (14.304220, 101.172605),
    }
    
    for name, (lat, lon) in gcp_points.items():
        dist = geodesic((cam_lat, cam_lon), (lat, lon)).meters
        features[f'dist_{name}'] = dist
    
    return features

# [3:30-4:00] Create full feature matrix
train_bbox = pd.read_csv('train_bbox_features.csv')

train_full_features = []
for i in tqdm(range(1, 439)):
    img_path = f'datasets/DATA_TRAIN/image/img_{i:04d}.jpg'
    csv_path = f'datasets/DATA_TRAIN/csv/img_{i:04d}.csv'
    
    # Bbox features
    bbox_feat = train_bbox[train_bbox.img_num == i].iloc[0].to_dict()
    
    # Combined features
    features = extract_combined_features(img_path, i, bbox_feat)
    
    # Labels
    df = pd.read_csv(csv_path)
    features['lat'] = df.Latitude.values[0]
    features['lon'] = df.Longitude.values[0]
    features['alt'] = df.Altitude.values[0]
    
    train_full_features.append(features)

X_train_full = pd.DataFrame(train_full_features)
X_train_full.to_csv('train_full_features.csv', index=False)
print(f"✅ Full features: {X_train_full.shape}")  # 438 x ~20 features
```

**Output**: ✅ Combined features (bbox + image + GCP)

---

### 🤖 **Hour 4-6: Train XGBoost Models**
```python
# [4:00-4:30] Prepare data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

df = pd.read_csv('train_full_features.csv')

# Features
feature_cols = [c for c in df.columns if c not in ['lat', 'lon', 'alt', 'img_num']]
X = df[feature_cols]
y_lat = df['lat']
y_lon = df['lon']
y_alt = df['alt']

# Split
X_train, X_val, y_lat_train, y_lat_val = train_test_split(X, y_lat, test_size=0.2, random_state=42)
_, _, y_lon_train, y_lon_val = train_test_split(X, y_lon, test_size=0.2, random_state=42)
_, _, y_alt_train, y_alt_val = train_test_split(X, y_alt, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
joblib.dump(scaler, 'scaler.pkl')

# [4:30-6:00] Train 3 models
print("Training models with bbox features...")

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Latitude
model_lat = xgb.XGBRegressor(**params)
model_lat.fit(X_train_s, y_lat_train, 
              eval_set=[(X_val_s, y_lat_val)],
              early_stopping_rounds=20, verbose=False)
joblib.dump(model_lat, 'model_lat.pkl')

# Longitude
model_lon = xgb.XGBRegressor(**params)
model_lon.fit(X_train_s, y_lon_train,
              eval_set=[(X_val_s, y_lon_val)],
              early_stopping_rounds=20, verbose=False)
joblib.dump(model_lon, 'model_lon.pkl')

# Altitude
model_alt = xgb.XGBRegressor(**params)
model_alt.fit(X_train_s, y_alt_train,
              eval_set=[(X_val_s, y_alt_val)],
              early_stopping_rounds=20, verbose=False)
joblib.dump(model_alt, 'model_alt.pkl')

print("✅ Models trained and saved!")

# Quick eval
from sklearn.metrics import mean_absolute_error

y_lat_pred = model_lat.predict(X_val_s)
y_lon_pred = model_lon.predict(X_val_s)
y_alt_pred = model_alt.predict(X_val_s)

print(f"\n📊 Validation MAE:")
print(f"Latitude:  {mean_absolute_error(y_lat_val, y_lat_pred):.6f}")
print(f"Longitude: {mean_absolute_error(y_lon_val, y_lon_pred):.6f}")
print(f"Altitude:  {mean_absolute_error(y_alt_val, y_alt_pred):.2f}m")

# Angle error
def calc_azimuth(lat1, lon1, lat2, lon2):
    import math
    dlon = math.radians(lon2 - lon1)
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

cam_lat, cam_lon = 14.305029, 101.173010
angles_true = [calc_azimuth(cam_lat, cam_lon, lat, lon) for lat, lon in zip(y_lat_val, y_lon_val)]
angles_pred = [calc_azimuth(cam_lat, cam_lon, lat, lon) for lat, lon in zip(y_lat_pred, y_lon_pred)]
angle_errors = [abs(at - ap) for at, ap in zip(angles_true, angles_pred)]
alt_error = mean_absolute_error(y_alt_val, y_alt_pred)

print(f"\nAngle Error: {np.mean(angle_errors):.2f}°")
print(f"Alt Error: {alt_error:.2f}m")
print(f"Total Score: {0.7*np.mean(angle_errors) + 0.3*alt_error:.2f}")
```

**Output**: ✅ 3 XGBoost models (with bbox features!)

---

### 📊 **Hour 6-7: Test Prediction**
```python
# [6:00-6:30] Extract test bbox features
print("Detecting drones in test set...")
test_bbox_features = []
for i in tqdm(range(1, 265)):
    img_path = f'datasets/DATA_TEST/img_{i:04d}.jpg'  # ⚠️ ไม่มี /images/ ซ้อน!
    features = extract_bbox_features(img_path, best_model)
    features['img_num'] = i
    test_bbox_features.append(features)

test_bbox = pd.DataFrame(test_bbox_features)

# [6:30-7:00] Extract full test features
test_full_features = []
for i in tqdm(range(1, 265)):
    img_path = f'datasets/DATA_TEST/img_{i:04d}.jpg'  # ⚠️ Path ถูกต้อง
    bbox_feat = test_bbox[test_bbox.img_num == i].iloc[0].to_dict()
    features = extract_combined_features(img_path, i, bbox_feat)
    features['ImageName'] = f'img_{i:04d}.jpg'
    test_full_features.append(features)

X_test = pd.DataFrame(test_full_features)

# Predict
feature_cols = [c for c in X_test.columns if c not in ['ImageName', 'img_num']]
X_test_features = X_test[feature_cols]
X_test_s = scaler.transform(X_test_features)

lat_pred = model_lat.predict(X_test_s)
lon_pred = model_lon.predict(X_test_s)
alt_pred = model_alt.predict(X_test_s)

# Clip
lat_pred = np.clip(lat_pred, 14.29, 14.31)
lon_pred = np.clip(lon_pred, 101.15, 101.18)
alt_pred = np.clip(alt_pred, 20, 60)

# Create submission
submission = pd.DataFrame({
    'ImageName': X_test['ImageName'],
    'Latitude': lat_pred,
    'Longitude': lon_pred,
    'Altitude': alt_pred
})

submission.to_csv('submission_v1.csv', index=False)
print(f"✅ Submission v1 created: {len(submission)} rows")
```

**Output**: ✅ submission_v1.csv

---

### 📈 **Hour 7-8: Visualization + Quick Analysis**
```python
# [7:00-7:30] Feature importance
import matplotlib.pyplot as plt

# Lat model
importance = model_lat.feature_importances_
indices = np.argsort(importance)[::-1][:10]
feature_names = feature_cols

plt.figure(figsize=(10, 6))
plt.bar(range(10), importance[indices])
plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.title('Top 10 Feature Importance (Latitude Model)')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("✅ Feature importance saved")

# [7:30-8:00] Map visualization
import folium

m = folium.Map(location=[14.305, 101.173], zoom_start=14)

# Predictions (first 30)
for _, row in submission.head(30).iterrows():
    folium.CircleMarker(
        [row.Latitude, row.Longitude],
        radius=3, color='red', fill=True,
        popup=f"{row.ImageName}<br>Alt: {row.Altitude:.1f}m"
    ).add_to(m)

# Camera
folium.Marker([cam_lat, cam_lon], popup='Camera', 
              icon=folium.Icon(color='green')).add_to(m)

m.save('prediction_map_v1.html')
print("✅ Map saved")
```

**Output**: ✅ Visualizations

---

### 🔧 **Hour 8-9: Fine-tuning Loop Setup**

```python
# [8:00-8:30] Label more images (manual - use Roboflow/LabelImg)
# เพิ่ม labels จาก 50 → 100 → 150 รูป (ค่อยๆ ทำ)

# [8:30-9:00] Create fine-tune script
fine_tune_script = """
# fine_tune.py - Run เรื่อยๆ เมื่อมี labels เพิ่ม

from ultralytics import YOLO

# Load previous best model
model = YOLO('runs/detect/drone_detector/weights/best.pt')

# Fine-tune with more labels
model.train(
    data='datasets/data.yaml',
    epochs=30,
    imgsz=640,
    batch=16,
    resume=True,  # Continue from previous
    name='drone_detector_v2',
    exist_ok=True
)

print("✅ Fine-tuning done!")
print("Re-run feature extraction and prediction...")
"""

with open('fine_tune.py', 'w') as f:
    f.write(fine_tune_script)

# README
readme = '''
# Drone Localization - Hybrid Approach

## Architecture
1. **YOLO Detection** (YOLOv8n) - trained on ~50 labels
2. **Bbox Features** (8 features) - position, size, confidence
3. **XGBoost Regression** (3 models) - lat, lon, alt prediction
4. **Fine-tuning Loop** - label more → re-train → improve

## Current Status
- Labels: 50/438 (11%)
- YOLO trained: ✅
- XGBoost trained: ✅
- Submission v1: ✅

## Results (v1 - 50 labels)
- Validation Angle Error: [YOUR_VALUE]°
- Validation Alt Error: [YOUR_VALUE]m
- Expected Score: 6-7/9

## Improvement Plan
1. Label 50 more images → 100 total (23%)
2. Re-train YOLO → better bbox
3. Re-train XGBoost → better predictions
4. Repeat until satisfied

## Files
- `train_yolo.py`: YOLO training
- `train_xgboost.py`: XGBoost training
- `predict.py`: Test prediction
- `fine_tune.py`: Incremental improvement
- `submission_v1.csv`: First submission

## To Run
```bash
# Initial training
python train_yolo.py
python train_xgboost.py
python predict.py

# Fine-tuning (later)
# 1. Label more images in DATA_TRAIN/label/
# 2. Run: python fine_tune.py
# 3. Re-extract features and predict
```
'''

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme)

print("✅ Fine-tune script + README created!")
print("\n🎉 Sprint completed! submission_v1.csv ready!")
print("\n💡 Next: Label more images → fine-tune → improve score")
```

**Output**: ✅ Fine-tune framework + Documentation

---

## 🎯 **Summary: Hybrid 9-Hour Sprint**

| Time | Task | Output | Score Impact |
|------|------|--------|--------------|
| 0-1h | Setup + YOLO prep | data.yaml | - |
| 1-3h | Train YOLO (50 labels) + Extract bbox | bbox features | +10-15% |
| 3-4h | Combine features | 438 × ~20 features | - |
| 4-6h | Train XGBoost | 3 models | Base accuracy |
| 6-7h | Test prediction | submission_v1.csv ✅ | - |
| 7-8h | Visualization | Maps + plots | - |
| 8-9h | Fine-tune setup | Improvement loop | Future +20% |

### ✅ **Expected Results**

**Version 1 (50 labels)**:
- Angle Error: **12-18°** (ดีขึ้นเพราะมี bbox!)
- Altitude Error: **4-6m** (ดี)
- **Score: 6-7/9 คะแนน** ✨

**Version 2 (100 labels - ภายหลัง)**:
- Angle Error: **8-12°** (ดีมาก)
- Altitude Error: **3-4m** (ดีมาก)
- **Score: 7-8/9 คะแนน** 🚀

**Version 3 (150+ labels - optimal)**:
- Angle Error: **5-8°** (excellent!)
- Altitude Error: **2-3m** (excellent!)
- **Score: 8-9/9 คะแนน** 🏆

### 🔄 **Fine-tuning Loop** (ทำต่อเนื่อง):

```
1. Label +50 images (1-2 hours)
2. Fine-tune YOLO (20-30 min)
3. Re-extract features (15 min)
4. Re-train XGBoost (20 min)
5. Re-predict test set (10 min)
6. Compare scores → Repeat!
```

---

**กลยุทธ์สุดท้าย**:
- ✅ **Hour 0-7**: ได้ submission v1 แล้ว (ส่งได้!)
- ⚡ **Hour 7-9**: Setup fine-tune loop
- 🔄 **ภายหลัง**: Label เพิ่ม → fine-tune เรื่อยๆ → improve score!

**Advantage ของ Hybrid**:
- 🎯 ใช้ bbox จาก YOLO → แม่นยำขึ้น
- 🚀 Fine-tune ได้ง่าย → score ดีขึ้นเรื่อยๆ  
- 💪 Flexible → เพิ่ม labels ตามกำลัง

---

### 🎯 **Hour 0-2: Setup + EDA (Quick & Dirty)**
```python
# [0:00-0:30] Environment Setup
pip install xgboost scikit-learn pandas numpy matplotlib seaborn opencv-python geopy folium

# [0:30-1:00] Load Data
import pandas as pd
import glob

csv_files = glob.glob('datasets/DATA_TRAIN/csv/*.csv')
data = []
for f in csv_files:
    df = pd.read_csv(f)
    df['filename'] = f
    data.append(df)
train_df = pd.concat(data, ignore_index=True)

# [1:00-1:30] Quick EDA
print(train_df.describe())
print(f"Lat: {train_df.Latitude.min():.6f} - {train_df.Latitude.max():.6f}")
print(f"Lon: {train_df.Longitude.min():.6f} - {train_df.Longitude.max():.6f}")
print(f"Alt: {train_df.Altitude.min():.2f} - {train_df.Altitude.max():.2f}")

# [1:30-2:00] Calculate Basics
from geopy.distance import geodesic
import math

def calc_azimuth(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

cam_lat, cam_lon = 14.305029, 101.173010
train_df['azimuth'] = train_df.apply(lambda r: calc_azimuth(cam_lat, cam_lon, r.Latitude, r.Longitude), axis=1)
train_df['distance'] = train_df.apply(lambda r: geodesic((cam_lat, cam_lon), (r.Latitude, r.Longitude)).meters, axis=1)

print(f"Azimuth: {train_df.azimuth.mean():.2f}° ± {train_df.azimuth.std():.2f}°")
print(f"Distance: {train_df.distance.mean():.2f}m ± {train_df.distance.std():.2f}m")
```

**Output**: ✅ เข้าใจข้อมูลพื้นฐาน

---

### 🎨 **Hour 2-4: Feature Engineering (Essential Only)**
```python
# [2:00-3:00] Basic Features
import cv2
import numpy as np

def extract_quick_features(img_path, img_num):
    """Extract only essential features - ไม่ต้องซับซ้อน"""
    features = {}
    
    # 1. Sequential features
    features['img_number'] = img_num
    features['img_seq_norm'] = img_num / 438
    
    # 2. Image basic properties
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    features['width'] = w
    features['height'] = h
    features['aspect_ratio'] = w / h
    
    # 3. Color features (fast)
    features['mean_blue'] = np.mean(img[:,:,0])
    features['mean_green'] = np.mean(img[:,:,1])
    features['mean_red'] = np.mean(img[:,:,2])
    
    # 4. Brightness (fast)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features['brightness'] = np.mean(gray)
    features['contrast'] = np.std(gray)
    
    # 5. GCP distances (pre-calculated)
    gcp_points = {
        'center': (14.304742, 101.172577),
        'center_field': (14.304605, 101.172350),
        'right_field': (14.304981, 101.172100),
    }
    
    for name, (lat, lon) in gcp_points.items():
        dist = geodesic((cam_lat, cam_lon), (lat, lon)).meters
        features[f'dist_{name}'] = dist
    
    return features

# [3:00-4:00] Extract for all train images
from tqdm import tqdm

train_features = []
for i in tqdm(range(1, 439)):
    img_path = f'datasets/DATA_TRAIN/image/img_{i:04d}.jpg'
    csv_path = f'datasets/DATA_TRAIN/csv/img_{i:04d}.csv'
    
    # Get features
    features = extract_quick_features(img_path, i)
    
    # Get labels
    df = pd.read_csv(csv_path)
    features['lat'] = df.Latitude.values[0]
    features['lon'] = df.Longitude.values[0]
    features['alt'] = df.Altitude.values[0]
    
    train_features.append(features)

# Save
X_train = pd.DataFrame(train_features)
X_train.to_csv('train_features.csv', index=False)
print(f"✅ Features: {X_train.shape}")
```

**Output**: ✅ Feature matrix (438 × ~15 features)

---

### 🤖 **Hour 4-6: Train Baseline Models (No Tuning!)**
```python
# [4:00-4:30] Prepare Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load
df = pd.read_csv('train_features.csv')

# Split features and targets
feature_cols = [c for c in df.columns if c not in ['lat', 'lon', 'alt']]
X = df[feature_cols]
y_lat = df['lat']
y_lon = df['lon']
y_alt = df['alt']

# Train/Val split
X_train, X_val, y_lat_train, y_lat_val = train_test_split(X, y_lat, test_size=0.2, random_state=42)
_, _, y_lon_train, y_lon_val = train_test_split(X, y_lon, test_size=0.2, random_state=42)
_, _, y_alt_train, y_alt_val = train_test_split(X, y_alt, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save scaler
import joblib
joblib.dump(scaler, 'scaler.pkl')

# [4:30-5:30] Train 3 Models (Default Params)
print("Training Latitude model...")
model_lat = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model_lat.fit(X_train_scaled, y_lat_train, 
              eval_set=[(X_val_scaled, y_lat_val)],
              early_stopping_rounds=20,
              verbose=False)

print("Training Longitude model...")
model_lon = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model_lon.fit(X_train_scaled, y_lon_train,
              eval_set=[(X_val_scaled, y_lon_val)],
              early_stopping_rounds=20,
              verbose=False)

print("Training Altitude model...")
model_alt = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model_alt.fit(X_train_scaled, y_alt_train,
              eval_set=[(X_val_scaled, y_alt_val)],
              early_stopping_rounds=20,
              verbose=False)

# Save models
joblib.dump(model_lat, 'model_lat.pkl')
joblib.dump(model_lon, 'model_lon.pkl')
joblib.dump(model_alt, 'model_alt.pkl')
print("✅ Models saved!")

# [5:30-6:00] Quick Evaluation
from sklearn.metrics import mean_absolute_error

y_lat_pred = model_lat.predict(X_val_scaled)
y_lon_pred = model_lon.predict(X_val_scaled)
y_alt_pred = model_alt.predict(X_val_scaled)

print(f"\n📊 Validation MAE:")
print(f"Latitude:  {mean_absolute_error(y_lat_val, y_lat_pred):.6f}")
print(f"Longitude: {mean_absolute_error(y_lon_val, y_lon_pred):.6f}")
print(f"Altitude:  {mean_absolute_error(y_alt_val, y_alt_pred):.2f}m")

# Calculate angle error
angles_true = [calc_azimuth(cam_lat, cam_lon, lat, lon) for lat, lon in zip(y_lat_val, y_lon_val)]
angles_pred = [calc_azimuth(cam_lat, cam_lon, lat, lon) for lat, lon in zip(y_lat_pred, y_lon_pred)]
angle_errors = [abs(at - ap) for at, ap in zip(angles_true, angles_pred)]
print(f"Mean Angle Error: {np.mean(angle_errors):.2f}°")
print(f"Mean Alt Error: {mean_absolute_error(y_alt_val, y_alt_pred):.2f}m")
print(f"Total Score: {0.7*np.mean(angle_errors) + 0.3*mean_absolute_error(y_alt_val, y_alt_pred):.2f}")
```

**Output**: ✅ 3 trained models + baseline metrics

---

### 📊 **Hour 6-8: Test Prediction + Submission**
```python
# [6:00-7:00] Extract Test Features
test_features = []
for i in tqdm(range(1, 265)):
    img_path = f'datasets/DATA_TEST/img_{i:04d}.jpg'  # ⚠️ Path ถูกต้อง!
    features = extract_quick_features(img_path, i)
    features['ImageName'] = f'img_{i:04d}.jpg'
    test_features.append(features)

X_test = pd.DataFrame(test_features)
print(f"✅ Test features: {X_test.shape}")

# [7:00-7:30] Predict
# Load models
model_lat = joblib.load('model_lat.pkl')
model_lon = joblib.load('model_lon.pkl')
model_alt = joblib.load('model_alt.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare test data
feature_cols = [c for c in X_test.columns if c != 'ImageName']
X_test_features = X_test[feature_cols]
X_test_scaled = scaler.transform(X_test_features)

# Predict
lat_pred = model_lat.predict(X_test_scaled)
lon_pred = model_lon.predict(X_test_scaled)
alt_pred = model_alt.predict(X_test_scaled)

# [7:30-7:45] Post-processing (Clip values)
lat_pred = np.clip(lat_pred, 14.29, 14.31)
lon_pred = np.clip(lon_pred, 101.15, 101.18)
alt_pred = np.clip(alt_pred, 20, 60)

# [7:45-8:00] Create Submission
submission = pd.DataFrame({
    'ImageName': X_test['ImageName'],
    'Latitude': lat_pred,
    'Longitude': lon_pred,
    'Altitude': alt_pred
})

# บันทึกแบบ pandas default (ไม่มี space หลัง comma)
submission.to_csv('submission.csv', index=False)
print(f"✅ Submission created: {len(submission)} rows")
print(submission.head(10))

# ⚠️ ถ้าโจทย์เข้มงวดเรื่อง format (มี space หลัง comma) ใช้:
# submission.to_csv('submission.csv', index=False, sep=',', quoting=csv.QUOTE_NONE)

# Quick validation
print(f"\nSubmission Stats:")
print(f"Lat: {submission.Latitude.min():.6f} - {submission.Latitude.max():.6f}")
print(f"Lon: {submission.Longitude.min():.6f} - {submission.Longitude.max():.6f}")
print(f"Alt: {submission.Altitude.min():.2f} - {submission.Altitude.max():.2f}")
```

**Output**: ✅ `submission.csv` พร้อมส่ง!

---

### 📈 **Hour 8-9: Quick Visualization + Documentation**
```python
# [8:00-8:30] Quick Map Plot
import folium

# Plot on map
m = folium.Map(location=[14.305, 101.173], zoom_start=14)

# Add predictions
for _, row in submission.head(50).iterrows():  # plot first 50
    folium.CircleMarker(
        [row.Latitude, row.Longitude],
        radius=3,
        color='red',
        fill=True,
        popup=f"{row.ImageName}<br>Alt: {row.Altitude:.1f}m"
    ).add_to(m)

# Add camera
folium.Marker([cam_lat, cam_lon], 
              popup='Camera',
              icon=folium.Icon(color='green')).add_to(m)

m.save('prediction_map.html')
print("✅ Map saved: prediction_map.html")

# [8:30-9:00] Quick README
readme = """
# Drone Localization - XGBoost Regression

## Approach
- Features: Image properties + GCP distances (~15 features)
- Models: 3 XGBoost regressors (lat, lon, alt)
- No hyperparameter tuning (baseline)

## Results
- Validation Angle Error: [YOUR_VALUE]°
- Validation Alt Error: [YOUR_VALUE]m
- Total Score: [YOUR_VALUE]

## Files
- `train.py`: Training script
- `predict.py`: Prediction script
- `submission.csv`: Final predictions
- `prediction_map.html`: Visualization

## To Run
```bash
pip install xgboost scikit-learn pandas numpy opencv-python geopy folium
python train.py
python predict.py
```
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme)

print("✅ README created!")
print("\n🎉 DONE! Submission ready in 8-9 hours!")
```

**Output**: ✅ Visualization + Documentation

---

### 🔥 **Hour 9+: Optional Tuning (ถ้ามีเวลาเหลือ)**

```python
# เปิดทิ้งไว้รัน background
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0]
}

grid = GridSearchCV(
    xgb.XGBRegressor(random_state=42),
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)

print("🔄 Tuning running in background... (go do something else)")
grid.fit(X_train_scaled, y_lat_train)
print(f"✅ Best params: {grid.best_params_}")

# Re-train ด้วย best params ตอนมีเวลา
```

---

## 🎯 **Summary: 9-Hour Sprint Plan**

| Time | Task | Output |
|------|------|--------|
| 0-2h | Setup + Quick EDA | ความเข้าใจข้อมูล |
| 2-4h | Feature Engineering | 438 × ~15 features |
| 4-6h | Train Baseline Models | 3 models + metrics |
| 6-8h | Test Prediction | submission.csv ✅ |
| 8-9h | Visualization + Docs | Maps + README |
| 9h+ | Optional Tuning | Better models (ถ้ามีเวลา) |

### ✅ **Expected Results (Baseline)**
- Angle Error: **15-25°** (ยอมรับได้)
- Altitude Error: **5-8m** (พอใช้)
- Total Score: **13-20** → คะแนน **5-7/9**

### 🚀 **After Tuning (ถ้าทำ)**
- Angle Error: **10-15°** (ดี)
- Altitude Error: **3-5m** (ดีมาก)
- Total Score: **8-12** → คะแนน **7-8/9**

---

**กลยุทธ์**: 
1. ✅ **Hour 0-8**: ทำให้ได้ submission (MVP)
2. ⚡ **Hour 8+**: รัน tuning ทิ้งไว้ background
3. 🎯 **ภายหลัง**: กลับมาใช้ tuned models

**แนะนำ**: เน้นทำ Hour 0-8 ให้เสร็จก่อน แล้วค่อย optimize ทีหลัง!
