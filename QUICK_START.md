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
│
└── 📤 outputs/problem_3/
    ├── final/                         # ✅ Final output
    │   └── P3_OUTPUT_FINAL.mp4       
    ├── analysis/                      # 📊 Analysis results
    └── experiments/                   # 🧪 Old experiments
```

---

## ⚙️ Configuration

**ไฟล์:** `src/problem_3_pipeline.py`

```python
# แก้ไขตรงนี้ถ้าต้องการปรับแต่ง:

# Model
MODEL_PATH = 'runs/detect/drone_detect_v21_max_data/weights/best.pt'
CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.3

# Features
USE_TRACK_MERGING = True   # Merge track IDs
USE_ENHANCEMENT = False    # Image enhancement (ไม่แนะนำ)

# Processing
START_FRAME = 0
END_FRAME = None  # None = ทั้งหมด
```

---

## 📊 ผลลัพธ์ที่คาดหวัง

```
Detection Rate:    99.1%
Track IDs:         2 [1, 2]
Processing Speed:  12-15 FPS
File Size:         ~70 MB
Quality:          ✅ Production Ready
```

---

## 🎨 Visualization Features

- ✅ Bounding boxes (สีแดง=Drone 1, สีเขียว=Drone 2)
- ✅ Track IDs
- ✅ GPS coordinates
- ✅ Tracking paths (เส้นทางการเคลื่อนที่)
- ✅ Info panel (มุมบนซ้าย)
- ✅ Frame info (มุมล่าง)

---

## 🔧 Troubleshooting

### **ปัญหา: รันช้า**
```bash
# ใช้ GPU (ถ้ามี)
# แก้ใน detector.py เปลี่ยน device='cpu' -> device='cuda'
```

### **ปัญหา: File not found**
```bash
# ตรวจสอบว่าอยู่ใน root directory
cd tesa_problem_2/
python src/problem_3_pipeline.py
```

### **ปัญหา: Memory error**
```bash
# ลด frame range
# แก้ใน main(): END_FRAME = 500  # ลอง 500 frames ก่อน
```

---

## 📚 Documentation

| File | Description |
|------|-------------|
| `SUMMARY.md` | ภาพรวมโปรเจค |
| `PROJECT_STRUCTURE.md` | โครงสร้างโปรเจค |
| `outputs/problem_3/final/README.md` | Output documentation |
| `README.md` | Project README |

---

## ✅ Checklist

- [ ] ติดตั้ง dependencies (`pip install -r requirements.txt`)
- [ ] มี input video (`P3_VIDEO.mp4`)
- [ ] มี model weights (`runs/detect/...`)
- [ ] รัน `python src/problem_3_pipeline.py`
- [ ] ตรวจสอบ output (`outputs/problem_3/final/`)

---

## 💡 Tips

1. **ครั้งแรก:** รัน pipeline ครั้งเดียวแล้วดู output
2. **ต้องการปรับแต่ง:** แก้ config ใน `main()` function
3. **วิเคราะห์:** ใช้ scripts ใน `scripts/05_evaluation/`
4. **Experiments:** ผลลัพธ์ทดลองอยู่ใน `outputs/problem_3/experiments/`

---

## 🚀 Next Steps

1. ✅ รัน pipeline → ได้ output video
2. ✅ ตรวจสอบคุณภาพ → ดู detection rate, track IDs
3. ✅ Deliver → ส่ง `P3_OUTPUT_FINAL.mp4`

---

**ใช้เวลาทั้งหมด:** ~3 นาที  
**Ready to go!** 🎉
