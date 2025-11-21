import json
import cv2
import os
import numpy as np

# --- CONFIG ---
# Path to Roboflow COCO JSON annotations
coco_json_path = "./valid/_annotations.coco.json"

# Path to images folder (ต้องใช้เพื่อเอาขนาดภาพ)
images_folder = "./valid"

# Output folder สำหรับ YOLO OBB labels
output_labels_folder = "labelsvalid"

# สร้าง folder ถ้าไม่มี
os.makedirs(output_labels_folder, exist_ok=True)

# โหลด COCO JSON
with open(coco_json_path, 'r') as f:
    coco = json.load(f)

# Map image id -> file name & size
image_info = {}
for img in coco['images']:
    image_info[img['id']] = {
        'file_name': img['file_name'],
        'width': img['width'],
        'height': img['height']
    }

# แปลง polygon → YOLO OBB
for ann in coco['annotations']:
    img_id = ann['image_id']
    info = image_info[img_id]
    img_name = info['file_name']
    img_w = info['width']
    img_h = info['height']

    # Polygon points
    poly = np.array(ann['segmentation'][0]).reshape(-1, 2).astype(np.float32)

    # ใช้ OpenCV หา rotated rectangle
    rect = cv2.minAreaRect(poly)  # ((center_x, center_y), (w, h), angle)
    (cx, cy), (w, h), angle = rect

    # Normalize values (0-1)
    cx /= img_w
    cy /= img_h
    w /= img_w
    h /= img_h
    angle_rad = np.deg2rad(angle)  # แปลงเป็น radian

    # สร้าง string YOLO OBB format: class cx cy w h angle(rad)
    line = f"{ann['category_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {angle_rad:.6f}"

    # เขียนไฟล์ .txt ตามชื่อภาพ
    label_file = os.path.join(output_labels_folder, os.path.splitext(img_name)[0] + ".txt")
    with open(label_file, "a") as f:
        f.write(line + "\n")

print("✅ แปลง polygon → YOLO OBB เสร็จเรียบร้อย!")
