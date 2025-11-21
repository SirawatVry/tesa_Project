"""
plotbox.py

Read YOLO-style labels (including oriented bounding boxes) from ./labels and draw them on
corresponding images found in ./train. Supports label lines with either:

  class cx cy w h angle
  class cx cy w h

where cx,cy,w,h are normalized (0..1) and angle is either radians or degrees.

Usage examples (from project root):
  python plotbox.py --labels labels --images train --out out_images --show

The script will try to find the matching image for each label file by searching for
an image filename in the images directory that contains the label file's base id
(for labels like test_0006_jpg.rf.<hash>.txt it looks for files containing "test_0006").
"""

from __future__ import annotations

import argparse
import math
import os
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np


def parse_yolo_obb_line(line: str) -> Optional[Tuple[int, float, float, float, float, Optional[float]]]:
	parts = line.strip().split()
	if not parts:
		return None
	if parts[0].startswith('#'):
		return None
	# Expect either 5 or 6 numeric values after class
	if len(parts) < 5:
		return None
	try:
		cls = int(float(parts[0]))
		cx = float(parts[1])
		cy = float(parts[2])
		w = float(parts[3])
		h = float(parts[4])
		angle = float(parts[5]) if len(parts) >= 6 else None
		return cls, cx, cy, w, h, angle
	except Exception:
		return None


def angle_to_radians(a: float) -> float:
	# Heuristics: if abs(a) <= 2*pi assume radians. If abs(a) <= 360 treat degrees.
	if abs(a) <= 2 * math.pi:
		return a
	if abs(a) <= 360:
		return math.radians(a)
	# fallback: treat as radians
	return a


def obb_to_polygon(cx: float, cy: float, w: float, h: float, angle_rad: float) -> np.ndarray:
	# cx,cy center in pixels; w,h in pixels; angle in radians
	dx = w / 2.0
	dy = h / 2.0
	corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
	# rotation matrix
	c = math.cos(angle_rad)
	s = math.sin(angle_rad)
	R = np.array([[c, -s], [s, c]], dtype=np.float32)
	rotated = corners.dot(R.T)
	translated = rotated + np.array([cx, cy], dtype=np.float32)
	return translated.reshape((-1, 2)).astype(np.int32)


def find_image_for_label(label_name: str, images_dir: str) -> Optional[str]:
	# label_name: filename only (not path)
	base = label_name
	# try to extract pattern like test_0006 from 'test_0006_jpg.rf.1937...txt'
	m = re.match(r"(.+?)_jpg", label_name)
	if m:
		base = m.group(1)
	else:
		base = os.path.splitext(label_name)[0]

	# search images dir for files that contain the base id
	for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
		# direct match
		candidate = os.path.join(images_dir, base + ext)
		if os.path.exists(candidate):
			return candidate

	# fallback: any file that contains base
	for root, _, files in os.walk(images_dir):
		for f in files:
			if base in f.lower():
				return os.path.join(root, f)

	return None


def draw_obb_on_image(img: np.ndarray, polygon: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> None:
	pts = polygon.reshape((-1, 1, 2))
	cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def random_color_for_class(cls: int) -> Tuple[int, int, int]:
	# deterministic color per class
	np.random.seed(cls)
	return tuple(int(x) for x in (np.random.randint(30, 230), np.random.randint(30, 230), np.random.randint(30, 230)))


def process_all(labels_dir: str, images_dir: str, out_dir: Optional[str] = None, show: bool = False) -> None:
	os.makedirs(out_dir, exist_ok=True) if out_dir else None

	label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
	if not label_files:
		print(f'No .txt label files found in {labels_dir}')
		return

	for lbl in sorted(label_files):
		lbl_path = os.path.join(labels_dir, lbl)
		img_path = find_image_for_label(lbl, images_dir)
		if img_path is None:
			print(f'Warning: no image found for label {lbl} (searched base id)')
			continue

		img = cv2.imread(img_path)
		if img is None:
			print(f'Failed to read image {img_path}')
			continue
		h_img, w_img = img.shape[:2]

		with open(lbl_path, 'r', encoding='utf-8') as fh:
			for line in fh:
				parsed = parse_yolo_obb_line(line)
				if parsed is None:
					continue
				cls, cxn, cyn, wn, hn, angle = parsed
				# convert normalized to pixels
				cx = cxn * w_img
				cy = cyn * h_img
				w = wn * w_img
				h = hn * h_img

				angle_rad = 0.0
				if angle is not None:
					angle_rad = angle_to_radians(angle)

				poly = obb_to_polygon(cx, cy, w, h, angle_rad)
				color = random_color_for_class(cls)
				draw_obb_on_image(img, poly, color=color, thickness=2)
				# draw label
				cv2.putText(img, str(cls), tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

		out_path = os.path.join(out_dir, os.path.basename(img_path)) if out_dir else None
		if out_path:
			cv2.imwrite(out_path, img)
			print(f'Wrote {out_path}')

		if show:
			winname = f'plot: {os.path.basename(img_path)}'
			cv2.imshow(winname, img)
			key = cv2.waitKey(0)
			cv2.destroyWindow(winname)
			# if user pressed q or ESC break
			if key in (ord('q'), 27):
				print('User requested exit')
				return


def main() -> None:
	p = argparse.ArgumentParser(description='Plot YOLO OBB labels on images')
	p.add_argument('--labels', default='labels', help='directory with .txt label files')
	p.add_argument('--images', default='train', help='directory with images')
	p.add_argument('--out', default='out_images', help='output directory to save images (optional)')
	p.add_argument('--show', action='store_true', help='show images interactively')
	args = p.parse_args()

	if not os.path.isdir(args.labels):
		print(f'Labels directory not found: {args.labels}')
		return
	if not os.path.isdir(args.images):
		print(f'Images directory not found: {args.images}')
		return

	process_all(args.labels, args.images, out_dir=args.out, show=args.show)


if __name__ == '__main__':
	main()

