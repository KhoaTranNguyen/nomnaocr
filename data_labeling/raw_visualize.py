import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN

# Load the image
image_path = "../dataset/page81a/page81a.jpg"
with Image.open(image_path) as img:
    canvas_width, canvas_height = img.size  # width, height

TOLERANCE = 5  # pixels

# Create canvas
canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# Load column.txt (labeled quadrilaterals)
label_map = []
with open("../dataset/page81a/column.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 9:
            try:
                coords = list(map(float, parts[:8]))
                label = parts[8]
                x_center = sum(coords[::2]) / 4
                y_center = sum(coords[1::2]) / 4
                label_map.append((x_center, y_center, coords, label))
            except ValueError:
                continue

# Draw labeled quadrilateral boxes
for x_center, y_center, coords, label in label_map:
    pts = np.array([[int(coords[i]), int(coords[i+1])] for i in range(0, 8, 2)], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    text_pos = (int(x_center), int(y_center))
    #cv2.putText(canvas, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 0), 1, cv2.LINE_AA)

# Read and parse data.txt
bboxes = []
with open("../dataset/page81a/data.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            class_id, x, y, w, h, conf = map(float, line.split())
            center_x = x * canvas_width
            center_y = y * canvas_height
            bboxes.append({
                'line': line,
                'center_x': center_x,
                'center_y': center_y,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'conf': conf,
            })
        except ValueError:
            print(f"Skipping invalid line: {line}")

# DBSCAN clustering along X axis
X = np.array([[bbox['center_x']] for bbox in bboxes])
dbscan = DBSCAN(eps=15, min_samples=1)
clusters = dbscan.fit_predict(X)

# Group by cluster
cluster_dict = {}
for cluster_id, bbox in zip(clusters, bboxes):
    cluster_dict.setdefault(cluster_id, []).append(bbox)

# Sort clusters right-to-left, and boxes top-to-bottom
sorted_clusters = sorted(cluster_dict.items(), key=lambda c: np.mean([b['center_x'] for b in c[1]]), reverse=True)
for _, boxes in sorted_clusters:
    boxes.sort(key=lambda b: b['center_y'])

# Flatten sorted boxes
sorted_bboxes = [bbox for _, boxes in sorted_clusters for bbox in boxes]

# Draw bounding boxes
for idx, bbox in enumerate(sorted_bboxes, start=1):
    center_x = bbox['center_x']
    center_y = bbox['center_y']
    w = int(bbox['w'] * canvas_width)
    h = int(bbox['h'] * canvas_height)

    x1 = max(int(center_x - w // 2), 0)
    y1 = max(int(center_y - h // 2), 0)
    x2 = min(int(center_x + w // 2), canvas_width - 1)
    y2 = min(int(center_y + h // 2), canvas_height - 1)

    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
    label_pos = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20)
    cv2.putText(canvas, f"{idx}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Show result
cv2.imshow('DBSCAN + Column Visualization', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
