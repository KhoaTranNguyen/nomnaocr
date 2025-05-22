import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load the image
image_path = "../dataset/ban_tuc_62a/DVSKTT_ban_tuc_XVIII_62a.jpg"
with Image.open(image_path) as img:
    canvas_width, canvas_height = img.size  # width, height

# Step 1: Parse column.txt and extract bounding boxes and labels
columns = []
with open("../dataset/ban_tuc_62a/column.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 9:
            try:
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
                label = parts[8]
                xs = [x1, x2, x3, x4]
                ys = [y1, y2, y3, y4]
                xmin = min(xs)
                xmax = max(xs)
                ymin = min(ys)
                ymax = max(ys)
                columns.append({
                    "bbox": (xmin, xmax, ymin, ymax),
                    "label": label,
                    "boxes": []  # will hold matched boxes
                })
            except ValueError:
                continue

# Step 2: Parse data.txt and assign each box to a column based on center location
all_boxes = []
with open("../dataset/ban_tuc_62a/data.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        _, x, y, w, h, conf = parts
        x = float(x)
        y = float(y)
        x_abs = x * canvas_width
        y_abs = y * canvas_height
        box = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "conf": conf,
            "x_abs": x_abs,
            "y_abs": y_abs,
            "assigned": False  # track usage
        }
        # Assign box to a column if inside its bounding box
        for col in columns:
            xmin, xmax, ymin, ymax = col["bbox"]
            if xmin <= x_abs <= xmax and ymin <= y_abs <= ymax:
                col["boxes"].append(box)
                box["assigned"] = True
                break
        all_boxes.append(box)

# Step 3: For each column, sort boxes top to bottom and assign one character per box
labeled_boxes = []  # store labeled boxes for visualization

for col in columns:
    label_chars = list(col["label"])
    col_boxes = col["boxes"]

    # Sort boxes by y_abs (top to bottom)
    col_boxes.sort(key=lambda b: b["y_abs"])

    # Only match up to the number of characters
    for i, box in enumerate(col_boxes[:len(label_chars)]):
        ch = label_chars[i]
        labeled_boxes.append({
            "char": ch,
            "x_abs": box["x_abs"],
            "y_abs": box["y_abs"],
            "w_abs": float(box["w"]) * canvas_width,
            "h_abs": float(box["h"]) * canvas_height,
        })

# Step 3.5: Write labeled data to file
with open("../dataset/ban_tuc_62a/data_labeled.txt", "w", encoding="utf-8") as f_out:
    for box in labeled_boxes:
        x_norm = box["x_abs"] / canvas_width
        y_norm = box["y_abs"] / canvas_height
        w_norm = box["w_abs"] / canvas_width
        h_norm = box["h_abs"] / canvas_height
        line = f"{box['char']} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f} 1.0\n"
        f_out.write(line)

print("✅ Labeled data saved to data_labeled.txt")

# Step 4: Visualization using Pillow to support local OTF font
canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# Convert OpenCV BGR canvas to PIL RGB image
canvas_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(canvas_pil)

# Load your local OTF font, adjust path and size
font_path = "../data_labeling/NomNaTong-Regular.otf"  # <-- set your font file path here
font_size = 20
font = ImageFont.truetype(font_path, font_size)

for box in labeled_boxes:
    x_center = box["x_abs"]
    y_center = box["y_abs"]
    w = int(box["w_abs"])
    h = int(box["h_abs"])

    x1 = max(int(x_center - w // 2), 0)
    y1 = max(int(y_center - h // 2), 0)
    x2 = min(int(x_center + w // 2), canvas_width - 1)
    y2 = min(int(y_center + h // 2), canvas_height - 1)

    # Draw rectangle with OpenCV (blue box)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw text with PIL (red text)
    label_pos = (x1, y1 - font_size if y1 - font_size > 0 else y1 + 2)
    draw.text(label_pos, box["char"], font=font, fill=(255, 0, 0))

# Convert back to OpenCV BGR format and show
canvas = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)

cv2.imshow("Labeled boxes with custom font", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
# The above code reads bounding boxes and labels from "column.txt" and "data.txt",
# assigns boxes to columns based on their center location, sorts them, and visualizes the results.