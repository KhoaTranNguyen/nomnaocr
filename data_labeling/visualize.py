import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from ultralytics.utils.plotting import colors
import numpy as np
from PIL import Image

def visualize_image_annotations(image_path, txt_path, output_path, column_path):
    img = np.array(Image.open(image_path))
    img_height, img_width = img.shape[:2]
    dpi = 100  # or choose any dpi you want

    # Calculate figure size in inches to match image pixel dimensions
    fig_width = img_width / dpi
    fig_height = img_height / dpi

    annotations = []
    font_path = "assets/NomNaTong-Regular.otf"  # Adjust if needed
    noto_font = fm.FontProperties(fname=font_path)
    
    with open(txt_path) as folder:
        for line in folder:
            class_id = line.split()[0]
            x_center, y_center, width, height = map(float, line.split()[1:5])
            x = (x_center - width / 2) * img_width
            y = (y_center - height / 2) * img_height
            w = width * img_width
            h = height * img_height
            annotations.append((x, y, w, h, class_id))

    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height), dpi=dpi)
    ax.imshow(img)

    for x, y, w, h, label in annotations:
        color = tuple(c / 255 for c in colors(0, True))
        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        ax.text(x, y, label, color="black" if luminance < 0.5 else "black", fontsize=8, fontproperties=noto_font)

    columns = []
    with open(column_path) as folder:
        for line in folder:
            parts = line.strip().split(',')
            class_id = parts[8]
            coords = list(map(float, parts[:8]))
            x_coords = coords[::2]
            y_coords = coords[1::2]
            xmin = min(x_coords)
            ymin = min(y_coords)
            xmax = max(x_coords)
            ymax = max(y_coords)
            w = xmax - xmin
            h = ymax - ymin
            columns.append((xmin, ymin, w, h, class_id))

    for x, y, w, h, class_id in columns:
        color = tuple(c / 255 for c in (4, 42, 255))
        rect_col = Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor="none")
        ax.add_patch(rect_col)
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        ax.text(x, y - 5, class_id, color="white" if luminance < 0.5 else "black", backgroundcolor=(0, 0, 0, 0.01))

    ax.axis("off")

    # Save with dpi to match figure size exactly, no padding
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)
