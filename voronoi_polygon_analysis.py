# -*- coding: utf-8 -*-
"""
Voronoi-like image analysis with edge thinning.
This script takes an image, binarizes it, applies morphological
operations and extracts polygons. An extra skeletonize step is
introduced after morphological cleaning to keep edges thin so
polygon counts remain stable.

Dependencies:
  - OpenCV (cv2)
  - NumPy
  - Matplotlib
  - SciPy
  - Pandas
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import pandas as pd

# --- Config ---
img_path = "/content/8f29978f-0d7e-49be-878e-6e7426b3d339.png"
output_dir = "/content"
os.makedirs(output_dir, exist_ok=True)

# --- 1. load image ---
orig_bgr = cv2.imread(img_path)
orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

# --- 2. crop central 50% ---
h, w = orig_rgb.shape[:2]
y0, y1 = int(0.25*h), int(0.75*h)
x0, x1 = int(0.25*w), int(0.75*w)
crop_rgb = orig_rgb[y0:y1, x0:x1]

# --- 3. grayscale + threshold ---
gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
minv, maxv = int(gray.min()), int(gray.max())
thresh = (minv + maxv) // 2
_, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

# --- 4. morphology ---
inv = cv2.bitwise_not(binary)
kernel = np.ones((3, 3), np.uint8)
# shrink objects slightly to remove small artifacts
eroded = cv2.erode(inv, kernel, iterations=1)
cleaned = cv2.bitwise_not(eroded)
ellipse_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, ellipse_k)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, ellipse_k)

# --- edge thinning ---
def skeletonize(img: np.ndarray) -> np.ndarray:
    """Return morphological skeleton of a binary image."""
    img = img.copy()
    skel = np.zeros(img.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open_img)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel

thin = skeletonize(opened)

# --- 5. contours and polygons ---
contours, _ = cv2.findContours(thin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
polygons = []
side_counts = []
H2, W2 = thin.shape
for cnt in contours:
    if cv2.contourArea(cnt) <= 10:
        continue
    approx = cv2.approxPolyDP(cnt, epsilon=0.02 * cv2.arcLength(cnt, True), closed=True)
    pts = approx.reshape(-1, 2)
    n = len(pts)
    if 4 <= n <= 9 and pts[:,0].min() > 1 and pts[:,0].max() < W2-2 and pts[:,1].min() > 1 and pts[:,1].max() < H2-2:
        polygons.append(pts)
        side_counts.append(n)

# --- 6. signed angle differences ---
angle_diffs = []
for pts in polygons:
    n = len(pts)
    for i in range(n):
        v1 = pts[i - 1] - pts[i]
        v2 = pts[(i + 1) % n] - pts[i]
        cos_t = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
        ang = math.degrees(math.acos(cos_t))
        angle_diffs.append(ang - 120.0)
angle_diffs = np.array(angle_diffs)

# --------------------------------
# figure utilities
# --------------------------------

def save_and_show(fig, path):
    fig.savefig(path, bbox_inches='tight')
    plt.show()

# Figure 1: original
fig = plt.figure(figsize=(6, 6))
plt.imshow(orig_rgb)
plt.axis('off')
plt.title("Figure 1. Original Image")
f1 = os.path.join(output_dir, "figure1_original.png")
save_and_show(fig, f1)

# Figure 2: morphology + skeleton
fig = plt.figure(figsize=(6, 6))
plt.imshow(thin, cmap='gray')
plt.axis('off')
plt.title("Figure 2. Morphology with Thinning")
f2 = os.path.join(output_dir, "figure2_thin.png")
save_and_show(fig, f2)

# Figure 3: polygons colored by sides
color_img = np.full((H2, W2, 3), 255, dtype=np.uint8)
cmap10 = plt.get_cmap('tab10', 6)
for pts in polygons:
    n = len(pts)
    color = (np.array(cmap10(n-4)[:3]) * 255).astype(np.uint8).tolist()
    cv2.fillPoly(color_img, [pts], color)
    cv2.polylines(color_img, [pts], True, (0, 0, 0), 1)
fig = plt.figure(figsize=(6, 6))
plt.imshow(color_img)
plt.axis('off')
plt.title("Figure 3. Polygons Colored by Number of Sides")
f3 = os.path.join(output_dir, "figure3_color_polygons.png")
save_and_show(fig, f3)

# Figure 4: histogram of sides
fig = plt.figure(figsize=(6, 4))
bars = plt.bar(range(4, 10), [side_counts.count(i) for i in range(4, 10)], edgecolor='black')
for bar, n in zip(bars, range(4, 10)):
    bar.set_facecolor(cmap10(n-4)[:3])
plt.xlabel("Number of Sides")
plt.ylabel("Count")
plt.title("Figure 4. Polygon Counts by Number of Sides")
plt.grid(axis='y', linestyle='--', alpha=0.7)
f4 = os.path.join(output_dir, "figure4_side_counts.png")
save_and_show(fig, f4)

# Figure 5: signed angle difference histogram
fig = plt.figure(figsize=(6, 4))
bin_edges = np.linspace(angle_diffs.min(), angle_diffs.max(), 31)
heights, edges, _ = plt.hist(angle_diffs, bins=bin_edges, edgecolor='black', color='lightblue')
x = np.linspace(angle_diffs.min(), angle_diffs.max(), 200)
pdf = stats.norm.pdf(x, loc=0, scale=angle_diffs.std())
pdf_scaled = pdf * len(angle_diffs) * (edges[1] - edges[0])
plt.plot(x, pdf_scaled, 'r-', linewidth=2)
plt.xlabel("Signed Difference (°)")
plt.ylabel("Frequency")
plt.title("Figure 5. Histogram of Angle Differences with Normal Approximation")
plt.grid(True)
f5 = os.path.join(output_dir, "figure5_histogram_overlay.png")
save_and_show(fig, f5)

# Figure 6: Q-Q plot
(res_theor, res_samp), (slope, intercept, _) = stats.probplot(angle_diffs, dist="norm", plot=None)
fig = plt.figure(figsize=(6, 6))
plt.scatter(res_samp, res_theor, s=10, color='blue')
x_vals = np.array([res_samp.min(), res_samp.max()])
y_vals = (x_vals - intercept) / slope
plt.plot(x_vals, y_vals, 'r--', linewidth=1.5)
pcts = [0.001, 0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.9, 99.99]
ticks = stats.norm.ppf(np.array(pcts) / 100)
plt.yticks(ticks, [f"{pv}%" for pv in pcts])
plt.ylim(stats.norm.ppf(0.001), stats.norm.ppf(0.9999))
plt.xlabel("Ordered Angle Differences (°)")
plt.ylabel("Theoretical Quantiles (Normal Probability)")
plt.title("Figure 6. Q–Q Plot on Normal-Probability Scale")
plt.grid(True, which='both', linestyle='-')
f6 = os.path.join(output_dir, "figure6_qqplot.png")
save_and_show(fig, f6)

# Figure 7: log-scale histogram
fig = plt.figure(figsize=(6, 4))
counts7, edges7 = np.histogram(angle_diffs, bins=bin_edges)
centers7 = (edges7[:-1] + edges7[1:]) / 2
plt.scatter(centers7, counts7, color='blue', marker='o')
x7 = np.linspace(angle_diffs.min(), angle_diffs.max(), 200)
pdf7 = stats.norm.pdf(x7, loc=0, scale=angle_diffs.std())
pdf7_scaled = pdf7 * len(angle_diffs) * (edges7[1] - edges7[0])
plt.plot(x7, pdf7_scaled, 'r-', linewidth=2)
plt.yscale('log')
plt.grid(True, which='both', linestyle='-', alpha=0.7)
plt.xlabel("Signed Difference (°)")
plt.ylabel("Count (log scale)")
plt.title("Figure 7. Histogram on Log Scale with Normal Approximation")
f7 = os.path.join(output_dir, "figure7_log_histogram.png")
save_and_show(fig, f7)

# Table 1: polygon counts

df_counts = pd.DataFrame({"num_sides": list(range(4, 10)),
                          "count": [side_counts.count(i) for i in range(4, 10)]})
print("Table 1. Polygon Counts (4–9 sides)")
print(df_counts.to_string(index=False))
csv_path = os.path.join(output_dir, "table1_polygon_counts.csv")
df_counts.to_csv(csv_path, index=False)
print("Saved Table 1 CSV:", csv_path)

# output summary
print("Saved figures:")
print(" Figure 1:", f1)
print(" Figure 2:", f2)
print(" Figure 3:", f3)
print(" Figure 4:", f4)
print(" Figure 5:", f5)
print(" Figure 6:", f6)
print(" Figure 7:", f7)
