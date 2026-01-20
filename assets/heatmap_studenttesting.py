import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import os

# === REPLACE WITH YOUR CLICK COORDINATES ===
xs = np.array([147, 267, 520, 567, 587, 442, 436, 335, 98, 392, 474, 534, 307, 429, 163, 207, 225, 200, 271, 357, 424, 469, 151, 156, 143, 357, 339, 139, 310, 413, 463])
ys = np.array([1531, 1529, 1554, 1527, 1531, 748, 730, 606, 1528, 1521, 1508, 1547, 741, 741, 1525, 1557, 1508, 1525, 1503, 1545, 1530, 1515, 1543, 1519, 1538, 1515, 1521, 1526, 1535, 1553, 1528])

# Use these to create a an empty map to measure the clicks
# xs = np.array([0])
# ys = np.array([0])

# === PATHS ===
input_path = "/Users/maximilianwilke/geant-rag/assets/chatbot.png"
output_path = "/Users/maximilianwilke/geant-rag/assets/heatmap.png"

# === LOAD IMAGE ===
img = plt.imread(input_path)
h, w = img.shape[:2]

# === CREATE HEATMAP DENSITY GRID ===
# Create a 2D histogram of clicks (use full resolution)
heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=[w, h], range=[[0, w], [0, h]])

# Apply Gaussian smoothing for smooth gradients
sigma = 15  # Adjust for more/less blur (higher = smoother)
heatmap = gaussian_filter(heatmap, sigma=sigma)

# Normalize to 0-1 range
if heatmap.max() > 0:
    heatmap = heatmap / heatmap.max()

# === PLOT WITH EXACT DIMENSIONS ===
# Create figure with exact pixel dimensions
fig = plt.figure(figsize=(w/100, h/100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])  # Remove all margins

# Show background image
ax.imshow(img, origin="upper", extent=[0, w, h, 0])

# Overlay heatmap with research-style colormap
heatmap_overlay = ax.imshow(
    heatmap.T,
    origin="upper",
    extent=[0, w, h, 0],
    cmap='jet',  # Change to 'hot', 'YlOrRd', or 'plasma' as preferred
    alpha=0.5,   # Transparency (0.4-0.6 works well)
    interpolation='bilinear',
    vmin=0,
    vmax=1
)

ax.set_xlim(0, w)
ax.set_ylim(h, 0)
ax.axis("off")

# Save with exact resolution (will overwrite if exists)
plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
print(f"Heatmap saved to: {output_path}")

plt.show()