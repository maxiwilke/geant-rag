import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# === REPLACE WITH YOUR CLICK COORDINATES ===
xs = np.array([147, 267, 520, 567, 587, 442, 436, 335, 98, 392, 474, 534, 307, 429, 163, 207, 225, 200, 271, 357, 424, 469, 151, 156, 143, 357, 339, 139, 310, 413, 463])
ys = np.array([1531, 1529, 1554, 1527, 1531, 748, 730, 606, 1528, 1521, 1508, 1547, 741, 741, 1525, 1557, 1508, 1525, 1503, 1545, 1530, 1515, 1543, 1519, 1538, 1515, 1521, 1526, 1535, 1553, 1528])

# Use these to create a an empty map to measure the clicks
# xs = np.array([0])
# ys = np.array([0])

# === LOAD IMAGE ===
img = plt.imread("/Users/maximilianwilke/geant-rag/assets/chatbot.png")
h, w = img.shape[:2]

# === CREATE HEATMAP DENSITY GRID ===
# Create a 2D histogram of clicks
heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=[w//4, h//4], range=[[0, w], [0, h]])

# Apply Gaussian smoothing for smooth gradients
sigma = 15  # Adjust for more/less blur (higher = smoother)
heatmap = gaussian_filter(heatmap, sigma=sigma)

# Normalize to 0-1 range
if heatmap.max() > 0:
    heatmap = heatmap / heatmap.max()

# === PLOT ===
dpi = 15
fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)

# Show background image
ax.imshow(img, origin="upper", extent=[0, w, h, 0])

# Overlay heatmap with research-style colormap
# Popular options: 'jet', 'hot', 'YlOrRd', 'plasma', 'inferno'
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

# Optional: Add colorbar to show intensity scale
# plt.colorbar(heatmap_overlay, ax=ax, fraction=0.046, pad=0.04)

ax.set_xlim(0, w)
ax.set_ylim(h, 0)
ax.axis("off")
plt.tight_layout(pad=0)
plt.show()