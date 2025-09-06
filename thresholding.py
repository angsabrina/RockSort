import numpy as np
import matplotlib.pyplot as plt

import skimage as ski
from skimage import io

image_full_path = 'images\\small_test_rock.jpg'
image = io.imread(image_full_path, as_gray=True)  # Load image as grayscale

hist, hist_centers = ski.exposure.histogram(image)

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(image, cmap=plt.cm.gray)
axes[0].set_axis_off()
axes[1].plot(hist_centers, hist, lw=2)
axes[1].set_title('histogram of gray values')

fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharey=True)

axes[0].imshow(image > 0.5, cmap=plt.cm.gray)
axes[0].set_title('image > 0.5')

axes[1].imshow(image > 0.6, cmap=plt.cm.gray)
axes[1].set_title('image > 0.6')

axes[2].imshow(image > 0.7, cmap=plt.cm.gray)
axes[2].set_title('image > 0.7')

for a in axes:
    a.set_axis_off()

fig.tight_layout()


edges = ski.feature.canny(image)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(edges, cmap=plt.cm.gray)
ax.set_title('Canny detector')
ax.set_axis_off()

plt.show()