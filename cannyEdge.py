import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
 
image_dir = 'images'

# List all files in the directory
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

for image_file in image_files:
    print("image file: " +  image_file)
    img_path = os.path.join(image_dir, image_file)
    img = cv2.imread(img_path, 0)
    img = cv2.medianBlur(img, 5)

    edges = cv2.Canny(img,100,200)
 
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.savefig("with-edges" + image_file)