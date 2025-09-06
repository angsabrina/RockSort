import cv2
import os
import numpy as np


image_dir = 'images'

# List all files in the directory
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

for image_file in image_files:
    print("image file: " +  image_file)
    img_path = os.path.join(image_dir, image_file)
    img = cv2.imread(img_path, 0)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,30,
                                param1=40,param2=30,minRadius=100,maxRadius=130)
    # min_dist: increase this value to find circles with a larger distance
    # maxRadius: reduce this parameter to avoid circles that contain more than one crater
    # param1: higher values make the edge detector less sensitive; only clear cut edges will be used

    print("number of circles: ", len(circles[0]))  # Number of circles found in the image

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

        # cv2.imshow('detected circles', cimg)
        cv2.imwrite("with-circles" + image_file, cimg)  # Save image with detected circles
        cv2.waitKey(0)