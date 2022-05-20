import cv2
import os
import numpy as np


desired_height, desired_width = (900,900)

for image_name in os.listdir('images/originals/'):
    
    image_path = f"images/originals/{image_name}"
    img = cv2.imread(image_path)

    image_height, image_width = img.shape[0:2]
    # process crop width and height for max available dimension
    crop_width = desired_width if desired_width<image_width else image_width
    crop_height = desired_height if desired_height<image_height else image_height
    mid_x, mid_y = int(image_width/2), int(image_height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]

    print(crop_img.shape)
    cv2.imwrite(f"images/cropped/{image_name}", crop_img)

