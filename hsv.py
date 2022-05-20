# differentiate images according to HSV value

import os
from segmentation import SegmentFace
import cv2
import numpy as np
from PIL import Image
from image_preprocessing import ImageProcessor


flags = ['brightest', 'darkest', 'in-between']
images = ["lol0.png", "lol2.png", "lol3.png", "lol4.png"]

nose_crop_points = ((650, 250), (975, 100))
cheek_crop_points = ((880, 250), (550, 100))

crop_type = ('nose', 'cheek')


def crop_area(image_path, crop_type='nose'):
    image = cv2.imread(image_path)
    if crop_type=='nose':
        start_crop_point_y, stop_crop_point_y = nose_crop_points[0][0], nose_crop_points[0][0] + nose_crop_points[0][1]
        start_crop_point_x, stop_crop_point_x = nose_crop_points[1][0], nose_crop_points[1][0] + nose_crop_points[1][1]
        cropped_strip = image[start_crop_point_y:stop_crop_point_y, start_crop_point_x:stop_crop_point_x]
    if crop_type=='cheek':
        start_crop_point_y, stop_crop_point_y = cheek_crop_points[0][0], cheek_crop_points[0][0] + cheek_crop_points[0][1]
        start_crop_point_x, stop_crop_point_x = cheek_crop_points[1][0], cheek_crop_points[1][0] + cheek_crop_points[1][1]
        cropped_strip = image[start_crop_point_y:stop_crop_point_y, start_crop_point_x:stop_crop_point_x]

    # cv2.imshow(f'stripped {crop_type}', cropped_strip)

    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return cropped_strip

def categorize_strip(cropped_strip):
    hsv_image = cv2.cvtColor(cropped_strip, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv_image)
    brightness = hsv_image[...,2].mean()
    if brightness>180: 
        flag = flags[0]
    elif brightness<85:
        flag = flags[1]
    else:
        flag = flags[2]
    
    return (flag, brightness)

def main(image_path):
    cropped_strip = crop_area(image_path, 'nose')
    flag, brightness_nose =  categorize_strip(cropped_strip)
    # print(flag, brightness_nose)

    if flag == 'in-between':
        cropped_cheek = crop_area(image_path, 'cheek')
        flag, brightness_cheek =  categorize_strip(cropped_cheek)
        # print(flag, brightness_cheek)

        if brightness_cheek < brightness_nose-20:
            flag = 'darker_inb'
        else:
            flag = 'lighter_inb'
    print(flag, image_path)
        
    return flag



if __name__ == '__main__':

    for image in images:
        image_path = f"images/originals/{image}"
        main(image_path)
    pass







