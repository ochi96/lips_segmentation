import numpy as np
import os
from segmentation import SegmentFace
import cv2
from PIL import Image
import torch


final_size = 2048
image_name = 'lol2.png'

im = cv2.imread(f'images/cropped/{image_name}')
im =cv2.GaussianBlur(im, (3,3), cv2.BORDER_DEFAULT)
_, im = cv2.threshold(im, 20, 70, cv2.THRESH_TOZERO)


row, col = im.shape[:2]
print(row, col)

bottom = im[row-50:row, 0:col]
mean = cv2.mean(bottom)[0]

bottom_bordersize = round((final_size - col)/2)
print(bottom_bordersize)

top_bordersize = round((final_size - row)/2)
print(top_bordersize)

border = cv2.copyMakeBorder(
    im,
    top=top_bordersize,
    bottom=top_bordersize,
    left=bottom_bordersize,
    right=bottom_bordersize,
    borderType=cv2.BORDER_CONSTANT,
    value=[mean, mean, mean]
)



cv2.imwrite(f'{image_name}', border)

# SegmentFace(f'{image_name}').run()

