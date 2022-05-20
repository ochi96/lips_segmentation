
import cv2
import numpy as np
import os
from PIL import Image
# from email.policy import strict
import torch


from segmentation import SegmentFace




img = cv2.imread('images/cropped/lol3.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)


_, th1 = cv2.threshold(img, 30, 80, cv2.THRESH_TOZERO)
th2 = cv2.adaptiveThreshold(th1,50,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,2)
th3 = cv2.adaptiveThreshold(th1,130,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,2)

cv2.imshow('Original', img)
# cv2.imshow('th1', th1)
cv2.imshow('th2', th2)
cv2.imshow('th3', th3)


cv2.waitKey(0)
cv2.destroyAllWindows()



