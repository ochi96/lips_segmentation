import cv2
import numpy as np
# import os
# from PIL import Image
# from email.policy import strict
# import torch


from segmentation import SegmentFace


img = cv2.imread('lol2.png')
print(type(img))


kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
img = cv2.filter2D(img, -1, kernel) # applying the sharpening kernel to the input image & displaying it.
cv2.imshow('Image Sharpening', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img_gray.shape)

img_gray = cv2.GaussianBlur(img_gray, (3,3), cv2.BORDER_DEFAULT)
# _, img_gray = cv2.threshold(img_gray, 20, 70, cv2.THRESH_TOZERO)


# cv2.imshow('lol', img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




print(img_gray.shape)
print(img_gray[0])
print(len(img_gray[0]))

max_pixel = max([max(row) for row in img_gray])
min_pixel = min([min(row) for row in img_gray])
print(max_pixel)
print(min_pixel)

diff = max_pixel-min_pixel


lol_gray = np.array([[(float(pixel/max_pixel)*249) + 1 for pixel in row] for row in img_gray], dtype = np.dtype('f8'))

# _, lol_gray = cv2.threshold(lol_gray, round(min_pixel+(0.2*diff)), round(max_pixel-(0.2*diff)), cv2.THRESH_TOZERO)
_, lol_gray = cv2.threshold(lol_gray, 30, 80, cv2.THRESH_TOZERO)




lol_gray = np.array([[round(pixel)+1 for pixel in row] for row in lol_gray], dtype = np.dtype('f8'))
# _, lol_gray = cv2.threshold(lol_gray, 30, 80, cv2.THRESH_TOZERO)

print('first row',lol_gray[0])

cv2.imwrite('lol_gray.png', lol_gray)

print('shape',lol_gray.shape)


# new = np.array([[[50%pixel,200%pixel,50%pixel] if pixel>=40 else [0,0,0] for pixel in row] for row in lol_gray], dtype = np.dtype('f8'))
new = np.array([[[90%pixel,255%pixel,120%pixel] if pixel>=50 else [70%pixel,80%pixel,105%pixel] for pixel in row] for row in lol_gray], dtype = np.dtype('f8'))



# print(type(new))
# dt = np.dtype('f8')
# new=np.array(new,dtype=dt)

# new = cv2.cvtColor(new, cv2.COLOR_BGRA2GRAY)


# cv2.COLOR_BGR




cv2.imwrite('new.png', new)

# SegmentFace('new.png').run()

# SegmentFace('lol2.png').run()

# SegmentFace('lol_gray.png').run()