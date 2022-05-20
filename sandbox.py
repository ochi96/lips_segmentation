import os
from segmentation import SegmentFace
import cv2
import numpy as np
from image_preprocessing import ImageProcessor


img = cv2.imread("bet_darker.png")

image= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

image = cv2.resize(image, (512, 512))
# cv2.imwrite('krool.png', image)

cv2.imshow('lol', image)

cv2.waitKey()
cv2.destroyAllWindows()

# ProcessImage("orig_2.png", 'brightest' ).run()

# if __name__ == '__main__':
#     SegmentFace('recolored_image.png').run()
    # SegmentFace('new.png').run()

# SegmentFace('recolored_image.png').run()
# SegmentFace('border.png').run()




