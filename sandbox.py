import os
from segmentation import SegmentFace
import cv2
import numpy as np
from image_preprocessing import ImageProcessor


# img = cv2.imread("bet_darker.png")

# image= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# image = cv2.resize(image, (512, 512))
# # cv2.imwrite('krool.png', image)

# cv2.imshow('lol', image)

# cv2.waitKey()
# cv2.destroyAllWindows()

# ProcessImage("orig_2.png", 'brightest' ).run()

# if __name__ == '__main__':
#     SegmentFace('bordered.png').run()
    # SegmentFace('new.png').run()

# SegmentFace('recolored_image.png').run()
SegmentFace('original.png', 'cropped.png', 'processed.png').run()



# from PIL import Image
  
# # Opening the primary image (used in background)
# img1 = Image.open(r"bordered.png").convert('RGB')
  
# # Opening the secondary image (overlay image)
# img2 = Image.open(r"final.png").convert('RGB')
  
# # Pasting img2 image on top of img1 
# # starting at coordinates (0, 0)
# img1.paste(img2, (0,0), mask = img2)
  
# # Displaying the image
# img1.show()




