import os
from segmentation import SegmentFace
import cv2
import numpy as np
from processor import ImageProcessor


# img = cv2.imread("bet_darker.png")

# image= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# image = cv2.resize(image, (512, 512))
# # cv2.imwrite('krool.png', image)

# cv2.imshow('lol', image)

# cv2.waitKey()
# cv2.destroyAllWindows()


if __name__ == '__main__':
    # (cropped_face, processed_face) = ImageProcessor("images/originals/lol2.png").run()
    # cv2.imwrite('cropped.png', cropped_face)
    # cv2.imwrite('processed.png', processed_face)
    SegmentFace("images/originals/lol2.png", 'cropped.png', 'processed.png').run()

# SegmentFace('recolored_image.png').run()
# SegmentFace('original.png', 'cropped.png', 'processed.png').run()


