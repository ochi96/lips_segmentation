import cv2
import numpy as np

from processor import ImageProcessor
import os
from segmentation import SegmentFace


image_to_run = ImageProcessor("images/originals/lol4.png").run()
cv2.imwrite('lol.png', image_to_run)
SegmentFace('lol.png').run()

# cv2.imshow('final image', np.array(image_to_run))

# cv2.waitKey()
# cv2.destroyAllWindows()
