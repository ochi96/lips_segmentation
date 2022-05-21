import cv2
import numpy as np

from processor import ImageProcessor
import os
from segmentation import SegmentFace



# cropped face --->add margin, detect boundary lines--->croppinng dimensions needed
# ....do the same for the mask and see if the lip areas coincide


# processed_img = ImageProcessor("images/originals/lol0.png").run()
# cv2.imwrite('lol2.png', processed_img)
# SegmentFace('lol2.png').run()

# processed_img, original_img = ImageProcessor("images/originals/lol2.png").run()
# cv2.imwrite('processed.png', processed_img)
# # cv2.imwrite('original.png', original_img)

# print('lol')


# def reverse_processed_image(processed_img):
#     img = cv2.resize(processed_img, (1536, 1536))
#     center_x, center_y = int(1536/2), int(1536/2)
#     half_crop_width, half_crop_height = int(450), int(450)
#     cropped_face = img[center_y-half_crop_height:center_y+half_crop_height, center_x-half_crop_width:center_x + half_crop_width]

#     # cv2.imwrite('cropped.png', cropped_face)

#     cropped_image_height, cropped_image_width = cropped_face.shape[0:2]

#     mean = cv2.mean(cropped_face[cropped_image_height-30:cropped_image_height, 0:cropped_image_width])[0]
#     bottom_bordersize, top_bordersize = round((1536 - cropped_image_width)/2), \
#                                         round((1536 - cropped_image_height)/2)

#     bordered_image = cv2.copyMakeBorder(cropped_face, top=top_bordersize,
#                             bottom=top_bordersize, left=bottom_bordersize,
#                             right=bottom_bordersize, borderType=cv2.BORDER_CONSTANT,
#                             value=[mean, mean, mean])
    
#     cv2.imwrite('original.png', bordered_image)

#     pass

# reverse_processed_image(original_img)


def reverse_processed_image2(img_path):
    img = cv2.imread(img_path)
    # img = cv2.resize(processed_img, (1536, 1536))
    # center_x, center_y = int(1536/2), int(1536/2)
    # half_crop_width, half_crop_height = int(450), int(450)
    # cropped_face = img[center_y-half_crop_height:center_y+half_crop_height, center_x-half_crop_width:center_x + half_crop_width]

    # cv2.imwrite('cropped.png', cropped_face)

    image_height, image_width = img.shape[0:2]

    # mean = cv2.mean(img[cropped_image_height-30:cropped_image_height, 0:cropped_image_width])[0]
    bottom_bordersize, top_bordersize = round((2048 - image_width)/2), \
                                        round((2048 - image_height)/2)

    bordered_image = cv2.copyMakeBorder(img, top=top_bordersize,
                            bottom=top_bordersize, left=bottom_bordersize,
                            right=bottom_bordersize, borderType=cv2.BORDER_CONSTANT,
                            value=[255, 255, 255])
    
    cv2.imwrite('results/original_mask_resized.png', bordered_image)

    pass

# reverse_processed_image2('results/original_mask.jpg')

def crop_orig(img_path):
    img = cv2.imread(img_path)
    center_x, center_y = int(2048/2), int(2048/2)
    half_crop_width, half_crop_height = int(768), int(768)
    cropped_face = img[center_y-half_crop_height:center_y+half_crop_height, center_x-half_crop_width:center_x + half_crop_width]

    cv2.imwrite('lol2_cropped.png', cropped_face)
    pass

# crop_orig('lol2.png')


from PIL import Image

def paste_into_it(img_path_1, img_path2):
    im1 = Image.open('lol2.png').convert('RGBA')
    im2 = Image.open('lol2_cropped_segmented.jpg').convert('RGBA')

    back_im = im1.copy()
    back_im.paste(im2, (256, 256))
    back_im.save('final.png', quality=95)

    pass

paste_into_it('lol2.png', 'lol2_cropped_segmented.png')
