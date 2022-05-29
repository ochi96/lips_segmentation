import cv2
import numpy as np
from face_type import categorize_image


# brightest_strategy: no recoloring, border_size = 1280 gaussian blur (5,5)...kernel = 9, ==>1536, 900

# darkest_strategy: kernel = 9, border_size = 1536 recolor = np.array([[[100%pixel,70%pixel,150%pixel] if pixel>=5 else [0%pixel,0%pixel,0%pixel] for pixel in row] for row in scaled_gray], dtype = np.dtype('f8'))
                # : blurr image (3,3)===>1536, 900

# the inbetweens:.....lighter...> face_size = 1024,  blur=(3,3)...1536, 900
                # self.recolored_image = np.array([[[255%pixel,120%pixel,140%pixel] if pixel>=30 else [70%pixel,80%pixel,105%pixel] for pixel in row] for row in scaled_gray], dtype = np.dtype('f8'))
                # turn to grayscale, then SEgment face, cropped_image_height=30

# the inbetweens: ....darker...> desired_width = (900, 900), bordered_image_size = 1344, kernel=9: mean_height =100

                # self.recolored_image = np.array([[[180%pixel,190%pixel,125%pixel] if pixel>=0 else [255%pixel,0%pixel,0%pixel] for pixel in row] for row in scaled_gray], dtype = np.dtype('f8'))



# lighter inb without grayscaling afterwards
# self.bordered_image_size = 1360
#             self.pixel_threshold = 30
#             self.mean_height = 100
#             self.bgr = {'b' : 120, 'g' : 240, 'r' : 130}
#             self.bgr_alt = {'b': 30, 'g' : 40,  'r' : 60}


class ImageProcessor():

    def __init__(self, image_path) -> None:
        self.image = cv2.imread(image_path)
        self.image_height, self.image_width = self.image.shape[0:2]
        self.desired_height, self.desired_width = (900,900)
        self.bordered_image_size = 1536
        self.cropped_image = self.crop_image()
        self.face_type = self.get_face_type()
        self.gaussian_blur=(3,3)
        self.kernel_sharpener = 9
        self.mean_height = 40
        self.pixel_threshold = 0



        if self.face_type=="brightest":
            self.bordered_image_size = 1468
            # self.desired_height, self.desired_width = (768, 768)
  
        if self.face_type=="darkest":
            self.bgr = {'b' : 100,  'g' : 70,  'r' : 150}
            self.bgr_alt = {'b' : 0,  'g' : 0,  'r' : 0}
        
        if self.face_type=="darker_inb":
            self.bordered_image_size = 1350           #1344 works best for now
            self.mean_height = 40                      #100 works decently
            self.bgr = {'b' : 180,  'g' : 180,  'r' : 125}
            self.bgr_alt = {'b' : 255, 'g' : 0,  'r' : 0}
        
        if self.face_type=="lighter_inb":
            self.bordered_image_size = 1536  #1485 DECENT

            self.pixel_threshold = 0
            self.mean_height = 100
            self.bgr = {'b' : 0, 'g' : 120, 'r' : 170}
            self.bgr_alt = {'b': 180, 'g' : 0,  'r' : 0}
            # ....then grayscale before segment face


            # self.bordered_image_size = 1440
            # self.pixel_threshold = 30
            # self.mean_height = 100
            # self.bgr = {'b' : 120, 'g' : 240, 'r' : 130}
            # self.bgr_alt = {'b': 30, 'g' : 40,  'r' : 60}
        
        pass

    def crop_face(self):

        crop_width = self.desired_width if self.desired_width<self.image_width else self.image_width
        crop_height = self.desired_height if self.desired_height<self.image_height else self.image_height
        center_x, center_y = int(self.image_width/2), int(self.image_height/2)
        half_crop_width, half_crop_height = int(crop_width/2), int(crop_height/2)
        self.cropped_face = self.image[center_y-half_crop_height:center_y+half_crop_height, center_x-half_crop_width:center_x + half_crop_width]

        return self.cropped_face

    
    def create_border(self):

        image =cv2.GaussianBlur(self.cropped_face, (self.gaussian_blur), cv2.BORDER_DEFAULT)
        cropped_image_height, cropped_image_width = self.cropped_face.shape[0:2]

        mean = cv2.mean(self.image[cropped_image_height-self.mean_height:cropped_image_height, 0:cropped_image_width])[0]
        bottom_bordersize, top_bordersize = round((self.bordered_image_size - cropped_image_width)/2), \
                                            round((self.bordered_image_size - cropped_image_height)/2)

        self.bordered_image = cv2.copyMakeBorder(image, top=top_bordersize,
                                bottom=top_bordersize, left=bottom_bordersize,
                                right=bottom_bordersize, borderType=cv2.BORDER_CONSTANT,
                                value=[mean, mean, mean])
        # cv2.imshow('bordered face', self.bordered_image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return self.bordered_image

    def recolor_image(self):
        # sharpen the image
        sharpening_kernel = np.array([
                                    [-1,-1,-1], 
                                    [-1, 9,-1],
                                    [-1,-1,-1]
                                    ])
        image_ = cv2.filter2D(self.bordered_image, -1, sharpening_kernel)

        image = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
        # image = cv2.GaussianBlur(image, self.gaussian_blur, cv2.BORDER_DEFAULT)

        max_pixel = max([max(row) for row in image])
        scaled_gray = np.array([[(float(pixel/max_pixel)*249) + 1 for pixel in row] for row in image], dtype = np.dtype("float32"))

        # add one to each pixel.....for division purposes later.(avoiding division by zero)
        scaled_gray = np.array([[round(pixel)+1 for pixel in row] for row in scaled_gray], dtype = np.dtype("float32"))

        # cv2.imwrite("scaled_gray.png", scaled_gray)

        if self.face_type == "brightest":
            self.recolored_image = scaled_gray
            print(self.face_type)
        
        elif self.face_type == "darkest" or self.face_type=="darker_inb" or self.face_type == "lighter_inb":
            self.recolored_image = np.array([[[self.bgr['b']%pixel, self.bgr['g']%pixel, self.bgr['r']%pixel] if pixel>=self.pixel_threshold else \
                [self.bgr_alt['b']%pixel,self.bgr_alt['g']%pixel,self.bgr_alt['r']%pixel] \
                    for pixel in row] for row in scaled_gray], dtype = np.dtype(np.uint8))

            if self.face_type == "lighter_inb":
                self.recolored_image = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
            # #     print(self.recolored_image)
            # #     # print(scaled_gray)
            #     _, self.recolored_image = cv2.threshold(self.recolored_image, 120, 140, cv2.THRESH_BINARY)
            # #     # print(self.recolored_image)
            #     self.recolored_image = cv2.adaptiveThreshold(self.recolored_image, 80, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 3)
                # cv2.imwrite("rec_gray.png", self.recolored_image)
        print(self.face_type)
        return self.recolored_image
 
    def get_face_type(self):
        face_type = categorize_image(self.image)
        return face_type

    def crop_image(self):
        center_x, center_y = int(self.image_height/2), int(self.image_width/2)
        half_crop_width, half_crop_height = int(self.bordered_image_size/2), int(self.bordered_image_size/2)
        cropped_image = self.image[center_y-half_crop_height:center_y+half_crop_height, center_x-half_crop_width:center_x + half_crop_width]
        return cropped_image

    
    def run(self):
        self.crop_face()
        self.create_border()
        return self.cropped_image, self.recolor_image()

# if __name__ == '__main__':
#     image_processor = ImageProcessor("images/originals/lol2.png")
#     cropped_face = image_processor.crop_image()
#     bordered_face = image_processor.create_border()
#     recolored_face = image_processor.recolor_image()
#     pass
