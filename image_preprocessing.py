import cv2
import numpy as np

# brightest_strategy: no recoloring, border_size = 1280 gaussian blur (5,5)...kernel = 9, ==>1536, 900

# darkest_strategy: kernel = 9, border_size = 1536 recolor = np.array([[[100%pixel,70%pixel,150%pixel] if pixel>=5 else [0%pixel,0%pixel,0%pixel] for pixel in row] for row in scaled_gray], dtype = np.dtype('f8'))
                # : blurr image (3,3)===>1536, 900

# the inbetweens:.....brighter...> border_size = 1024, kernel==7, blur=(3,3)...1536, 900
                # self.recolored_image = np.array([[[255%pixel,120%pixel,140%pixel] if pixel>=30 else [70%pixel,80%pixel,105%pixel] for pixel in row] for row in scaled_gray], dtype = np.dtype('f8'))
                # turn to grayscale, then SEgment face, cropped_image_height=30

# the inbetweens: ....darker...> desired_width = (900, 900), bordered_image_size = 1344, kernel=9: cropped_imageheight =100

                # self.recolored_image = np.array([[[180%pixel,190%pixel,125%pixel] if pixel>=10 else [255%pixel,0%pixel,0%pixel] for pixel in row] for row in scaled_gray], dtype = np.dtype('f8'))


class ImageProcessor():

    def __init__(self, image_path, flag=None) -> None:
        self.image = cv2.imread(image_path)
        self.image_height, self.image_width = self.image.shape[0:2]
        self.desired_height, self.desired_width = (900,900)
        self.bordered_image_size = 1344
        self.flag = flag
        pass

    def crop_image(self):

        crop_width = self.desired_width if self.desired_width<self.image_width else self.image_width
        crop_height = self.desired_height if self.desired_height<self.image_height else self.image_height
        center_x, center_y = int(self.image_width/2), int(self.image_height/2)
        half_crop_width, half_crop_height = int(crop_width/2), int(crop_height/2)
        self.cropped_face = self.image[center_y-half_crop_height:center_y+half_crop_height, center_x-half_crop_width:center_x + half_crop_width]

        return self.cropped_face

    def create_border(self):

        image =cv2.GaussianBlur(self.cropped_face, (3, 3), cv2.BORDER_DEFAULT)
        # _, image = cv2.threshold(image, 20, 70, cv2.THRESH_TOZERO)

        cropped_image_height, cropped_image_width = self.cropped_face.shape[0:2]

        mean = cv2.mean(self.image[cropped_image_height-100:cropped_image_height, 0:cropped_image_width])[0]

        bottom_bordersize, top_bordersize = round((self.bordered_image_size - cropped_image_width)/2), round((self.bordered_image_size - cropped_image_height)/2)

        self.bordered_image = cv2.copyMakeBorder(
            image,
            top=top_bordersize,
            bottom=top_bordersize,
            left=bottom_bordersize,
            right=bottom_bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[mean, mean, mean]
        )

        return self.bordered_image

    def recolor_image(self):

        # sharpen the image

        sharpening_kernel = np.array([
                                    [-1,-1,-1], 
                                    [-1, 9,-1],
                                    [-1,-1,-1]
                                    ])
        image= cv2.filter2D(self.bordered_image, -1, sharpening_kernel) 


        image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (3,3), cv2.BORDER_DEFAULT)
        # cv2.imshow('blurred', image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        max_pixel = max([max(row) for row in image])
        min_pixel = min([min(row) for row in image])

        diff = max_pixel-min_pixel


        scaled_gray = np.array([[(float(pixel/max_pixel)*249) + 1 for pixel in row] for row in image], dtype = np.dtype('f8'))

        # _, scaled_gray = cv2.threshold(scaled_gray, round(min_pixel+(0.2*diff)), round(max_pixel-(0.2*diff)), cv2.THRESH_TOZERO)
        # _, scaled_gray = cv2.threshold(scaled_gray, 30, 80, cv2.THRESH_TOZERO)



        # add one to each pixel.....for division purposes later.
        scaled_gray = np.array([[round(pixel)+1 for pixel in row] for row in scaled_gray], dtype = np.dtype('f8'))

        # use a human like face----red lips, white to brown skin complexion.(not necessarily, whatever works by trial and error)
        if self.flag == 'brightest':
            self.recolored_image = scaled_gray
        else:
            self.recolored_image = np.array([[[180%pixel,190%pixel,125%pixel] if pixel>=10 else [255%pixel,0%pixel,0%pixel] for pixel in row] for row in scaled_gray], dtype = np.dtype('f8'))

        self.recolored_image = cv2.resize(self.recolored_image, (1024, 1024))

        cv2.imwrite('recolored_image.png', self.recolored_image)
        print('lol')

        return self.recolored_image


    def recolor_image_darkest(self):

        # sharpen the image

        sharpening_kernel = np.array([
                                    [-1,-1,-1], 
                                    [-1, 9,-1],
                                    [-1,-1,-1]
                                    ])
        image= cv2.filter2D(self.bordered_image, -1, sharpening_kernel) 


        image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)

        max_pixel = max([max(row) for row in image])
        min_pixel = min([min(row) for row in image])

        diff = max_pixel-min_pixel


        scaled_gray = np.array([[(float(pixel/max_pixel)*249) + 1 for pixel in row] for row in image], dtype = np.dtype('f8'))

        # _, scaled_gray = cv2.threshold(scaled_gray, round(min_pixel+(0.2*diff)), round(max_pixel-(0.2*diff)), cv2.THRESH_TOZERO)
        # _, scaled_gray = cv2.threshold(scaled_gray, 30, 80, cv2.THRESH_TOZERO)



        # add one to each pixel.....for division purposes later.
        scaled_gray = np.array([[round(pixel)+1 for pixel in row] for row in scaled_gray], dtype = np.dtype('f8'))

        # use a human like face----red lips, white to brown skin complexion.(not necessarily, whatever works by trial and error)
        if self.flag == 'brightest':
            self.recolored_image = scaled_gray
        else:
            self.recolored_image = np.array([[[100%pixel,70%pixel,150%pixel] if pixel>=5 else [0%pixel,0%pixel,0%pixel] for pixel in row] for row in scaled_gray], dtype = np.dtype('f8'))

        self.recolored_image = cv2.resize(self.recolored_image, (1024, 1024))

        cv2.imwrite('recolored_image.png', self.recolored_image)
        print('lol')

        return self.recolored_image

    def run(self):
        self.crop_image()
        self.create_border()
        recolored_image = self.recolor_image()
        return recolored_image


if __name__ == "__main__":
    image_processor = ImageProcessor("bet_darker_2.png")
    cropped_face = image_processor.crop_image()
    bordered_face = image_processor.create_border()
    recolored_face = image_processor.recolor_image()
    # cv2.imshow('cropped_face', cropped_face)
    # cv2.imshow('bordered_face', bordered_face)
    # cv2.imshow('recolored_face', recolored_face)

    # cv2.waitKey()
    # cv2.destroyAllWindows()
    pass


    


    


