# differentiate images according to HSV value

import cv2

flags = ['brightest', 'darkest', 'in-between']
nose_crop_points = ((650, 250), (975, 100))
cheek_crop_points = ((880, 250), (550, 100))


class CheckBrightness():

    def __init__(self, image, crop_type='nose') -> None:
        self.flags = ['brightest', 'darkest', 'in-between']
        self.nose_crop_points = ((650, 250), (975, 100))
        self.cheek_crop_points = ((880, 250), (550, 100))
        self.image = image
        self.crop_type = crop_type
        self.face_area = self.nose_crop_points
        if self.crop_type=='cheek':
            self.face_area = self.cheek_crop_points
        pass

    def crop_area(self):
        start_crop_point_y, stop_crop_point_y = self.face_area[0][0], self.face_area[0][0] + self.face_area[0][1]
        start_crop_point_x, stop_crop_point_x = self.face_area[1][0], self.face_area[1][0] + self.face_area[1][1]
        self.cropped_strip = self.image[start_crop_point_y:stop_crop_point_y, start_crop_point_x:stop_crop_point_x]

        return self.cropped_strip

    def categorize_strip(self):
        hsv_image = cv2.cvtColor(self.cropped_strip, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(hsv_image)
        self.brightness = hsv_image[...,2].mean()
        if self.brightness>180: 
            self.flag = flags[0]
        elif self.brightness<85:
            self.flag = flags[1]
        else:
            self.flag = flags[2]
        
        return self.flag, self.brightness



def categorize_image(image):
    check_nose = CheckBrightness(image)
    check_nose.crop_area()
    flag, brightness_nose =  check_nose.categorize_strip()

    if flag == 'in-between':
        check_cheek = CheckBrightness(image, "cheek")
        check_cheek.crop_area()
        flag, brightness_cheek =  check_cheek.categorize_strip()
        # print(flag, brightness_cheek)
        if brightness_cheek < brightness_nose-20:
            flag = 'darker_inb'
        else:
            flag = 'lighter_inb'
    return flag


if __name__ == '__main__':
    image_path = f"images/originals/lol4.png"
    flag = categorize_image(image_path)
    print(flag)
    pass







