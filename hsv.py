# differentiate images according to HSV value
import cv2

flags = ['brightest', 'darkest', 'in-between']
nose_crop_points = ((650, 250), (975, 100))
cheek_crop_points = ((880, 250), (550, 100))

def crop_area(image, crop_type='nose'):
    # image = cv2.imread(image_path)
    if crop_type=='nose':
        start_crop_point_y, stop_crop_point_y = nose_crop_points[0][0], nose_crop_points[0][0] + nose_crop_points[0][1]
        start_crop_point_x, stop_crop_point_x = nose_crop_points[1][0], nose_crop_points[1][0] + nose_crop_points[1][1]
        cropped_strip = image[start_crop_point_y:stop_crop_point_y, start_crop_point_x:stop_crop_point_x]
    if crop_type=='cheek':
        start_crop_point_y, stop_crop_point_y = cheek_crop_points[0][0], cheek_crop_points[0][0] + cheek_crop_points[0][1]
        start_crop_point_x, stop_crop_point_x = cheek_crop_points[1][0], cheek_crop_points[1][0] + cheek_crop_points[1][1]
        cropped_strip = image[start_crop_point_y:stop_crop_point_y, start_crop_point_x:stop_crop_point_x]

    return cropped_strip

def categorize_strip(cropped_strip):
    hsv_image = cv2.cvtColor(cropped_strip, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv_image)
    brightness = hsv_image[...,2].mean()
    if brightness>180: 
        flag = flags[0]
    elif brightness<85:
        flag = flags[1]
    else:
        flag = flags[2]
    
    return (flag, brightness)

def categorize_image(image):
    cropped_strip = crop_area(image, 'nose')
    flag, brightness_nose =  categorize_strip(cropped_strip)
    # print(flag, brightness_nose)
    if flag == 'in-between':
        cropped_cheek = crop_area(image, 'cheek')
        flag, brightness_cheek =  categorize_strip(cropped_cheek)
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







