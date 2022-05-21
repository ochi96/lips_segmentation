import cv2
import os
import numpy as np
from PIL import Image


import torch
import torchvision.transforms as transforms


from model import BiSeNet
# from image_preprocessing import ProcessImage

class SegmentFace(object):
	def __init__(self, original_image_path, cropped_image_path, processed_image_path) -> None:
		self.__original_image_path = original_image_path
		self.__cropped_image_path = cropped_image_path
		self.__processed_image_path = processed_image_path
		self.__table = {'upper_lip': 12,
						'lower_lip': 13,
						'left_eyebrow': 2,
						'right_eyebrow': 3}
		self.__cp = 'models/79999_iter.pth'
		self.__segmentation_color = [0, 200, 200]
		self.__mask_background_color = (255, 255, 255)
		self.__n_classes = 19
		self.__results_dir = 'results'
		
		if not os.path.exists(self.__results_dir):
			os.mkdir(self.__results_dir)
		
		try:
			self.__original_image = Image.open(self.__original_image_path).convert('RGB')
			self.__cropped_image = Image.open(self.__cropped_image_path).convert('RGB')
			self.__processed_image = Image.open(self.__processed_image_path).convert('RGB')
			self.__image_name = os.path.basename(self.__original_image_path).split('.')[0]
		except FileNotFoundError:
			print('Image not found. Please enter a valid path.')
			quit()

	
	def mask(self, image, parsing, part=17):
		b, g, r = self.__segmentation_color     
		tar_color = np.zeros_like(image)
		tar_color[:, :, 0] = r
		tar_color[:, :, 1] = g
		tar_color[:, :, 2] = b

		image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)
		image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]

		changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
		changed[parsing != part] = image[parsing != part]
		
		return changed

	def run(self):
		image = cv2.resize(np.array(self.__original_image), (1536, 1536))

		image_mask = image.copy()
		image_mask = cv2.rectangle(image_mask, (0, 0), (1080, 1080), self.__mask_background_color, thickness = 1080)
		
		parsing = self.evaluate()
		
		parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)
		# cv2.imshow('lol', parsing.astype(np.float32))

		# cv2.waitKey()
		# cv2.destroyAllWindows()

		for part in self.__table.values():
			image = self.mask(image, parsing, part)									#mask over the original photo
			image_mask = self.mask(image_mask, parsing, part)
		
		cv2.imwrite(f'{self.__results_dir}/{self.__image_name}_mask.jpg', image_mask)
		cv2.imwrite(f'{self.__results_dir}/{self.__image_name}_segmented.jpg', image)


		print(f'Segmentation  of {self.__image_path_orig} complete! Results in {self.__results_dir}')
		pass

	def evaluate(self):
		net = BiSeNet(self.__n_classes)
		net.load_state_dict(torch.load(self.__cp, map_location = 'cpu'))
		net.eval()

		to_tensor = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])

		with torch.no_grad():
			image = self.__image_trans.resize((512, 512), Image.BILINEAR)     #with the processed photo
			img = to_tensor(image)
			img = torch.unsqueeze(img, 0)
			out = net(img)[0]
			parsing = out.squeeze(0).cpu().numpy().argmax(0)
		
		return parsing

	
	def overlay_image(self):
		original_image = Image.open(self.__original_image_path).convert('RGBA')
		cropped_image = Image.open(self.__cropped_image_path).convert('RGBA')
		starting_position = (256, 256)

		self.final_image = original_image.copy()
		self.final_image.paste(cropped_image, starting_position)
		# self.final_image.save('final.png', quality=100)

		return self.final_image



	
	
	
	
