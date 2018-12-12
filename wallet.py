import numpy as np
import cv2
import time
import os
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import model_from_json

FRAME_WIDTH = 720
path = '/home/sebastien/Euros_Recognition/scans'

# load json and create model
json_file = open('model_test3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_test3.h5")
print("Loaded model from disk")
optimizer = Adam(lr=1e-3)
loaded_model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])


def collect_euro_coin(img):
	"""Collect the euro coins from an image."""

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Adaptive Thresholding
	gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
	thresh = cv2.adaptiveThreshold(gray_blur, 255,
	cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

	# Circle detection
	circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 64, param1=20, param2=40, minRadius=24, maxRadius=96)

	if circles is not None:
		i = 0
		count = 0
		for c in circles[0]:
			cimg = img[int(c[1]-c[2]):int(c[1]+c[2]), int(c[0]-c[2]):int(c[0]+c[2])]
			cimg = cv2.resize(cimg, (64, 64))
			#gimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
			data = cimg.reshape(1,64, 64,3)
			model_out = loaded_model.predict([data])
			
			if np.argmax(model_out) == 0:
				title='1cent'
				count += 0.01
			elif np.argmax(model_out) == 1:
				title='2cent'
				count += 0.02
			elif np.argmax(model_out) == 2:
				title='5cent'
				count += 0.05
			elif np.argmax(model_out) == 3:
				title='10cent'
				count += 0.10
			elif np.argmax(model_out) == 4:
				title='20cent'
				count += 0.20
			elif np.argmax(model_out) == 5:
				title='50cent'
				count += 0.50
			elif np.argmax(model_out) == 6:
				title='1eur'
				count += 1
			elif np.argmax(model_out) == 7:
				title='2eur'
				count += 2
						
			name = (title + "." + str(int((time.time() - 1400000000) * 1000))[-4:] + str(int(time.clock() * 100000000000))[-4:] + str(i))	
			cv2.imwrite(os.path.join(path,  name +'.jpeg'), cimg)
			i += 1


# Start the video camera and show the detection results in real-time.

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
	ret, frame = cap.read()
	height, width, depth = frame.shape
	roi = cv2.resize(frame, (FRAME_WIDTH, FRAME_WIDTH * height / width))
	cv2.imshow('Video', roi)
	
	if cv2.waitKey(1) & 0xFF == ord(' '):
		collect_euro_coin(roi)
		print('Il y a ' + count + '€ dans ce porte monnaie')
	

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
