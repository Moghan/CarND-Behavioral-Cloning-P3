import csv
import cv2
import numpy as np


def load_data():
	images = []
	steering_angles = []

	recorded_data_file = 'recorded_data/driving_log.csv'

	list = []

	with open(recorded_data_file) as csvfile:
	    reader = csv.reader(csvfile)
	    for row in reader:
	    	list.append(row)

	

	path = 'recorded_data\IMG\\'


	for row in list:
		# get all images center, left and right
		full_source_path = row[0]
		img_file_name = full_source_path.split('\\')[-1]
		image = cv2.imread(path + img_file_name)
		images.append(image)

		full_source_path = row[1]
		img_file_name = full_source_path.split('\\')[-1]
		image = cv2.imread(path + img_file_name)
		images.append(image)

		full_source_path = row[2]
		img_file_name = full_source_path.split('\\')[-1]
		image = cv2.imread(path + img_file_name)
		images.append(image)

		# get all labels (steering angles)
		steering_angle = float(row[3])
		correction = 0.2
		
		steering_angles.append(steering_angle)
		steering_angles.append(steering_angle + correction)
		steering_angles.append(steering_angle - correction)

	print ('===========================')
	print ('Rawdata loaded:')
	print ('%i Images' % len(images))
	print ('%i Labels' % len(steering_angles))
	print ('---------------------------')

	print(path + img_file_name)
	print(images[0])
	print(image)

	return images, steering_angles

def augmentation(images, steering_angles):
	augmented_images = []
	augmented_steering_angles = []

	for image, angle in zip(images, steering_angles):
		augmented_images.append(image)
		augmented_steering_angles.append(angle)

		flipped_image = cv2.flip(image, 1)
		flipped_angle = angle * -1
		augmented_images.append(flipped_image)
		augmented_steering_angles.append(flipped_angle)

	print ('===========================')
	print ('Data augmented:')
	print ('%i Images' % len(augmented_images))
	print ('%i Labels' % len(augmented_steering_angles))
	print ('---------------------------')

	return augmented_images, augmented_steering_angles


images, steering_angles = load_data()
augmented_images, augmented_steering_angles = augmentation(images, steering_angles)

X_train = np.array(augmented_images)
y_train = np.array(augmented_steering_angles)

# X_train = np.array(images)
# y_train = np.array(steering_angles)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

print ('Imported Keras')

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 -0.5, input_shape = (160, 320, 3)))
model.add(Convolution2D(6, 5, 5, activation = 'relu'))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
