import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os.path

import glob

PRE_TRAINED_MODEL_WEIGHTS_FILE_PATH = 'pre_trained_model/model_weights.h5'
CSV_FILE_NAME_FINETUNING = 'recorded_data_finetuning/driving_log.csv'
CSV_FILE_NAME = 'recorded_data/driving_log.csv'
IMAGES_PATH = 'recorded_data/IMG/'
IMAGES_PATH_FINETUNING = 'recorded_data_finetuning/IMG/'

IS_FINETUNING = False

if IS_FINETUNING:
	assert os.path.exists(PRE_TRAINED_MODEL_WEIGHTS_FILE_PATH), 'IS_FINETUNING is set to True, but the file "pre_trained_model/model_weights.h5" does not exist.'
	assert os.path.exists(CSV_FILE_NAME_FINETUNING), 'IS_FINETUNING is set to True, but the file "recorded_data_finetuning/driving_log.csv" does not exist.'

def right_transform(img):
    w_height = img.shape[0]
    w_width = img.shape[1]

    offset = 40   

    src = np.float32([[0,offset],[w_width,offset],[w_width, w_height-offset],[0, w_height -offset]])
    dst = np.float32([[0,offset],[w_width,0],[w_width, w_height],[0, w_height-offset]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w_width, w_height), flags=cv2.INTER_LINEAR)

    return warped

def left_transform(img):
    w_height = img.shape[0]
    w_width = img.shape[1]

    offset = 40   

    src = np.float32([[0,offset],[w_width,offset],[w_width, w_height-offset],[0, w_height -offset]])
    dst = np.float32([[0,0],[w_width,offset],[w_width, w_height-offset],[0, w_height]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w_width, w_height), flags=cv2.INTER_LINEAR)

    return warped

# nummber of images increase x (10/3)
def augmentation(images, steering_angles):
	'''Left and right camera image are transformed to simulate a di
	'''
	augmented_images = []
	augmented_steering_angles = []

	correction = 0.3

	count = 0
	for image, angle in zip(images, steering_angles):
		if (count % 3) == 1: # left camera image
			lt_image = left_transform(image)
			augmented_images.append(lt_image)
			augmented_steering_angles.append(angle + correction)

			flipped_image = cv2.flip(lt_image, 1)
			flipped_angle = (angle+correction) * -1
			augmented_images.append(flipped_image)
			augmented_steering_angles.append(flipped_angle)

		elif (count % 3) == 2: # right camera image
			rt_image = right_transform(image)
			augmented_images.append(rt_image)
			augmented_steering_angles.append(angle - correction)

			flipped_image = cv2.flip(rt_image, 1)
			flipped_angle = (angle-correction) * -1
			augmented_images.append(flipped_image)
			augmented_steering_angles.append(flipped_angle)

		count += 1

	
	for image, angle in zip(images, steering_angles):
		augmented_images.append(image)
		augmented_steering_angles.append(angle)

		flipped_image = cv2.flip(image, 1)
		flipped_angle = angle * -1
		augmented_images.append(flipped_image)
		augmented_steering_angles.append(flipped_angle)

		

	return augmented_images, augmented_steering_angles

def load_batch_data(batch_samples):
	if IS_FINETUNING:
		path = IMAGES_PATH_FINETUNING
	else:
		path = IMAGES_PATH

	images = []
	angles = []

	for batch_sample in batch_samples:
		center_angle = float(batch_sample[3])

		#center camera
		name = path + batch_sample[0].split('\\')[-1]
		center_image = cv2.imread(name)
		images.append(center_image)

		#left camera
		name = path + batch_sample[1].split('\\')[-1]
		left_image = cv2.imread(name)
		images.append(center_image)

		#right camera
		name = path + batch_sample[2].split('\\')[-1]
		right_image = cv2.imread(name)
		images.append(center_image)

		#append center, left and right angle
		correction = 0.3
		angles.append(center_angle)
		angles.append(center_angle + correction)
		angles.append(center_angle - correction)

	return images, angles


def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				
				#load images and angles
				images, angles = load_batch_data(batch_samples)

				#augment 
				images, angles = augmentation(images, angles)

			X_train = np.array(images)
			y_train = np.array(angles)

			yield shuffle(X_train, y_train)


def load_CSV(file_name):
	list = []

	with open(file_name) as csvfile:
	    reader = csv.reader(csvfile)
	    for row in reader:
	    	list.append(row)

	print('======================================')
	print('Data statistics\n')
	print('Number of RAW samples: %i' % len(list))
	print('Number of images: %i' % (len(list)*3))
	print('Augmented no images: %i' % (len(list)*10))
	print('======================================')

	return list


if(IS_FINETUNING):
	samples = load_CSV(CSV_FILE_NAME_FINETUNING)
else:
	samples = load_CSV(CSV_FILE_NAME)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 -0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70, 26), (0, 0))))

# nvidias end-to-end self-driving-car CNN
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))


if IS_FINETUNING:
	model.load_weights(PRE_TRAINED_MODEL_WEIGHTS_FILE_PATH)

model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*10, validation_data=validation_generator, nb_val_samples=len(validation_samples)*10, nb_epoch=5)

model.save('model.h5')
model.save_weights('model_weights.h5')
