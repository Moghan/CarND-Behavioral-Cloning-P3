import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

CSV_file_name = 'recorded_data/driving_log.csv'

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

	return augmented_images, augmented_steering_angles

def load_batch_data(batch_samples):
	images = []
	angles = []

	for batch_sample in batch_samples:
		center_angle = float(batch_sample[3])

		#center camera
		name = './recorded_data/IMG/'+batch_sample[0].split('\\')[-1]
		center_image = cv2.imread(name)
		images.append(center_image)
		# print('name' + name)
		# print(center_image.shape)

		#left camera
		name = './recorded_data/IMG/'+batch_sample[1].split('\\')[-1]
		left_image = cv2.imread(name)
		images.append(center_image)

		#right camera
		name = './recorded_data/IMG/'+batch_sample[2].split('\\')[-1]
		right_image = cv2.imread(name)
		images.append(center_image)

		#append center, left and right angle
		correction = 0.2
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
				# center_angle = float(batch_sample[3])

				# #center camera
				# name = './recorded_data/IMG/'+batch_sample[0].split('\\')[-1]
				# center_image = cv2.imread(name)
				# images.append(center_image)
				# # print('name' + name)
				# # print(center_image.shape)

				# #left camera
				# name = './recorded_data/IMG/'+batch_sample[1].split('\\')[-1]
				# left_image = cv2.imread(name)
				# images.append(center_image)

				# #right camera
				# name = './recorded_data/IMG/'+batch_sample[2].split('\\')[-1]
				# right_image = cv2.imread(name)
				# images.append(center_image)

				# #append center, left and right angle
				# correction = 0.2
				# angles.append(center_angle)
				# angles.append(center_angle + correction)
				# angles.append(center_angle - correction)

				#load images and angles
				images, angles = load_batch_data(batch_samples)

				#augment 
				# images, angles = augmentation(images, angles)


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
	print('Augmented no images: %i' % (len(list)*6))

	return list


samples = load_CSV(CSV_file_name)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


train_generator = generator(train_samples)
validation_generator = generator(validation_samples)


import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

print ('Imported Keras')

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 -0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# nvidias end-to-end self-driving-car
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(24, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

# model.add(Convolution2D(24, 3, 3, activation = 'relu'))
# model.add(Flatten())
# model.add(Dense(50))
# model.add(Dense(1))


model.compile(optimizer='adam', loss='mse')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*3, validation_data=validation_generator, nb_val_samples=len(validation_samples)*3, nb_epoch=3)

model.save('model.h5')
