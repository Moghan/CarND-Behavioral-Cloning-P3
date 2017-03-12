[//]: # (Image References)

[model_visual]: ./examples/cnn-architecture-768x1095.png "Model Visualization"
[right_image]: ./examples/right_image.jpg "Right camera image"
[rec_image3]: ./examples/rec_image1.jpg "Recovery Image"
[rec_image1]: ./examples/rec_image2.jpg "Recovery Image"
[rec_image2]: ./examples/rec_image3.jpg "Recovery Image"
[center_image]: ./examples/center_image.jpg "Center Image"
[flipped_right_image]: ./examples/flipped_right_image.jpg "Flipped Image"

[orig_image]: ./examples/origin.jpg "Original image"
[hls_image]: ./examples/hls.jpg "HLS image"
[s_channel]: ./examples/s_channel.jpg "S-channel"
[gaussian_blurred]: ./examples/gaussian.jpg "Gaussian blurred S-channel"

## Writeup for P3 - Behavioral Cloning
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
and run the simulator in autonomous mode.  

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy
####1. An appropriate model architecture has been employed

Data is normalized and cropped in the model using Keras lambda layers (code line 159-160).


####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Train/validation/test splits are made (code line 144).

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (code line 177).

#### Appropriate training data
Nvidias approach, which also had cameras on the side , was to augment using shift and rotation to teach how to recover from poor position and orientation.

I started with the noble goal that I would do the same, but in the end, my training data is almost only images from recovering.

During training (and from Nvidias blog) I also found out that to much training data from driving straight will make the network biased to drive straight. It seems like the network forgets how to produce higher steering angles.

Training data have been augmented with flipped images. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I started with designing my tensorflow network from P2, but Keras made everything very simple, so it was not much fun.

After I read a blog post at Nvidia, [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) , I decided to try on their achritecture. This network architecture is most likely way overdimensioned, but time is up and the quest for the smallest possible network will be a future project.

Early on I tried solving the bad driving with increasing the training-set. I did not take long before generators was needed because of high memory usage.
Finally I found out that it was possible to keep the car on the road with a very small training-set, as long it was a good set with a lot of recovery practise. All it took was some wobbling over the road, back and forth.

Setting up the network was the easy part. Implementing generators, a smaller speedbumb. The main issue was to realize how to create a good set of training data.

At the end of the process, I had problem with the left curve, just after crossing the bridge. There it seemed like the main road and the dirt road looked to much the same. I got a tip from my mentor to play with the images, so I set everything up to use the s-channel of a HLS image instead, and added Gaussian blur to them. Like magic, the network now performed excellent.

####2. Final Model Architecture

The final model architecture (model.py lines 158-171)

This image is from Nvidias blog post mentioned above:

![Model architecture][model_visual]

####3. Creation of the Training Set & Training Process

Almost all training-set images are from the vehicle recovering to the middle. Some smooth driving is done in the curves. I small part of the training-set is from "wobbling", using the full width of the road with maximum steering angle.

Example of recovery training from the left side of the road, getting back to the middle:

![recover training][rec_image1]
![recover training][rec_image2]
![recover training][rec_image3]

The training process created 9421 data points and three images for every data point.
Total size of training-set after augmentation is 56526 images

The augmentation pipeline consist of flipping all images.

Flipped example:
![image1][right_image]
![flipped_image1][flipped_right_image]


Images are preprocessed to make the road stand out from the surroundings more clearly.

1. Convert from BGR to HLS
2. Filter out S-channel
3. Add a Gaussian blur

Original image:

![original image][orig_image]

Converted to HLS:

![image converted to HLS][hls_image]

S-channel of HLS image:

![S-channel of HLS image][s_channel]

S-channel with Gaussian blur:

![blurred S-channel][gaussian_blurred]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

Through trial and error, number of epochs was set to 5.
