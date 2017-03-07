* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model_visual]: ./examples/cnn-architecture-768x1095.png "Model Visualization"
[right_image]: ./examples/placeholder_small.png "Right camera image"
[t_right_image]: ./examples/placeholder_small.png "Image transformed"
[rec_image1]: ./examples/placeholder_small.png "Recovery Image"
[rec_image2]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
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

Data is normalized in the model using a Keras lambda layer (code line 194). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Train/validation/test splits are made (code line 179).

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (code line 212).

#### Appropriate training data
Training data consist of images from all three cameras (center, left, right). There was one value for steering angle so an angle for left and right images was calculated with a correction of (+/-) 0.3. This teach the network to recover from poor position.

Driving training for recovery of poor orientation has been done.

Training data have been augmented with flipped images. This so the network gets equal amount of left and right steering.

Had some struggle before I realized that driving the track perfectly during training was a bad idea. The car got biased to drive straight I guess, not knowing how to recover when entering a curve.
With this in mind a wobbled the car from side to side for one lap, and then added another trying to stay in the middle. Great relief to see the car finally get around. Not a beauty sight, but stable.

Nvidias approach, as I see it, was to augment using shift and rotation to teach how to recover from poor position and orientation. I imagined driving a real car like I did, in my first succesful attempt teaching it to drive, and felt strongly I wanted to do better. Since perspective transformation is brought up in P4, I looked into it, and added transformation(rotation) into my augmentation pipe. 

Pretty sure the network do not need as much recory training as before. A future test would be with a larger test-set to see if recovey training can be done with augmentaton only.


###Model Architecture and Training Strategy

####1. Solution Design Approach

I started with designing my tensorflow network from P2, but Keras made everything very simple, so it was not much fun.

Afer I read a blod post at Nvidia, [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) , I decided to try on their achritecture.

Early on I tried solving the bad driving with a larger training-set. I did not take long before generators was needed because of high memory usage.
Finally I found out that it was possible to keep the car on the road with a very small training-set, so long it was a good set with a lot of recovery practise. All it took was some wobbling over the road, back and forth.

To reduce the need of wobbling, transformation was put in the augmentation pipe.

The final step was to experiment with creating a good training set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 197-206)
(image is from the blog post mentioned above)

![Model architecture][model_visual]

####3. Creation of the Training Set & Training Process

I first recorded two laps on track one using center lane driving.
Example image:

![ex_center_drive][center_image]

I then drove half a lap, wobbling back and forth to learn the car how to recover from bad orientation.

![recover training][rec_image1]
![recover training][rec_image2]
![recover training][rec_image3]

After the collection process, I had 3388 number of data points (10164 images).


To augmention pipeline consist of flipping all images and transform the left and right camera images.

Flipped example:
![image1][image1]
![flipped_image1][flipped_image1]

Transormation example:
![right image][right_image]
![transormed right image][t_right_image]

Total size of training-set (after augmentation) = 33880 images


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

Through trial and error, number of epochs was set to 5.
