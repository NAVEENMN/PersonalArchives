Regression and Convolutional neural network based Face keypoints detection
=============

files:
facekey.py : Two layer neural net
facekey_CNN.py : Two layer neural net with Three layer CNN

## Dataset

These models were trained using dataset from Kaggle.
https://www.kaggle.com/c/facial-keypoints-detection/leaderboard

download them and save them in two directories training/, test/

The training dataset for the Facial Keypoint Detection challenge consists of 7,049 96x96 gray-scale images. For each image, we're supposed learn to find the correct position (the x and y coordinates) of 15 keypoints, such as left_eye_center, right_eye_outer_corner, mouth_center_bottom_lip, and so on.
![Alt text](/images/input.jpg?raw=true "Training with key points")

## Training
Training was done on two models. One using just two layer nnet with an optimizer and other with CNN. 

## Testing

After Traning for a test sample. The network outputs the pixel locations of where it things those facekey points might be

![Alt text](/images/outout.jpg?raw=true "Testing on samples")
