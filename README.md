# KNTU_CV_2021

This project was designed for the "Fundamentals of Computer Vision" course held at K. N. Toosi University of Technology in Spring 2021. You can access the course website [here](https://wp.kntu.ac.ir/nasihatkon/teaching/cvug/s2021/).

The purpose of this project was to allow students to use classic image processing and computer vision methods together with deep learning to achieve an acceptable results in the given task. 

In this project, many students worked with background subtraction, morphological operations, perspective transformation, labeling tools such as [labelImg](https://github.com/tzutalin/labelImg), convolution neural networks' training and testing with Keras, and the Google Colaboratory environment for training their model.

The full project description can be found in the PDF file on this repo. In addition, there is a basic implementation of the part 1 (player detection) in Python. The code was just a quick and simple implementation for me to test whether the goal can be achieved by students. 

Full implementation can be accessed from the sample implementations at the end of this document.
## Part 1

Given an input video of a soccer match from a static angle, students are asked to provide a 2D map of the field (bird's eye view). Therefore, they should implement a fairly accurate detection method, and then use perspective transform to project the positions on a 2D map. Another approach is to first warp the image with perspective transform, and then apply a detection method.

For this part, we suggested using background subtraction methods. Of course, students had to come up with ways to handle noise, scale, and irrelevant objects such as the ball.

## Part 2

Here, the students are asked to use Convolutional Neural Networks to classify the detected players in two teams. Of course, such model will be able to only descriminate the team colors available in the input dataset. However, the purpose of this part is to allow students to gain hands-on experience with data acquisition, training and testing.

## Part 3-Extra Points

We also offered multiple extra tasks for enthusiasts.
1. In addition to the players, classify the referee as well. 
2. Instead of running the detection method for every frame, run it in intervals and use object tracking methods for the intermediate frames.
3. Instead of a single angle, apply detection for all three angles of the field and gennerate a full 2D map, in which every player can be observed.

## Sample Implementations

Here are some of the best projects presented by the students:

1- [TroddenSpade/Soccer-Players-Tracking](https://github.com/TroddenSpade/Soccer-Players-Tracking) Parsa and Hashem had one of the most elaborate projects. Not only did they get full mark from the project, but they also implemented every bonus point.

2- [itsAliSali/soccerField](https://github.com/itsAliSali/soccerField) : Ali came up with a novel method handle to the variant size of the near and distant players. His detection results were among the most accurate ones.

3- [shamohamin/football_player_tracking](https://github.com/shamohamin/football_player_tracking) : Amin and Ghazal's detection and classification were among the best.
