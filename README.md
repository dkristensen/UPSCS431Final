# UPSCS431Final
# Authors
 * [Drew Kristensen](https://github.com/dkristensen)
 * [Patrick Ryan](https://github.com/pjryan513)
# Project Goal
The goal of this project is to predict the number of sea lions in an aerial photograph using a convolutional neural network. This is a relaxed version of the [Kaggle competition]( https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/data) sponsored by the NOAA. This serves as the final project for the CS 431 - Artificial Intelligence class at the University of Puget Sound. You can read our final write up of the project [here](./resources/paper.pdf)

## Packages Used
 * [Keras](https://keras.io/)
 * [Theano](http://deeplearning.net/software/theano/)
 * [NumPy](http://www.numpy.org/)
 * [SciPy](https://www.scipy.org/)

## Process
To accomplish our goal of identifying and enumerating the number of sea lions in the images, we used a convolution neural network with the Keras framework with the Theano backend. We naively search the image by passing a sliding window over the image, capturing a 64 by 64 pixel window of the image and overlapping 1/4 of the previous window to make sure we don't miss any sea lions (ie shift 48 pixels after each window test). From this, we can make an estimate on the quantity of sea lions in the image given how many positive results our CNN output and where they were located in relation to each other.

### Future Steps
Ideally, we would like to continue the project by downloading and training on the majority of the data that was made available by the NOAA. We were unable to utilize this data due to internet usage constraints, lack of space on our personal hard drives, and time.
