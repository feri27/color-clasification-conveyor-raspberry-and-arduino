#!/usr/bin/python
from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import itemfreq


def color_histogram_of_training_image(img_name):

    # mendeteksi folder gambar sebagai label
    if 'matang' in img_name:
        data_source = 'matang'
    elif 'kosong' in img_name:
        data_source = 'kosong'
    elif 'mentah' in img_name:
        data_source = 'mentah'

    # load the image
    image = cv2.imread(img_name)

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    with open('training.csv', 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')


    # matang color training images
for f in os.listdir('./TRAINING_IMAGE/matang'):
      color_histogram_of_training_image('./TRAINING_IMAGE/matang/' + f)

    # sedang color training images
for f in os.listdir('./TRAINING_IMAGE/kosong'):
      color_histogram_of_training_image('./TRAINING_IMAGE/kosong/' + f)

    # mentah color training images
for f in os.listdir('./TRAINING_IMAGE/mentah'):
      color_histogram_of_training_image('./TRAINING_IMAGE/mentah/' + f)

