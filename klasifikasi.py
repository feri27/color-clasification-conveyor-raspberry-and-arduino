#!/usr/bin/python
import cv2
import knn
import os
import os.path
import numpy as np

cap = cv2.VideoCapture(0)
(ret, frame) = cap.read()
prediction = 'n.a.'


def color_histogram_of_test_image(test_src_image):

    # load gambar dari  frame
    image = test_src_image

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

    with open('recent.file', 'w') as myfile:
        myfile.write(feature_data)
    print(feature_data)



while True:

    # mengcapture frame realtime
    (ret, frame) = cap.read()

    cv2.putText(
        frame,
        'Prediksi: ' + prediction,
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)

    # window output
    cv2.imshow('Sistem Pendeteksi warna kematangan tomat', frame)

    color_histogram_of_test_image(frame)

    prediction = knn.main('training.csv', 'recent.file')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()		
