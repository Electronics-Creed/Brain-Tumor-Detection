import cv2
import tensorflow as tf
import numpy as np
from brain_img_preprocess import crop_brain_contour

model = tf.keras.models.load_model('C:\\Users\\shobh\\PycharmProjects\\Brain Tumor project\\brain_tumor_model.h5')
model.load_weights('C:\\Users\\shobh\\PycharmProjects\\Brain Tumor project\\brain_tumor_weights.h5')

# test_img = cv2.imread('C:\\Users\\shobh\\PycharmProjects\\ML projects\\Brain-Tumor-Detection-master\\yes\\Y104.jpg')
test_img = cv2.imread('C:\\Users\\shobh\\PycharmProjects\\ML projects\\Brain-Tumor-Detection-master\\no\\31 no.jpg')

cv2.imshow('Image', test_img)

test_img = crop_brain_contour(test_img)
test_img = cv2.resize(test_img, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
test_img = test_img / 255.

test_img = np.expand_dims(test_img, axis=0)

pred = model.predict(test_img)

if pred >= 0.9:
    print('Tumor found!!')
else:
    print('No tumour found!!')

cv2.waitKey(0)
cv2.destroyAllWindows()
