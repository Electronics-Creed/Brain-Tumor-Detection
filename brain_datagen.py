from keras.preprocessing.image import ImageDataGenerator
import cv2
from os import listdir


def augment_data(file_dir, n_generated_samples, save_to_dir):
    data_gen = ImageDataGenerator(rotation_range=10,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.1,
                                  brightness_range=(0.3, 1.0),
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest'
                                  )

    for filename in listdir(file_dir):
        image = cv2.imread(file_dir + '\\' + filename)
        image = image.reshape((1,) + image.shape)
        save_prefix = 'aug_' + filename[:-4]
        i = 0
        for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir,
                                   save_prefix=save_prefix, save_format='jpg'):
            i += 1
            if i > n_generated_samples:
                break


augmented_data_path = 'C:\\Users\\shobh\\PycharmProjects\\ML projects\\Brain-Tumor-Detection-master\\augmented data\\'

yes_path = augmented_data_path + 'yes'
no_path = augmented_data_path + 'no'

augment_data(file_dir=yes_path, n_generated_samples=6, save_to_dir=augmented_data_path + 'yes')

augment_data(file_dir=no_path, n_generated_samples=9, save_to_dir=augmented_data_path + 'no')
