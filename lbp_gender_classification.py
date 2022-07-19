""" Face-based Gender Classification based on Local Binary Patterns (LBP)"""
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from random import sample
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import local_binary_pattern
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import SVC



class GenderClassification:
    MALE = 0
    FEMALE = 1
    ATTR_TRUE = 1
    ATTR_FALSE = -1

    def __init__(self, images, method='uniform', radius=3, num_images=10000):
        self.images = images
        self.attributes_file = 'img_align_celeba/list_attr_celeba.txt'
        self.method = method
        self.radius = radius
        self.num_points = radius * 8
        self.num_images = num_images

        self.attributes = self.read_attributes()
        self.svc = SVC(kernel='rbf', C=1000, gamma=1)

    def classify(self):
        """ Convenience method to perform all necessary steps to train and predict the images """
        lbps, classification = gc.load_directory(self.images)

        # Split data for training / testing
        x_train, x_test, y_train, y_test = train_test_split(lbps, classification)

        self.train_svm(x_train, y_train)

        prediction = gc.predict(x_test)
        self.summary(y_test, prediction)

    def create_local_binary_pattern(self, image):
        """ Retrieves features of image using local binary pattern

        Args:
            image (str): path of the image to convert

        Returns:
            image features
        """
        img = imread(image)
        resized_img = resize(img, (178, 218))
        gray_image = rgb2gray(resized_img)
        lbp = local_binary_pattern(gray_image, self.num_points, self.radius, self.method)
        features = np.unique(lbp, return_counts=True)
        (_, features) = normalize(features, copy=False)
        return features

    def load_directory(self, image_directory):
        """ Loads random sample of images from the given directory and tracks images classification

        Args:
            image_directory (str): Directory to extract images from

        Returns:
            list, list: image features, image labels
        """
        lbps = []
        labels = []

        for index, infile in enumerate(sample(glob.glob(os.path.join(f"{image_directory}/*.jpg")), self.num_images)):
            if self.attributes.loc[f"{os.path.basename(infile)}", 'Male'] == GenderClassification.ATTR_TRUE:
                gender = GenderClassification.MALE
            else:
                gender = GenderClassification.FEMALE

            labels.append(gender)
            lbps.append(gc.create_local_binary_pattern(infile))

            divisor = 10 if self.num_images <= 10000 else 100
            if (index + 1) % (self.num_images / divisor) == 0:
                print(f"Processed {index + 1} images.")

        return lbps, labels

    def train_svm(self, x_train, y_train):
        """ Trains the svm with the given data

        Args:
            x_train (list): List of image features
            y_train (list): List of image labels
        """
        self.svc.fit(x_train, y_train)

    def predict(self, x_test):
        """ Predicts the given image features

        Args:
            x_test (list): Image features to predict

        Returns:
            list: predictions
        """
        return self.svc.predict(x_test)

    def read_attributes(self):
        """ Loads attribute file into a dataframe

        Returns:
            the attributes dataframe
        """
        return pd.read_csv(self.attributes_file, header=1, index_col=0, delim_whitespace=True)

    def summary(self, y_test, prediction):
        """ Outputs a summary of the prediction of the svm

        Args:
            y_test:
            prediction: prediction values returned from the svm
        """
        print(f"Prediction: {prediction}")

        print(confusion_matrix(y_test, prediction))
        print(classification_report(y_test, prediction, target_names=[f'Male - {GenderClassification.MALE}',
                                                                      f'Female - {GenderClassification.FEMALE}']))


if __name__ == '__main__':
    image_path = 'img_align_celeba/img_align_celeba'

    gc = GenderClassification(image_path)
    gc.classify()



