"""
Gender Classification utilizing Histogram of Gradients (HOG).
"""
import os
import pandas as pd
from skimage.feature import hog
from skimage.io import imread
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from sklearn.svm import SVC


class GenderClassificationHOG:
    ATTR_TRUE = 1
    ATTR_FALSE = -1
    MALE = 1
    FEMALE = 0

    def __init__(self):
        self.svc = SVC(kernel='linear', C=1)

    def create_hog(self, image):
        img = imread(image)
        img = resize(img, (64, 64))

        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                            visualize=True, channel_axis=2) # multichannel=True)

        return fd

    def classify(self, image_directory, attributes):
        hog_images, labels = self.prepare_data(image_directory, attributes)

        x_train, x_test, y_train, y_test = train_test_split(hog_images, labels)

        self.train_svm(x_train, y_train)
        prediction = self.predict(x_test)

        self.summary(y_test, prediction)

        return prediction

    def predict(self, test):
        return self.svc.predict(test)

    def prepare_data(self, image_directory, attributes):
        hog_images = []

        attributes.loc[attributes['Male'] == GenderClassificationHOG.ATTR_FALSE, ["Male"]] = GenderClassificationHOG.FEMALE
        labels = attributes['Male'].tolist()
        image_names = attributes['image_id'].tolist()

        for index, image in enumerate(image_names):
            hog_images.append(self.create_hog(os.path.join(image_directory, image)))

            divisor = 10
            if (index + 1) % (len(image_names) / divisor) == 0:
                print(f"Processed {index + 1} images.")

        return hog_images, labels

    def summary(self, y_test, prediction):
        """ Outputs a summary of the prediction of the svm

        Args:
            y_test:
            prediction: prediction values returned from the svm
        """
        report = classification_report(y_test, prediction, target_names=[f'Male - {GenderClassificationHOG.MALE}',
                                                                         f'Female - {GenderClassificationHOG.FEMALE}'])
        print(report)

        return report

    def train_svm(self, x_train, y_train):
        """ Trains the svm with the given data

        Args:
            x_train (list): List of image features
            y_train (list): List of image labels
        """
        self.svc.fit(x_train, y_train)


if __name__ == '__main__':
    image_dir = 'C:/Users/Cole/Downloads/img_align_celeba/img_align_celeba'
    attr_file = 'C:/Users/Cole/Downloads/img_align_celeba/list_attr_celeba.csv'

    data = pd.read_csv(attr_file)
    data = data.sample(25000)

    gc = GenderClassificationHOG()
    gc.classify(image_dir, data)
