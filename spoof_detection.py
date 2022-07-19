"""
Fingerprint spoof detection system based on two-class support vector machine
"""

import numpy as np
import pathlib
from PIL import Image
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def convert_images(img_dir, img_size):
    """ Converts images in given directories to a list of 1D arrays

    Args:
        img_dir (pathlib.Path): Directory containing images to convert
        img_size (tuple): The size to resize images to

    Returns:
        numpy.array: resized, converted images as numpy arrays
    """
    print(f"Converting .png images in {img_dir}")
    converted = []
    for img in img_dir.glob("**/*.png"):
        img = Image.open(img)
        img = img.resize(img_size)
        img_data = np.array(img).reshape(-1)
        converted.append(img_data)

    return np.asarray(converted)


class SpoofDetection:
    LIVE = 1
    SPOOF = 0

    def __init__(self, training_live, training_spoof, testing_live, testing_spoof, image_size):
        # Create Training Data
        self.training_x = convert_images(training_live, image_size)
        self.training_y = [self.LIVE] * len(self.training_x)

        _spoof = convert_images(training_spoof, image_size)
        self.training_x = np.concatenate((self.training_x, _spoof))
        self.training_x.reshape(-1, 1)

        self.training_y.extend([self.SPOOF] * len(_spoof))

        # Create Testing Data
        self.testing_x = convert_images(testing_live, image_size)
        self.testing_y = [self.LIVE] * len(self.testing_x)

        _spoof = convert_images(testing_spoof, image_size)
        self.testing_x = np.concatenate((self.testing_x, _spoof))
        self.testing_x.reshape(-1, 1)

        self.testing_y.extend([self.SPOOF] * len(_spoof))

    def polynomial_kernel_svm(self, degree=3, coef0=1, c=1.0):
        """ Trains a polynomial kernel svm using the training data and then predicts and outputs the prediction metrics
        using the testing data.

        Args:
            degree (int): Degree of polynomial kernel function. Defaults to 3.
            coef0 (int): Independent term in kernel function. Defaults to 1.
            c (float): Regularization parameter. Defaults to 1.0.
        """
        poly_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="poly", degree=degree, coef0=coef0, C=c))])

        poly_kernel_svm_clf.fit(self.training_x, self.training_y)
        prediction = poly_kernel_svm_clf.predict(self.testing_x)

        print(f"Polynomial Kernel SVM {poly_kernel_svm_clf}:\n "
              f"{metrics.classification_report(self.testing_y, prediction, target_names=['spoof', 'live'])}\n")
        print(f"{prediction}\n")
        print(f"{self.testing_y}\n")


if __name__ == '__main__':
    spoof_detection = SpoofDetection(
        testing_live=pathlib.Path("Testing Biometrika Live/live"),
        testing_spoof=pathlib.Path("Testing Biometrika Spoof/Testing Biometrika Spoof/spoof"),
        training_live=pathlib.Path("Training Biometrika Live/live"),
        training_spoof=pathlib.Path("Training Biometrika Spoof/Training Biometrika Spoof/spoof"),
        image_size=(8, 8))

    spoof_detection.polynomial_kernel_svm(degree=3, coef0=1, c=1.0)

# output from python3 spoof_detection.py
#
# Polynomial Kernel SVM Pipeline(steps=[('scaler', StandardScaler()),
#                 ('svm_clf', SVC(C=1, coef0=1, kernel='poly'))]):
#                precision    recall  f1-score   support
#
#        spoof       0.90      0.98      0.94       200
#         live       0.98      0.89      0.93       200
#
#     accuracy                           0.94       400
#    macro avg       0.94      0.94      0.93       400
# weighted avg       0.94      0.94      0.93       400
