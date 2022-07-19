"""
Gender Classification utilizing a Convolutional Neural Network (CNN).
"""
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd


class GenderClassificationCNN:
    ATTR_TRUE = 1
    ATTR_FALSE = -1
    MALE = 1
    FEMALE = 0

    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.cnn = self.create_cnn()
        self.compile()

    def classify(self, image_dir, attributes):
        """ Convenience function that runs all necessary steps to perform gender classification

        Args:
            image_dir (str): The directory containing the image data.
            attributes (dataframe): The dataframe containing the image attributes.

        Returns:
            the history object from fitting the CNN model, The evaluation results of the trained CNN
        """
        test, train = self.prepare_data(image_dir, attributes)
        history = self.fit(test, train)
        results = self.evaluate(test)

        return history, results

    def create_cnn(self):
        model = Sequential()
        model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))  # Input layer
        model.add(MaxPooling2D((2, 2)))
        model.add(Convolution2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())  # Allow for fully connected layer
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(1, activation='sigmoid'))  # Output Layer

        print(model.summary())

        return model

    def compile(self):
        self.cnn.compile(optimizer='adam', loss='binary_crossentropy',  metrics=['accuracy'])

    def evaluate(self, test):
        """ Evaluates the model

        Args:
            test: The validation image generator

        Returns:
            the results of evaluating the model
        """
        test.reset()
        results = self.cnn.evaluate(test)
        return results

    def fit(self, train, test, epochs=25):
        """ Fits the model

        Args:
            train: Training image data generator
            test: Validation image data generator
            epochs: Number of epochs to perform while fitting

        Returns:
            the result values of fitting the model
        """
        history = self.cnn.fit(
            train,
            validation_data=test,
            epochs=epochs,
            callbacks=[EarlyStopping(patience=10)]
        )

        return history

    def prepare_data(self, image_directory, attributes):
        # Update values to be binary
        attributes.loc[attributes['Male'] == GenderClassificationCNN.ATTR_TRUE, ["Male"]] = str(GenderClassificationCNN.MALE)
        attributes.loc[attributes['Male'] == GenderClassificationCNN.ATTR_FALSE, ["Male"]] = str(GenderClassificationCNN.FEMALE)

        # Define Image Data Generators
        train_data_generator = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.25,
        )

        train_generator = train_data_generator.flow_from_dataframe(
            attributes,
            directory=image_directory,
            x_col='image_id',
            y_col='Male',
            target_size=(64, 64),
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True,
            seed=42,
            subset='training'
        )

        test_generator = train_data_generator.flow_from_dataframe(
            attributes,
            directory=image_directory,
            x_col='image_id',
            y_col='Male',
            target_size=(64, 64),
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True,
            seed=42,
            subset='validation'
        )

        return train_generator, test_generator


if __name__ == '__main__':
    image_dir = 'C:/Users/Cole/Downloads/img_align_celeba/img_align_celeba'
    attr_file = 'C:/Users/Cole/Downloads/img_align_celeba/list_attr_celeba.csv'

    data = pd.read_csv(attr_file)
    data = data.sample(10000)

    gc = GenderClassificationCNN()
    gc.classify(image_dir, data)

