"""
Use transfer learning on VGG-16 pretrained on ImageNet for gender classification task.
"""

from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd


class GenderClassification:
    ATTR_TRUE = 1
    ATTR_FALSE = -1

    def __init__(self, batch_size=32, num_images=20000, split=.25):
        # Initialize VGG model
        self.vgg = self.define_vgg()
        self.compile()

        self.training_size = int(num_images * split)
        self.test_size = int(num_images * split)
        self.batch_size = batch_size

    def compile(self):
        """ Compile the constructed model """
        self.vgg.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy'])

    def define_vgg(self):
        """ Defines the model using vgg as a base and adds fully connected hidden and output layers """
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
        # Freeze Layers
        for layer in vgg.layers:
            layer.trainable = False

        # add flatten layer, so we can add the fully connected layer later
        x = Flatten()(vgg.output)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(vgg.input, x)

        # Output model structure
        print(model.summary())
        return model

    def evaluate(self, test, step_size=500):
        """ Evaluates the model

        Args:
            test: The validation image generator
            step_size (int): Step size to take when evaluating

        Returns:
            the results of evaluating the model
        """
        test.reset()
        results = self.vgg.evaluate(test, steps=step_size)
        return results

    def fit(self, train, test, epochs):
        """ Fits the model

        Args:
            train: Training image data generator
            test: Validation image data generator
            epochs: Number of epochs to perform while fitting

        Returns:
            the result values of fitting the model
        """
        history = self.vgg.fit(
            train,
            validation_data=test,
            validation_steps=self.test_size // self.batch_size,
            steps_per_epoch=self.training_size // self.batch_size,
            epochs=epochs,
            callbacks=[EarlyStopping(patience=10)]
        )

        return history

    def plot(self, history):
        """ Plots the accuracy of the epochs of fitting the model

        Args:
            history: results from model fitting
        """
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def output_evaluate(self, eval):
        """ Outputs the results of the model evaluation

        Args:
            eval: The model's evaluation results
        """
        print(f"Evaluate: {self.vgg.metrics_names[1], eval[1] * 100}")

    def prepare_data(self, image_directory, attribute_file):
        """ Loads random sample of images from the given directory and tracks images classification via
        keras ImageDataGenerator.

        Args:
            image_directory (str): Directory to extract images from
            attribute_file (str): CSV file containing file attributes

        Returns:
            list, list: image features, image labels
        """
        # Read in attribute file and update Male column values
        data = pd.read_csv(attribute_file)
        data.loc[data['Male'] == GenderClassification.ATTR_TRUE, ["Male"]] = 'male'
        data.loc[data['Male'] == GenderClassification.ATTR_FALSE, ["Male"]] = 'female'

        # Define Image Data Generators
        train_data_generator = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.25,
        )

        train_generator = train_data_generator.flow_from_dataframe(
            data,
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
            data,
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
    # Dataset information
    image_path = 'C:/Users/Cole/Downloads/img_align_celeba/img_align_celeba'
    attribute_file = 'C:/Users/Cole/Downloads/img_align_celeba/list_attr_celeba.csv'

    gc = GenderClassification()
    train_gen, test_gen = gc.prepare_data(image_path, attribute_file)
    hist = gc.fit(train_gen, test_gen, 10)
    gc.plot(hist)
    eval = gc.evaluate(test_gen)
    gc.output_evaluate(eval)

    # Save Model
    gc.vgg.save('model.h5')
