"""
Create a commitee of three neural networks using Fashion MNIST dataset.
Report their individual accuracy rates along with the final average accuracy of the ensemble.
Average accuracy of the ensemble will be obtained by averaging the probabilities of the three neural
networks for each test sample.
"""

import numpy
from sklearn.metrics import accuracy_score
from tensorflow import keras


class NeuralNetworks:
    def __init__(self):
        # Get Data
        fashion_mnist = keras.datasets.fashion_mnist
        (X_train_full, y_train_full), (self.X_test, self.y_test) = fashion_mnist.load_data()

        # Prepare/Reshape data
        self.X_train_full = X_train_full / 255.0

        # Split data
        self.X_valid, self.X_train = X_train_full[:5000], X_train_full[5000:]
        self.y_valid, self.y_train = y_train_full[:5000], y_train_full[5000:]

        # Data Classifications
        self.class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
                            "Shirt", "Sneaker", "Bag", "Ankle boot"]

    def model_1(self):
        """ Sequential NN with a single hidden layer

        Returns:
            created model
        """
        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=self.X_train.shape[1:]))
        model.add(keras.layers.Dense(256, activation="relu"))
        model.add(keras.layers.Dense(len(self.class_names), activation="softmax"))

        # Output summary of the network
        print(model.summary())

        # Configure the created model for training
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    def model_2(self):
        """ Sequential NN with two hidden layers

        Returns:
            created model
        """
        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=self.X_train.shape[1:]))
        model.add(keras.layers.Dense(256, activation="relu"))
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dense(len(self.class_names), activation="softmax"))

        # Output summary of the network
        print(model.summary())

        # Configure the created model for training
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    def model_3(self):
        """ Sequential model with two hidden layers each followed by BatchNormalization

        Returns:
            created model
        """
        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=self.X_train.shape[1:]))
        model.add(keras.layers.Dense(256, activation="relu"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(len(self.class_names), activation="softmax"))

        # Configure the created model for training
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    def train_model(self, model):
        """ Trains the given model, maximum of 100 epochs, utilizes EarlyStopping callback

        Args:
            model: The model to train.
        """
        model.fit(self.X_train, self.y_train, epochs=100, validation_data=(self.X_valid, self.y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])

    def evaluate_model(self, model):
        """ Evaluate the model

        Args:
            model: keras model to evaluate

        Returns:
            tuple: (accuracy, loss)
        """
        return model.evaluate(self.X_test, self.y_test)

    def predict_models(self, models):
        """ Predicts the input variables and returns the models accuracy.

        Args:
            models: Model(s) to predict with.

        Returns:
            float: Calculated accuracy of the model(s)
        """
        # Predict all models and create a numpy array
        values = numpy.array([model.predict(self.X_test) for model in models])

        # Sum across all model predictions
        summed_values = numpy.sum(values, axis=0)

        return accuracy_score(self.y_test, numpy.argmax(summed_values, axis=1))


if __name__ == "__main__":
    nn = NeuralNetworks()

    # Initialize models
    _models = [nn.model_1(), nn.model_2(), nn.model_3()]

    # Train all the models
    for _model in _models:
        nn.train_model(_model)

    # Validate Models
    print(f"\n{'*' * 50}")
    print(f"Validation{'-' * 40}")
    for i, _model in enumerate(_models, start=1):
        accuracy, loss = nn.evaluate_model(_model)
        print(f"Model {i}: Validation Loss: {loss}, Accuracy {accuracy}")

    # Predict Models By themselves
    print(f"Prediction{'-' * 40}")
    for i, _model in enumerate(_models, start=1):
        accuracy = nn.predict_models([_model])
        print(f"Model {i}: Accuracy {accuracy}")

    # Predict the Ensemble Accuracy
    print(f"Ensemble Accuracy: {nn.predict_models(_models)}")
