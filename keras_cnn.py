import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import tensorflow as tf


class SentimentAnalysisAnalyzer(object):

    def __init__(self, features_in_words, words_in_review):
        self.n_features_in_word = features_in_words
        self.n_words_in_review = words_in_review
        self.input_channels = 1 # no "color" channels since this is not a picture

    def fit(self, x_train, y_train, batch_size, epochs):
        raise NotImplementedError("'fit' has to be implemented")

    def evaluate(self, x, y, batch_size):
        raise NotImplementedError("'evaluate' has to be implemented")

class WithKeras(SentimentAnalysisAnalyzer):

    def __init__(self, features_in_words, words_in_review):

        SentimentAnalysisAnalyzer.__init__(self, features_in_words, words_in_review)

        # # n_features_in_word = mw.model.vector_size
        # n_features_in_word = 300 # TODO: replace this by previous line
        # n_words_in_review = 10 # number of words to take from each review

        self.model = Sequential()
        # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
        # this applies 1 convolution filter(s) of size 3x3 each.
        filter_depth = 5
        self.model.add(Conv2D(filter_depth, (3,  self.n_features_in_word), activation='relu', input_shape=(self.n_words_in_review, self.n_features_in_word, self.input_channels), strides=(1, 1), padding='valid'))
        self.model.add(MaxPooling2D(pool_size=(2, 1), strides = (1,1)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], optimizer=sgd)

    def fit(self, x_train, y_train, batch_size, epochs):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    def evaluate(self, x, y, batch_size):
        return self.model.evaluate(x, y, batch_size=batch_size)


if __name__ == "__main__":
    cnn_k = WithKeras(features_in_words=300, words_in_review=10)
    num_examples_training = 98
    x_train = np.random.random((num_examples_training, cnn_k.n_words_in_review, cnn_k.n_features_in_word, cnn_k.input_channels))
    y_train = keras.utils.to_categorical(np.random.randint(2, size=(num_examples_training, 1)), num_classes=2)
    num_examples_testing = 17
    x_test = np.random.random((num_examples_testing, cnn_k.n_words_in_review, cnn_k.n_features_in_word, cnn_k.input_channels))
    y_test = keras.utils.to_categorical(np.random.randint(2, size=(num_examples_testing, 1)), num_classes=2)
    cnn_k.fit(x_train, y_train, batch_size = 16, epochs = 11)
    score = cnn_k.evaluate(x_test, y_test, batch_size = 16)
    print("\n\n ====> score is {}".format(score))

