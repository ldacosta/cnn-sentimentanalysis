import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Merge
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers import merge, add


class SentimentAnalysisAnalyzer(object):

    def __init__(self, features_in_words, words_in_review):
        self.n_features_in_word = features_in_words
        self.n_words_in_review = words_in_review
        self.input_channels = 1 # no "color" channels since this is not a picture

    def fit(self, x_train, y_train, batch_size, epochs):
        raise NotImplementedError("'fit' has to be implemented")

    def evaluate(self, x, y, batch_size):
        raise NotImplementedError("'evaluate' has to be implemented")


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import Model

class WithKeras(SentimentAnalysisAnalyzer):

    def __init__(self, features_in_words, words_in_review):

        SentimentAnalysisAnalyzer.__init__(self, features_in_words, words_in_review)
        # inputs
        inputs = Input(shape=(self.n_words_in_review, self.n_features_in_word, self.input_channels))
        # intermediate results
        # 2D 3x3 convolution followed by a maxpool
        branch1 =Conv2D(
                filters=5,
                kernel_size=(3, self.n_features_in_word),
                activation='relu',
                input_shape=(self.n_words_in_review, self.n_features_in_word, self.input_channels),
                strides=(1, 1),
                padding='valid')(inputs)
        branch1 = MaxPooling2D(pool_size=(2, 1), strides = (1,1))(branch1)
        branch1 = Dropout(0.25)(branch1)

        # 2D 4x4 convolution followed by a maxpool
        branch2 =Conv2D(
                filters=5,
                kernel_size=(4, self.n_features_in_word),
                activation='relu',
                input_shape=(self.n_words_in_review, self.n_features_in_word, self.input_channels),
                strides=(1, 1),
                padding='valid')(inputs)
        branch2 = MaxPooling2D(pool_size=(2, 1), strides = (1,1))(branch2)
        branch2 = Dropout(0.25)(branch2)

        # now let's go fully-connected to a 2-way classification:
        merged_branches = keras.layers.concatenate([branch1, branch2], axis = 1) # , axis=-1)

        # fully-connected
        fully_connected = Flatten()(merged_branches)
        fully_connected = Dense(256, activation='relu')(fully_connected)
        fully_connected = Dropout(0.5)(fully_connected)

        # ok. done
        categorization = Dense(2, activation='softmax')(fully_connected)

        # all good. Let's build the model, then
        self.model = Model(inputs=inputs, outputs=categorization)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], optimizer=sgd)

    def summary(self):
        return self.model.summary()

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

