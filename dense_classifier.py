from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation
from keras.callbacks import TensorBoard


class Classifier:
    def __init__(self, constants_file=None):
        self._model = None

        if constants_file is not None:
            # load model with constants
            pass

    def model(self, input_length, n_labels):
        # This returns a tensor
        inputs = Input(shape=(input_length,), name='input')

        # a layer instance is callable on a tensor, and returns a tensor
        x = Dense(500, activation='relu')(inputs)
        x = Dense(500, activation='relu')(x)
        predictions = Dense(n_labels, activation='softmax')(x)

        # This creates a model that includes
        # the Input layer and three Dense layers
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self, data, validation_data, labels):
        model = self.model(input_length=len(data[0]),
                           n_labels=len(labels[0]))
        tensorboard_callback = TensorBoard(log_dir='./test_logdir',
                                           histogram_freq=0,
                                           write_graph=True,
                                           write_images=False,
                                           write_grads=True,
                                           embeddings_freq=1,
                                           embeddings_layer_names=['input'])
        model.fit(data,
                  labels,
                  validation_data=validation_data,
                  callbacks=[tensorboard_callback],
                  epochs=3)
