from keras.layers import Dense, Activation, Dropout, Conv1D, LSTM, MaxPooling1D
from keras.layers import Flatten, Conv2D, MaxPooling2D, GRU, BatchNormalization
from keras.layers import Conv3D, MaxPool3D, Reshape, Input, AveragePooling2D
from keras.models import Sequential, load_model, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.losses import CategoricalCrossentropy
from utils import *
import tensorflow as tf


class SequentialModel:
    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, config):
        """ Virtual Function """
        return

    def train(self, x, y, x_val, y_val, config, save_dir):
        """ Virtual Function """
        return

    def evaluate(self, x_test, y_test, verbose=1):
        return self.model.evaluate(x_test,  y_test, verbose=verbose)

    def predict(self, x_test, verbose=1):
        return self.model.predict(x_test, verbose=verbose)


class SimpleGRUConv(SequentialModel):
    def __init__(self):
        super(SimpleGRUConv, self).__init__()

    def build_model(self, config):
        # replace hardcoded dimensions with config dictionary
        model = self.model
        model.add(GRU(22, input_shape=(config['input_shape'][0], config['input_shape'][1]), return_sequences=True))
        model.add(Conv1D(22, 10))
        model.add(Flatten())
        model.add(Dropout(config['dropout']))
        model.add(Dense(4, activation='softmax'))
        optimizer = Adam(learning_rate=config['lr'], beta_1=0.9, beta_2=0.999,
                         amsgrad=False)
        model.compile(optimizer=optimizer,
                      loss=CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.summary()
        print("Model compiled.")

    def train(self, x, y, x_val, y_val, config, save_dir):
        ensure_dir(save_dir)
        file_path = join(save_dir, 'SimpleGRUConv_best_val.hdf5')
        cp_callback = ModelCheckpoint(filepath=file_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max', verbose=0)

        history = self.model.fit(x, y, epochs=config['epochs'], batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                 callbacks=[cp_callback])

        return history


class ConvMixGRU(SequentialModel):
    def __init__(self):
        super(ConvMixGRU, self).__init__()

    def build_model(self, config):
        input_shape = config['input_shape']
        model = self.model
        model.add(Conv1D(22, 10,
                         input_shape=(input_shape[0], input_shape[1]),
                         kernel_regularizer=l2(config['l2'])))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling1D(2))
        if config['LSTM']:
            model.add(LSTM(44, kernel_regularizer=l2(config['l2']), return_sequences=True))
        else:
            model.add(GRU(44, kernel_regularizer=l2(config['l2']), return_sequences=True))
        model.add(Dropout(config['dropout']))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dropout(config['dropout']))
        model.add(Dense(4, activation='softmax'))
        optimizer = Adam(learning_rate=config['lr'], beta_1=0.9, beta_2=0.999,
                         amsgrad=False)
        model.compile(optimizer=optimizer,
                      loss=CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.summary()
        print("Model compiled.")

    def train(self, x, y, x_val, y_val, config, save_dir):
        ensure_dir(save_dir)
        file_path = join(save_dir, 'ConvMixGRU_best_val.hdf5')
        cp_callback = ModelCheckpoint(filepath=file_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max', verbose=0)

        history = self.model.fit(x, y, epochs=config['epochs'],
                                 batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                 callbacks=[cp_callback], verbose=1)
        return history


class SimpleGridModel(SequentialModel):
    def __init__(self):
        super(SimpleGridModel, self).__init__()

    def build_model(self, config):
        input_shape = config['input_shape']
        # input_shape is (None, rows, cols, timesteps)
        model = self.model

        model.add(Conv3D(32, kernel_size=(3, 2, 4), strides=(1, 1, 2),
                         input_shape=(input_shape[1],
                                      input_shape[2],
                                      input_shape[3], 1)))
        model.add(BatchNormalization(axis=2))
        model.add(MaxPool3D(pool_size=(2, 2, 1)))
        model.add(Conv3D(1, kernel_size=(1, 1, 1)))
        model.add(Reshape(target_shape=(
        model.layers[-1].output_shape[1] * model.layers[-1].output_shape[2],
        model.layers[-1].output_shape[3])))
        model.add(GRU(20))
        model.add(Dense(32))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='softmax'))
        optimizer = Adam(learning_rate=config['lr'], beta_1=0.9, beta_2=0.999,
                         amsgrad=False)
        model.compile(optimizer=optimizer,
                      loss=CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.summary()
        print("Model compiled.")

    def train(self, x, y, x_val, y_val, config, save_dir):
        ensure_dir(save_dir)
        file_path = join(save_dir, 'SimpleGrid_best_val.hdf5')
        cp_callback = ModelCheckpoint(filepath=file_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max', verbose=0)

        history = self.model.fit(x, y, epochs=config['epochs'],
                                 batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                 callbacks=[cp_callback])
        return history


class SingleFrameFourier(SequentialModel):
    def __init__(self):
        super(SingleFrameFourier, self).__init__()

    def build_model(self, config):
        if config['MLP']:
            input_shape = config['input_shape']
            model = self.model
            #model.add(Flatten())
            model.add(Reshape(input_shape=(input_shape[1], input_shape[2], input_shape[3]),
                              target_shape=((input_shape[1]*input_shape[2]*input_shape[3],))))
            #model.add(Dense(126, activation='relu'))
            model.add(Dense(100, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(100, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(4, activation='softmax'))
            optimizer = Adam(learning_rate=config['lr'], beta_1=0.9, beta_2=0.999,
                             amsgrad=False)
            model.compile(optimizer=optimizer,
                          loss=CategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            model.summary()

        else:
            input_shape = config['input_shape']
            # input_shape is (batch, rows, cols, channel)
            model = self.model
            model.add(Conv2D(128, kernel_size=(2, 2), input_shape=(
            input_shape[1], input_shape[2], input_shape[3])))
            print(model.layers[-1].output_shape)
            # kernel_regularizer=l2(config['l2'])))
            original_shape = model.layers[-1].output_shape
            model.add(Reshape(
                (original_shape[1] * original_shape[2], original_shape[3])))
            model.add(BatchNormalization(axis=2))
            model.add(Reshape((original_shape[1],
                               original_shape[2],
                               original_shape[3])))
            model.add(Conv2D(64, kernel_size=(
            2, 2)))  # , kernel_regularizer=l2(config['l2'])))
            # model.add(Conv2D(64, kernel_size=(2, 2)))
            model.add(MaxPooling2D())
            model.add(Flatten())
            model.add(Dropout(0.2))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(4, activation='softmax'))

            # input_shape = config['input_shape']
            # # input_shape is (batch, rows, cols, channel)
            # model = self.model
            # model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(
            # input_shape[1], input_shape[2], input_shape[3])))
            # model.add(Conv2D(32, kernel_size=(2, 2)))
            # model.add(MaxPooling2D())
            # model.add(Flatten())
            # model.add(Dropout(0.5))
            # model.add(Dense(128))
            # model.add(Dropout(0.5))
            # model.add(Dense(4, activation='softmax'))
            # model.summary()




            optimizer = Adam(learning_rate=config['lr'],
                                                 beta_1=0.9, beta_2=0.999,
                                                 amsgrad=False)
            model.compile(optimizer=optimizer,
                          loss=CategoricalCrossentropy(
                              from_logits=True),
                          metrics=['accuracy'])
            model.summary()

    def train(self, x, y, x_val, y_val, config, save_dir):
        ensure_dir(save_dir)
        file_path = join(save_dir, 'SimpleFourier_best_val.hdf5')
        cp_callback = ModelCheckpoint(filepath=file_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max', verbose=0)

        history = self.model.fit(x, y, epochs=config['epochs'],
                                 batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                 callbacks=[cp_callback])
        return history


class SpatioTempCNN(SequentialModel):
    def __init__(self):
        #super(SpatioTempCNN, self).__init__()
        super().__init__()
        self.model = tf.keras.models.Sequential()

    def build_model(self, config):
        model = self.model
        model.add(tf.keras.layers.Reshape((22, 100, 1), input_shape=(22, 100)))
        model.add(tf.keras.layers.Conv2D(16, (3, 1), activation='elu',kernel_regularizer=l2(0.04)))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.MaxPooling2D((2, 1)))
        model.add(tf.keras.layers.Conv2D(16, (1, 10), activation='elu',kernel_regularizer=l2(0.04)))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.MaxPooling2D((1, 2)))
        model.add(tf.keras.layers.Conv2D(16, (2, 1), activation='elu',kernel_regularizer=l2(0.04)))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Conv2D(16, (1, 10), activation='elu',kernel_regularizer=l2(0.04)))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.MaxPooling2D((1, 2)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu',
                               kernel_regularizer=l2(0.03)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(4))

        optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr'], beta_1=0.9, beta_2=0.999,
                         amsgrad=False)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.summary()

    def train(self, x, y, x_val, y_val, config, save_dir):
        ensure_dir(save_dir)
        file_path = join(save_dir, 'SpatioTempCNN.hdf5')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max', verbose=0)

        history = self.model.fit(x, y, epochs=config['epochs'],
                                 batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                 callbacks=[cp_callback])
        return history


class AvgPoolCNN(SequentialModel):
    def __init__(self):
        super(AvgPoolCNN, self).__init__()

    def build_model(self, config):

        inputs = Input(shape=(22, 100))

        layer = inputs

        layer = Reshape((22, 100, 1))(layer)

        layer = Conv2D(48, (1, 10), activation='elu',
                       kernel_regularizer=l2(config['l2']))(layer)
        layer = BatchNormalization()(layer)
        # layer = Conv2D(15, (1, 15), activation='elu',kernel_regularizer=regularizers.l2(0.02))(layer)
        layer = Dropout(0.1)(layer)
        layer = Conv2D(40, (22, 1), activation='elu',
                       kernel_regularizer=l2(config['l2']))(layer)
        layer = BatchNormalization()(layer)
        # layer = Conv2D(14, (22, 1), activation='elu')(layer)
        layer = AveragePooling2D((1, 25), strides=(1, 4))(layer)

        layer = Flatten()(layer)

        layer = Dense(4)(layer)
        outputs = Activation('softmax')(layer)
        self.model = Model(inputs=inputs, outputs=outputs)

        optimizer = Adam(learning_rate=config['lr'], beta_1=0.9, beta_2=0.999,
                         amsgrad=False)
        self.model.compile(optimizer=optimizer,
                      loss=CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        self.model.summary()
        print("Model compiled.")

    def train(self, x, y, x_val, y_val, config, save_dir):
        ensure_dir(save_dir)
        file_path = join(save_dir, 'AvgPoolCNN_best_val.hdf5')
        cp_callback = ModelCheckpoint(filepath=file_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max', verbose=0)

        history = self.model.fit(x, y, epochs=config['epochs'],
                                 batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                 callbacks=[cp_callback])
        return history


class VanillaRNN(SequentialModel):
    def __init__(self):
        super(VanillaRNN, self).__init__()

    def build_model(self, config):
        # replace hardcoded dimensions with config dictionary
        model = self.model
        if config['LSTM']:
            model.add(LSTM(22, input_shape=(config['input_shape'][0], config['input_shape'][1]), return_sequences=True))
        else:
            model.add(GRU(22, input_shape=(
            config['input_shape'][0], config['input_shape'][1]),
                           return_sequences=True))
        model.add(Flatten())
        model.add(Dropout(config['dropout']))
        model.add(Dense(4, activation='softmax'))
        optimizer = Adam(learning_rate=config['lr'], beta_1=0.9, beta_2=0.999,
                         amsgrad=False)
        model.compile(optimizer=optimizer,
                      loss=CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.summary()
        print("Model compiled.")

    def train(self, x, y, x_val, y_val, config, save_dir):
        ensure_dir(save_dir)
        file_path = join(save_dir, 'VanillaGRU_best_val.hdf5')
        cp_callback = ModelCheckpoint(filepath=file_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max', verbose=0)

        history = self.model.fit(x, y, epochs=config['epochs'], batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                 callbacks=[cp_callback])

        return history