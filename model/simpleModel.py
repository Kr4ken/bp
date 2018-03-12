import time
from model.baseModel import baseModel
import seaborn as sns
sns.despine()
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras import regularizers


class simpleModel(baseModel):
    fileName = 'Straight.h5'
    name = 'Straight'
    loss_function = ''
    optimiser_function = ''
    layers =[]

    def __init__(self, model_filepath, data,steps_per_epoch = 0,epochs = 1,refresh = False):
        super(simpleModel, self).__init__(model_filepath,data,steps_per_epoch,epochs,refresh)
        self.loss_function = 'mse'
        self.optimiser_function = 'Nadam'
        self.layers = [self.model_data.ncols, 150, 150, 1]

    def build_network(self):
        super(simpleModel, self).build_network()
        self.model = Sequential()
        self.model.add(Dense(
            input_dim=4,
            output_dim=64,
                        activity_regularizer=regularizers.l2(0.01)))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Dense(output_dim = 16,
                        activity_regularizer=regularizers.l2(0.01)))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Dense(1))
        self.model.add(Activation('linear'))

        start = time.time()
        self.model.compile(
            loss=self.loss_function,
            optimizer=self.optimiser_function)

        print("> Время компиляции : ", time.time() - start)
        return self.model
