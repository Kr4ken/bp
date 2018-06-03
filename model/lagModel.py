import time
from model.baseModel import baseModel
import seaborn as sns
sns.despine()
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras import regularizers
import os


class lagModel(baseModel):
    fileName = 'Lag.h5'
    name = 'Lag'

    def __init__(self, model_filepath, data,steps_per_epoch = 0,epochs = 1,refresh = False):
        super(lagModel, self).__init__(model_filepath, data, steps_per_epoch, epochs, refresh)

    def build_network(self):
        return self.model

    def __fit_model_threaded(self):
        print('> Тренируем модель ' + self.name)
        print('> Тут сеть не строится, так что и билдить нечего, возвращаем None')
        return self.model

    def get_network(self):
        if (not self.refresh and os.path.isfile(self.filename)):
            return self.__load_network()
        else:
            return self.__fit_model_threaded()

    def get_predictions_true_data_with_norm(self):
        true_values = []
        predictions = []
        norms =[]

        data_gen_test = self.model_data.get_generator_clean_data_test_with_norm()

        for x, y, n in data_gen_test:
            if len(x) == 0:
                break
            true_values += list(y)
            norms += list(n)

        predictions = [true_values[0]] + true_values[:-1]

        true_values = [(a+1)*b for a, b in zip(true_values, norms)]
        predictions = [(a+1)*b for a, b in zip(predictions, norms)]

        return predictions, true_values

    def get_predictions_true_data(self):
        true_values = []
        predictions = []

        data_gen_test = self.model_data.get_generator_clean_data_test()

        for x, y in data_gen_test:
            if len(x) == 0:
                break
            true_values += list(y)

        predictions += [true_values[0]] + true_values[:-1]

        return predictions, true_values