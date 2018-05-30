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


    def get_predictions_true_data(self):
        true_values = []

        data_gen_test = self.model_data.get_generator_clean_data()

        def generator_strip_xy(data_gen, true_values):
            for x, y in data_gen:
                true_values += list(y)
                yield x

        steps_test = int(self.model_data.ntest / self.model_data.batch_size)
        print('> Тестируем модель на', self.model_data.ntest, 'строках с', steps_test, 'шагами')
        temp = list(generator_strip_xy(data_gen_test,true_values))
        predictions = [true_values[0]] + true_values[:-1]
        # predictions = self.model.predict_generator(
        #     generator_strip_xy(data_gen_test, true_values),
        #     steps=steps_test
        # )

        return predictions, true_values