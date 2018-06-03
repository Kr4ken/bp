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

    def get_multistep_predictions_true_data_with_norm(self,steps=1):
        true_values = []
        multi_predictions=[]
        data_gen_test = self.model_data.get_generator_clean_data_test_with_norm()

        steps_test = int(self.model_data.ntest / self.model_data.batch_size)
        print('> Тестируем модель на ', self.model_data.ntest, ' строках с ', steps_test, ' шагами')
        for x,y,n in data_gen_test:
            if len(x) == 0:
                break
            for i in range(len(x)-steps):
                true_value=[]
                multi_prediction = []
                n_curr = n[i]
                for step in range(steps):
                    prediction = y[i]
                    y_curr = y[i+step]
                    # true_value.append(y_curr)
                    true_value.append([(a+1)*b for a, b in zip(y_curr,n_curr)])
                    # multi_prediction.append(prediction)
                    multi_prediction.append([(a+1)*b for a, b in zip(prediction,n_curr)])
                true_values.append(true_value)
                # true_values.append([(a+1)*b for a,b in zip(true_value,n_curr)])
                multi_predictions.append(multi_prediction)
                # multi_predictions.append([(a+1)*b for a,b in zip(multi_prediction,n_curr)])

        return multi_predictions, true_values

    def get_multistep_predictions_true_data(self,steps=1):
        true_values = []
        multi_predictions=[]
        data_gen_test = self.model_data.get_generator_clean_data_test()

        steps_test = int(self.model_data.ntest / self.model_data.batch_size)
        print('> Тестируем модель на ', self.model_data.ntest, ' строках с ', steps_test, ' шагами')
        for x,y in data_gen_test:
            if len(x) == 0:
                break
            for i in range(len(x)-steps):
                true_value=[]
                multi_prediction = []
                for step in range(steps):
                    prediction = y[i]
                    y_curr = y[i+step]
                    true_value.append(y_curr)
                    multi_prediction.append(prediction)
                true_values.append(true_value)
                multi_predictions.append(multi_prediction)

        return multi_predictions, true_values
