import time
from model.baseModel import baseModel
import math
import seaborn as sns
sns.despine()
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras import regularizers
import os
import numpy as np


class randomWalkModel(baseModel):
    fileName = 'RandomWalk.h5'
    name = 'RandomWalk'
    mu = 0
    sigma = 0.1

    def __init__(self, model_filepath, data,steps_per_epoch = 0,epochs = 1,refresh = False,mu=0,sigma=0.1):
        super(randomWalkModel, self).__init__(model_filepath, data, steps_per_epoch, epochs, refresh)
        self.mu = mu
        self.sigma = sigma

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
        predictions = []

        data_gen_test = self.model_data.get_generator_clean_data()

        for x, y in data_gen_test:
            if len(x) == 0:
                break
            true_values += list(y)
            lx = list(x)[:-1]
            lx1 = list(x)[1:]
            diff =[abs(abs(lx[k]) - abs(lx1[k])) for k in range(len(lx))]
            mean = math.sqrt(np.array(diff).mean())
            predictions += [np.random.normal(self.mu,mean,1)[0] + list(pred)[-1] for pred in x]

        return predictions, true_values