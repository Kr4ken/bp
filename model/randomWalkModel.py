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

    # def get_multistep_predictions_true_data(self,steps=1):
    #     true_values = []
    #     multi_predictions=[]
    #     data_gen_test = self.model_data.get_generator_clean_data_test()
    #
    #     steps_test = int(self.model_data.ntest / self.model_data.batch_size)
    #     print('> Тестируем модель на ', self.model_data.ntest, ' строках с ', steps_test, ' шагами')
    #
    #     for x,y in data_gen_test:
    #         if len(x) == 0:
    #             break
    #         for i in range(len(x)-steps):
    #             true_value=[]
    #             multi_prediction = []
    #             x_curr = x[i]
    #             y_curr = y[i]
    #             for step in range(steps):
    #                 prediction = self.model.predict(x_curr.reshape(1,self.x_window,1))
    #                 x_curr = x_curr[1:]
    #                 x_curr = np.append(x_curr, prediction)
    #                 y_curr = y[i+step]
    #                 true_value.append(y_curr)
    #                 multi_prediction.append(prediction)
    #             true_values.append(true_value)
    #             multi_predictions.append(multi_prediction)
    #
    #     return multi_predictions, true_values
    #
    # def get_multistep_predictions_true_data_with_norm(self,steps=1):
    #     true_values = []
    #     multi_predictions=[]
    #     norms = []
    #     data_gen_test = self.model_data.get_generator_clean_data_test_with_norm()
    #
    #     steps_test = int(self.model_data.ntest / self.model_data.batch_size)
    #     print('> Тестируем модель на ', self.model_data.ntest, ' строках с ', steps_test, ' шагами')
    #
    #     for x,y,n in data_gen_test:
    #         if len(x) == 0:
    #             break
    #         norms+=list(n)
    #         for i in range(len(x)-steps):
    #             true_value=[]
    #             multi_prediction = []
    #             x_curr = x[i]
    #             y_curr = y[i]
    #             for step in range(steps):
    #                 prediction = self.model.predict(x_curr.reshape(1,self.x_window,1))
    #                 # x_curr = x_curr[1:] + [prediction]
    #                 x_curr = x_curr[1:]
    #                 # x_curr.put(self.x_window-2,prediction)
    #                 x_curr = np.append(x_curr,prediction)
    #                 y_curr = y[i+step]
    #                 true_value.append(y_curr)
    #                 multi_prediction.append(prediction)
    #             true_values.append(true_value)
    #             multi_predictions.append(multi_prediction)
    #
    #     multi_predictions = [(a+1)*b for a,b in zip(multi_predictions,norms)]
    #     true_values = [(a+1)*b for a,b in zip(true_values,norms)]
    #     return multi_predictions, true_values




    def get_predictions_true_data_with_norm(self):
        true_values = []
        predictions = []
        norms = []

        data_gen_test = self.model_data.get_generator_clean_data_test_with_norm()

        for x, y, n in data_gen_test:
            if len(x) == 0:
                break
            true_values += list(y)
            norms += list(n)
            lx = list(x)[:-1]
            lx1 = list(x)[1:]
            diff =[abs(abs(lx[k]) - abs(lx1[k])) for k in range(len(lx))]
            mean = math.sqrt(np.array(diff).mean())
            predictions += [np.random.normal(self.mu,mean,1)[0] + list(pred)[-1] for pred in x]
            # predictions += [np.random.normal(self.mu,mean,1)[0] + list(x)[-1]]

        true_values = [(a+1)*b for a, b in zip(true_values,norms)]
        predictions = [(a+1)*b for a, b in zip(predictions,norms)]
        return predictions, true_values


    def get_predictions_true_data(self):
        true_values = []
        predictions = []

        data_gen_test = self.model_data.get_generator_clean_data_test()

        for x, y in data_gen_test:
            if len(x) == 0:
                break
            true_values += list(y)
            lx = list(x)[:-1]
            lx1 = list(x)[1:]
            diff =[abs(abs(lx[k]) - abs(lx1[k])) for k in range(len(lx))]
            mean = math.sqrt(np.array(diff).mean())
            predictions += [np.random.normal(self.mu,mean,1)[0] + list(pred)[-1] for pred in x]
            # predictions += [np.random.normal(self.mu,mean,1)[0] + list(x)[-1]]

        return predictions, true_values


    # Использовать только для случая, когда y_window=1
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
                x_curr = x[i]
                y_curr = y[i]
                for step in range(steps):
                    print(step, end='\r')
                    lx = list(x_curr)[:-1]
                    lx1 = list(x_curr)[1:]
                    diff = [abs(abs(lx[k]) - abs(lx1[k])) for k in range(len(lx))]
                    mean = math.sqrt(np.array(diff).mean())
                    prediction = [np.random.normal(self.mu,mean,1)[0] + list(x_curr)[-1]]
                    x_curr = x_curr[1:]
                    x_curr = np.append(x_curr, prediction)
                    # x_curr += prediction
                    y_curr = y[i+step]
                    true_value.append(y_curr)
                    multi_prediction.append(prediction)
                true_values.append(true_value)
                multi_predictions.append(multi_prediction)

        return multi_predictions, true_values

    # Использовать только для случая, когда y_window=1
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
                x_curr = x[i]
                y_curr = y[i]
                n_curr = n[i]
                for step in range(steps):
                    print(step, end='\r')
                    lx = list(x_curr)[:-1]
                    lx1 = list(x_curr)[1:]
                    diff = [abs(abs(lx[k]) - abs(lx1[k])) for k in range(len(lx))]
                    mean = math.sqrt(np.array(diff).mean())
                    prediction = list([np.random.normal(self.mu, mean, 1)[0] + list(x_curr)[-1]])
                    x_curr = x_curr[1:]
                    x_curr = np.append(x_curr, prediction)
                    # x_curr += prediction
                    y_curr = y[i+step]
                    true_value.append([(a+1)*b for a, b in zip(y_curr, n_curr)])
                    multi_prediction.append([(a+1)*b for a, b in zip(prediction, n_curr)])
                # true_values.append([(a+1)*b for a,b in zip(true_value,n_curr)])
                # multi_predictions.append([(a+1)*b for a,b in zip(multi_prediction,n_curr)])
                true_values.append(true_value)
                multi_predictions.append(multi_prediction)

        return multi_predictions, true_values