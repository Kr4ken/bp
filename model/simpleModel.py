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
            input_dim=self.model_data.x_window_size,
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
            optimizer=self.optimiser_function,
            metrics=['accuracy']
        )

        print("> Время компиляции : ", time.time() - start)
        return self.model

    def __fit_model_threaded(self):
        print('> Тренируем модель ' + self.name)
        print('АЛЯРМА, я нужный метод')
        self.model = self.build_network()
        output_file = self.filename
        print('> Информация по данным nrows =  ' + str(self.model_data.nrows) + ' ntrain = ' + str(self.model_data.ntrain) + ' ntest = ' + str(self.model_data.ntest))
        print('> Параметры модели epochs =  ' + str(self.epochs) + ' Шагов за эпоху = ' + str(self.steps_per_epoch) + ' validation steps ' + str(self.validation_steps))
        self.history = self.model.fit_generator(
            self.model_data.get_generator_clean_data_reshape(),
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs)
            # validation_data=self.model_data.get_generator_clean_data_test(),
            # validation_steps=self.validation_steps
        self.model.save(output_file)
        print('> Модель создана! веса сохранены ', output_file)
        return self.model

    def get_network(self):
        if (not self.refresh and os.path.isfile(self.filename)):
            return self.__load_network()
        else:
            return self.__fit_model_threaded()


    def get_predictions_true_data(self):
        true_values = []

        data_gen_test = self.model_data.get_generator_clean_data_reshape()

        def generator_strip_xy(data_gen, true_values):
            for x, y in data_gen:
                true_values += list(y)
                yield x

        steps_test = int(self.model_data.ntest / self.model_data.batch_size)
        print('> Тестируем модель на', self.model_data.ntest, 'строках с', steps_test, 'шагами')

        predictions = self.model.predict_generator(
            generator_strip_xy(data_gen_test, true_values),
            steps=steps_test
        )

        return predictions, true_values