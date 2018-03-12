import warnings
import json
import os
from keras.models import load_model
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings


class baseModel(object):
    config = None
    filename = 'abstract.h5'
    name = 'abstract'
    model = None
    model_filepath = ''
    model_data = None

    steps_per_epoch = 0
    validation_steps = 0
    epochs = 0
    refresh = False
    history = None

    def __init__(self, model_filepath, data,steps_per_epoch = 0,epochs = 1,refresh = False):
        super(baseModel, self).__init__()
        print('> Инициализируем модель сети  ' + self.name)
        self.model_filepath = model_filepath
        self.model_data = data
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.filename = self.model_filepath + self.name + ".h5"
        self.refresh = refresh
        if steps_per_epoch == 0:
            self.steps_per_epoch = int(data.ntrain /epochs / data.batch_size)
        self.validation_steps = int(data.ntest /epochs / data.batch_size)
        # self.validation_steps = int(data.ntest / data.batch_size)



    def get_network(self):
        if (not self.refresh and os.path.isfile(self.filename)):
            return self.load_network()
        else:
            return self.fit_model_threaded()
            # return self.build_network()

    def build_network(self):
        print('> Собираем ' + self.name + ' модель сети')
        return None

    def load_network(self):
        # Load the h5 saved model and weights
        print('> Загружаем ' + self.name + ' модель сети')
        if (os.path.isfile(self.filename)):
            self.model = load_model(self.filename)
            return self.model
        else:
            print('ОШИБКА: "' + self.filename + '" Файл не содержит h5 модель\n')
            return None

    def fit_model_threaded_old(self):
        print('> Тренируем модель ' + self.name)
        self.model = self.build_network()
        output_file = self.filename
        print('> Параметры модели epochs =  ' + str(self.epochs) + ' Шагов за эпоху = ' + str(self.steps_per_epoch))
        self.history = self.model.fit_generator(
            self.model_data.get_generator_clean_data(),
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs
        )
        self.model.save(output_file)
        print('> Модель создана! веса сохранены ', output_file)
        return self.model

    def fit_model_threaded(self):
        print('> Тренируем модель ' + self.name)
        self.model = self.build_network()
        output_file = self.filename
        print('> Информация по данным nrows =  ' + str(self.model_data.nrows) + ' ntrain = ' + str(self.model_data.ntrain) + ' ntest = ' + str(self.model_data.ntest))
        print('> Параметры модели epochs =  ' + str(self.epochs) + ' Шагов за эпоху = ' + str(self.steps_per_epoch) + ' validation steps ' + str(self.validation_steps))
        self.history = self.model.fit_generator(
            self.model_data.get_generator_clean_data(),
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs
            # validation_data=self.model_data.get_generator_clean_data_test(),
            # validation_steps=self.validation_steps
        )
        self.model.save(output_file)
        print('> Модель создана! веса сохранены ', output_file)
        return self.model

    def get_predictions_true_data(self):
        true_values = []

        data_gen_test = self.model_data.get_generator_clean_data_test()

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