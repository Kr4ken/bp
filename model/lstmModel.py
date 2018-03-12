import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from model.baseModel import baseModel


class lstmModel(baseModel):
    fileName = 'LSTM.h5'
    name = 'LSTM'
    loss_function = ''
    optimiser_function = ''
    layers =[]

    def __init__(self, model_filepath, data,steps_per_epoch = 0,epochs = 1,refresh = False):
        super(lstmModel, self).__init__(model_filepath,data,steps_per_epoch,epochs,refresh)
        self.loss_function = 'mse'
        self.optimiser_function = 'Nadam'
        self.layers = [self.model_data.ncols, 150, 150, 1]

    def build_network(self):
        super(lstmModel, self).build_network()
        self.model = Sequential()

        self.model.add(LSTM(
            input_dim=self.layers[0],
            output_dim=self.layers[1],
            return_sequences=True
        ))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(
            self.layers[2],
            return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(
            output_dim=self.layers[3]))
        self.model.add(Activation("tanh"))

        start = time.time()
        self.model.compile(
            loss=self.loss_function,
            optimizer=self.optimiser_function)

        print("> Время компиляции : ", time.time() - start)
        return self.model
