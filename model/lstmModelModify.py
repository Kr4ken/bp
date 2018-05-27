import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from model.baseModel import baseModel


class lstmModelModify(baseModel):
    filename = 'LSTM.h5'
    name = 'LSTM'
    loss_function = ''
    optimiser_function = ''
    layers =[]
    x_window = 100
    y_window = 1

    def __init__(self, model_filepath, data,steps_per_epoch = 0,epochs = 1,refresh = False,x_window=100,y_window=1,layers=[150,150]):
        self.loss_function = 'mse'
        self.optimiser_function = 'Nadam'
        # self.layers = [data.ncols, 150, 150, 1]
        # self.layers = [data.ncols, 150, 150, 10]
        self.layers = [data.ncols]  + layers + [y_window]
        self.name = self.name + '_' + "_".join([str(x) for x in self.layers])+ "_epochs="+str(epochs)
        super(lstmModelModify, self).__init__(model_filepath=model_filepath,data=data,steps_per_epoch=steps_per_epoch,epochs=epochs,refresh=refresh,x_window=x_window,y_window=y_window)

    def build_network(self):
        super(lstmModelModify, self).build_network()
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
            optimizer=self.optimiser_function,
            metrics=['accuracy']
        )

        print("> Время компиляции : ", time.time() - start)
        return self.model
