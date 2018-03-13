from data.bitdata import bitdata
from model.lstmModel import lstmModel
from model.baseModel import baseModel
from model.simpleModel import simpleModel
import graph.plotUtils

epochs = 400
x_windows_size = 200
filter_cols = ["Close"]

# data = bitdata(
#     data_filepath='/home/kraken/projects/bitcoin_project/data/',
#     source_filename='coinbaseUSD',
#     start_date='12.01.2017',
#     end_date='01.01.2018',
#     filter_cols=["Open", "Close", "Volume_(BTC)", "Volume_(Currency)"],
#     refresh = False
# )

def name_generator(name,epochs,x_windows_size,filter_cols):
    return str(name +str(epochs) + ' epochs win_size ' + str(x_windows_size)+ ' used columns ' + str(filter_cols))

data = bitdata(
    data_filepath='/home/kraken/projects/bitcoin_project/data/',
    source_filename='coinbaseUSD',
    # start_date='12.01.2017',
    start_date='10.01.2017',
    end_date='01.01.2018',
    # filter_cols=["Close"],
    filter_cols=filter_cols,
    # filter_cols=["Open", "Close", "Volume_(BTC)", "Volume_(Currency)"],
    # batch_size=50,
    # x_windows_size=50,
    batch_size=x_windows_size,
    x_windows_size=x_windows_size,
    refresh = False
)

# model = lstmModel(
#     model_filepath='/home/kraken/projects/bitcoin_project/model/',
#     data=data,
#     # epochs=50,
#     epochs=epochs,
#     refresh=True
# )

model = simpleModel(
    model_filepath='/home/kraken/projects/bitcoin_project/model/',
    data = data,
    epochs=epochs,
    refresh=True
)

model_network = model.get_network()
prediction, true_value = model.get_predictions_true_data()
# graph.plotUtils.plot_results(prediction[:200], true_value[:200],'LSTM 15 epochs win_size 50',save=True)
title = name_generator(model.name,epochs,x_windows_size,filter_cols)
graph.plotUtils.plot_history(model.history,title,True)

graph.plotUtils.plot_results(prediction[:200], true_value[:200],title,save=True)
