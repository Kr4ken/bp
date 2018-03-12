from data.bitdata import bitdata
from model.lstmModel import lstmModel
from model.baseModel import baseModel
from model.simpleModel import simpleModel
import graph.plotUtils

# data = bitdata(
#     data_filepath='/home/kraken/projects/bitcoin_project/data/',
#     source_filename='coinbaseUSD',
#     start_date='12.01.2017',
#     end_date='01.01.2018',
#     filter_cols=["Open", "Close", "Volume_(BTC)", "Volume_(Currency)"],
#     refresh = False
# )

data = bitdata(
    data_filepath='/home/kraken/projects/bitcoin_project/data/',
    source_filename='coinbaseUSD',
    start_date='12.01.2017',
    end_date='01.01.2018',
    # filter_cols=["Close"],
    filter_cols=["Open", "Close", "Volume_(BTC)", "Volume_(Currency)"],
    batch_size=50,
    x_windows_size=50,
    refresh = True

)

model = lstmModel(
    model_filepath='/home/kraken/projects/bitcoin_project/model/',
    data=data,
    epochs=15,
    refresh=True
)

# model = simpleModel(
#     model_filepath='/home/kraken/projects/bitcoin_project/model/',
#     data = data,
#     epochs=100,
#     refresh=True
# )

model_network = model.get_network()
prediction, true_value = model.get_predictions_true_data()
graph.plotUtils.plot_results(prediction[:200], true_value[:200],'LSTM 15 epochs win_size 50',save=True)
# graph.plotUtils.plot_history(model.history)
