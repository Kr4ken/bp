from data.bitdata import bitdata
from model.lstmModel import lstmModel
from model.baseModel import baseModel
import graph.plotUtils

data = bitdata(
    data_filepath='/home/kraken/projects/bitcoin_project/data/',
    source_filename='coinbaseUSD',
    start_date='12.01.2017',
    end_date='01.01.2018',
    filter_cols=["Open", "Close", "Volume_(BTC)", "Volume_(Currency)"]
)

model = lstmModel(
    model_filepath='/home/kraken/projects/bitcoin_project/model/',
    data = data
)

model_network = model.get_network()
prediction, true_value = model.get_predictions_true_data()
graph.plotUtils.plot_results(prediction[:200], true_value[:200])
