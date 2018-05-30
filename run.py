from data.bitdata import bitdata
from model.lstmModel import lstmModel
from model.lstmModelModify import lstmModelModify
from model.lagModel import lagModel
from model.baseModel import baseModel
from model.simpleModel import simpleModel
import graph.plotUtils
from  config.bpconfig import BPConfig

import csv

def save_prediction_and_true(prediction,true_value,filename = 'result.cvs'):
    with open(filename,'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Prediction','True_value'])
        print('Размер тестовой выборки :' +str(len(prediction)) + ":"+str(len(true_value)))
        for i in range(0,len(prediction)):
            spamwriter.writerow([prediction[i],true_value[i]])

def plot_name_generator(name, epochs, x_windows_size, filter_cols):
    return str(name +str(epochs) + ' epochs win_size ' + str(x_windows_size)+ ' used columns ' + str(filter_cols))

def result_filename_generator(folder='/home/kraken/projects/bitcoin_project/results/',epochs=1,x_window=100,y_window=1,filter_cols=['Close'],layers=[]):
    return str(folder + 'result_epochs=' + str(epochs) + '_x_window' + str(x_window)+'_y_window' + str(y_window)+"|Layers:" + str(layers) + "|Cols:" + str(filter_cols) + '.csv')

# epochs = 400
# epochs = 10
# x_windows_size = 200
#     # filter_cols=["Open", "Close", "Volume_(BTC)", "Volume_(Currency)"],
# filter_cols = ["Open","Close"]
filter_colses=[
                  ["Open", "Close", "Volume_(BTC)", "Volume_(Currency)","Weighted_Price"],
                  # ["Open", "Close", "Volume_(BTC)","Weighted_Price"],
                  # ["Open", "Close", "Weighted_Price"],
                  ["Close", "Weighted_Price"],
                  ["Close"]
               ]
# epoches = [1,2,3,5,10,20,30,50]
epoches = [1,10,50]
# x_windows_sizes = [50,100,200,300,500]
x_windows_sizes = [50,300,500]
# y_windows_sizes = [1,2,3,5,10,20,50]
y_windows_sizes = [1,10,50]
layerses =[
    [50,50],
    # [100,100],
    [150,150],
    # [200,200],
    [500,500],
    # [500,100],
    # [100,500]
]



#
# data = bitdata(
#     data_filepath='/home/kraken/projects/bitcoin_project/data/',
#     source_filename='coinbaseUSD',
#     start_date='12.01.2017',
#     end_date='01.01.2018',
#     # filter_cols=["Open", "Close", "Volume_(BTC)", "Volume_(Currency)"],
#     filter_cols=filter_cols,
#     refresh = False
# )


# for epochs in epoches:
#     for x_windows_size in x_windows_sizes:
#         print("Работаем над сетью с параметрами epochs = " + str(epochs) + ' x_window_size=' + str(x_windows_size))
#
#         data = bitdata(
#             data_filepath='/home/kraken/projects/bitcoin_project/data/',
#             source_filename='coinbaseUSD',
#             # start_date='12.01.2017',
#             start_date='11.01.2017',
#             end_date='01.01.2018',
#             filter_cols=filter_cols,
#             batch_size=x_windows_size,
#             x_windows_size=x_windows_size,
#             refresh = False
#         )
#
#         model = lstmModel(
#             model_filepath='/home/kraken/projects/bitcoin_project/model/',
#             data=data,
#             epochs=epochs
#         )
#
#         model_network = model.get_network()
#         prediction, true_value = model.get_predictions_true_data()
#         save_prediction_and_true(prediction=prediction,true_value=true_value,filename=str('result_epochs=' + str(epochs) + '_x_window_size=' + str(x_windows_size)+'.csv'))



# epochs = 10
# x_windows_size = 100
# y_window_size = 1
# layers = [x_windows_size,x_windows_size]
# y_window_size = 1

import os


# for filter_cols in filter_colses:
#     for epochs in epoches:
#         for x_windows_size in x_windows_sizes:
#             for y_windows_size in y_windows_sizes:
#                 for layers in layerses:
#                     filename = result_filename_generator(epochs=epochs,x_window=x_windows_size,y_window=y_windows_size,filter_cols=filter_cols,layers=layers)
#                     print('=======================================')
#                     print('> Запуск с параметрами ')
#                     print('> filter_cols :  ' + str(filter_cols))
#                     print('> epochs :  ' + str(epochs))
#                     print('> x_windows_size :  ' + str(x_windows_size))
#                     print('> y_windows_size :  ' + str(y_windows_size))
#                     print('> layers :  ' + str(layers))
#                     if not os.path.isfile(filename):
#                         try:
#                             data = bitdata(
#                                         data_filepath='/home/kraken/projects/bitcoin_project/data/',
#                                         source_filename='coinbaseUSD',
#                                         # start_date='12.01.2017',
#                                         start_date='11.01.2017',
#                                         end_date='01.01.2018',
#                                         filter_cols=filter_cols,
#                                         batch_size=x_windows_size,
#                                         x_windows_size=x_windows_size,
#                                         y_windows_size=y_windows_size,
#                                         refresh = False
#                                     )
#
#                             model = lstmModelModify(
#                                 model_filepath='/home/kraken/projects/bitcoin_project/model/',
#                                 y_window=y_windows_size,
#                                 x_window=x_windows_size,
#                                 layers=layers,
#                                 data=data,
#                                 epochs=epochs
#                             )
#
#                             model_network = model.get_network()
#                             prediction, true_value = model.get_predictions_true_data()
#                             save_prediction_and_true(prediction=prediction,true_value=true_value,filename=result_filename_generator(epochs=epochs,x_window=x_windows_size,y_window=y_windows_size,filter_cols=filter_cols,layers=layers))
#                         except :
#                             print('> Ошибка, но наплевать, продолжаем')
#                     else:
#                         print('> Уже сделано')

data = bitdata(
    data_filepath='/home/kraken/projects/bitcoin_project/data/',
    source_filename='coinbaseUSD',
    # start_date='12.01.2017',
    # start_date='11.01.2017',
    start_date='12.31.2017',
    end_date='01.01.2018',
    filter_cols=['Close'],
    batch_size=100,
    x_windows_size=100,
    y_windows_size=1,
    # type='MONTHS',
    refresh = False
)
model = lagModel(
    model_filepath='/home/kraken/projects/bitcoin_project/model/',
    data=data
)

model_network = model.get_network()
prediction, true_value = model.get_predictions_true_data()
save_prediction_and_true(prediction=prediction,true_value=true_value,filename=result_filename_generator(epochs=1,x_window=1,y_window=1,filter_cols=[1],layers=[1]))


# graph.plotUtils.plot_results(prediction[:200], true_value[:200],'LSTM 15 epochs win_size 50',save=True)
# title = plot_name_generator(model.name, epochs, x_windows_size, filter_cols)

# graph.plotUtils.plot_history(model.history,title,True)
#
# graph.plotUtils.plot_results(prediction[:200], true_value[:200],title,save=True)
# config = BPConfig()
# config.save()
