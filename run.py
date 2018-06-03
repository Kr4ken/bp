from data.bitdata import bitdata,period
from model.lstmModel import lstmModel
from model.lstmModelModify import lstmModelModify
from model.lagModel import lagModel
from model.baseModel import baseModel
from model.simpleModel import simpleModel
from model.randomWalkModel import randomWalkModel
import graph.plotUtils
from  config.bpconfig import BPConfig
import os

import csv

def save_prediction_and_true(prediction,true_value,filename = 'result.cvs'):
    with open(filename,'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        header = []
        print('> Предсказания :' +str(len(prediction[0])) + " истинные значнения : " +  str(len(true_value[0])))
        for k in range(len(prediction[0])):
            header.append('prediction_'+str(k))
        for k in range(len(true_value[0])):
            header.append('true_value_'+str(k))
        # spamwriter.writerow(['Prediction','True_value'])
        spamwriter.writerow(header)
        print('Размер тестовой выборки :' +str(len(prediction)) + ":"+str(len(true_value)))
        for i in range(0,len(prediction)):
            data_row =[]
            for r in prediction[i]:
                data_row.append(r)
            for r in true_value[i]:
                data_row.append(r)
            spamwriter.writerow(data_row)

def plot_name_generator(name, epochs, x_windows_size, filter_cols):
    return str(name +str(epochs) + ' epochs win_size ' + str(x_windows_size)+ ' used columns ' + str(filter_cols))

def result_filename_generator(folder='/home/kraken/projects/bitcoin_project/results/',epochs=1,x_window=100,y_window=1,filter_cols=['Close'],layers=[],model_name='',normalization=""):
    return str(folder + 'result_epochs=' + str(epochs) + '_x_window' + str(x_window)+'_y_window' + str(y_window)+"|Net:" + model_name+"|Layers:" + str(layers) + "|Cols:" + str(filter_cols) +normalization+ '.csv')

filter_colses=[
                  ["Open", "Close", "Volume_(BTC)", "Volume_(Currency)","Weighted_Price"],
                  ["Close", "Weighted_Price"],
                  ["Close"]
               ]
epoches = [1,10,50]
x_windows_sizes = [50,300,500]
y_windows_sizes = [1,10,50]
layerses =[
    [50,50],
    [150,150],
    [500,500],
]


# for filter_cols in filter_colses:
#     for epochs in epoches:
#         for x_windows_size in x_windows_sizes:
#             for y_windows_size in y_windows_sizes:
#                 for layers in layerses:
#                     filename = result_filename_generator(epochs=epochs,x_window=x_windows_size,y_window=y_windows_size,filter_cols=filter_cols,layers=layers)
#                     print('=======================================')
#                     print('>>> Запуск с параметрами ')
#                     print('>>> filter_cols :  ' + str(filter_cols))
#                     print('>>> epochs :  ' + str(epochs))
#                     print('>>> x_windows_size :  ' + str(x_windows_size))
#                     print('>>> y_windows_size :  ' + str(y_windows_size))
#                     print('>>> layers :  ' + str(layers))
#                     if not os.path.isfile(filename):
#                         try:
#                             print('>>> Готовим данные ')
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
#                             # LSTM model
#                             print('>>> Данные готовы')
#                             model = lstmModelModify(
#                                 model_filepath='/home/kraken/projects/bitcoin_project/model/',
#                                 y_window=y_windows_size,
#                                 x_window=x_windows_size,
#                                 layers=layers,
#                                 data=data,
#                                 epochs=epochs
#                             )
#
#                             # WO NORM
#                             print('>>> Готовим модель ' + model.name)
#                             model_network = model.get_network()
#                             print('>>> Модель готова')
#                             print('>>> Предсказываем и сохраняем:' + model.name)
#                             prediction, true_value = model.get_predictions_true_data()
#                             result_filename = filename=result_filename_generator(epochs=epochs,
#                                                                                  x_window=x_windows_size,
#                                                                                  y_window=y_windows_size,
#                                                                                  filter_cols=filter_cols,
#                                                                                  layers=layers,
#                                                                                  model_name=model.name)
#                             save_prediction_and_true(prediction=prediction,true_value=true_value,filename=result_filename)
#                             # NORM
#                             print('>>> Предсказываем и сохраняем с нормой:' + model.name)
#                             prediction, true_value = model.get_predictions_true_data_with_norm()
#                             result_filename = filename=result_filename_generator(epochs=epochs,
#                                                                                  x_window=x_windows_size,
#                                                                                  y_window=y_windows_size,
#                                                                                  filter_cols=filter_cols,
#                                                                                  layers=layers,
#                                                                                  model_name=model.name,
#                                                                                  normalization='|norm')
#                             save_prediction_and_true(prediction=prediction,true_value=true_value,filename=result_filename)
#
#                             # Если окно 1 то пробуем експериментальное предсказание
#                             if(y_windows_size==1):
#                             # WO NORM
#                                 print('>>> Предсказываем и сохраняем множественно :' + model.name)
#                                 model_network = model.get_network()
#                                 prediction, true_value = model.get_multistep_predictions_true_data(10)
#                                 result_filename = filename=result_filename_generator(epochs=epochs,
#                                                                                  x_window=x_windows_size,
#                                                                                  y_window=y_windows_size,
#                                                                                  filter_cols=filter_cols,
#                                                                                  layers=layers,
#                                                                                  model_name=model.name)
#                                 save_prediction_and_true(prediction=prediction,true_value=true_value,filename=result_filename)
#                                 # NORM
#                                 print('>>> Предсказываем и сохраняем с нормой:' + model.name)
#                                 prediction, true_value = model.get_multistep_predictions_true_data_with_norm(10)
#                                 result_filename = filename=result_filename_generator(epochs=epochs,
#                                                                                  x_window=x_windows_size,
#                                                                                  y_window=y_windows_size,
#                                                                                  filter_cols=filter_cols,
#                                                                                  layers=layers,
#                                                                                  model_name=model.name,
#                                                                                  normalization='|norm')
#                                 save_prediction_and_true(prediction=prediction,true_value=true_value,filename=result_filename)
#
#                             # Случайные шаги
#                             if(y_windows_size==1):
#                                 model = randomWalkModel(
#                                 model_filepath='/home/kraken/projects/bitcoin_project/model/',
#                                     data=data,
#                                     epochs=epochs
#                                 )
#                                 print('>>> Готовим модель ' + model.name)
#                                 print('>>> Модель готова')
#                                 print('>>> Предсказываем и сохраняем:' + model.name)
#                                 result_filename = filename=result_filename_generator(epochs=epochs,
#                                                                                  x_window=x_windows_size,
#                                                                                  y_window=y_windows_size,
#                                                                                  filter_cols=filter_cols,
#                                                                                  layers=layers,
#                                                                                  model_name=model.name)
#                                 save_prediction_and_true(prediction=prediction,true_value=true_value,filename=result_filename)
#
#                                 print('>>> Предсказываем и сохраняем с нормой:' + model.name)
#                                 prediction, true_value = model.get_predictions_true_data_with_norm()
#                                 result_filename = filename=result_filename_generator(epochs=epochs,
#                                                                                  x_window=x_windows_size,
#                                                                                  y_window=y_windows_size,
#                                                                                  filter_cols=filter_cols,
#                                                                                  layers=layers,
#                                                                                  model_name=model.name,
#                                                                                  normalization='|norm')
#                                 save_prediction_and_true(prediction=prediction,true_value=true_value,filename=result_filename)
#                                 # WO NORM
#                                 print('>>> Предсказываем и сохраняем множественно :' + model.name)
#                                 model_network = model.get_network()
#                                 prediction, true_value = model.get_multistep_predictions_true_data(10)
#                                 result_filename = filename=result_filename_generator(epochs=epochs,
#                                                                                      x_window=x_windows_size,
#                                                                                      y_window=y_windows_size,
#                                                                                      filter_cols=filter_cols,
#                                                                                      layers=layers,
#                                                                                      model_name=model.name)
#                                 save_prediction_and_true(prediction=prediction,true_value=true_value,filename=result_filename)
#                                 # NORM
#                                 print('>>> Предсказываем и сохраняем множественно с нормой:' + model.name)
#                                 prediction, true_value = model.get_multistep_predictions_true_data_with_norm(10)
#                                 result_filename = filename=result_filename_generator(epochs=epochs,
#                                                                                      x_window=x_windows_size,
#                                                                                      y_window=y_windows_size,
#                                                                                      filter_cols=filter_cols,
#                                                                                      layers=layers,
#                                                                                      model_name=model.name,
#                                                                                      normalization='|norm')
#                                 save_prediction_and_true(prediction=prediction,true_value=true_value,filename=result_filename)
#
#                             # С запаздыванием
#                             if(y_windows_size==1):
#                                 model = lagModel(
#                                     model_filepath='/home/kraken/projects/bitcoin_project/model/',
#                                     data=data,
#                                     epochs=epochs
#                                 )
#                                 print('>>> Готовим модель ' + model.name)
#                                 print('>>> Модель готова')
#                                 print('>>> Предсказываем и сохраняем:' + model.name)
#                                 result_filename = filename=result_filename_generator(epochs=epochs,
#                                                                                      x_window=x_windows_size,
#                                                                                      y_window=y_windows_size,
#                                                                                      filter_cols=filter_cols,
#                                                                                      layers=layers,
#                                                                                      model_name=model.name)
#                                 save_prediction_and_true(prediction=prediction,true_value=true_value,filename=result_filename)
#
#                                 print('>>> Предсказываем и сохраняем с нормой:' + model.name)
#                                 prediction, true_value = model.get_predictions_true_data_with_norm()
#                                 result_filename = filename=result_filename_generator(epochs=epochs,
#                                                                                      x_window=x_windows_size,
#                                                                                      y_window=y_windows_size,
#                                                                                      filter_cols=filter_cols,
#                                                                                      layers=layers,
#                                                                                      model_name=model.name,
#                                                                                      normalization='|norm')
#                                 save_prediction_and_true(prediction=prediction,true_value=true_value,filename=result_filename)
#                                 # WO NORM
#                                 print('>>> Предсказываем и сохраняем множественно :' + model.name)
#                                 model_network = model.get_network()
#                                 prediction, true_value = model.get_multistep_predictions_true_data(10)
#                                 result_filename = filename=result_filename_generator(epochs=epochs,
#                                                                                      x_window=x_windows_size,
#                                                                                      y_window=y_windows_size,
#                                                                                      filter_cols=filter_cols,
#                                                                                      layers=layers,
#                                                                                      model_name=model.name)
#                                 save_prediction_and_true(prediction=prediction,true_value=true_value,filename=result_filename)
#                                 # NORM
#                                 print('>>> Предсказываем и сохраняем множественно с нормой:' + model.name)
#                                 prediction, true_value = model.get_multistep_predictions_true_data_with_norm(10)
#                                 result_filename = filename=result_filename_generator(epochs=epochs,
#                                                                                      x_window=x_windows_size,
#                                                                                      y_window=y_windows_size,
#                                                                                      filter_cols=filter_cols,
#                                                                                      layers=layers,
#                                                                                      model_name=model.name,
#                                                                                      normalization='|norm')
#                                 save_prediction_and_true(prediction=prediction,true_value=true_value,filename=result_filename)
#
#
#
#                         except :
#                             print('>>> Ошибка, но наплевать, продолжаем')
#                     else:
#                         print('>>> Уже сделано')

start_date = '01.01.2017'
end_date = '01.01.2018'
filter_cols = ['Close']
layers = [150, 150]
x_window = 150
y_window = 1
epochs = 10
refresh = False
multi_step = 10
type=period.HOURS

data = bitdata(
    data_filepath='/home/kraken/projects/bitcoin_project/data/',
    source_filename='coinbaseUSD',
    start_date=start_date,
    end_date=end_date,
    filter_cols=filter_cols,
    batch_size=x_window,
    x_windows_size=x_window,
    y_windows_size=y_window,
    refresh=refresh,
    type=type
)

# model = lstmModelModify(
#     model_filepath='/home/kraken/projects/bitcoin_project/model/',
#     x_window=x_window,
#     y_window=y_window,
#     layers=layers,
#     data=data,
#     epochs=epochs,
#     refresh=refresh
# )



# model = randomWalkModel(
#     model_filepath='/home/kraken/projects/bitcoin_project/model/',
#     data=data
# )

model = lagModel(
    model_filepath='/home/kraken/projects/bitcoin_project/model/',
    data=data
)

model_network = model.get_network()
prediction, true_value = model.get_predictions_true_data()
result_filename = result_filename_generator(epochs=epochs,
                                            x_window=x_window,
                                            y_window=y_window,
                                            filter_cols=filter_cols,
                                            layers=layers,
                                            model_name=model.name)
save_prediction_and_true(prediction=prediction, true_value=true_value, filename=result_filename)

prediction, true_value = model.get_predictions_true_data_with_norm()
result_filename = result_filename_generator(epochs=epochs,
                                            x_window=x_window,
                                            y_window=y_window,
                                            filter_cols=filter_cols,
                                            layers=layers,
                                            model_name=model.name,
                                            normalization='|norm')
save_prediction_and_true(prediction=prediction, true_value=true_value, filename=result_filename)

prediction, true_value = model.get_multistep_predictions_true_data(multi_step)
result_filename = result_filename_generator(epochs=epochs,
                                            x_window=x_window,
                                            y_window=y_window,
                                            filter_cols=filter_cols,
                                            layers=layers,
                                            model_name=model.name,
                                            normalization='|multi_'+ str(multi_step))
save_prediction_and_true(prediction=prediction, true_value=true_value, filename=result_filename)

prediction, true_value = model.get_multistep_predictions_true_data_with_norm(multi_step)
result_filename = result_filename_generator(epochs=epochs,
                                            x_window=x_window,
                                            y_window=y_window,
                                            filter_cols=filter_cols,
                                            layers=layers,
                                            model_name=model.name,
                                            normalization='|multi_'+str(multi_step)+'_norm')
save_prediction_and_true(prediction=prediction, true_value=true_value, filename=result_filename)

