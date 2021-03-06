# BITCOIN_PROJECT

## Краткое описание классов и методов

### Data

#### data.bitdata.py
Класс для работы с данными, нужен для подготовки данных, их очистки, формирования файлов с готовыми наборами входы-выходы для тренировки и последующей проверки сети.

Методы 
Конструктор
```python
__init__(self,
        data_filepath,
        source_filename,
        start_date,
        end_date,
        filename_extension='h5',
        source_extension = 'csv',
        batch_size = 100,
        x_windows_size = 100,
        y_windows_size = 1,
        filter_cols = None,
        normalize = True,
        y_col = 'Close',
        refresh = False)
```
data_filepath - Путь к папке где лежат все данные и куда в последующем будут сохраняться новые сгенерированные файлы
source_filename - Имя файла с исходными данными в этой папке
start_date - Время начала отбора значений
end_date - Время окончания отбора значений
filename_extension='h5' - расширение с которым будут сохраняться обработанные куски
source_extension = 'csv' - расширение файла источника данных
batch_size = 100 - размер вытягиваемого колличества данных в генераторах
x_windows_size = 100 - Размер входного окна, с помощью которого будут генерироваться данные
y_windows_size = 1 - размер выходного окна
filter_cols = None - Список столбцов входных данных, которые необходимо учитывать, если пусто то учитываюстя все 
normalize = True - производить ли нормализацию значений
y_col = 'Close' - столбец прогнозируемых значений
refresh = False - насильно обновлять данные, даже если уже есть подготовленный файл

В результате работы генерируются следующие файлы:
`clean_<source>` - файл анологичный исходному очищенный от очевидно невалидных данных
`source|<start_date>-<end_date>` - Файл который отфильтрован по датам, которые внутри себя содержит
`source|<start_date>-<end_date>_x=<x_window>_y=<y_window>` - Файл содрежащий результирующие наборы данных по которым и будет тренироваться сеть, x_window,y_window - соответственные размеры окон

Также происходит деление всех данных на тренировочные и тестируемые, по умолчанию 10% ПОСЛЕДНИХ значений, считаются проверочными, на которых и происходит проверка подготовленной модели.

Еще внутри класса на этапе очистки данных происходит нормализация данных, нормализация сейчас доступна 2 типов, относительно первого элемента, и мин-макс. По умолчанию используется относительно первого элемента.


```python
get_generator_clean_data(self)
``` 
Получение генератора подчищенных исходных данных, без невалидных значений
```python
get_generator_clean_data_reshape(self)
``` 
Получение генератора подчищенных исходных данных, без невалидных значений. При этом изменена форма входного массива с [cols,batch,x_window] на [batch,x_window]
Проверить.

```python
get_generator_clean_data_test(self)
``` 
Генератор тестируемых данных

```python
get_generator_clean_data_test_reshape(self)
``` 
Генератор тестируемых данных, reshape


### Graph

#### graph.plotUtils.py

Набор вспомогательных методов для отрисовки типовых графиков и сохранения их со сгенерированными именами в указанную папку.


```python
plot_graph(data, label, title='',save=False)
```
Нарисовать и вывести на экран 
data - данные
label - подписи
title - Наименование графа
save - сохранить граф в файл, или вывести на экран

```python
plot_results(predicted_data, true_data, title='', save=False)
```
Нарисовать и вывести на экран предсказанные и исходные данные для сравнения
predicted_data - предсказанные данные
true_data - исходные данные
title - Наименование графа
save - сохранить граф в файл, или вывести на экран

```python
plot_history(history, title='', save=False):
```
Нарисовать историю обучниея  
history - 


```python
plot_results_multiple(predicted_data, true_data, prediction_len)
```
Нарисовать исхходные и предсказанные данные
predicted_data - предсказанные данные
true_data - исходные данные
prediction_len - длинна предсказываемого участка

### Model

#### model.baseModel.py
Класс, который является исходным классом для моделей сетей

Конструктор
```python
__init__(self, model_filepath, data,steps_per_epoch = 0,epochs = 1,refresh = False)
```
model_filepath - путь к модели
data - Данные на основе которых будет производиться тренировка модели
steps_per_epoch - Шагов за эпоху
epochs - колличество эпох
refresh - насильно обновить  файл, даже если он уже сгенерирован

```python
get_network(self)
```
Получить модель тренированной сети

```python
build_network(self)
```
Построить сеть

```python
load_network(self)
```
Загрузить сеть из фала

```python
get_predictions_true_data(self)
```
Получить два набора данных, сперва предсказанные, затем исходные данные из тестового набора.
