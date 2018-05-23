import matplotlib.pyplot as plt

plot_directory = '/home/kraken/projects/bitcoin_project/plot/'

#TODO составить представление о наиболее интересных вариантах отрисовки результатов
#TODO сделать этот вывод результатов


def __savePlot(plot, filename):
    plot.savefig(filename)


def plot_graph(data, label, title='',save=False):
    plt.figure(figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(data, label=label)
    plt.legend()
    if (save):
        __savePlot(plt, plot_directory + title + ".png")
    else:
        plt.show()




def plot_results(predicted_data, true_data, title='', save=False):
    fig = plt.figure(figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='Исходные данные')
    plt.plot(predicted_data, label='Предсказание')
    plt.title(title)
    plt.legend()
    if (save):
        __savePlot(plt, plot_directory + title + ".png")
    else:
        plt.show()


# Отрисовывает ошибки на каждой эпохе во время обучения
def plot_history(history, title='', save=False):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    if (save):
        __savePlot(plt, plot_directory + title + ".png")
    else:
        plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='Исходные данные')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Предсказание')
        plt.legend()
    plt.show()
