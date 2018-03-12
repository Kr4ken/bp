import matplotlib.pyplot as plt


plot_directory = '/home/kraken/projects/bitcoin_project/plot/'


def plot_graph(data,label):
	plt.figure(figsize=(18, 12), dpi= 80, facecolor='w', edgecolor='k')
	plt.plot(data,label=label)
	plt.legend()
	plt.show()

def plot_results(predicted_data, true_data,title='',save = False):
    fig=plt.figure(figsize=(18, 12), dpi= 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='Исходные данные')
    plt.plot(predicted_data, label='Предсказание')
    plt.title(title)
    plt.legend()
    if(save):
        plt.savefig(plot_directory + title + ".png")
    plt.show()

def plot_history(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()

    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()



def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig=plt.figure(figsize=(18, 12), dpi= 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='Исходные данные')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Предсказание')
        plt.legend()
    plt.show()