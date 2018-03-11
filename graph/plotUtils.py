import matplotlib.pyplot as plt

def plot_graph(data,label):
	plt.figure(figsize=(18, 12), dpi= 80, facecolor='w', edgecolor='k')
	plt.plot(data,label=label)
	plt.legend()
	plt.show()

def plot_results(predicted_data, true_data):
    fig=plt.figure(figsize=(18, 12), dpi= 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='Исходные данные')
    plt.plot(predicted_data, label='Предсказание')
    plt.legend()
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