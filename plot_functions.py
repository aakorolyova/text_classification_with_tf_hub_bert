import matplotlib.pyplot as plt


def plot_label(history, label):
    plt.plot(history.history[label])
    plt.plot(history.history['val_' + label ])
    plt.title('model ' + label)
    plt.ylabel(label)
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.show()