import matplotlib.pyplot as plt
import numpy as np
from utils import *

folder = './'


def graph():
    logs = load_pickle(folder + 'log.pkl')
    for i in range(50):
        print(i, logs['generator_loss'][i])
    print(logs.keys())
    print(logs['validation loss'])
    c = []
    c.append('blue')
    c.append('green')
    c.append('red')
    c.append('black')
    losses = list(logs.keys())
    for i in range(len(losses)):
        data = logs[losses[i]]
        x = range(1, 1 + len(data))
        y = data
        x = x[:2000]
        y = y[:2000]
        if losses[i] == 'validation loss':
            x = np.array(x) * 20
            x = x[:100]
            y = y[:100]
        plt.plot(x, y, color=c[i])
    plt.savefig(folder + 'log.jpg')
    plt.clf()
    for i in range(len(losses)):
        data = logs[losses[i]]
        x = range(1, 1 + len(data))
        y = data
        x = x[:2000]
        y = y[:2000]
        if losses[i] == 'validation loss':
            x = np.array(x) * 20
            x = x[:100]
            y = y[:100]
        plt.plot(x, y, color=c[i])
        plt.xlabel('number of epochs')
        plt.ylabel('Reconstruction Loss')
        plt.savefig(folder + losses[i] + '.jpg')
        plt.clf()


graph()
