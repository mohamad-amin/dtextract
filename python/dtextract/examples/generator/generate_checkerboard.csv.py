from sklearn.datasets import make_checkerboard
from matplotlib import pyplot
import numpy

from generateCsv import *

n_clusters = 2
shape = (100, 100)

PLOT_DATA = False

if __name__ == "__main__":

    data, rows, columns = make_checkerboard(shape, n_clusters, noise=0, shuffle=True)

    if (PLOT_DATA != True):

        resMap = {}
        for i in range(shape[0]):
            for j in range(shape[1]):
                val = data[i][j]
                if (val not in resMap):
                    resMap[val] = len(resMap)

        X = numpy.zeros((shape[0] * shape[1], 2))
        for i in range(X.shape[0]):
            X[i][0] = (i / shape[1]) + 1
            X[i][1] = (i % shape[1]) + 1

        Y = numpy.zeros((X.shape[0], 1))
        for i in range(Y.shape[0]):
            row = i / shape[1]
            col = i % shape[1]
            Y[i] = resMap[data[row][col]]

        generateCsv('checkerboard.csv', X, Y)

    else:

        print("Rows shape -> " + str(rows.shape))
        print("Columns shape -> " + str(columns.shape))
        print("\nRows -> ")
        print(str(rows))
        print("Columns -> ")
        print(str(columns))
        print("\nData -> ")
        print(str(data))

        pyplot.matshow(data, cmap=pyplot.cm.Blues)
        pyplot.show()
