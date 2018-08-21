from sklearn.datasets import make_checkerboard
from matplotlib import pyplot
import numpy

from generateCsv import *

n_clusters = (2, 2)
shape = (100, 100)

PLOT_DATA = False

if __name__ == "__main__":

    data, rows, columns = make_checkerboard(shape, n_clusters, noise=0, shuffle=False)

    if (PLOT_DATA != True):

        resMap = {}
        vals = []
        q = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                val = data[i][j]
                if (val not in resMap):
                    resMap[val] = len(resMap)
                    vals.append(val)
                    q += 1

        X = numpy.zeros((shape[0] * shape[1], 2))
        for i in range(X.shape[0]):
            X[i][0] = (i / shape[1]) + 1
            X[i][1] = (i % shape[1]) + 1

        resMap[vals[2]] = resMap[vals[1]]
        resMap[vals[3]] = resMap[vals[0]]

        Y = numpy.zeros((X.shape[0], 1))
        for i in range(Y.shape[0]):
            row = i / shape[1]
            col = i % shape[1]
            Y[i] = resMap[data[row][col]]

        print(str(X))
        print('\n')
        print(str(Y))

        for i in range(shape[0]):
            for j in range(shape[1]):
                data[i][j] = resMap[data[i][j]]

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
