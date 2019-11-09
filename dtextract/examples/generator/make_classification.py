from sklearn.datasets import make_classification
from matplotlib import pyplot
from pandas import DataFrame

from generateCsv import *

PLOT_DATA = True

if __name__ == "__main__":

    X, Y = make_circles(n_samples=1000, noise=0.2, factor=0.5)

    if PLOT_DATA == False:
        generateCsv("circle.csv", X, Y)
    else:
        # scatter plot, dots colored by class value
        df = DataFrame(dict(x=X[:,0], y=X[:,1], label=Y))
        colors = {0:'red', 1:'blue'}
        fig, ax = pyplot.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
        pyplot.show()

