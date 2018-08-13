import numpy as np

def generateCsv(name, X, Y):
    Z = np.column_stack((X, Y))
    np.savetxt("../../../../tmp/" + name, Z, delimiter=',')
