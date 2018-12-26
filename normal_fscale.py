# Implementing the normal Feature Scaling

def nfscale(X, col = []):
    import math
    import numpy as np
    if len(np.shape(X)) == 1:
        X = X.reshape(len(X), 1)
        col = [0]
    if len(col) != 0:
        mean = [sum(X[:,col[i]])/len(X) for i in col]
        std = []
        for j in col:
            cur = 0
            for i in range(len(X)):
                cur += ((X[i, j] - mean[col.index(j)])**2)            
            std.append(math.sqrt(cur/len(X)))
        for j in col:
            for i in range(len(X)):
                X[i, j] = (X[i, j]-mean[col.index(j)])/(std[col.index(j)])
    else:
        print("Feature Scaling is not Possible")