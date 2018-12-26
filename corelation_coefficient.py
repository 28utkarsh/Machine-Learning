# Finding out the coefficient of corelation
def corelation(X, y):
    import numpy as np
    import math
    X_mean = [sum(X[:,i])/len(X) for i in range(X.shape[1])]
    y_mean = sum(y)/len(y)
    X_x = []
    y_y = []
    X_sq = []
    y_sq = []
    XY = []
    for j in range(X.shape[1]):
        X_x.append([x-X_mean[j] for x in X[:,j]])
        if j == 0:
            y_y = [x-y_mean for x in y]
    temp = []
    for j in range(X.shape[1]):
        temp = X_x[j]
        X_sq.append([z**2 for z in temp])
        if j == 0:
            y_sq = [z**2 for z in y_y]       
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            temp = X_x[j][i] * y_y[i]
            if i == 0:
                XY.append([])
            XY[j].append(temp)
    sum_X_sq = []
    sum_XY = []
    sum_Y_sq = sum(y_sq)
    for i in range(len(X_sq)):
        temp = X_sq[i]
        sum_X_sq.append(sum(temp))
        temp = XY[i]
        sum_XY.append(sum(temp))
    del(temp, y, y_mean, y_sq, y_y, X, X_mean, X_sq, X_x, i ,j, XY)
    r = []
    for i in range (len(sum_XY)):
        r.append(sum_XY[i] / (math.sqrt(sum_X_sq[i]) * math.sqrt(sum_Y_sq)))
    return r