# Multiple Linear Regression Model
def mlr(X, y):
    import numpy as np
    X_transpose = np.matrix.transpose(X)
    result = [[0] * len(X_transpose)] * len(X_transpose)
    cur=[]
    sum_x = 0
    
    result = []
    for i in range(len(X_transpose)):
        cur1 = 0
        cur = []
        for j in range(X.shape[1]):
            for k in range(X_transpose.shape[1]):
                a = X_transpose[i][k]
                b = X[k][j]
                cur1 += a * b
            cur.append(cur1)
            cur1 = 0
        result.append(cur)
    beta = np.linalg.inv(result)
    
    result = []
    for i in range(len(beta)):
        cur1 = 0
        cur = []
        for j in range(X_transpose.shape[1]):
            for k in range(beta.shape[1]):
                a = beta[i][k]
                b = X_transpose[k][j]
                cur1 += a * b
            cur.append(cur1)
            cur1 = 0
        result.append(cur)
    y = y.reshape(50,1)
    result = []
    for i in range(len(beta)):
        cur1 = 0
        cur = []
        for j in range(y.shape[1]):
            for k in range(beta.shape[1]):
                a = beta[i][k]
                b = y[k][j]
                cur1 += a * b
            cur.append(cur1)
            cur1 = 0
        result.append(cur)    
    
    
    
    beta = np.matmul(beta, y)
    return beta