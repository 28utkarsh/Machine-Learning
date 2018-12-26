def poly(X, y, degree = 2, plot = False, t = 'Real Dataset', xlab = 'X-Variable', ylab = ' Y-Variable'):
    k = 0
    n = len(X)
    s = [0] * (degree * 3 + 1)
    for i in range(n):
        k = 0
        for j in range( (degree * 2)-1, -1, -1 ):
            s[k] += X[i,0] ** (j+1)
            k += 1
        for j in range((degree * 3)-1, (degree * 2)-1, -1):
            s[j] += (X[i, 0] ** ((degree * 3)-j)) * y[i]
        s[degree * 3] += y[i]
    mat = [[0]*(degree + 2)] * ( degree + 1)
    temp = []
    for i in range(degree + 1):
        for j in range(degree + 1):
            temp.append(s[j+i])
        temp.append(s[degree * 2 + i])
        mat[i] = temp
        temp = []
    mat[degree][degree] = n
    
    # Implementing the Gauss Elimination Method to achieve the echelon form
    for i in range(degree + 1):
        mat[i] = [x / mat[i][i] for x in mat[i]]
        for j in range(degree - i):
            mat[j+i+1] = [y - mat[j+i+1][i] * x for x,y in zip(mat[i], mat[j+i+1])]
            
    # Finding out the coefficients
    coef = []
    for i in range(degree, -1, -1):
        cur = mat[i][degree + 1]
        for j in range(degree - i):
            cur -= coef[j] * mat[i][degree - j]
        coef.append(cur)
    coef = coef[::-1]
    if plot == True:
        itr = 0.1
        step = 0.01
        equ = []
        sc = []
        for i in range(0,1000):
            cur = 0
            sc.append(itr)
            for j in range(len(coef)):
                cur += coef[j] * (itr ** (degree - j))
            equ.append(cur)
            itr = itr + step
        import matplotlib.pyplot as plt    
        plt.scatter(X, y, color = 'red')
        plt.plot(sc, equ, color = 'blue')
        plt.title(t)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.show()
    return coef
def predict(r, y):
    pred = []
    cur = 0
    degree = len(r) - 1
    for i in range(len(y)):
        cur = 0
        for j in range(len(r)):
            cur += r[j] * (y[i] ** (degree - j))
        pred.append(cur)
    return pred
    