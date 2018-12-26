# Categorising the given data

def category(X, col = []):
    count = 0
    types = []
    cat = []
    ind = 0
    while len(col) != 0:
        count = 0
        for i in range(len(X)):
            if X[i, col[0]] not in types:
                types.append(X[i, col[0]])
                cat.append(count)
                X[i, col[0]] = count
                count += 1
            else:
                ind = types.index(X[i, col[0]])
                X[i, col[0]] = cat[ind]
        col.pop(0)