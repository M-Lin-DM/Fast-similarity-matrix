import numpy as np


def dist(x, y):
    return np.sqrt(np.sum((x-y)**2))

def sigm(d):
    return np.exp(-d**2/(2*0.3**2))

def row_similarity(dat, row, tau):
    #dat: data matrix
    #row: what row index this worker is in charge of looping across
    #tau: similarity threshold
    row_arr = []
    col_arr = []
    data = []
    for j in np.arange(row):
        if sigm(dist(dat[row, :], dat[j, :])) >= tau:
            data.append(sigm(dist(dat[row, :], dat[j, :])))
            row_arr.append(row)
            col_arr.append(j)
    return data, row_arr, col_arr

def componentwise_similarity(dat, dim, N):
    #dim: dimension of input vectors to take the difference in
    #dat: data matrix
    col_tile = np.tile(np.expand_dims(dat[:, dim], axis=1), (1, N))
    row_tile = np.transpose(col_tile)
    return (col_tile - row_tile)**2
