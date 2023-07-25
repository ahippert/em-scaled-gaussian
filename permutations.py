#!/usr/bin/python
import numpy as np
import copy

def permut_col(Y):
    """ Permutes block of missing variables to the right
    of matrix Y
    """
    X = copy.copy(Y)
    M = [not any(np.isnan(X[:,i])) for i in range(len(X[0]))]
    ind = np.arange(len(X[0]), dtype=int)
    M = np.array(M)
    ind[ind[M]] = np.arange(np.sum(M))
    ind[ind[M==False]] = np.arange(np.sum(M),len(X[0]))
    X[:,ind] = X[:,np.arange(len(X[0]))] # permutation
    return X

def permut_row(Y):
    """ Permutes block of missing variables to the bottom
    of matrix Y
    """
    X = copy.copy(Y)
    M = [not np.isnan(X[i]) for i in range(len(X))]
    ind = np.arange(len(X), dtype=int)
    M = np.array(M)
    ind[ind[M]] = np.arange(np.sum(M), dtype=int)
    ind[ind[M==False]] = np.arange(np.sum(M), len(X), dtype=int)
    X[ind] = X[np.arange(len(X))] # permutation
    return X, ind

def permut_missing_block(X):
    """ Permutes on rows and columns so that the block of missing
    data is at the lower-right corner of matrix X
    """
    return permut_row(permut_col(X))

#def permut_vector(y):
#     """ Permutes missing values (NaN) to the right of vector
#     """
# X = np.array([1., np.nan, 3., np.nan, 5.])
# M = [not any(np.isnan(X[i])) for i in range(len(X))]
# ind = np.arange(len(X))
# M = np.array(M)
# ind[ind[M]] = np.arange(np.sum(M))
# ind[ind[M==False]] = np.arange(np.sum(M),len(X))
# X[ind] = X[np.arange(len(X))] # permutation

