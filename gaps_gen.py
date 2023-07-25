#!/usr/bin/env python

# gaps_gen.py
# Generate random or space-correlated gaps (missing values) in a given dataset
#
# AH 05/09/2018
# Last update 23/11/2018

import numpy as np


def gen_random_gaps(mask, x, y, gaps, verbose=False):
    """Generate mask of random missing data
    """
    for k in gaps :
        xgap = np.random.randint(0, x)
        ygap = np.random.randint(0, y)
        try :
            while (mask[xgap, ygap] == True) :
                if verbose : print("redundancy found for mask[%d, %d]" % (xgap, ygap))
                xgap = np.random.randint(0, x)
                ygap = np.random.randint(0, y)
                if verbose : print("new gap at : mask[%d, %d]" % (xgap, ygap))
        except ValueError :
            if verbose : print ("no redundancy found")
        mask[xgap, ygap] = True
    return mask

def generate_random_gaps(X, missing_rate, rng=42):
    """ Generate Missing Completely At Random data
    """

    p, N = X.shape
    shape = (p-1, N)

    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)

    M = rng.binomial(1, missing_rate, shape)
    np.putmask(X[:p-1,:], M, np.nan)

    return X

def generate_general_pattern(X, n):
    """ Generate blocks of missing data (general pattern)
    """

    ni = int(n/64)
    d1, d2, d3 = ni*np.random.randint(10,60), ni*np.random.randint(10,60), ni*np.random.randint(10,60)
    X[:2,5+d1:15+d1] = np.nan
    X[1:3,n-34+d2:n-29+d2] = np.nan
    X[1:3,n-18+d3:n-13+d3] = np.nan
    X[4:6,n-55:n-30] = np.nan
    X[4:7,n-15:] = np.nan
    X[7,6:40] = np.nan
    X[8:,n-20:] = np.nan
    X[9,5+d2:35+d2] = np.nan
    X[11:13,20+d3:35+d3] = np.nan
    X[12,6+d1:15+d1] = np.nan
    X[14,20+d2:30+d2] = np.nan

    return X

def generate_one_block(X, n):
    """ Generate one block of missing data (monotone pattern)
    """
    X[10:,n-40:] = np.nan
    return X

def gen_correlated_gaps(field, seuil, tstart, tend):
    """Generate mask of missing data correlated in space
    A threshold on the given field's low values is applied to give
    the mask its shape
    """
    mask = np.zeros((field.shape[0], field.shape[1], field.shape[2]), dtype=bool)
    for i in range(field.shape[0]):
        if i >= tstart and i <= tend :
            mask[i][field[i] < seuil] = True
    return mask


def gen_cv_mask(mask, N, gaps, verbose=False):
    """Generate mask for cross validation points (same principle than gen_random_gaps())
    """
    for k in gaps :
        val = np.random.randint(0, N)
        try :
            while mask[val]:
                if verbose : print ('redundancy found')
                val = np.random.randint(0, N)
        except ValueError :
            if verbose : print ("no redundancy found")
        mask[val] = True
    return mask

# Apply mask on array and fill masked values with user-defined value
#
def mask_field(displ, mask, fval):
    return np.ma.filled(np.ma.array(displ, mask=mask, fill_value=0.0), fill_value=fval)


""" Gen holes directly on spatio_temporal matrix """ 
# X = np.reshape(field_s.T, (nb_obs, nb_time))
# mask_0 = np.zeros((nb_obs, nb_time), dtype=bool) 
# mask = holes.gen_holes_mask(mask_0, nb_obs, nb_time, tirage, remise, holes_list, verbose=False)
# field_zeros = np.ma.filled(np.ma.array(X, mask=mask, fill_value=0.0),
#                            fill_value=fill_val)
# field_A = copy.copy(field_zeros)
