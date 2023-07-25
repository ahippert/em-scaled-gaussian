import numpy as np

def compute_incomplete_log_likelihood_1(X, S, P, M):
    L1, L2 = 0., 0.
    N = X.shape[1]
    for i in range(N):
        D = P[i]@S@P[i].T
        det_D = np.linalg.det(D[:M[i],:M[i]])
        L1 += np.log(det_D)
        L2 += X[:M[i],i]@np.linalg.inv(D[:M[i],:M[i]])@np.vstack(X[:M[i],i])
    return (-L1-L2)

def compute_incomplete_log_likelihood_2(X, S, M):
    L1, L2 = 0., 0.
    N = X.shape[1]
    for i in range(N):
        #D = P[i]@S@P[i].T
        det_S = np.linalg.det(S[M[i],M[i]])
        L1 += np.log(det_S)
        L2 += X[M[i],i]@np.linalg.inv(S[M[i],M[i]])@np.vstack(X[M[i],i])
    return (-L1-L2)

def compute_incomplete_log_likelihood_compound(X, S, tau, P, M):
    L1, L2, L3 = 0., 0., 0.
    p, N = X.shape
    for i in range(N):
        D = P[i]@S@P[i].T
        det_D = np.linalg.det(D[:M[i],:M[i]])
        L1 += np.log(det_D)
        L2 += p*np.log(tau[i])
        L3 += (1./tau[i])*X[:M[i],i]@np.linalg.inv(D[:M[i],:M[i]])@np.vstack(X[:M[i],i])
    return (-L1-L2-L3)

def compute_log_likelihood_compound(X, S, tau):
    L1, L2, L3 = 0., 0., 0.
    N = X.shape[1]
    for i in range(N):
        det_S = np.linalg.det(S)
        L1 += np.log(det_S)
        L2 += p*np.log(tau[i])
        L3 += (1./tau[i])*X[:,i]@np.linalg.inv(S)@np.vstack(X[:,i])
    return (-L1-L2-L3)

def compute_log_likelihood(X, S, tau):
    p, N = X.shape
    N_tab = np.arange(N)
    N_mis = N_tab[np.sum(np.isnan(X), axis=0)==True]
    N_obs = N_tab[np.sum(np.isnan(X), axis=0)==False]
    M = p - np.sum(np.isnan(X), axis=0)
    L3, L4 = 0., 0.
    L1 = -N*np.log(np.linalg.det(S))
    L2 = -p*np.sum(np.log(tau))
    for i in N_obs:
        L3 += (1./tau[i])*X[:,i]@np.linalg.inv(S)@np.vstack(X[:,i])
    for i in N_mis:
        L4 += (1./tau[i])*X[:M[i],i]@np.linalg.inv(S[:M[i],:M[i]])@np.vstack(X[:M[i],i])
    return L1+L2-L3-L4
