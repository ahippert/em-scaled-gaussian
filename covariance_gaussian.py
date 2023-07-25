import numpy as np
import copy
import permutations as perm

def EM_gaussian(X, tol, iter_max, lowrank, rank, init=None):
    """ A function that estimates the covariance matrix of an incomplete dataset with general missing pattern under a Gaussian distribution
        Inputs:
            * 𝐗 = a matrix of size p*N with each observation along column dimension
            * low_rank = if True, perform low-rank estimation
            * rank = rank in R++ of covariance matrix
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
            * init = initial estimate for 𝚺, default is the identity matrix
        Outputs:
            * 𝚺 = covariance matrix estimate
            * tau = textures estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p, N) = X.shape
    if init is None:
        # Initialisation of Sigma
        S = np.eye(p)
    else:
        S = init

    # Get row indices where values are missing
    N_mis_indices = np.where(np.sum(np.isnan(X), axis=0))[0]
    N_obs_indices = np.where(np.sum(np.isnan(X), axis=0)==False)[0]
    N_mis = len(N_mis_indices)
    N_obs = len(N_obs_indices)

    # Get number of observed components where observations are missing
    M = p - np.sum(np.isnan(X[:,N_mis_indices]), axis=0)

    # Permute missing data at the bottom of each vector
    Y = copy.copy(X)
    ind = np.empty((N_mis,p), dtype=int)
    for i, i_mis in enumerate(N_mis_indices):
        Y[:,i_mis], ind[i] = perm.permut_row(Y[:,i_mis])

    # Form permutation matrices
    P = np.array([np.eye(p)[ind[i]].T for i in range(N_mis)])

    # Initialisation of the covariance
    S = np.eye(p)

    # Expectation-Maximization loop
    conv = 0       # Iteration number
    delta = np.inf # Distance error between two iterations
    while (delta>tol) and (conv<iter_max):

        # E step: compute conditional probabilities in matrix B
        B = e_step(Y, S, P, N_mis, N_mis_indices, M)

        # M step: update parameter using conditional propabilities
        S_hat = m_step(Y, B, P, N_obs_indices, N_mis, lowrank, rank)

        # Compute distance
        delta = np.linalg.norm(S_hat - S, 'fro') / np.linalg.norm(S, 'fro')

        # Replace previous covariance by current one
        S = S_hat

        conv = conv + 1

    return S

def e_step(Y, S, P, N_mis, N_mis_indices, M):

    (p, N) = Y.shape

    # Init matrix B (block matrix which is part of the M-step solution)
    B = np.zeros((N_mis,p,p))

    # Fill matrix B
    for (i, i_mis), j in zip(enumerate(N_mis_indices), M):
        D = P[i]@S@P[i].T # Apply P on each side of covariance
        D_mm = D[j:,j:]
        D_mo = D[j:,:j]
        D_oo = D[:j,:j]
        D_om = D[:j,j:]

        cov = D_mm - D_mo@np.linalg.inv(D_oo)@D_om
        mu_mis = D_mo@np.linalg.inv(D_oo)@np.vstack(Y[:j,i_mis])

        B[i, :j, :j] = Y[:j,i_mis,np.newaxis] @ Y[np.newaxis,:j,i_mis] # Upper left block
        B[i, j:, j:] = cov + mu_mis @ mu_mis.T # Lower right block
        B[i, j:, :j] = mu_mis @ Y[np.newaxis,:j,i_mis]
        B[i, :j, j:] = Y[:j,i_mis,np.newaxis] @ mu_mis.T

    return B

def m_step(Y, B, P, N_obs_indices, N_mis, lowrank, rank):

    (p, N) = Y.shape

    # Compute covariance and normalize by number of observations
    C = Y[:,N_obs_indices]@Y[:,N_obs_indices].T
    for i in range(N_mis):
        C += P[i].T@B[i]@P[i]
    Sigma_new = C.T/N

    # Low-rank approximation
    if lowrank: #and rank>=opt_rank:
        Sigma_new = low_rank_covariance(Sigma_new, rank)

    return Sigma_new

def low_rank_covariance(S, rank):
    """ Performs Low Rank (LR) reconstruction from Algorithm 5 of Sun and Palomar (2016)
        Input:
            * S => covariance matrix
            * rank => desired rank for LR
        Ouput:
            * R => Low rank covariance matrix """

    # Get shape of covariance
    p = S.shape[0]

    # EVD of covariance matrix
    l, v = np.linalg.eig(S)

    # Get mean of last p-rank eigenvalues
    sig = np.mean(l[rank:])

    # Reconstruct R with Low Rank structure
    R = (l[:rank]-sig) * v[:,:rank]@v[:,:rank].T
    R += sig*np.eye(p) # Low eigenvalues part of the signal
    #R /= np.trace(R) # Normalize by trace

    return R

def estim_gaussian_aubry(Y, A, A_bar, n, p, tol, iter_max):
    """ Estimation of the covariance matrix of an incomplete dataset with
    general missing pattern under a Gaussian distribution.
    """

    # Initialisation of the covariance
    M = np.eye(p)

    # EM loop
    conv = 0 # Iteration number
    LL = [] # Log-likelihood values are going here
    delta = np.inf # Distance between two iterations
    while (delta>tol) and (conv<iter_max):

        Sigma = 0.

        # E-step
        for i in range(n):
            y = np.vstack(Y[i])
            if A_bar[i].size == 0:
                A_bar[i] = np.eye(p)

            mu = (A_bar[i]@M@A[i].T)@(np.linalg.inv(A[i]@M@A[i].T))@y
            G = (A_bar[i]@M@A_bar[i].T) - (A_bar[i]@M@A[i].T)@(np.linalg.inv(A[i]@M@A[i].T))@(A[i]@M@A_bar[i].T)
            D = (A[i].T@y + A_bar[i].T@mu)
            C = D@D.T + A_bar[i].T@G@A_bar[i]
            Sigma += C
        Sigma /= n

        # M-step
        Q = -n*(p*np.log(np.pi) + np.log(np.linalg.det(M))
                + np.trace(np.linalg.inv(M)@Sigma))

        # Compute distance
        delta = np.linalg.norm(Sigma - M, 'fro') / np.linalg.norm(M, 'fro')
        #print(delta)
        # Replace previous covariance by current one
        #S = S_hat
        M = Sigma

        conv = conv + 1

        #LL.append(compute_incomplete_log_likelihood(X, S, P, M))

    return M

def estim_gaussian_ghahramani(X, tol, iter_max, lowrank, rank):

    #
    p, n = X.shape
    n_mis = np.where(np.sum(np.isnan(X), axis=0))[0]
    n_obs = np.where(np.sum(np.isnan(X), axis=0)==False)[0]

    # Initialisation of Sigma
    S = np.eye(p)

    # EM loop
    conv = 0 # Iteration number
    delta = np.inf # Distance between two iterations
    while (delta>tol) and (conv<iter_max):
        X_temp = copy.copy(X)
        cov_total = np.zeros((p,p))
        # Computation of conditional probabilities
        for i in n_mis:
            p_obs = np.where(np.isnan(X[:,i])==False)[0]
            p_mis = np.where(np.isnan(X[:,i]))[0]

            S_mm = S[p_mis][:,p_mis]
            S_mo = S[p_mis][:,p_obs]
            S_om = S[p_obs][:,p_mis]
            S_oo = S[p_obs][:,p_obs]

            mu_m = S_mo@np.linalg.inv(S_oo)@np.vstack(X[p_obs,i])
            #X_temp[p_mis,i] = np.hstack(mu_m)
            X_temp[:,i] = np.concatenate((X_temp[p_obs,i], np.hstack(mu_m)), axis=0)
            cov_mm = S_mm - S_mo@np.linalg.inv(S_oo)@S_om
            cov = np.block([[np.zeros((len(p_obs),len(p_obs))), np.zeros((len(p_obs),len(p_mis)))],
                            [np.zeros((len(p_mis),len(p_obs))), cov_mm]])

            cov_total = cov_total + cov

        # Compute new covariance
        S_hat = (X_temp @ X_temp.T + cov_total)/n

        # Low-rank approximation
        if lowrank: #and rank>=opt_rank:
            S_hat = low_rank_covariance(S_hat, rank)

        # Compute distance
        delta = np.linalg.norm(S_hat - S, 'fro') / np.linalg.norm(S, 'fro')

        S = S_hat

        conv = conv + 1

    return S


def estim_gaussian_test(Y, tol, iter_max, lowrank, rank):
    """ Estimation of the covariance matrix of an incomplete dataset with
    general missing pattern under a Gaussian distribution.
    """

    # Initialisation
    (p, N) = Y.shape

    # Get row indices where values are missing
    N_mis_indices = np.where(np.sum(np.isnan(Y), axis=0))[0]
    N_obs_indices = np.where(np.sum(np.isnan(Y), axis=0)==False)[0]
    N_mis = len(N_mis_indices)

    # Initialisation of the covariance
    S = np.eye(p)

    # Expectation-Maximization loop
    conv = 0       # Iteration number
    delta = np.inf # Distance error between two iterations
    while (delta>tol) and (conv<iter_max):

        # Init matrix B (block matrix which is part of the M-step solution)
        B = np.zeros((N_mis,p,p))

        for i_mis, i in enumerate(N_mis_indices):

            p_obs = np.where(np.isnan(Y[:,i])==False)[0]
            p_mis = np.where(np.isnan(Y[:,i]))[0]
            D_mm = S[p_mis][:,p_mis]
            D_mo = S[p_mis][:,p_obs]
            D_om = S[p_obs][:,p_mis]
            D_oo = S[p_obs][:,p_obs]

            # Compute conditional probabilities (E-step)
            cov_mis = D_mm - D_mo@np.linalg.inv(D_oo)@D_om
            mu_mis = D_mo@np.linalg.inv(D_oo)@np.vstack(Y[p_obs,i])
            D = Y[p_obs,i,np.newaxis] @ Y[np.newaxis,p_obs,i] # Upper left block
            E = Y[p_obs,i,np.newaxis] @ mu_mis.T
            F = mu_mis @ Y[np.newaxis,p_obs,i]
            G = cov_mis + mu_mis @ mu_mis.T # Lower right block

            B[i_mis] = np.block([[D, E],
                                 [F, G]])

        # Compute estimates (M-step)
        S_hat = (Y[:,N_obs_indices] @ Y[:,N_obs_indices].T + np.sum(B, axis=0))/N

        # Low-rank approximation
        if lowrank: #and rank>=opt_rank:
            S_hat = low_rank_covariance(S_hat, rank)

        # Compute distance for convergence
        delta = np.linalg.norm(S_hat - S, 'fro') / np.linalg.norm(S, 'fro')

        # Replace previous covariance by current one
        S = S_hat

        conv = conv + 1

    return S
