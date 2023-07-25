import numpy as np
import copy
import permutations as perm
from covariance_gaussian import low_rank_covariance

def tyler_estimator_covariance_normalisedet(𝐗, low_rank, rank, tol=1e-4, iter_max=50, init=None):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
                    and normalisation by determinant
                    Inputs:
                        * 𝐗 = a matrix of size p*N with each observation along column dimension
                        * tol = tolerance for convergence of estimator
                        * iter_max = number of maximum iterations
                        * init = Initialisation point of the fixed-point, default is identity matrix
                    Outputs:
                        * 𝚺 = the estimate
                        * δ = the final distance between two iterations
                        * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    if init is None:
        𝚺 = np.eye(p) # Initialise estimate to identity
    else:
        𝚺 = init
    iteration = 0

    τ=np.zeros((p,N))
    # Recursive algorithm
    while (δ>tol) and (iteration<iter_max):

        # Computing expression of Tyler estimator (with matrix multiplication)
        v=np.linalg.inv(np.linalg.cholesky(𝚺))@𝐗
        a=np.mean(v*v.conj(),axis=0)

        τ[0:p,:] = np.sqrt(np.real(a))
        𝐗_bis = 𝐗 / τ
        𝚺_new = (1/N) * 𝐗_bis@𝐗_bis.conj().T

        # Performs Low Rank decomposition
        if low_rank:
            𝚺_new = low_rank_covariance(𝚺_new, rank)

        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro')
        iteration = iteration + 1

        # Updating 𝚺
        𝚺 = 𝚺_new

    # Calcul textures
    𝛕 = np.zeros(N)
    v = np.linalg.inv(np.linalg.cholesky(𝚺))@𝐗
    a = np.mean(v*v.conj(),axis=0)
    𝛕 = np.real(a)
    c = np.linalg.det(𝚺)**(1/p)
    𝛕 = 𝛕*c

    # Imposing det constraint: det(𝚺) = 1
    𝚺 = 𝚺/c

    return 𝚺, 𝛕

def EM_tyler_estimator(X, tol, iter_max, lowrank, rank, init=None):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        and normalisation by determinant with missing data
        Inputs:
            * 𝐗 = a matrix of size p*N with each observation along column dimension
            * low_rank = if True, perform low-rank estimation
            * rank = rank in R++ of covariance matrix
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
            * init = tuple of initial estimates for 𝚺 and tau
                     Default are the identity matrix and a vector of ones
        Outputs:
            * 𝚺 = covariance matrix estimate
            * tau = textures estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p, N) = X.shape
    if init is None:
        # Initialisation of Sigma and tau
        S = np.eye(p)
        tau = np.ones(N)
    else:
        S, tau = init
    iteration = 0

    # Get row indices where values are missing
    N_mis_indices = np.where(np.sum(np.isnan(X), axis=0))[0]
    N_obs_indices = np.where(np.sum(np.isnan(X), axis=0)==False)[0]
    N_mis = len(N_mis_indices)
    N_obs = len(N_obs_indices)

    # Get number of observed components where observations are missing
    M = p - np.sum(np.isnan(X[:,N_mis_indices]), axis=0)

    # Permute missing data at the bottom of each vector
    ind = np.empty((N_mis,p), dtype=int)
    Y = copy.copy(X)
    for i, i_mis in enumerate(N_mis_indices):
        Y[:,i_mis], ind[i] = perm.permut_row(X[:,i_mis])

    # Form permutation matrices
    P = np.array([np.eye(p)[ind[i]].T for i in range(N_mis)])

    # Expectation-Maximization loop
    conv = 0
    LL, LL_full_estim, err = [], [], []
    delta_em = np.inf # Distance between two iterations
    while (delta_em>tol) and (conv<iter_max):

        # E step: compute conditional probabilities in matrix C
        C = e_step_scaled_gaussian(Y, S, tau, P, N_mis, N_mis_indices, N_obs_indices, M)

        # M step: update parameter using conditional propabilities
        S_em = m_step_scaled_gaussian(C, S, p, N, lowrank, rank)

        # Compute delta(Sigma_new - Sigma_old)
        delta_em = np.linalg.norm(S_em - S, 'fro') / np.linalg.norm(S, 'fro')

        conv = conv + 1

        # Update Sigma and tau
        S = S_em

        # Using newly estimated Sigma to compute tau
        for i in range(N):
            tau[i] = (1./p) * np.trace(C[i]@np.linalg.inv(S))

    # Normalize by determinant to imposing det constraint: det(S) = 1
    c = np.linalg.det(S)**(1/p)
    tau = tau*c
    S = S/c

    return S, tau

def e_step_scaled_gaussian(Y, S, tau, P, N_mis, N_mis_indices, N_obs_indices, M):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        and normalisation by determinant with missing data
        Inputs:
            * Y = a PERMUTED matrix of size p*N with each observation y_i = (y_i^o, y_i^m)^T
        Outputs:
            * C = matrix containing conditional probabilities """

    # Initialisation
    (p, N) = Y.shape

    # Init matrix B (block matrix which is part of the M-step solution)
    B = np.zeros((N_mis,p,p))
    C = np.zeros((N,p,p))

    # Compute covariance and normalize by number of observations
    for i_obs in N_obs_indices:
        C[i_obs] = Y[:,i_obs,np.newaxis] @ Y[np.newaxis,:,i_obs]

    # Fill matrix B
    for (i, i_mis), j in zip(enumerate(N_mis_indices), M):
        D = P[i]@S@P[i].T # Apply them on each side of covariance
        D_mm = D[j:,j:]
        D_mo = D[j:,:j]
        D_oo = D[:j,:j]
        D_om = D[:j,j:]
        inv_D_oo = np.linalg.inv(D_oo)

        # Conditional covariance
        cov = D_mm - D_mo@inv_D_oo@D_om

        # Conditional mean of missing data given observed data
        mu_mis = D_mo@inv_D_oo@np.vstack(Y[:j,i_mis])

        # Fill matrix B_i with conditional probabilities
        B[i, :j, :j] = Y[:j,i_mis,np.newaxis] @ Y[np.newaxis,:j,i_mis] # Upper left block
        B[i, j:, j:] = tau[i_mis]*cov + mu_mis @ mu_mis.T              # Lower right block
        B[i, j:, :j] = mu_mis @ Y[np.newaxis,:j,i_mis]
        B[i, :j, j:] = Y[:j,i_mis,np.newaxis] @ mu_mis.T

        # Permute each B_i to match
        C[i_mis] = P[i].T@B[i]@P[i]

    return C

def m_step_scaled_gaussian(C, Sigma_old, p, N, lowrank, rank):
    """ M step of the EM algorithm: update covariance matrix using a Fixed Point estimator
        Inputs:
            * C = matrix containing conditional probabilities
            * Sigma_old = estimate of the covariance matrix at the (t-1)-th iteration of the EM
        Outputs:
            * Sigma_new = updated estimate of the covariance at the (t)-th iteration of the EM """

    # Parameters for convergence
    delta_fp = np.inf
    iteration = 0
    tol = 1e-4     # Can be decreased, but it is enough
    iter_max = 50  # 50 is more than enough

    # Initialized covariance matrix
    S_em = Sigma_old

    # Fixed point loop
    while (delta_fp>tol) and (iteration<iter_max):

        # Compute fixed point covariance (S_fp)
        S_fp = np.zeros((p,p))
        for i in range(N):
            S_fp += C[i].T/np.trace(C[i]@np.linalg.inv(S_em))

        S_fp = (p/N)*S_fp

        # Performs Low Rank decomposition
        if lowrank: #and rank>=opt_rank:
            S_fp = low_rank_covariance(S_fp, rank)

        # Condition for stopping
        delta_fp = np.linalg.norm(S_fp - S_em, 'fro') / np.linalg.norm(S_em, 'fro')
        iteration = iteration + 1

        # Updating 𝚺
        S_em = S_fp

    Sigma_new = S_em

    return Sigma_new

# def EM_tyler_estimator_test(X, tol, iter_max, lowrank, rank):
#     """
#     m : variable index where missing data start
#     r : same for observation index
#     """

#     # Initialisation
#     (p, N) = X.shape

#     # Get row indices where values are missing
#     N_mis_indices = np.where(np.sum(np.isnan(X), axis=0))[0]
#     N_obs_indices = np.where(np.sum(np.isnan(X), axis=0)==False)[0]
#     N_mis = len(N_mis_indices)
#     N_obs = len(N_obs_indices)

#     # Get number of observed components where observations are missing
#     M = p - np.sum(np.isnan(X[:,N_mis_indices]), axis=0)

#     # Permute missing data at the bottom of each vector
#     ind = np.empty((N_mis,p), dtype=int)
#     Y = copy.copy(X)
#     # for i, i_mis in enumerate(N_mis_indices):
#     #     Y[:,i_mis], ind[i] = perm.permut_row(X[:,i_mis])

#     # Form permutation matrices
#     #P = np.array([np.eye(p)[ind[i]].T for i in range(N_mis)])

#     # Initialisation of Sigma and tau
#     S = np.eye(p)
#     tau = np.ones(N)

#     # EM loop
#     conv = 0
#     LL, LL_full_estim, err = [], [], []
#     delta_em = np.inf # Distance between two iterations
#     while (delta_em>tol) and (conv<iter_max):

#         # Init matrix B (block matrix which is part of the M-step solution)
#         B = np.zeros((N_mis,p,p))
#         C = np.zeros((N,p,p))

#         # Compute covariance and normalize by number of observations
#         for i_obs in N_obs_indices:
#             C[i_obs] = Y[:,i_obs,np.newaxis] @ Y[np.newaxis,:,i_obs]

#         # Fill matrix B
#         for (i, i_mis), j in zip(enumerate(N_mis_indices), M):
#             p_obs = np.where(np.isnan(Y[:,i])==False)[0]
#             p_mis = np.where(np.isnan(Y[:,i]))[0]
#             D_mm = S[p_mis][:,p_mis]
#             D_mo = S[p_mis][:,p_obs]
#             D_om = S[p_obs][:,p_mis]
#             D_oo = S[p_obs][:,p_obs]

#             # Compute conditional probabilities (E-step)
#             cov_mis = D_mm - D_mo@np.linalg.inv(D_oo)@D_om
#             mu_mis = D_mo@np.linalg.inv(D_oo)@np.vstack(Y[p_obs,i])
#             D = Y[p_obs,i,np.newaxis] @ Y[np.newaxis,p_obs,i] # Upper left block
#             E = Y[p_obs,i,np.newaxis] @ mu_mis.T
#             F = mu_mis @ Y[np.newaxis,p_obs,i]
#             G = cov_mis + mu_mis @ mu_mis.T # Lower right block

#             B[i] = np.block([[D, E],
#                              [F, G]])
#             C[i_mis] = B[i]

#         # Fixed point loop
#         delta_fp = np.inf
#         iteration = 0
#         S_em = S
#         while (delta_fp>tol) and (iteration<iter_max):

#             # Compute fixed point covariance (S_fp)
#             S_fp = np.zeros((p,p))
#             for i in range(N):
#                 S_fp += C[i].T/np.trace(C[i]@np.linalg.inv(S_em))

#             S_fp = (p/N)*S_fp

#             # Performs Low Rank decomposition
#             if lowrank:
#                 S_fp = low_rank_covariance(S_fp, rank)

#             # Condition for stopping
#             delta_fp = np.linalg.norm(S_fp - S_em, 'fro') / np.linalg.norm(S_em, 'fro')
#             iteration = iteration + 1

#             # Updating 𝚺
#             S_em = S_fp

#         # Compute delta Sigma
#         delta_em = np.linalg.norm(S_em - S, 'fro') / np.linalg.norm(S, 'fro')
#         #print(delta_em)
#         err.append(delta_em)
#         conv = conv + 1

#         # Update Sigma and tau
#         S = S_em

#         # Using estimated S to compute tau_new
#         for i in range(N):
#             tau[i] = (1./p) * np.trace(C[i]@np.linalg.inv(S))
#     tau = tau*(np.linalg.det(S)**(1/p))
#     S = S/(np.linalg.det(S)**(1/p))

#     return (S, tau)
