# -*- coding: utf-8 -*
##############################################################################
# A file for comparing the estimation error (natural distance) evolution as
# the number of dates grows
# Authored by Ammar Mian, 13/05/2019
# e-mail: ammar.mian@centralesupelec.fr
##############################################################################
# Copyright 2019 @CentraleSupelec
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
import sys
import numpy as np
import scipy as sp
import scipy.special
import warnings
from matrix_operators import sqrtm, invsqrtm, expm, logm, powm, inv
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm
#import tikzplotlib
import warnings
warnings.simplefilter('once', UserWarning)
#from estim_param import estim_block_normalizedet

# Just for having a nice color style when plotting
# Can be commented
#import seaborn as sns
#sns.set_style("darkgrid")

# ---------------------------------------------------------------------------------------------------------------
# Definition of functions for Generation of Data
# ---------------------------------------------------------------------------------------------------------------
def ToeplitzMatrix(rho, p):
    """ A function that computes a Hermitian semi-positive matrix.
            Inputs:
                * rho = a scalar
                * p = size of matrix
            Outputs:
                * the matrix """

    return sp.linalg.toeplitz(np.power(rho, np.arange(0, p)))

def multivariate_complex_normal_samples(mean, covariance, N, pseudo_covariance=0):
    """ A function to generate multivariate complex normal vectos as described in:
        Picinbono, B. (1996). Second-order complex random vectors and normal
        distributions. IEEE Transactions on Signal Processing, 44(10), 2637–2640.
        Inputs:
            * mean = vector of size p, mean of the distribution
            * covariance = the covariance matrix of size p*p(Gamma in the paper)
            * pseudo_covariance = the pseudo-covariance of size p*p (C in the paper)
                for a circular distribution omit the parameter
            * N = number of Samples
        Outputs:
            * Z = Samples from the complex Normal multivariate distribution, size p*N"""

    (p, p) = covariance.shape
    Gamma = covariance
    C = pseudo_covariance

    # Computing elements of matrix Gamma_2r
    Gamma_x = 0.5 * np.real(Gamma + C)
    Gamma_xy = 0.5 * np.imag(-Gamma + C)
    Gamma_yx = 0.5 * np.imag(Gamma + C)
    Gamma_y = 0.5 * np.real(Gamma - C)

    # Matrix Gamma_2r as a block matrix
    Gamma_2r = np.block([[Gamma_x, Gamma_xy], [Gamma_yx, Gamma_y]])

    # Generating the real part and imaginary part
    mu = np.hstack((mean.real, mean.imag))
    v = np.random.multivariate_normal(mu, Gamma_2r, N).T
    X = v[0:p, :]
    Y = v[p:, :]
    return X + 1j * Y

def generate_data_compound_Gaussian(𝛕, p, N, 𝛍, 𝚺, pseudo_𝚺):
    # Re-seed the random number generator
    #np.random.seed()
    𝐗 = np.empty((p,N), dtype=complex)
    𝐗 = np.sqrt(𝛕)[None,:] * multivariate_complex_normal_samples(𝛍, 𝚺, N, pseudo_𝚺)
    return 𝐗

# ---------------------------------------------------------------------------------------------------------------
# Definition of functions for Covariance Estimation
# ---------------------------------------------------------------------------------------------------------------

def SCM(x, *args):
    """ A function that computes the SCM for covariance matrix estimation
            Inputs:
                * x = a matrix of size p*N with each observation along column dimension
            Outputs:
                * Sigma = the estimate"""

    (p, N) = x.shape
    return (x @ x.conj().T) / N

def tyler_estimator_covariance_normalisedet(𝐗, tol=0.001, iter_max=20, init=None):
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

        # Imposing det constraint: det(𝚺) = 1 DOT NOT WORK HERE
        # 𝚺 = 𝚺/(np.linalg.det(𝚺)**(1/p))

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
    𝛕 = 𝛕*(np.linalg.det(𝚺)**(1/p))

    # Imposing det constraint: det(𝚺) = 1
    𝚺 = 𝚺/(np.linalg.det(𝚺)**(1/p))

    return 𝚺, 𝛕

# ---------------------------------------------------------------------------------------------------------------
# Definition of functions for Computing Distance
# ---------------------------------------------------------------------------------------------------------------

def δ_𝒮ℋplusplus(𝚺_0, 𝚺_1):
    """ Fonction for computing the Riemannian distance between two PDH matrices
        ------------------------------------------------------------------------
        Inputs:
        --------
            * 𝚺_0 = PDH matrix of dimension p
            * 𝚺_1 = PDH matrix of dimension p

        Outputs:
        ---------
            * δ = the distance
        """

    isqrtm𝚺_0 = invsqrtm(𝚺_0)
    δ = np.linalg.norm( logm( isqrtm𝚺_0 @ 𝚺_1 @ isqrtm𝚺_0 ), 'fro') # A changer
    return δ

def δ_ℛplusplus(𝛕_0, 𝛕_1):
    """ Fonction for computing the Riemannian distance between two texture parameters
        ------------------------------------------------------------------------------
        Inputs:
        --------
            * 𝛕_0 = vector in ℛ+^N
            * 𝛕_1 = vector in ℛ+^N

        Outputs:
        ---------
            * δ = the distance
        """

    δ = np.linalg.norm( np.log(𝛕_1/𝛕_0), 2)
    return δ


def δ_𝓜pn(𝚺_0, 𝚺_1, 𝛕_0, 𝛕_1):
    """ Fonction for computing the Riemannian distance between two parameters in manifold
        PDH x {R+}^N
        ----------------------------------------------------------------------------------
        Inputs:
        --------
            * 𝚺_0 = PDH matrix of dimension p
            * 𝚺_1 = PDH matrix of dimension p
            * 𝛕_0 = vector in ℛ+^N
            * 𝛕_1 = vector in ℛ+^N

        Outputs:
        ---------
            * δ = the distance
        """

    δ2 = δ_𝒮ℋplusplus(𝚺_0, 𝚺_1)**2 + δ_ℛplusplus(𝛕_0, 𝛕_1)**2
    return np.sqrt(δ2)

def δ_𝓜p1(𝚺_0, 𝚺_1, τ_0, τ_1):
    """ Fonction for computing the Riemannian distance between two parameters in manifold
        PDH x R+
        ----------------------------------------------------------------------------------
        Inputs:
        --------
            * 𝚺_0 = PDH matrix of dimension p
            * 𝚺_1 = PDH matrix of dimension p
            * τ_0 = scalar in ℛ+
            * τ_1 = scalar in ℛ+

        Outputs:
        ---------
            * δ = the distance
        """

    δ2 = δ_𝒮ℋplusplus(𝚺_0, 𝚺_1)**2 + (τ_0-τ_1)**2
    return np.sqrt(δ2)

# ---------------------------------------------------------------------------------------------------------------
# Definition of functions for Monte - Carlo
# ---------------------------------------------------------------------------------------------------------------

def one_monte_carlo(trial_no,𝛕, p, n, 𝛍, 𝚺, pseudo_𝚺):

        np.random.seed(trial_no)

        𝐗 = generate_data_compound_Gaussian(𝛕, p, n, 𝛍, 𝚺, pseudo_𝚺)

        m = 9
        r = n-10
        X_obs = X[:,:r]

        #𝚺_SCM = SCM(𝐗)
        #𝚺_SCM = 𝚺_SCM / (np.linalg.det(𝚺_SCM)**(1/p))                       # Normalise shape matrix by determinant
        𝚺_Tyl_full, 𝛕_Tyl_full = tyler_estimator_covariance_normalisedet(X)
        𝚺_Tyl_obs, 𝛕_Tyl_obs = tyler_estimator_covariance_normalisedet(X_obs)
        𝚺_EM, 𝛕_EM, e, ll = estim_block_normalizedet(𝐗, m, r, tol=1e-6, iter_max=100)

        # δ_𝚺_SCM = δ_𝒮ℋplusplus(𝚺, 𝚺_SCM) # Natural distance to true value, only shape matrix
        δ_𝚺_Tyl_full = δ_𝒮ℋplusplus(𝚺, 𝚺_Tyl_full) # Natural distance to true value, only shape matrix
        δ_𝚺_Tyl_obs = δ_𝒮ℋplusplus(𝚺, 𝚺_Tyl_obs) # Natural distance to true value, only shape matrix
        δ_θ_Tyl = 0#δ_𝓜p1(𝚺, 𝚺_Tyl, np.mean(𝛕), np.mean(𝛕_Tyl))
        δ_𝛕_Tyl = 0#δ_ℛplusplus(𝛕, 𝛕_Tyl)
        δ_mean_𝛕_Tyl = 0#(np.mean(𝛕)-np.mean(𝛕_Tyl))**2
        δ_𝚺_EM = δ_𝒮ℋplusplus(𝚺, 𝚺_EM) # Natural distance to true value, only shape matrix

        return [δ_𝚺_Tyl_full, δ_𝚺_Tyl_obs, δ_θ_Tyl, δ_𝛕_Tyl, δ_mean_𝛕_Tyl, δ_𝚺_EM]

def parallel_monte_carlo(𝛕, p, n, 𝛍, 𝚺, pseudo_𝚺, number_of_threads, number_of_trials, Multi):

    # Looping on Monte Carlo Trials
    if Multi:
        results_parallel = Parallel(n_jobs=number_of_threads)(delayed(one_monte_carlo)(iMC,𝛕, p, n, 𝛍, 𝚺, pseudo_𝚺) for iMC in range(number_of_trials))
        results_parallel = np.array(results_parallel)
        δ_𝚺_SCM = np.mean(results_parallel[:,0], axis=0)
        δ_𝚺_Tyl = np.mean(results_parallel[:,1], axis=0)
        δ_θ_Tyl = np.mean(results_parallel[:,2], axis=0)
        δ_𝛕_Tyl = np.mean(results_parallel[:,3], axis=0)
        δ_mean_𝛕_Tyl = np.mean(results_parallel[:,4], axis=0)
        δ_EM = np.mean(results_parallel[:,5], axis=0)
        return δ_𝚺_SCM, δ_𝚺_Tyl, δ_θ_Tyl, δ_𝛕_Tyl, δ_mean_𝛕_Tyl, δ_EM
    else:
        # Results container
        results = []
        for iMC in range(number_of_trials):
            results.append(one_monte_carlo(iMC,𝛕, p, n, 𝛍, 𝚺, pseudo_𝚺))

        results = np.array(results)
        δ_𝚺_SCM = np.mean(results[:,0], axis=0)
        δ_𝚺_Tyl = np.mean(results[:,1], axis=0)
        δ_θ_Tyl = np.mean(results[:,2], axis=0)
        δ_𝛕_Tyl = np.mean(results[:,3], axis=0)
        δ_mean_𝛕_Tyl = np.mean(results[:,4], axis=0)
        δ_EM = np.mean(results_parallel[:,5], axis=0)
        return δ_𝚺_SCM, δ_𝚺_Tyl, δ_θ_Tyl, δ_𝛕_Tyl, δ_mean_𝛕_Tyl, δ_EM

# ---------------------------------------------------------------------------------------------------------------
# Definition of Program Principal
# ---------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    # ---------------------------------------------------------------------------------------------------------------
    # Simulation Parameters
    # ---------------------------------------------------------------------------------------------------------------
    number_of_threads = -1                                  # to use the maximum number of threads : -1
    Multi = True
    p = 10                                                  # Dimension of data
    n_vec = np.unique(np.logspace(1.8,2.5,6).astype(int))  # Number of dates
    number_of_trials = 1                               # Number of trials for each point of the MSE
    𝛍 = np.zeros(p)                                         # Mean of Gaussian distribution
    pseudo_𝚺 = 0                                            # Pseudo-covariance of Gaussian distribution
    ρ = 0.999 * (1+1j)/np.sqrt(2)                           # Toeplitz coefficient for shape matrix
    𝚺 = ToeplitzMatrix(ρ, p)                                # Toeplitz shape matrix
    𝚺 = 𝚺 / (np.linalg.det(𝚺)**(1/p))                       # Normalise shape matrix by determinant
    α = 1                                                # Shape parameter of Gamma texture
    β = 1/α                                                 # Scale parameter of Gamma texture

    # ---------------------------------------------------------------------------------------------------------------
    # Doing estimation for an increasing T and saving the natural distance to true value
    # ---------------------------------------------------------------------------------------------------------------
    # print( '|￣￣￣￣￣￣￣￣￣￣|')
    # print( '|   Launching      |')
    # print( '|   Monte Carlo    |')
    # print( '|   simulation     |' )
    # print( '|＿＿＿＿＿＿＿＿＿＿|')
    # print( ' (\__/) ||')
    # print( ' (•ㅅ•) || ')
    # print( ' / 　 づ')
    print(u"Parameters: p=%d, n=%s, rho=%.2f+i*%.2f, alpha=%.2f, beta=%.2f" % (p,n_vec,ρ.real,ρ.imag,α,β))
    t_beginning = time.time()

    # Distance containers
    δ_𝚺_SCM_container = np.zeros(len(n_vec))
    δ_𝚺_Tyl_container = np.zeros(len(n_vec))
    δ_θ_Tyl_container = np.zeros(len(n_vec))
    δ_𝛕_Tyl_container = np.zeros(len(n_vec))
    δ_mean_𝛕_Tyl_container = np.zeros(len(n_vec))
    δ_EM_container = np.zeros(len(n_vec))

    for i_n, n in enumerate(tqdm(n_vec)):
        # generation textures
        𝛕 = np.random.gamma(α, β, n_vec[i_n])
        δ_𝚺_SCM_container[i_n], δ_𝚺_Tyl_container[i_n], δ_θ_Tyl_container[i_n], δ_𝛕_Tyl_container[i_n], δ_mean_𝛕_Tyl_container[i_n], δ_EM_container[i_n]  = parallel_monte_carlo(𝛕, p, n_vec[i_n], 𝛍, 𝚺, pseudo_𝚺, number_of_threads, number_of_trials, Multi)

    print('Done in %f s'%(time.time()-t_beginning))

    # ---------------------------------------------------------------------------------------------------------------
    # Plotting using Matplotlib
    # ---------------------------------------------------------------------------------------------------------------
    markers = ['o', 's' , '*']#, '8', 'P', 'D', 'X']

    # Natural distance to true value, only texture
    # plt.figure(figsize=(16, 7), dpi=80, facecolor='w')
    plt.figure()
    # plt.subplot(1, 3, 1)
    plt.loglog(n_vec, δ_𝚺_SCM_container, marker=markers[0], label='SCM')
    plt.loglog(n_vec, δ_𝚺_Tyl_container, marker=markers[1], label='Tyler')
    plt.loglog(n_vec, δ_EM_container, marker=markers[2], label='EM')
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\delta_{\mathcal{SH}_{++}}\left( \mathbf{\xi}, \hat{\mathbf{\xi}} \right)$')
    plt.legend()
    plt.title(r"Parameters: $p=%d$, $\rho=%.2f+i*%.2f$, $\alpha=%.2f$, $\beta=%.2f$" % (p,ρ.real,ρ.imag,α,β))

    plt.figure()
    # plt.subplot(1, 3, 1)
    plt.loglog(n_vec, δ_θ_Tyl_container, marker=markers[2], label='Tyler Covar + Mean Textures')
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\delta_{\mathcal{M}_{p,n}}\left( θ, \hat{θ} \right)$')
    plt.legend()
    plt.title(r"Parameters: $p=%d$, $\rho=%.2f+i*%.2f$, $\alpha=%.2f$, $\beta=%.2f$" % (p,ρ.real,ρ.imag,α,β))

    plt.figure()
    # plt.subplot(1, 3, 1)
    plt.loglog(n_vec, δ_𝛕_Tyl_container, marker=markers[2], label='Tyler Textures')
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\delta_{\mathcal{R}_{n}}\left( 𝛕, \hat{𝛕} \right)$')
    plt.legend()
    plt.title(r"Parameters: $p=%d$, $\rho=%.2f+i*%.2f$, $\alpha=%.2f$, $\beta=%.2f$" % (p,ρ.real,ρ.imag,α,β))

    plt.figure()
    # plt.subplot(1, 3, 1)
    plt.loglog(n_vec, δ_mean_𝛕_Tyl_container, marker=markers[2], label='Mean Tyler Textures')
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\delta_{\mathcal{R}}\left( E(𝛕), \hat{E(𝛕)} \right)$')
    plt.legend()
    plt.title(r"Parameters: $p=%d$, $\rho=%.2f+i*%.2f$, $\alpha=%.2f$, $\beta=%.2f$" % (p,ρ.real,ρ.imag,α,β))

    #tikzplotlib.save('Optimization_H0_scale.tex')
    plt.show()
