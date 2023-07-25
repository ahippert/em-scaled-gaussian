import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import time
from joblib import Parallel, delayed
from tqdm import tqdm
from TestTyler_Parallel import δ_𝒮ℋplusplus, ToeplitzMatrix, SCM
import gaps_gen as gg
from covariance_scaled_gaussian import tyler_estimator_covariance_normalisedet, EM_tyler_estimator
from covariance_gaussian import EM_gaussian, low_rank_covariance
import os
import boxplot

import seaborn as sns
sns.set(style="ticks",font_scale=1.25)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

def generate_data_for_estimation(p, N, 𝛍, 𝚺, 𝛕):
    𝐗 = np.empty((p,N))
    𝐗 = np.sqrt(𝛕)[None,:] * np.random.multivariate_normal(𝛍, 𝚺, N).T
    return 𝐗

def one_monte_carlo(trial_no, S, 𝛕, 𝛍, n, p):

    np.random.seed(trial_no)

    # Generate multivariate data without missing data
    X = generate_data_for_estimation(p, n, 𝛍, S, 𝛕)

    # Make a copy of the full data set
    X_full = copy.copy(X)

    # Generate missing data

    # Random pattern of missing values
    #percent = int(40*(63/n))
    #X = gg.generate_random_gaps(X, percent/100.)

    # General pattern of missing values
    X = gg.generate_general_pattern(X, n)

    # Generate one block of missing values
    #X = gg.generate_one_block(X, n)

    # Make copies of the data with missing values
    Y = copy.copy(X)
    Z = copy.copy(X)

    # Get observation indices where values are missing and observed
    N = np.arange(n)
    N_obs = N[np.sum(np.isnan(X), axis=0)==0]
    P = np.arange(p)
    P_mis = P[np.sum(np.isnan(X),axis=1)>0]

    # Various estimators are computed in the following

    # EM-Tyl estimator and its low-rank version
    EM_Tyler, tau_hat = EM_tyler_estimator(Y, tol=1e-4, iter_max=50, lowrank=False, rank=5)
    EM_Tyler_lr, tau_hat_lr = EM_tyler_estimator(Y, tol=1e-4, iter_max=50, lowrank=True, rank=5)

    # Mean imputation
    X_mean_imput = copy.copy(X) # Data on which mean imputation is performed
    X_mult_imput = copy.copy(X) # ---- -- ----- multiple imputation -- ---------
    mean_p = np.nanmean(X, axis=1)
    var_p = np.nanvar(X, axis=1)
    for i in P_mis:
        len_mis = len(X[i][np.isnan(X[i])])
        to = np.random.gamma(1, 1, len_mis)
        imput_array = np.sqrt(to)*np.random.normal(mean_p[i], var_p[i], len_mis)
        X_mean_imput[i][np.isnan(X_mean_imput[i])] = mean_p[i] # Mean imputation

    # Perform Multiple imputation and compute Tyler's estimator on imputed data
    trials = 5
    Tyler_k, Tyler_k_lr = 0., 0.
    for k in range(trials):
        for i in P_mis:
            len_mis = len(X[i][np.isnan(X[i])])
            to = np.random.gamma(1, 1, len_mis)
            X_mult_imput[i][np.isnan(X[i])] = np.sqrt(to)*np.random.normal(mean_p[i], var_p[i], len_mis)
        Tyler_k += tyler_estimator_covariance_normalisedet(X_mult_imput, False, 5)[0]
        Tyler_k_lr += tyler_estimator_covariance_normalisedet(X_mult_imput, True, 5)[0]
    Tyler_mult_imput = Tyler_k/trials
    Tyler_mult_imput_lr = Tyler_k_lr/trials

    # Tyler's estimator on clairvoyant data (no missing data)
    Tyler, tau = tyler_estimator_covariance_normalisedet(X_full, False, 5)
    Tyler_lr, tau = tyler_estimator_covariance_normalisedet(X_full, True, 5)

    # SCM estimator on clairvoyant data
    Scm = SCM(X_full)
    Scm_lr = low_rank_covariance(Scm, rank=5)

    # Tyler's and SCM estimators on observed data only
    if X_full[:,N_obs].shape[1] < 2*X_full[:,N_obs].shape[0]:
         Tyler_obs, tau_obs = [np.nan], [np.nan]
         Scm_obs = [np.nan]
    else:
        Tyler_obs, tau_obs = tyler_estimator_covariance_normalisedet(X_full[:,N_obs], False, 5)
        Scm_obs = SCM(X_full[:,N_obs])

    # Tyler's estimator on data imputed by the mean
    Tyler_mean_imput, _ = tyler_estimator_covariance_normalisedet(X_mean_imput, False, 5)
    Tyler_mean_imput_lr, _ = tyler_estimator_covariance_normalisedet(X_mean_imput, True, 5)

    # EM-SCM estimator and its low-rank version
    EM_SCM = EM_gaussian(Y, tol=1e-4, iter_max=50, lowrank=False, rank=5)
    EM_SCM_lr = EM_gaussian(Y, tol=1e-4, iter_max=50, lowrank=True, rank=5)

    estimators = [EM_Tyler, EM_SCM, Tyler, Scm, Tyler_obs, Scm_obs,
                  Tyler_mean_imput, Tyler_mult_imput, EM_Tyler_lr, EM_SCM_lr,
                  Tyler_lr, Scm_lr, Tyler_mean_imput_lr, Tyler_mult_imput_lr]

    deltas = []
    for estimator in estimators:
        if len(estimator) == p:
            deltas.append(δ_𝒮ℋplusplus(S, estimator))
        else:
            deltas.append(np.nan)
    return deltas

def parallel_monte_carlo(𝚺, 𝛕, mean, n, p, number_of_threads, number_of_trials, Multi):

    # Looping on Monte Carlo Trials
    if Multi:
        results_parallel = Parallel(n_jobs=number_of_threads)(delayed(one_monte_carlo)(iMC, 𝚺, 𝛕, mean, n, p) for iMC in range(number_of_trials))
        results_parallel = np.array(results_parallel)
        return results_parallel
    else:
        # Results container
        results = []
        for iMC in range(number_of_trials):
            results.append(one_monte_carlo(iMC, 𝚺, 𝛕, mean, n, p))
        results = np.array(results)
        return results

#  TESTS

# Generate real mean and covariance
p = 15                                 # nb of variables
mean = np.zeros(p)
rho = 0.999/np.sqrt(2)                 # correlation (Toeplitz) oefficient
𝚺  = ToeplitzMatrix(rho, p)
𝚺 = 𝚺 / (np.linalg.det(𝚺)**(1/p))     # Normalise shape matrix by determinant

# Low-rank structure of matrix
rank = 5
R = low_rank_covariance(𝚺, rank)
R = R / (np.linalg.det(R)**(1/p))

number_of_threads = -1              # to use the maximum number of threads
Multi = True
n_vec = np.unique(np.logspace(1.8,3,6).astype(int))
α = 1                               # Shape parameter of Gamma texture
β = 1/α                             # Scale parameter of Gamma texture
number_of_trials = 2

# # # ---------------------------------------------------------------------------------------------------------------
# # # Doing estimation for an increasing N and saving the natural distance to true value
# # # ---------------------------------------------------------------------------------------------------------------
print( '|￣￣￣￣￣￣￣￣￣￣|')
print( '|   Launching        |')
print( '|   Monte Carlo      |')
print( '|   simulation       |' )
print( '|＿＿＿＿＿＿＿＿＿＿|')
print( ' (\__/) ||')
print( ' (•ㅅ•) || ')
print( ' / 　 づ')
print(u"Parameters: p=%d, n=%s" % (p,n_vec))

t_beginning = time.time()

# Distance container cube
number_of_δ = 14
δ_𝚺_container = np.zeros((len(n_vec),number_of_trials,number_of_δ))

for i_n, n in enumerate(tqdm(n_vec)):
    𝛕 = np.random.gamma(α, 1/α, n_vec[i_n])
    δ_𝚺_container[i_n] = parallel_monte_carlo(R, 𝛕, mean, n_vec[i_n], p, number_of_threads, number_of_trials, Multi)

print('Done in %f s'%(time.time()-t_beginning))

# Compute mean of Monte Carlo experiments
data = []
for i in range(number_of_δ):
    data.append(np.nanmean(δ_𝚺_container[:,:,i], axis=1))

# Plot boxplots
np.save("boxplot", δ_𝚺_container[2,:,:])
# showfig, savefig = True, False
# boxplot.plot_boxplot(showfig, savefig)

# Put data in text file saved in current directory
savefig = False
if savefig:
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    g = open(os.path.join(__location__, "./results/distances_to_plot.txt"), "w")
    g.write("N\t"+"EM_Tyler\t"+"EM_SCM\t"+"Tyler\t"+"SCM\t"+"Tyler_obs\t"+"SCM_obs\t"+"Mean\t"+"Mult\t"+"EM_tyler_lr\t"+"EM_SCM_lr\t"+"Tyler_lr\t"+"SCM_lr\t"+"Mean_lr\t"+"Mult_lr\n")
    for i in range(len(n_vec)):
        g.write("%d\t"%n_vec[i]+"%0.8f\t"%(data[0][i])+"%0.8f\t"%(data[1][i])+"%0.8f\t"%(data[2][i])+"%0.8f\t"%(data[3][i])+"%0.8f\t"%(data[4][i])+"%0.8f\t"%(data[5][i])+"%0.8f\t"%(data[6][i])+"%0.8f\t"%(data[7][i])+"%0.8f\t"%(data[8][i])+"%0.8f\t"%(data[9][i])+"%0.8f\t"%(data[10][i])+"%0.8f\t"%(data[11][i])+"%0.8f\t"%(data[12][i])+"%0.8f\n"%(data[13][i]))
    g.close()

