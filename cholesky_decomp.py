import numpy as np
import math
from scipy.optimize import least_squares as ls

def single_exp(params, t): # model function
    return params[0]*math.exp(-params[1]*t)

def double_exp(params, t): # model function
    return params[0]*math.exp(-params[1]*t) + params[2]*math.exp(-params[3]*t)

def GEVP_fn(params, t, t_0):
    # Add a penalty term for the constraint (E'_n < E_n)
    if params[1] >= params[2]:
        penalty = 1e6  # Choose a large penalty term to discourage violation of the constraint
        return (1-params[0])*np.exp(-params[1]*(t-t_0)) + params[0]*np.exp(-params[2]*(t-t_0)) + penalty

    else:
        return (1-params[0])*np.exp(-params[1]*(t-t_0)) + params[0]*np.exp(-params[2]*(t-t_0))

def single_exp_fit(t, data, inv_cov, init_guess, evaluation_number, bounds):

    ''''Function which applies a single exponential fit to data using Cholesky decomposition. 
        The bounds argument provides bounds for the minimisation parameters, this stops the minimistaion
        routine running off to unphysical results (lower bounds are currently set to zero).'''
    
    b = bounds

    def LD(params, t, data, inv_cov):

        diff_vec = single_exp(params, t) - data
        cholesky_decomp = np.linalg.cholesky(inv_cov)
        return np.dot(cholesky_decomp.T, diff_vec)
    
    result = ls(LD, init_guess, args=(t, data, inv_cov), max_nfev = evaluation_number, bounds=([0,0],[b,b]))
    return result.x, np.dot(np.transpose(LD(result.x, t, data, inv_cov)), LD(result.x, t, data, inv_cov))

def double_exp_fit(t, data, inv_cov, init_guess, evaluation_number, bounds):

    ''''Function which applies a double exponential fit to data using Cholesky decomposition. 
        The bounds argument provides bounds for the minimisation parameters, this stops the minimistaion
        routine running off to unphysical results (lower bounds are currently set to zero).'''
    
    b = bounds

    def LD(params, t, data, inv_cov):

        diff_vec = double_exp(params, t) - data
        cholesky_decomp = np.linalg.cholesky(inv_cov)
        return np.dot(cholesky_decomp.T, diff_vec) #THIS COULD BE WRONG, MAY NEED TRANSPOSING
    
    result = ls(LD, init_guess, args=(t, data, inv_cov), max_nfev = evaluation_number, bounds=([0,0,0,0],[b,b,b,b]))
    return result.x, np.dot(np.transpose(LD(result.x, t, data, inv_cov)), LD(result.x, t, data, inv_cov))

def GEVP_fit(t, t_0, data, inv_cov, init_guess, evaluation_number, bounds = np.inf):
    
    b = bounds
    def LD(params, t, data, inv_cov):
        diff_vec = GEVP_fn(params, t, t_0) - data
        cholesky_decomp = np.linalg.cholesky(inv_cov)
        return np.dot(cholesky_decomp.T, diff_vec)
    
    result = ls(LD, init_guess, args=(t, data, inv_cov), max_nfev = evaluation_number, bounds=([0,0,0],[b,b,b]))
    return result.x, np.dot(np.transpose(LD(result.x, t, data, inv_cov)), LD(result.x, t, data, inv_cov))

def cov_mat_bs(configs, x):

    ''''Function to determine covariance matrix for bootstrap sampling.
        If the matrix is singular only the diagonal is returned. 
        
        If x = True 1/N normalisation is used, else 1/(N-1) is used.'''
    
    cov_mat = np.cov(configs, rowvar=False, bias=x)

    if np.linalg.det(cov_mat) == 0:
        return np.diag(np.diag(cov_mat))
    
    else: 
        return cov_mat

