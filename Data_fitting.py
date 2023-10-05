import multiprocess

import numpy as np
import pandas as pd
import itertools as it

from scipy import linalg
from tqdm import tqdm
from Data_reader import final_data as data
from Main_GEVP_code import GEVP_eigenvalues
from cholesky_decomp import GEVP_fit
from cholesky_decomp import cov_mat_bs

def GEVP_fitting(bs_samples, data, min_t_0, max_t_0, min_op, max_op, max_t_min, min_t_max, max_t_max, no_params):

    '''This function computes and fits GEVP data for all permutations of operator number, t_0, t_min and t_max
    based off the inputted parameters. This function also employs multiprocessing.'''

    output_parameters = []

    for t_0 in tqdm(np.arange(min_t_0, max_t_0 + 1)):

        #### perform GEVP to get eigenvalues to fit ####
        eigenvalues = GEVP_eigenvalues(data, t_0, bs_samples) #(no_bs, no_ts, 10)
        
        #### generate all parameters needed for fitting, in a format compatible with multiprocessing syntax. 
        # These parameters include the time range to be fitted over, t_0, the eigenvalues we're fitting, the 
        # inverse covariance matrix, the initial guess for fitting, the maximum number of desired fitting itterations,
        # the bounds on the optimized parameters and a list of [op_no, t_0, t_min, t_max] which is used to track valid
        # combinations of such parameters. ####

        bs_no = np.arange(bs_samples)
        tmin = np.arange(t_0, max_t_min +1)
        tmax = np.arange(min_t_max, max_t_max + 1)
        ops = np.arange(min_op, max_op + 1)
        combinations = it.product(ops, tmin, tmax, bs_no) # all combinations of operator, tmin, tmax and bs number.
        filtered_combs = [i for i in combinations if i[2] - no_params > i[1] + 2] # filter out invalid combinations
        non_bs_combs = filtered_combs[::bs_samples] # disregard combinations which only differ in bs number

        time = np.arange(t_0, max_t_max +1)
        cov_mats = [np.cov(eigenvalues[:, i[1]:i[2]+1, i[0]], rowvar=False, bias=True) for i in filtered_combs] #[(2d array), (2d array), ...]
        cov_dets = [linalg.det(i) for i in cov_mats[::bs_samples]]
        inv_covs = [linalg.inv(i) for i in cov_mats]

        time_and_eigs = [(time[i[1]-t_0 : i[2] - t_0+1], eigenvalues[i[3]][i[1]: i[2]+1, i[0]]) for i in filtered_combs]

        fitting_params = [(time_and_eigs[i][0], t_0, time_and_eigs[i][1], inv_covs[i], [0.4,0.45,1.2], 10000, 10) 
                          for i in range(len(filtered_combs))]
        
        #### Compute fitting parameters using multiprocessing. ####
        with multiprocess.Pool() as pool:
            res = (pool.starmap(GEVP_fit, fitting_params))

        #### Unpack and analyse the chis to get reduced chi-squareds.
        chisqs = [inner_tuple[1] for inner_tuple in res]
        chi_bs_avg = [sum(chisqs[i: i+bs_samples]) / bs_samples for i in range(0,len(chisqs),bs_samples)]
        reduced_chis = [chi_bs_avg[i]/(non_bs_combs[i][2] - non_bs_combs[i][1] - no_params) for i in range(len(chi_bs_avg))]

        #### Unpack and analyse the fitting parameters and compute the errors.
        params = np.array([inner_tuple[0] for inner_tuple in res]) #shape = (no_bs, no_params)
        params_bs_avg = [np.mean(params[i:i+bs_samples], axis=0) for i in range(0,len(params),bs_samples)]
        param_errs = [np.var(params[i:i+bs_samples], axis=0)**(1/2) for i in range(0,len(params),bs_samples)]

        output_res = [[non_bs_combs[i], reduced_chis[i], params_bs_avg[i], param_errs[i], t_0, cov_dets[i]] 
                      for i in range(len(reduced_chis))]
        output_parameters.append(output_res)

    #### Flatten everything into simple 1-dimensional lists. ####
    flat_output_params = [i for inner in output_parameters for i in inner]
    flat_t_0s = [i[4] for i in flat_output_params]
    flat_red_chis = [i[1] for i in flat_output_params]
    flat_cov_dets = [i[5] for i in flat_output_params]

    flat_time_params = np.array([i[0] for i in flat_output_params])
    flat_params = np.array([i[2] for i in flat_output_params])
    flat_errs = np.array([i[3] for i in flat_output_params])

    #### Output data to excel ####

    output_data = {'Operator': flat_time_params[:,0],
                    't_0' : flat_t_0s,
                    't_min' : flat_time_params[:,1], 
                    't_max' : flat_time_params[:,2],
                    'A_n' : flat_params[:,0],
                    'A_n error' : flat_errs[:,0], 
                    'E_n' : flat_params[:,1],
                    'E_n error' : flat_errs[:,1],
                    'E*_n': flat_params[:,2],
                    'E*_n error' : flat_errs[:,2],
                    'Reduced chi-squared': flat_red_chis,
                    'Covariance det': flat_cov_dets}
    df = pd.DataFrame(output_data)
    return df.to_excel('48_with_det_cov.xlsx', startcol=2, startrow=2, index=False)

if __name__ == '__main__':
    print(GEVP_fitting(500, data, 3,5,0,5,12,15,35,3))
    #print(GEVP_fitting(5, data, 3,3,0,1,12,15,20,3))