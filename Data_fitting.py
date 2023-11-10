import itertools as it

import multiprocess
import numpy as np
import pandas as pd
import uncertainties as unc
from cholesky_decomp import GEVP_fit
from Data_reader import final_data as data
from Main_GEVP_code import GEVP_eigenvalues
from scipy import linalg, special, stats
from tqdm import tqdm


def ottb_combinations(
    bs_samples, min_tmin, max_tmin, min_tmax, max_tmax, min_op, max_op, no_params
):
    """ottb combinations, stands for Operator, tmin, tmax, bootstrap combinations. That is we create an array called filtered_combs
    which is of the form (op value, tmin value, tmax value, bs sample), more explicitly:

    [ (min_op, min_tmin, min_tmax, 1st bs_sample), ... ,(min_op, min_tmin, min_tmax, final bs_sample),
    (min_op, min_tmin, (min_tmax + 1), 1st bs_sample), ..., (min_op, min_tmin, (min_tmax + 1), final bs_sample),
    ... ,
    (min_op, min_tmin, max_tmax, final bs_sample),
    ... ,
    (max_op, max_tmin, max_tmax, final bs_sample) ]

    which can be itterated over when doing many fits."""

    #### Generate the ranges of tmin, tmax, operators and bs samples we wish to consider ####
    bs_no = np.arange(bs_samples)
    tmin = np.arange(min_tmin, max_tmin + 1)
    tmax = np.arange(min_tmax, max_tmax + 1)
    ops = np.arange(min_op, max_op + 1)

    #### Produce all possible combinations using itertools ####
    combinations = it.product(ops, tmin, tmax, bs_no)

    #### Filter out invalid combinations, such as those which cannot be fitted due to an insufficient number of degrees of freedom ####
    filtered_combs = [i for i in combinations if i[2] - no_params > i[1]]

    return np.array(filtered_combs)


def det_and_inv_cov(bs_samples, ottb_combinations, eigenvalues):
    """Generate all required inverse covariance matrices. This is done by utilising the output from the ottb_combinations function.
    Function returns a list containing matrices."""

    #### i[0] = operator, i[1] = tmin and i[2] = tmax. The +1 is to maintain inclusivity.

    #### the ottb_combinations[::bs_samples] indexing filters out combinations which only differ in the bs parameter, this is done
    #### as this parameter has no bearing on the covariance matrix. Hence not filtering it would result in many identical covariance
    #### matrices being calculated unnecessarily. ####
    unique_combs = ottb_combinations[::bs_samples]
    cov_mats = [
        np.cov(eigenvalues[:, i[1] : i[2] + 1, i[0]], rowvar=False, bias=True)
        for i in unique_combs
    ]
    cov_dets = [linalg.det(i) for i in cov_mats]
    inv_covs = [linalg.inv(i) for i in cov_mats]

    return inv_covs, cov_dets


def ot_tt_p_values(Ndof, chisq):
    """This function returns a so-called one-tailed and two-tailed
    p-value. The two-tailed p-value is whatever is smaller the p-value
    or (1 - p-value), where the p-value and the one-tailed p-value are
    the same thing.

    For a reasonable fit one_tailed ~ 0.5, while two_tailed ~ 1."""

    #### the 1/2 factors come the default definition of the gammaincc function
    one_tailed = special.gammaincc(Ndof / 2.0, chisq / 2.0)
    two_tailed = 2 * np.min([one_tailed, 1 - one_tailed], axis=0)

    return one_tailed, two_tailed


def mean_fit(
    bs_samples,
    max_tmax,
    no_params,
    t_0,
    init_guess,
    ottb_combinations,
    eigenvalues,
    inv_covs,
):
    """This function will compute the mean eigenvalue configurations for each operator and then fit them. It will return the
    optimized paramenetrs and the two-tailed p-value.

    This is done by first constructing a list of tuples. Each tuple contains a time array over a t-range determined by the
    ottb_combination, t_0, an eigenvalue array (for a certain operator and certain bs_sample) over the same t-range,
    the inverse covariance matrices, the initial guess and the maximum number of iterations performed in the minimisation.

    We need this peculiar arrangement as it needs to fit the vectorized syntax associated with multiprocessing.
    """

    #### Compute average configurations. ####
    avg_config = np.mean(eigenvalues, axis=0)

    #### Remove ottb_combinations varying in only bs_sample (same process as used in the det_and_inv_cov function). ####
    unique_combinations = ottb_combinations[::bs_samples]

    #### Generate the time and eigenvalue arrays described in the docstring.
    #### From ottb_combinations i[0] = operator, i[1] = tmin, i[2] = tmax. ####
    time_and_eigs = [
        (np.arange(0, max_tmax + 1)[i[1] : i[2] + 1], avg_config[i[1] : i[2] + 1, i[0]])
        for i in unique_combinations
    ]

    #### Combine into final fitting parameters. ####
    fitting_params = [
        (time_and_eigs[i][0], t_0, time_and_eigs[i][1], inv_covs[i], init_guess, 15000)
        for i in range(len(unique_combinations))
    ]

    #### Perform fitting using multiprocessing and return results. ####
    with multiprocess.Pool() as pool:
        results = pool.starmap(GEVP_fit, fitting_params)

    #### Separate parameters and chisqs. Then calculate the degrees of
    # freedom for each chisq. ####
    parameters = np.array([inner_tuple[0] for inner_tuple in results])
    chisqs = np.array([inner_tuple[1] for inner_tuple in results])
    ndofs = np.array(
        [
            unique_combinations[i][2] - unique_combinations[i][1] - no_params + 1
            for i in range(len(chisqs))
        ]
    )
    reduced_chisqs = [chisqs[i] / ndofs[i] for i in range(len(chisqs))]

    #### Calculate two tailed p-values. ####
    ot_p_vals, tt_p_vals = ot_tt_p_values(ndofs, chisqs)

    #### Output is a list of two arrays. The first array is an array of lists,
    # in which are the optimized parameters. The second array just contains
    # the two tailed p-vauels.
    return parameters, ot_p_vals, tt_p_vals, reduced_chisqs, ndofs


def parameter_errors(
    bs_samples, max_tmax, t_0, init_guesses, eigenvalues, ottb_combinations, inv_covs
):
    """This function does the same thing as the mean_fit function, but on all bootstrap samples,
    not just the mean configurations.
    """

    #### From ottb_combinations i[0] = operator, i[1] = tmin, i[2] = tmax and i[3] = bs_sample.####

    time_and_eigs = [
        (
            np.arange(0, max_tmax + 1)[i[1] : i[2] + 1],
            eigenvalues[i[3]][i[1] : i[2] + 1, i[0]],
        )
        for i in ottb_combinations
    ]

    #### Extend the inv_covs. We need to do this as we will need to iterate over bs_samples, however, our det_and_inv_cov function
    #### does not account for this as we would need to calculate the same matrix for each bs_sample. Instead we don't do this, but
    #### instead just repeat the result for each bs_sample. We do the same for the init_guesses ####

    # extended_inv_covs = np.repeat(inv_covs, bs_samples, axis=0)
    extended_inv_covs = list(it.chain(*[[item] * bs_samples for item in inv_covs]))
    # result = list(chain(*[[item] * repeat_factor for item in original_list]))
    extended_init_guesses = np.repeat(init_guesses, bs_samples, axis=0)

    #### Combine with other required parameters. ####
    fitting_params = [
        (
            time_and_eigs[i][0],
            t_0,
            time_and_eigs[i][1],
            extended_inv_covs[i],
            extended_init_guesses[i],
            15000,
        )
        for i in range(len(ottb_combinations))
    ]

    #### Use multiprocessing to do fits. ####
    with multiprocess.Pool() as pool:
        results = pool.starmap(GEVP_fit, fitting_params)

    #### Separate the outputted parameters from the outputted chi-squared values.####
    output_params = np.array([inner_tuple[0] for inner_tuple in results])

    #### Compute and return the errors.
    # The unusual indexing: output_params[i : i + bs_samples] when iterated
    # over groups the [ith, (i + bs_samples)th] elements in output_params
    # the std is then performed on the group. The
    # 'for i in range(0, len(output_params), bs_samples)' then prevents i
    # from going from 0 to 1 but instead from 0 to bs_samples. All together
    # we calcuate the std for each bs_sample. ####
    param_errs = [
        stats.tstd(output_params[i : i + bs_samples], axis=0)
        for i in range(0, len(output_params), bs_samples)
    ]
    return param_errs


def ufloat_errors(value, error):
    """This function takes in a value (or list of values) and an
    error (or list of errors) and returns an appropriate ufloat (or
    list of ufloats)."""

    size = len(value)
    if size == 1:
        return unc.ufloat(value, error)

    else:
        return [unc.ufloat(value[i], error[i]) for i in range(size)]


def shorthand_error(value, error):
    """This function takes in a value and its corresponding error and
    returns them in shorthand notation. Example:
    Measurement = 0.125 +/- 0.012555
    Value = 0.125
    Error = 0.012555
    Function output = 0.125(13)."""

    #### Create ufloat uncertainty. ####
    pm_error = unc.ufloat(value, error)

    #### Transform to shorthand format. Note this function returns a
    # string. ####
    short_error = "{:.2uS}".format(pm_error)
    return short_error


#############################################################################
if __name__ == "__main__":
    data = np.delete(np.delete(data, 4, axis=2), 4, axis=3)  # del op4
    data = np.delete(np.delete(data, 4, axis=2), 4, axis=3)  # del op5
    data = np.delete(np.delete(data, 4, axis=2), 4, axis=3)  # del op6
    data = np.delete(np.delete(data, 4, axis=2), 4, axis=3)  # del op7
    # data = np.delete(np.delete(data, 4, axis=2), 4, axis=3)  # del op8
    data = data[:, :, 0:5, 0:5]
    bs_samples = 500
    min_t0 = 3
    max_t0 = 4
    # min_tmin = 2
    max_tmin = 11
    min_tmax = 6
    max_tmax = 14
    min_op = 0
    max_op = 4
    no_params = 3
    init_guess_central = [0, 0.3, 1.5]

    final_results = []

    for t_0 in tqdm(np.arange(min_t0, max_t0 + 1)):
        min_tmin = t_0
        #### Generate ottb combinations. ####
        combs = ottb_combinations(
            bs_samples,
            min_tmin,
            max_tmin,
            min_tmax,
            max_tmax,
            min_op,
            max_op,
            no_params,
        )

        #### Acquire eigenvalues. ####
        # eigenvals = np.load("GEVP/GEVP_output.npy")[0, :, :, :]  # (t_0, bs, t, ops)
        # eigenvals = GEVP_eigenvalues(data, t_0, bs_samples)  # (no_bs, no_ts, 10)
        eigenvals = np.load("C:/Users/jm1n22/test_sftp/GEVP_ops_01238.npy")[
            t_0 - 3, :, :, :
        ]

        #### Acquire inverse covariance matrices. ####
        invs_and_dets = det_and_inv_cov(bs_samples, combs, eigenvals)
        invcovs = invs_and_dets[0]
        cov_dets = invs_and_dets[1]

        #### Compute central values. ####
        fit_params_and_pvals = mean_fit(
            bs_samples,
            max_tmax,
            no_params,
            t_0,
            init_guess_central,
            combs,
            eigenvals,
            invcovs,
        )

        #### Feed central values into subsequent fits to speed up computation. ####
        fit_params = fit_params_and_pvals[0]
        otpvals = fit_params_and_pvals[1]
        ttpvals = fit_params_and_pvals[2]
        red_chis = fit_params_and_pvals[3]
        dofs = fit_params_and_pvals[4]

        #### Set initial guess for next t_0 loop to be the first set of fitted
        # parameters from the first loop. This is logical as the first set will
        # be of the form ottb = (op, tmin, tmax, bs) at t_0 and it will be
        # being used as the initial guess for ottb = (op, tmin, tmax, bs) at
        # t_0 + 1.
        init_guess_central = fit_params[0]

        #### Compute parameter errors. ####
        fit_errors = parameter_errors(
            bs_samples, max_tmax, t_0, fit_params, eigenvals, combs, invcovs
        )

        #### Output relevant parameters. ####
        # We need unique combinations, but don't care about the bs parameter anymore
        # as we have now averaged over them where necessary. Thus define non_bs_combs.
        non_bs_combs = combs[::bs_samples]
        output_params = [
            [
                non_bs_combs[i],
                t_0,
                fit_params[i],
                fit_errors[i],
                otpvals[i],
                ttpvals[i],
                red_chis[i],
                dofs[i],
            ]
            for i in range(len(non_bs_combs))
        ]
        final_results.append(output_params)

    #### final results can be a list of lists of lists. So we flatten
    # and the distribute everything. ####
    flat_results = [i for inner in final_results for i in inner]

    flat_t_0s = [i[1] for i in flat_results]
    flat_otpvals = [i[4] for i in flat_results]
    flat_ttpvals = [i[5] for i in flat_results]
    flat_red_chis = [i[6] for i in flat_results]
    flat_dofs = [i[7] for i in flat_results]

    flat_time_params = np.array([i[0] for i in flat_results])
    flat_params = np.array([i[2] for i in flat_results])
    flat_errs = np.array([i[3] for i in flat_results])

    #### Make parameters and their errors into ufloats. ####
    A_ufloats = ufloat_errors(flat_params[:, 0], flat_errs[:, 0])
    A_ufloats = ["{:.2uS}".format(i) for i in A_ufloats]
    E_ufloats = ufloat_errors(flat_params[:, 1], flat_errs[:, 1])
    E_ufloats = ["{:.2uS}".format(i) for i in E_ufloats]
    Est_ufloats = ufloat_errors(flat_params[:, 2], flat_errs[:, 2])
    Est_ufloats = ["{:.2uS}".format(i) for i in Est_ufloats]

    #### Arrange data into a DataFrame. ####
    output_data = {
        "Operator": flat_time_params[:, 0],
        "t_0": flat_t_0s,
        "t_min": flat_time_params[:, 1],
        "t_max": flat_time_params[:, 2],
        "one tailed p-value": flat_otpvals,
        "two tailed p-value": flat_ttpvals,
        "Reduced chi-squareds": flat_red_chis,
        "dofs": flat_dofs,
        "A_n": A_ufloats,
        "E_n": E_ufloats,
        "E*_n": Est_ufloats,
    }
    df = pd.DataFrame(output_data)

    #### Drop combinations with < 4 dofs for operators 0,1,2,3. ####
    df0123 = df[(df["Operator"] < 4) & (df["dofs"] > 3)]
    df8 = df[(df["Operator"] == 4) & (df["dofs"] > 1)]

    df_1cut = pd.concat([df0123, df8])

    #### Apply tmin and tmax cuts. ####
    df_tmin = df_1cut[(df_1cut["t_min"] < 8) & (df_1cut["t_min"] > 4)]
    df_op01 = df_tmin[(df["Operator"] == 0) | (df["Operator"] == 1)]
    df_op2 = df_tmin[(df["Operator"] == 2) & (df["t_max"] < 14)]
    df_op3 = df_tmin[(df["Operator"] == 3) & (df["t_max"] < 13)]
    df_op8 = df_tmin[(df["Operator"] == 4) & (df["t_max"] < 11)]

    df_2cut = pd.concat([df_op01, df_op2, df_op3, df_op8])

    #### Apply two-tailed p-value cut. ####
    df_3cut = df_2cut[
        (df_2cut["one tailed p-value"] < 0.9) & (df_2cut["one tailed p-value"] > 0.1)
    ]

    # Create an ExcelWriter object
    excel_file = "48_01238_t0_34(TEST).xlsx"
    with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Raw Data", index=False)
        df_1cut.to_excel(writer, sheet_name="Dofs cut", index=False)
        df_2cut.to_excel(writer, sheet_name="t-range cut", index=False)
        df_3cut.to_excel(writer, sheet_name="P-values cut", index=False)

    ##### Sort the data frame first by Operator, the by t_0 and finally by ...
    # sorted_df = df.sort_values(
    #    by=["Operator", "t_0", "two_tailed_p-value"], ascending=[True, True, False]
    # )
    #

    # result_df.to_excel(
    #    "48_0_38_t0_2_6_no3summary.xlsx", startcol=2, startrow=2, index=False
    # )
    # df = df.drop(df[(df.score < 50) & (df.score > 20)].index)
