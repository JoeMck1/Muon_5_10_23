import GEVP_functions as GEVP
import numpy as np

"""GEVP outline:

Starting GEVP: 
C(t)v = lambda C(t_0)v,

where C(t) is the matrix of correlator, v is the eigenvector, lambda is
the eigenvalue and C(t_0) is the correlator at time t_0.

Decomposition:
C(t_0) can, due to its positive-definiteness, be decomposed to
C(t_0) = U Sigma U^T, where Sigma is a matrix with eigenvalues along
the diagonal and U is a matrix whose columns are the normalized
eigenvectors of C(t_0).

Algebra:

C(t)v = lambda U Sigma U^T v

can be rearranged to 

(Sigma^(-1/2) U^T C(t) U Sigma^(-1/2)) Sigma^(1/2) U^T v
= lambda Sigma^(1/2) U^T v

Redefine variables:

let C_tilde(t) = Sigma^(-1/2) U^T C(t) U Sigma^(-1/2) and
v_tilde = Sigma^(1/2) U^T v. Thus the GEVP has been reduced to an
ordinary eigenvalue problem:

C_tilde(t) v_tilde = lambda v_tilde.

We the solve for lambda."""


def GEVP_eigenvalues(data, t_0, no_bs):
    #### Generate bootstrapped configurations. ####
    bs_configs = GEVP.bs_mat(data, no_bs)

    #### We now enforce symmetrization upon the correlator matrices.
    # This is permitted since it is only statistical fluctuations which
    # casue C(t) to not be symmetric.
    # Note, GEVP.config_symmetrizer symmetrizes all matrices in a given
    # configuration. ####
    symm = np.array([GEVP.config_symmetrizer(x) for x in bs_configs])

    #### Fold data across time-slices. ####
    folding = np.array([GEVP.fold_mats(x) for x in symm])

    #### number of time-slices after folding, will be needed later ####
    no_ts = len(folding[0])

    #### Acquire average correlator matrix at time t_0 to use as
    # reference for later reordering of all C(t_0) eigenvalues and
    # eigenvectors.
    C_a_t0 = GEVP.C_a_t0(folding, t_0)

    #### Acquire eigenvalues for C_a_t0 ####
    C_a_t0_evecs = np.linalg.eigh(C_a_t0)[1]

    #### Acquire eigenvalues and eigenvectors for all C(t_0). These
    # eigenvalues and eigenvectors will be unordered. ####
    C_t0_evals, C_t0_evecs = np.linalg.eigh(folding[:, t_0, :, :])

    #### Generate the reordering templates to properly order the C(t_0)
    # eigenvalues and eigenvectors. ####
    templates = [GEVP.reordering_template(C_a_t0_evecs, x) for x in C_t0_evecs]

    #### Order the C_t0 eigenvalues and eigenvectors using the above
    # templates. ####
    o_C_t0_evals, o_C_t0_evecs = zip(
        *[
            GEVP.order_mat(templates[i], 0, C_t0_evals[i], C_t0_evecs[i])
            for i in range(no_bs)
        ]
    )

    #### Construct C_tilde(t) = Sigma^(-1/2) U^T C(t) U Sigma^(-1/2).
    # First need the Sigma^(-1/2) which we call diag_evals ####
    inv_sqrt_evals = np.array(o_C_t0_evals) ** (-1 / 2)
    diag_evals = [np.diag(x) for x in inv_sqrt_evals]

    C_tilde = [
        diag_evals[i] @ o_C_t0_evecs[i].T @ folding[i] @ o_C_t0_evecs[i] @ diag_evals[i]
        for i in range(no_bs)
    ]

    #################################################################
    # First half of GEVP is now done. Now just use the same methods #
    # to solve the resulting ordinary eigenvalue equation.
    #################################################################

    #### Find all average C_tilde(t). ####
    C_a_tilde = np.mean(C_tilde, axis=0)

    #### Find unordered eigenvalues and eigenvectors of all
    # C_a_tilde(t). ####
    C_a_tilde_evals, C_a_tilde_evecs = np.linalg.eigh(C_a_tilde)

    #### Use C_a_tilde(t_0 + 1) as a reference to reorder
    # C_a_tilde(t)'s. First will need reordering templates again. ####

    C_a_tilde_temps = [
        GEVP.reordering_template(C_a_tilde[t_0 + 1], i) for i in C_a_tilde_evecs
    ]

    #### Reorder C_a_tilde eigenvectors using above templates. ####
    o_C_a_tilde_evecs = [
        GEVP.order_mat(C_a_tilde_temps[i], 0, C_a_tilde_evals[i], C_a_tilde_evecs[i])[1]
        for i in range(no_ts)
    ]

    #### Find unordered eigenvalues and eigenvectors of all C_tilde(t).
    C_tilde_evals, C_tilde_evecs = zip(*[np.linalg.eigh(i) for i in C_tilde])

    #### Now we use the ordered C_a_tilde eigenvectors as references to
    # reorder the all the C_tilde(t) eigenvalues and eigenvectors.
    # First step is again to generate the templates. ####
    C_tilde_temps = [
        [
            GEVP.reordering_template(o_C_a_tilde_evecs[i], C_tilde_evecs[j][i])
            for i in range(no_ts)
        ]
        for j in range(no_bs)
    ]

    #### Reorder C_tilde eigenvalues using the above templates. ####
    o_C_tilde_evals = [
        [
            GEVP.order_mat(
                C_tilde_temps[j][i], 0, C_tilde_evals[j][i], C_tilde_evecs[j][i]
            )[0]
            for i in range(no_ts)
        ]
        for j in range(no_bs)
    ]

    #### Reverse order of C_tilde eigenvalues. ####
    return np.flip(o_C_tilde_evals, axis=2)
