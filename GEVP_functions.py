import random

import numpy as np

"""Functions to be used in GEVP data analysis. They are written in order of usage too"""


def reality_fn(array):
    """Function which removes negligible real or imaginary parts from arrays of complex numbers, based
    ONLY on the first element of the array."""

    if abs(array[0].real) > abs(array[0].imag):
        return array.real

    elif abs(array[0].real) < abs(array[0].imag):
        return array.imag


def bs_mat(configs, no_bs):
    """This function takes in configurations of data and generates a
    number of bootstrap samples, given by the no_samples argument."""

    confs = len(configs)  # number of configurations

    bs = [
        (sum([random.choice(configs) for i in range(confs)])) / confs
        for j in range(no_bs)
    ]

    return np.array(bs)


def fold_mats(array):
    """Function which symmetrically folds an array. The zeroth element is
    not folded, but is retained."""

    if len(array) % 2 == 0:
        folded_matrices = np.array(
            [(array[i] + array[-i]) / 2 for i in range(int(len(array) / 2))]
        )

    elif len(array) % 2 == 1:
        folded_matrices = np.array(
            [(array[i] + array[-i]) / 2 for i in range(int((len(array)) / 2))]
        )

    return folded_matrices


def matrix_symmetrizer(mat):
    """function which takes a matrix as input and returns it symmetrized across the diagonal"""

    return (1 / 2) * (mat + mat.T)


def config_symmetrizer(config):
    """function which takes a configuration of matrice as input and
    then returns the configuration with the matrices symmetrized."""

    symmetrized_mats = [matrix_symmetrizer(i) for i in config]
    return np.array(symmetrized_mats)


def max_indices(matrix):
    """returns the indices of the maximum argument in a matrix e.g. if max argument is in the nth row
    at the mth column, fn will return (n, m). PYTHON INDEXING IS USED I.E. START FROM ZEROTH ROW/COLUMN
    """

    row_of_max_arg = (
        np.argmax(matrix) // matrix.shape[1]
    )  # indices of maximum argument of sparse_mat
    column_of_max_arg = np.argmax(matrix) - (row_of_max_arg * matrix.shape[1])

    return (row_of_max_arg, column_of_max_arg)


def C_a_t0(configurations, t0):
    """Function takes in configurations of matrices and provides the average correlator matrix at time-slice t0"""

    avg_configuration = np.mean(configurations, axis=0)

    return avg_configuration[t0, :, :]


def reordering_template(ref_mat, unordered_vecs):
    """This fn provides a template for the reordering process. Basically all it does is take a
    matrix full of elements roughly equal to 0 and 1 and returns the same matrix but with 0's and
    1's exactly - just smooths it. Hence the output matrix is just the identity matrix with columns permutated.
    """

    template_matrix = np.zeros(
        (unordered_vecs.shape[0], unordered_vecs.shape[0]), dtype=float
    )  # zero matrix
    product = abs(np.matmul(ref_mat.T, unordered_vecs))

    max_val = np.amax(product)
    while max_val > 0:
        row_of_max_arg = (
            np.argmax(product) // product.shape[1]
        )  # indices of maximum argument of sparse_mat
        column_of_max_arg = np.argmax(product) - (row_of_max_arg * product.shape[1])

        template_matrix[row_of_max_arg, column_of_max_arg] = 1.0

        # replace corresponding column and row of product matrix with zeros

        product[row_of_max_arg, :] = np.zeros([unordered_vecs.shape[0]], dtype=float)
        product[:, column_of_max_arg] = np.zeros([unordered_vecs.shape[0]], dtype=float)

        max_val = np.amax(product)

    return template_matrix


def column_swapper(unordered_mat, index, index_destination):
    """swaps the column given by index with the one at index_destination"""

    unordered_mat[:, [index, index_destination]] = unordered_mat[
        :, [index_destination, index]
    ]
    return unordered_mat


def list_swapper(list, index, index_destination):
    """swaps the (index)th element of list with the (index_destination)th element"""

    list[index], list[index_destination] = list[index_destination], list[index]
    return list


def order_mat(template, start_val, eigenvalues, eigenvectors):
    """Function which reorders a template matrix such that it becomes equal to the identity matrix.
    If a column is all zeros it will be placed first. These reordering transformations are then applied
    to the unordered eigenvalues and unordered eigenvectors.

    template is the matrix to be reordered.
    start_val is the column which the algorithm starts upon. I can't see why this would ever not be zero but,
    for generality, I've included it as a parameter anyway"""

    identity = np.identity(template.shape[0])

    while np.array_equal(template, identity) == False:
        max_index = np.argmax(template[:, start_val])

        if start_val == max_index:
            pass

        elif start_val != max_index:
            template = column_swapper(template, start_val, max_index)
            eigenvectors = column_swapper(eigenvectors, start_val, max_index)
            eigenvalues = list_swapper(eigenvalues, start_val, max_index)

        elif start_val > template.shape[0]:
            print(
                "I've cycled through once without being able to produce the matrices of the desired form. Hence two largest column values must be in the same place."
            )

        start_val = (start_val + 1) % len(template[0])

    return eigenvalues, eigenvectors
