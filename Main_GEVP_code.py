import numpy as np
import GEVP_functions as GEVP

def GEVP_eigenvalues(data, t_0, no_bs_samples):
    
    '''This function takes in configurations of lattice data. Where each configuration contains an nxn correlator matrix
    at every time-slice. It then solves the generalised eigenvalue problem and returns configurations of eigenvalues, from
    which energies can be extracted.
    
    t_0 is the time-slice around which the GEVP is solved. Note folding is currently assumed, thus the zeroth time-slice is
    currently being omitted.
    
    no_bs_samples is the number of bootstrap samples employed.'''

    no_ts = int(len(data[0])/2) # number of time-slices AFTER FOLDING
    #Bootstrapping
    bs_configurations = GEVP.bs_mat(data, no_bs_samples) #shape = (bs_number,96,10,10)

    #folding across time-slices 
    folding = np.array([GEVP.fold_matrices(bs_configurations[i]) for i in range(no_bs_samples)]) #shape = (bs_number,48,10,10)

    #Symmetrizing matrices
    symmetrization = np.array([[GEVP.matrix_symmetrizer(folding[j][i]) for i in range(no_ts)] for j in range(no_bs_samples)]) #shape = (bs_number,48,10,10)

    #Acquire average correlator matrix at time t_0 to use as reference for later reordering
    avg_C_t0 = GEVP.C_a_t0(symmetrization, t_0)

    #Acquire eigenvalues and eigenvectors for C_t0's and average C_t0
    avg_eigvals, avg_eigvecs = np.linalg.eigh(avg_C_t0)

    tuple_eigvals, tuple_eigvecs = zip(*[np.linalg.eigh(symmetrization[i][t_0]) for i in range(no_bs_samples)]) # Note, these are unordered

    #Turn tuples into arrays
    eigvals, eigvecs = np.array(tuple_eigvals), np.array(tuple_eigvecs) # these are unordered. shape = (bs_number,10), (bs_number,10,10)

    #Generate reordering templates
    templates = np.array([GEVP.reordering_template(avg_eigvecs, eigvecs[i]) for i in range(no_bs_samples)])# shape = (bs_number,10,10)

    #Reorder the unordered C_t0 eigenvalues and eigenvectors based off the templates
    tuple_o_eigvals, tuple_o_eigvecs = zip(*[GEVP.matrix_reorderer(templates[i], 0, eigvals[i], eigvecs[i]) for i in range(no_bs_samples)])

    #Turn tuples into arrays
    o_eigvals, o_eigvecs = np.array(tuple_o_eigvals), np.array(tuple_o_eigvecs) # shape = (bs_number,10), (bs_number,10,10)

    #Construct C tilde
    #first we need diag{1/sqrt(eigenvalues)} 
    diag_eigvals = np.array([np.diag(1/np.sqrt(o_eigvals[i])) for i in range(no_bs_samples)]) # shape = (bs_number,10,10)

    C_tilde = np.array([np.matmul(diag_eigvals[i], np.matmul(o_eigvecs[i].T,
                         np.matmul(symmetrization[i], np.matmul(o_eigvecs[i], diag_eigvals[i])))) for i in range(no_bs_samples)]) #shape (bs_number, 48, 10,10)

    #find average C_tilde's
    C_a_tilde = np.mean(C_tilde, axis=0) # shape = (48,10,10)

    #find unordered eigenvalues and eigenvectors of C_tilde and C_a_tilde
    C_a_tilde_evals, C_a_tilde_evecs = np.linalg.eigh(C_a_tilde) # shape = (48,10), (48,10,10)

    tuple_C_tilde_evals, tuple_C_tilde_evecs = zip(*[np.linalg.eigh(C_tilde[i]) for i in range(no_bs_samples)])

    #turn tuples into arrays
    C_tilde_evals = np.array(tuple_C_tilde_evals) # shape = (bs_number, 48, 10)
    C_tilde_evecs = np.array(tuple_C_tilde_evecs) # shape = (bs_number, 48, 10, 10)

    #use C_a_tilde(t_0 + 1) as a reference to reorder C_a_tilde(t)'s. 

    #first need reordering templates again
    a_tilde_templates = np.array([GEVP.reordering_template(C_a_tilde[t_0 + 1], C_a_tilde_evecs[i]) for i in range(no_ts)])

    #Reorder C_a_tilde_evals and C_a_tilde_evecs

    o_a_tilde_evals_tup, o_a_tilde_evecs_tup = zip(*[GEVP.matrix_reorderer(a_tilde_templates[i], 0
                                                    , C_a_tilde_evals[i], C_a_tilde_evecs[i]) for i in range(no_ts)])

    #o_a_tilde_evals = np.array(o_a_tilde_evals_tup) # (48,10)
    o_a_tilde_evecs = np.array(o_a_tilde_evecs_tup) # (48,10,10)

    # Now we use the ordered evals and evecs of C_a_tilde to reorder the evals and evecs of C_tilde at every time slice
    # again, first need reordering templates

    tilde_templates = np.array([[GEVP.reordering_template(o_a_tilde_evecs[i], C_tilde_evecs[j][i]) 
                                 for i in range(no_ts)] for j in range(no_bs_samples)]) # (bs_number,48,10,10)

    ordered_eigens = [[GEVP.matrix_reorderer(tilde_templates[j][i], 0, C_tilde_evals[j][i], 
                                    C_tilde_evecs[j][i]) for i in range(no_ts)] for j in range(no_bs_samples)]

    o_C_tilde_evals = np.array([[i[0] for i in inner_list] for inner_list in ordered_eigens]) #(bs_number, 48,10)
    #o_C_tilde_evecs = np.array([[i[1] for i in inner_list] for inner_list in ordered_eigens]) #(bs_number, 48,10,10)

    # reverse order of C_tilde eigenvalues

    return np.flip(o_C_tilde_evals, axis=2)