import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Data_reader import final_data as data
from Main_GEVP_code import GEVP_eigenvalues

t_min = 4
t_max = 25
t_0 = 4
no_bs = 500
no_ops = 6
time = np.linspace(t_min, t_max - 1, (t_max - t_min))
eigenvalues = GEVP_eigenvalues(data, t_0, no_bs) #(no_bs, no_ts, 10)

#### plot average eigenvalues against time ####

#acquire averages
avg_eigenvalues = np.mean(eigenvalues, axis = 0) #(no_ts, 10)

#acquire errors for errorbars
eigen_errs = np.sqrt(np.var(eigenvalues, axis= 0, ddof=1))

#compute effective masses

meff = np.array([[[-np.log(eigenvalues[x][t+1][i] / eigenvalues[x][t][i]) for t in range(len(avg_eigenvalues) - 1)] 
                 for i in range(len(avg_eigenvalues[0]))] for x in range(no_bs)])

avg_meff = np.array([[-np.log(avg_eigenvalues[j+1][i] / avg_eigenvalues[j][i]) for j in range(len(avg_eigenvalues) - 1)] 
                     for i in range(len(avg_eigenvalues[0]))])

meff_errs = np.sqrt(np.var(meff, axis=0, ddof=1))
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle("n =" f'{[i for i in range(no_ops)]}' " levels at "r'$t_0=$'f"{t_0}, " r'$t = $'f"{[t_min,t_max]}")
ax2.set_ylim((0.2,0.6))
for i in range(no_ops):
    ax1.errorbar(time, avg_eigenvalues[t_min:t_max,i], eigen_errs[t_min:t_max,i], fmt = 'o', mec = 'black', capsize = 2, label = f"n = {i}")
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$\lambda (t, t_0)$')
    ax1.legend()
    ax2.errorbar(time, avg_meff[i, t_min:t_max], meff_errs[i, t_min:t_max], fmt = 's:', mec = 'black', capsize = 2)
    ax2.set_ylabel(r'$-ln \left( \frac{\lambda(t+1, t_0)}{\lambda(t,t_0)} \right)$')
plt.xticks(np.arange(0, t_max+1, 5))
plt.xlabel(r'$t$')
plt.show()