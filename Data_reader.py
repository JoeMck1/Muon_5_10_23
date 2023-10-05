import h5py
import os

import numpy as np

from GEVP_functions import reality_fn

'''This script imports the data and rearranges it into a more useful format.'''

#folder_path = '/home/jm1n22/GEVP/48I'
folder_path = 'C:/Users/jm1n22/test_sftp/GEVP/48I'
file_names = os.listdir(folder_path)

all_configs = []

for file_name in file_names:

    file = h5py.File(f"{folder_path}/{file_name}", 'r')
    group_names = list(file.keys()) # Get the list of group names
    group = file['I1_T1M'] # Access a specific group
    subgroup_names = list(group.keys())
    all_data = [] # 121 x 96 i.e. [(all 96 time-slices for operator 1), (all 96 time-slices for operator 2), ... , (all 96 time-slices for operator 121)]

    for subgroup_name in subgroup_names:

        subgroup = group[subgroup_name]
        dataset = subgroup['corr'] # Read data from a specific dataset within the group
        data = dataset[:]  # Retrieve all the data
        all_data.append(data)

    all_configs.append(all_data)

    file.close()

data_array = np.array(all_configs) # shape (27, 121, 96) = [1000.h5, 1040.h5, ...], 1000.h5 = [[op1 t=1, op1 t=2,...], [op2 t=1, op2 t=2,...]]
no_operators = int(np.sqrt(data_array.shape[1]))
time_slices = data_array.shape[2]

smoothed_data = np.array([[reality_fn(data_array[j][i]) for i in range(len(data_array[0]))] for j in range(len(data_array))]) #removes negligible real or imaginary components

reshaped_data = np.array([np.reshape(smoothed_data[i].T, (time_slices, no_operators, no_operators)) for i in range(len(smoothed_data))]) # (27,96,11,11)

final_data = np.delete(np.delete(reshaped_data, 8, axis=2), 8, axis=3) # remove local vectors

# final_data represents 27 configurations, containing 96 10 x 10 matrices, one for each time-slice.