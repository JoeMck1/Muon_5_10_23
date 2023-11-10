import h5py

"""This function will read the data for any h5 group, regardless of its structure. However, it doesn't
output the data in a very user friendly way, so it's best use is to just establish the structure of an 
unknown group."""


def print_h5_objects(obj):
    for key in obj.keys():
        if isinstance(obj[key], h5py.Group):
            print(f"Group: {key}")
            print_h5_objects(obj[key])
        else:
            print(f"Dataset: {key}")
            data = obj[key][:]
            print(data)


file = h5py.File("C:/Users/jm1n22/test_sftp/GEVP/48I_new/1000 (1).h5")

print_h5_objects(file)
