import csv
import os

# File paths
# TC
ids = [f'1-{i}' for i in range(1, 11)]
for id in ids:
    dat_file_path = f"dataset/TC1-all_including-GT/TC{id}/{id}.dat"

    # Read data from the .dat file
    with open(dat_file_path, "r") as dat_file:
        data = [line.strip() for line in dat_file] 
    data = [line.replace('\t', '   ').strip() for line in data]
    data_ = []

    for d in data:
        dlist = d.split('   ')
        dlist = [int(dl) - 1 for dl in dlist]
        data_.append(dlist)

    import pandas as pd 
    df = pd.DataFrame(data_)
    df.columns = ['node_1', 'node_2']
    df.to_csv(f'dataset/TC1-all_including-GT/TC{id}/{id}.csv',index=False)

# real world 
folders = os.listdir('dataset/real-world dataset')
print(folders)

for folder in folders: 
    dat_file_path = f"dataset/real-world dataset/{folder}/network.dat"

    # Read data from the .dat file
    with open(dat_file_path, "r") as dat_file:
        data = [line.strip() for line in dat_file] 

    data = [line.replace('\t', ' ').strip() for line in data]
    data_ = []

    for d in data:
        dlist = d.split(' ')
        dlist = [int(dl) - 1 for dl in dlist]
        data_.append(dlist)
        
    import pandas as pd 

    df = pd.DataFrame(data_)

    print(df)

    df.columns = ['node_1', 'node_2']

    df.to_csv(f'dataset/real-world dataset/{folder}/{folder}.csv',index=False)