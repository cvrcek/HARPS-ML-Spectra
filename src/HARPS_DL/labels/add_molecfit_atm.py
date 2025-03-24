import pandas as pd
from pdb import set_trace as st
import numpy as np
import os
from tqdm import tqdm

import sys

project_folder = '/home/vcvrcek/Python/spectra_DL/'
sys.path.append(project_folder + 'tools/')
from molecfit_postprocess import parse_atm_file

mol_folder = '/diska/vcvrcek/meta_results/molecfit_output_old/'
csv_file = '/home/vcvrcek/Python/spectra_DL/labels/harps_metadata_and_labels.csv'
csv_all_file = '/home/vcvrcek/Python/spectra_DL/labels/harps_labels_w_atm.csv'

df_base = pd.read_csv(csv_file) # base csv without molecfit
df_base.set_index('dp_id', inplace=True)
watched_vars = ['*PRE [mbar]', '*TEM [K]', '*H2O [ppmv]', '*O2 [ppmv]']
names_vars = ['PRE', 'TEM', 'H2O', 'O2']


first = True
N = 51 # elements per variable
for name in tqdm(os.listdir(mol_folder)):
    atm_file = mol_folder + name
    name, ext = os.path.splitext(os.path.split(name)[1])        
    if ext != '.atm':
        continue
    dp_id = name[:-4] 
    row = parse_atm_file(atm_file, watched_vars, names_vars)
    st()
    if first:
        for var in names_vars:
            for i in range(0, N):
                df_base[var + '_' + str(i)] = np.nan
        
    for var in names_vars:
        for i in range(0, N):
            df_base.at[dp_id, var + '_' + str(i)] = row[var + '_' + str(i)]
    #print(df_base.loc[dp_id,:])
    
df_base.to_csv(csv_all_file)
