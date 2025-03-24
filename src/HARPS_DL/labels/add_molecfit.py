import pandas as pd
from pdb import set_trace as st
import numpy as np
import os
from tqdm import tqdm

def parse_res_file(res_file, name):
    # 1) go through res file until the line "MOLECULAR GAS COLUMNS IN PPMV"
    # 2) start collecting molecular information
    # 3) newline signals end of PPMV data    
    columns = ['dp_id']
    PPMVs = []
    with open(res_file, 'r') as f:
        start_collection = False
        for line in f:
            if start_collection:                
                if line == '\n':
                    break
                line = line.split()
                columns.append(line[0][:-1])
                PPMVs.append(float(line[1]))                
            if line == 'MOLECULAR GAS COLUMNS IN PPMV:\n':
                start_collection = True                
    df = pd.DataFrame(columns=columns)
    df.loc[0] = [name] + PPMVs    
    return df

res_folder = '/diska/vcvrcek/meta_results/molecfit_comp_atm_res/'
csv_file = '/home/vcvrcek/Python/spectra_DL/labels/harps_metadata_and_labels.csv'
csv_all_file = '/home/vcvrcek/Python/spectra_DL/labels/harps_labels_complete.csv'

df_base = pd.read_csv(csv_file) # base csv without molecfit
df_base.set_index('dp_id', inplace=True)
molecules = ['H2O', 'O2'] # molecules we are using

for mol in molecules:
    df_base[mol] = np.nan

for name in tqdm(os.listdir(res_folder)):
    res_file = res_folder + name
    name, ext = os.path.splitext(os.path.split(name)[1])        
    if ext != '.res':
        continue
    dp_id = name[:-4] 
    row = parse_res_file(res_file, dp_id)
    row = parse_res_file(res_file, dp_id)
    for mol in molecules:
        df_base.at[dp_id, mol] = row[mol]
    #print(df_base.loc[dp_id,:])
    
df_base.to_csv(csv_all_file)
