import logging
import pandas as pd
import os
import uuid
import tempfile
import fnmatch

import numpy as np
from statsmodels import robust

desiredMinW = 3785 #inclusive
desiredMaxW = 6910 #inclusive

from HARPS_DL.project_config import LINELIST_FILE, CSV_REAL_FILE

def get_lambda_range(df):
    # get wavelength ranges that cover range change due to radial velocity shift
    radvel = df['radvel'].values
    radvel = radvel[~np.isnan(radvel)]
    radvel_mad = robust.mad(radvel)
    radvel_median = np.median(radvel)
    print(radvel_mad)
    print(radvel_median)

    #
    lambda_1 = desiredMinW
    lambda_2 = desiredMaxW

    print('lambda_1: %f, lambda_2: %f' % (lambda_1, lambda_2))
    lambda_1 = lambda_1*(1 - (5*radvel_mad + radvel_median)/299792.458)
    lambda_2 = lambda_2*(1 + (5*radvel_mad + radvel_median)/299792.458)
    print('stretching lambda range to incorporate possible RV shifts')
    print('lambda_1: %f, lambda_2: %f' % (lambda_1, lambda_2))
    return lambda_1, lambda_2

def process_parameters(parameters, params_glob, model='ATLAS9'):
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = "harps_" + uuid.uuid1().hex
        para = (parameters['Teff'], parameters['logg'], parameters['[M/H]']) +\
               (params_glob['lambda_1'], params_glob['lambda_2'], params_glob['resol']) +\
               (filename, params_glob['pixel_size'])
        with open(tmpdirname + "/input.kur3", 'w') as outfile:
            if model == 'ATLAS9':
                outfile.write('synth    %4i   %5.2f   %5.2f  0  2  2.2  linelist.xkur  abu.var %4i %4i %6i 0  %s  0   %6i\n' % para)
            elif model == 'MARCS':
                outfile.write('synthm    %4i   %5.2f   %5.2f  0  2  2.2  linelist.xkur  abu.var %4i %4i %6i 0  %s  0   %6i\n' % para)
            else:
                raise('Unknown model (use ATLAS9 or MARCS)')
            outfile.write('\n')
            outfile.write('AT_STUFF -6.875 72 5 no\n')
            outfile.write('VERBOSE 1\n')
            outfile.write('OUTPUT 1\n')
            outfile.write('HEADER yes\n')
            outfile.close()
        os.system(f'cp {LINELIST_FILE} {tmpdirname}/linelist.xkur')
        path_curr =os.popen('pwd').read()
        data_file = []
        debug = False
        while len(data_file) != 1 and not debug:
            os.system(f'cd {tmpdirname} && autokur input.kur3')
            dirs = os.listdir(tmpdirname)
            filtered_dirs = fnmatch.filter(dirs, '*.dat')
            filtered_dirs_neg = fnmatch.filter(filtered_dirs, '*_lin.dat')
            data_file = set(filtered_dirs) - set(filtered_dirs_neg)
            if len(data_file) != 1:
                logging.warning('autokur output not found, trying again')


        if not debug:
            data_file = data_file.pop()
            output = process_autokur_output(tmpdirname + '/' + data_file)
        else:
            output = ([], [])
        return {'wave': output[0], 'spectrum': output[1], 'parameters': parameters}, filename

def process_autokur_output(filename):
    wave = []
    line_flux = []
    cont_flux = []
    norm_flux = []
    with open(filename, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            l = line.split()
            wave.append(float(l[0]))
            line_flux.append(float(l[1]))
            cont_flux.append(float(l[2]))
            norm_flux.append(float(l[3]))
    return wave, line_flux, cont_flux, norm_flux

def process_parameters_standalone(parameters):    
    df_real = pd.read_csv(CSV_REAL_FILE)
    lambda_1, lambda_2 = get_lambda_range(df_real)
    resol =  115000
    pixel_size =  0 # Harps_spec_info.wave_resolution
    params_glob = {
                   'lambda_1': lambda_1,
                   'lambda_2': lambda_2,
                   'resol': resol,
                   'pixel_size': pixel_size
                   }

    return process_parameters(parameters, params_glob, model='ATLAS9')
