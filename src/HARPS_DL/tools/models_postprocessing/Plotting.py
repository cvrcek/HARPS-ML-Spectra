import json
import sys
import os
import pickle
import requests
import cgi

import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns

from pdb import set_trace
import pandas as pd

from HARPS_DL.project_config import MODELS_BANK

from HARPS_DL.models_structured.model_infoVAE import info_VAE as infoVAE
from HARPS_DL.models_structured.model_composite import composite_VAE
from HARPS_DL.models_structured.decoder import Decoder as Decoder_ResNet
from HARPS_DL.models_structured.encoder import Encoder as Encoder_CNN

from HARPS_DL.datasets.Labels import Labels
from HARPS_DL.datasets.Dataset_mixin import Dataset_mixin
from HARPS_DL.datasets.Spectra_split_data_module import Spectra_split_data_module

from HARPS_DL.generate_simulations.generate_pickled_harps import process_parameters, get_lambda_range, process_autokur_output
from HARPS_DL.generate_simulations.ETC_processing.etc_tools import precompute

from HARPS_DL.tools.spectra_tools import doppler_shift, DER_SNR

import neptune.new as neptune
import yaml
from tqdm import tqdm

import torch
from astropy import units as u

from learning.infoVAE.learn_infoVAE import MyLightningCLI

from HARPS_DL.models_structured.model_recognition import recognition_NN

from HARPS_DL.datasets.Spectra_real_data_module import Spectra_real_data_module
from HARPS_DL.tools.quick_fix_tools.update_config import update_config

class Plotting():
    def labels_plots(self, model_name2skip=[], quantity='MAE', visualization='boxplot', log_scale=False):
        labels_all = self.get_labels_all()
        descs = self.get_descs()

        for idx_label, label in enumerate(labels_all):
            err_label = pd.DataFrame()
            for idx_model, model in enumerate(self.models):        
                if model['name'] in model_name2skip:
                    continue

                if label not in model['err'].keys():
                    continue

                if quantity == 'MAE':
                    err_label[descs[idx_model]] = model['err'][label]
                elif quantity == 'residuals':
                    err_label[descs[idx_model]] = model['residuals'][label]
                else:
                    raise Exception(f'unknown quantity {quantity}')
                    
                if label == 'radvel' or label == 'BERV':
                    ylabel_str = 'absolute difference error [km/s]'
                elif label == 'Teff':
                    ylabel_str = 'absolute difference error [K]'
                else:
                    ylabel_str = 'absolute difference error [-]'

                if log_scale:
                    ylabel_str = 'log of ' + ylabel_str

            plt.figure()
            sns.set_theme(style="white")
            if visualization == 'boxplot':
                sns.boxplot(data=err_label, showfliers=False)
                plt.ylabel(ylabel_str)
            elif visualization == 'boxenplot':
                sns.boxenplot(data=err_label, orient="v", palette="Set2", showfliers=False)
                plt.ylabel(ylabel_str)
            elif visualization == 'KDE':
                err_label_with_offset = err_label + 1e-6
                #sns.histplot(data=err_label_with_offset, element="step", fill=False, log_scale=True, kde=True)
                lower_bound = err_label.quantile(0.01, axis=0)
                upper_bound = err_label.quantile(0.99, axis=0)
            
                # Clip the values in the DataFrame based on the calculated percentiles
                clipped_data = err_label.clip(lower=lower_bound, upper=upper_bound, axis=1)
                sns.kdeplot(data=clipped_data, log_scale=log_scale)
                plt.xlabel(ylabel_str)
            elif visualization == 'ECDF':
                # Calculate the lower and upper bounds for the x-axis
                lower_bound_x = np.min(err_label.quantile(0.01)) # 1st percentile
                upper_bound_x = np.max(err_label.quantile(0.95)) # 99th percentile
                
                sns.ecdfplot(data=err_label, log_scale=log_scale)
                plt.xlim(lower_bound_x, upper_bound_x) # Set the x-axis limits
                plt.xlabel(ylabel_str)
            else:
                raise Exception(f'unknown visualization choice {visualization}')
            if quantity == 'MAE':
                plt.title(f'{visualization} for {label} error')
            elif quantity == 'residuals':
                plt.title(f'{visualization} for {label} residuals')
            else:
                raise Exception(f'unknown quantity {quantity}')


            # output to folder named after visualization
            folder_out = visualization
            if not os.path.exists(folder_out):
                os.makedirs(folder_out)

            if label == '[M/H]':
                file_out = f'{folder_out}/{quantity}_metallicity.pdf'
            else:
                file_out = f'{folder_out}/{quantity}_{label}.pdf'

            plt.savefig(file_out, format='pdf') 

    def reconstruction_plot(self, visualization='boxplot', log_scale=False):
        descs = self.get_descs()
        err_reconstruction = pd.DataFrame()
        for idx_model, model in enumerate(self.models):        
            if model['type'] == 'supervised':
                continue
            err_reconstruction[descs[idx_model]] = model['rec_MAE']

        plt.figure()
        sns.set_theme(style="white")
        if visualization == 'boxplot':
            sns.boxplot(data=err_reconstruction, showfliers=False)
            plt.title(f'boxplots for reconstruction errors')
            plt.ylabel('mean absolute difference error [-]')
        elif visualization == 'boxenplot':
            sns.boxenplot(data=err_reconstruction, orient="v", palette="Set2", showfliers=False)
            plt.title(f'boxenplots for reconstruction errors')
            plt.ylabel('mean absolute difference error [-]')
        elif visualization == 'KDE':
            err_reconstruction_with_offset = err_reconstruction + 1e-6
            #sns.histplot(data=err_reconstruction_with_offset, element="step", fill=False, log_scale=True, kde=True)
            lower_bound = err_reconstruction.quantile(0.01, axis=0)
            upper_bound = err_reconstruction.quantile(0.99, axis=0)
        
            # Clip the values in the DataFrame based on the calculated percentiles
            clipped_data = err_reconstruction.clip(lower=lower_bound, upper=upper_bound, axis=1)
            sns.kdeplot(data=clipped_data, log_scale=log_scale)
            plt.title(f'KDE for reconstruction errors')
            plt.xlabel('mean absolute difference error [-]')
        elif visualization == 'ECDF':
            # Calculate the lower and upper bounds for the x-axis
            lower_bound_x = np.min(err_reconstruction.quantile(0.01)) # 1st percentile
            upper_bound_x = np.max(err_reconstruction.quantile(0.95)) # 99th percentile
            
            sns.ecdfplot(data=err_reconstruction, log_scale=log_scale)
            plt.xlim(lower_bound_x, upper_bound_x) # Set the x-axis limits
            plt.title(f'ECDFs for reconstruction errors')
            plt.xlabel('mean absolute difference error [-]')
        else:
            raise Exception(f'unknown visualization choice {visualization}')

        # output to folder named after visualization
        folder_out = visualization
        if not os.path.exists(folder_out):
            os.makedirs(folder_out)


        file_out = f'{folder_out}/reconstruction_errors.pdf'

        plt.savefig(file_out, format='pdf') 



    def GIS_plots(self):
        sns.set_style("white")
        for label2investigate in self.get_GIS_labels():        
            boxplot_dir = {}
            for model in self.models:        
                if model['type'] == 'supervised' or model['type'] == 'unsupervised' :
                    continue
                boxplot_dir[model['desc']] = model['GIS'][label2investigate]
            boxplot_data = pd.DataFrame(boxplot_dir)

            if len(boxplot_data) != 0:
                boxplot_data_melted = boxplot_data.melt(var_name='model', value_name=label2investigate)
                plt.figure()
                sns.boxplot(x='model', y=label2investigate, data=boxplot_data_melted, showfliers=False)
                plt.title(f'Boxplot for {label2investigate}')
                plt.show()
                if label2investigate == '[M/H]':
                    plt.savefig('models_GIS_metallicity.pdf', format='pdf') 
                else:
                    plt.savefig('models_GIS_'+ label2investigate + '.pdf', format='pdf') 
            else:
                print('no data for ', label2investigate)

            
    def RVIS_plots(self):
        sns.set_style("white")
        boxplot_dir = {}
        for model in self.models:        
            if model['type'] == 'supervised':
                continue
            boxplot_dir[model['desc']] = model['RVIS']
        boxplot_data = pd.DataFrame(boxplot_dir)

        boxplot_data_melted = boxplot_data.melt(var_name='model', value_name='RVIS')
        plt.figure()
        sns.boxplot(x='model', y='RVIS', data=boxplot_data_melted, showfliers=False)
        plt.title(f'Boxplot for RVIS')
        plt.show()
        plt.savefig('models_RVIS.pdf', format='pdf') 