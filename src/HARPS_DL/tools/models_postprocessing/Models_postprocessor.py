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
from HARPS_DL.datasets.Spectra_etc_data_module import Spectra_data_module as Spectra_etc_data_module
from HARPS_DL.tools.quick_fix_tools.update_config import update_config

from HARPS_DL.tools.models_postprocessing.Precomputations import Precomputations
from HARPS_DL.tools.models_postprocessing.Tabling import Tabling
from HARPS_DL.tools.models_postprocessing.Plotting import Plotting
from HARPS_DL.tools.models_postprocessing.Unsupervised_postprocess import Unsupervised_postprocess
from HARPS_DL.tools.models_postprocessing.Simulations_helper import Simulations_helper

class Models_postprocessor(Simulations_helper, Precomputations, Tabling, Plotting, Unsupervised_postprocess):
    def __init__(self, models):
        """ 
            models: list of dicts with keys 'name', 'desc',and 'type'
        """
        self.models = models
        self.load_models()
        self.get_bottlenecks()

    def get_bottlenecks(self):
        for idx, model in enumerate(self.models):
            model['bottleneck'] = model['model'].bottleneck

    def load_models(self):
        for idx, model in enumerate(self.models):            
            if model['name'] == 'reference':
                # reference model from Sedaghat et al. 2021
                model['model'], reference_datamodule  =  composite_VAE.load_reference_model()
                model['dataset_test'] = reference_datamodule.real_test_dataset
                model['datamodule'] = reference_datamodule
                model['dataset_unique'] = reference_datamodule.real_unique_dataloader().dataset # necessary for MI!
                assert(model['dataset_test'].dataset.dataset_name == 'real')
            elif model['type'] == 'semi-supervised' or model['type'] == 'unsupervised':
                datamodule_class = Spectra_split_data_module
                self.models[idx] = self.load_model(model, datamodule_class)
            elif model['type'] == 'supervised':
                data_module_classes = [Spectra_real_data_module, Spectra_split_data_module, Spectra_etc_data_module]
                #data_module_classes = [Spectra_etc_data_module]

                # try:
                #     self.models[idx] = self.load_model(model, datamodule_class=Spectra_real_data_module)
                # except:
                #     try:
                #         self.models[idx] = self.load_model(model, datamodule_class=Spectra_etc_data_module)
                #     except:
                #         raise Exception('Could not load model')
                for data_module_class in data_module_classes:
                    print(f"Trying to load model using {data_module_class.__name__}")
                    try:
                        self.models[idx] = self.load_model(model, datamodule_class=data_module_class)
                        model_loaded = True
                        print(f"Model loaded successfully with {data_module_class.__name__}")
                        break  # Exit the loop if model is loaded successfully
                    except SystemExit:
                        print(f"Failed to load using {data_module_class.__name__}")

                if not model_loaded:
                    raise Exception('Could not load model')

            elif model['type'] == 'simulation':
                self.models[idx] = self.load_simulation_model(model)
            else:
                raise ValueError(f"""Unknown model type: {model['type']},
                                it can be either 'semi-supervised' or 'supervised'""")
    
    def load_model(self, model, datamodule_class):
        if model['type'] == 'semi-supervised' or model['type'] == 'unsupervised':
            model_class = infoVAE
        elif model['type'] == 'supervised':
            model_class = recognition_NN
        else:
            raise ValueError(f"""Unknown model type: {model['type']},
                            it can be either 'semi-supervised' or 'supervised'""")
        name = model['name']

        model_folder = MODELS_BANK + name + '/'
        config_file = model_folder + 'config.yaml'

        model_ckpt = ""
        for filename in os.listdir(model_folder):
            if filename[-4:] == 'ckpt':
                model_ckpt = model_folder + filename

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        #config = update_config(config)
        config['trainer']['devices'] = [0]

        state_dict = torch.load(model_ckpt, map_location='cpu')['state_dict']


        cli = MyLightningCLI(model_class,
                                datamodule_class=datamodule_class,
                                parser_kwargs={"error_handler": None},
                                save_config_callback=None,
                                args=config,
                                run=False,
                                )

        cli.model.load_state_dict(state_dict)
        cli.model.eval()


        dataset_test = cli.datamodule.test_dataloader().dataset
        assert(dataset_test.dataset.dataset_name == 'real')

        labels = cli.datamodule.labels

        model['model'] = cli.model
        model['datamodule'] = cli.datamodule

        if hasattr(cli.datamodule, "real_unique_dataloader"):
            model['dataset_unique'] = cli.datamodule.real_unique_dataloader().dataset # necessary for MI!

        model['dataset_test'] = dataset_test
        model['labels'] = labels

        
        if model['type'] == 'semi-supervised':
            dataset_gen = cli.datamodule.etc_gen_dataset
            df_gen =  pd.read_csv(cli.datamodule.etc_gen_csv_file)
            model['dataset_gen'] = dataset_gen
            model['df_gen'] = df_gen
    

        return model
        
    def get_descs(self):
        descs = []
        for model_dict in self.models:
            descs.append(model_dict['desc'])
        return descs

    def get_labels_all(self):
        labels_all = []
        for model_dict in self.models:
            labels_all += model_dict['labels'].labels
        labels_all = np.unique(labels_all)
        labels_all = labels_all[labels_all != 'H2O_pwv']

        return labels_all

    def get_GIS_labels(self):
        return ['Teff', '[M/H]', 'logg']


    def count_labels(self):
        model = self.models[0]
        labels = model['labels_gt']

        # count elements in columns that are not nan 
        counts = np.sum(~np.isnan(labels), axis=1)

        counts_dic = {}
        for idx, label in enumerate(model['labels'].labels):
            counts_dic[label] = counts[idx]

        return counts_dic