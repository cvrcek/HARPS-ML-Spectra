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

from HARPS_DL.tools.models_postprocessing.Precomputations import Precomputations
from HARPS_DL.tools.models_postprocessing.Tabling import Tabling
from HARPS_DL.tools.models_postprocessing.Plotting import Plotting
from HARPS_DL.tools.models_postprocessing.Unsupervised_postprocess import Unsupervised_postprocess

from HARPS_DL.models_structured.model_simulation import simulator_NN
from HARPS_DL.datasets.Spectra_etc_data_module import Spectra_data_module



class Simulations_helper():
    def load_simulation_model(self, model):
        assert(model['type'] == 'simulation')
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


        cli = MyLightningCLI(simulator_NN,
                                datamodule_class=Spectra_data_module,
                                parser_kwargs={"error_handler": None},
                                save_config_callback=None,
                                args=config,
                                run=False,
                                )

        cli.model.load_state_dict(state_dict)
        cli.model.eval()


#        dataset_test = cli.datamodule.test_dataloader().dataset
#        assert(dataset_test.dataset.dataset_name == 'real')


        model['model'] = cli.model
        model['datamodule'] = cli.datamodule

        model['dataset_test'] = cli.datamodule.etc_val_dataset

#        model['dataset_test'] = dataset_test
#        model['labels'] = labels

        return model
        
