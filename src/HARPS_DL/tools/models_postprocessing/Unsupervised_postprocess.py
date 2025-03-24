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




class Unsupervised_postprocess():
    def downstream_learn(self, model, datamodule, labels_sel, metric='MSE'):
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.preprocessing import StandardScaler

        original_device = model.device
        if torch.cuda.is_available():
            model.to('cuda:0')

        dataset_train = datamodule.real_train_dataset
        assert(dataset_train.dataset.dataset_name == 'real')

        dataset_test = datamodule.real_test_dataset
        assert(dataset_test.dataset.dataset_name == 'real')

        # precomputation is left for top level
        nodes_mu, _ = model.get_nodes(dataset_train)    
        nodes_mu_test, _ = model.get_nodes(dataset_test)    

        model.to(original_device)


        df = pd.read_csv(datamodule.csv_real_file)

        df_train = df.loc[dataset_train.indices, labels_sel]
        df_test = df.loc[dataset_test.indices, labels_sel]
        # Select data for training and testing
        df_train = df.loc[dataset_train.indices, labels_sel]
        df_test = df.loc[dataset_test.indices, labels_sel]

        # Combine the features and labels for each set
        features_train = nodes_mu.T
        features_test = nodes_mu_test.T

        # Drop rows with NaN values from the training dataset
        combined_df_train = pd.concat([df_train, pd.DataFrame(features_train)], axis=1)
        combined_df_train = combined_df_train.dropna()
        df_train_cleaned = combined_df_train[labels_sel]
        features_train_cleaned = combined_df_train.iloc[:, len(labels_sel):].values

        # Drop corresponding rows from y_train to maintain consistency
        y_train_cleaned = df_train_cleaned.values

        # Optionally, you can standardize the data using StandardScaler
        if 0:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(features_train_cleaned)
            X_test = scaler.transform(features_test)
        else:
            X_train = features_train_cleaned
            X_test = features_test
        # Extract the labels from the dataframes
        y_test = df_test.values

        if metric == 'MSE':
            # Initialize Linear Regression model
            lin_reg = LinearRegression()

            # Fit the model on the scaled training data
            lin_reg.fit(X_train, y_train_cleaned)

            # Use the model to make predictions on the scaled test data
            y_pred = lin_reg.predict(X_test)
        else:
            from sklearn.linear_model import SGDRegressor
            from sklearn.multioutput import MultiOutputRegressor

            # Initialize the SGDRegressor with MAE loss
            base_regressor = SGDRegressor(loss='epsilon_insensitive', epsilon=0, max_iter=1000, tol=1e-3)
            multi_regressor = MultiOutputRegressor(base_regressor)

            # Fit the model
            multi_regressor.fit(X_train, y_train_cleaned)

            # Predict on test data
            y_pred = multi_regressor.predict(X_test)


        return y_test, y_pred

    def plot_nodes_activation(self):
        fig, ax = plt.subplots()

        for model_dict in self.models:
            nodes_mu = model_dict['nodes_mu']

            medians = np.median(nodes_mu, axis=1)[:, np.newaxis]
            mads = np.median(np.abs(nodes_mu - medians), axis=1).ravel()

            # Compute CCDF without binning
            sorted_mads = np.sort(mads)
            ccdf = np.linspace(1, 0, sorted_mads.size)
            
            plt.plot(sorted_mads, ccdf, label=model_dict['desc'])

        ax.set_xscale("log")
        ax.legend(title='Models')
        plt.xlabel("MAD value (log scale)")
        plt.ylabel("Proportion of nodes with MAD > value")

    def plot_nodes_mads_ECDF(self):
        fig, ax = plt.subplots()

        for model_dict in self.models:
            nodes_mu = model_dict['nodes_mu']

            assert(nodes_mu.shape[1] == model_dict['model'].bottleneck)
            medians = np.median(nodes_mu, axis=0)
            mads = np.median(np.abs(nodes_mu - medians), axis=0).ravel()

            # Compute CCDF without binning
            sorted_mads = np.sort(mads)
            ccdf = np.linspace(1, 0, sorted_mads.size)
            active_nodes = ccdf*model_dict['model'].bottleneck
            
            plt.plot(sorted_mads, active_nodes, label=model_dict['desc'])

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend(title='Models')
        plt.xlabel("MAD value")
        plt.ylabel("number of nodes with MAD > value")

    def plot_nodes_activation(self, cutoffs=0.1):
        if type(cutoffs) is float:
            cutoffs = [cutoffs]*len(self.models)
        
        assert(len(cutoffs) == len(self.models))

        idxs_selection = {}
            
        fig, ax = plt.subplots()

        for idx, model_dict in enumerate(self.models):
            nodes_mu = model_dict['nodes_mu_unique']

            assert(nodes_mu.shape[1] == model_dict['model'].bottleneck)
            medians = np.median(nodes_mu, axis=0)
            mads = np.median(np.abs(nodes_mu - medians), axis=0).ravel()

            sorted_indices = np.argsort(mads)
            
            plt.scatter(range(len(mads)), np.sort(mads), label=model_dict['desc'])

            # print the indexes with MAD > cutoff as dataframe
            valid_indices = np.where(np.sort(mads) > cutoffs[idx])[0]
            df = pd.DataFrame({'index': sorted_indices[valid_indices],
                                'mad': np.sort(mads)[valid_indices]})
            idxs_selection[model_dict['name']] = sorted_indices[valid_indices]

            print(df)

        ax.legend(title='Models')
        plt.xlabel("Node Index (sorted by MAD value)")
        plt.ylabel("Sorted MAD values")
        return idxs_selection

    def plot_MI_labels_nodes(self, idxs_selection=None):
        """ plot MI for each label vs each node """

        for model_dict in self.models:
            plt.figure()
            MI = model_dict['MI'].T
            if idxs_selection is not None:
                MI = MI[:, idxs_selection[model_dict['name']]]

            labels = model_dict['labels'].labels
        
            sns.heatmap(MI,
                        cmap='viridis',
                        xticklabels=idxs_selection[model_dict['name']],
                        yticklabels=labels,
                        )
            plt.title(model_dict['desc'])

    def plot_labels_vs_nodes(self, selection_dic):
        """ plot label values vs selected nodes
        """
        for model_dict in self.models:
            if model_dict['name'] not in selection_dic:
                continue
            nodes_mu = model_dict['nodes_mu']
            labels_gt = model_dict['labels_gt'].T
            for idx_i, (label, idx_node) in enumerate(selection_dic[model_dict['name']]):
                plt.figure()
                idx_label = model_dict['labels'].labels.index(label)
                sns.scatterplot(x=nodes_mu[:, idx_node],
                                y=labels_gt[:, idx_label],
                                hue=labels_gt[:, idx_label],
                                palette='viridis',
                                )
                plt.title(f'{model_dict["desc"]}: {label} vs node {idx_node}')

    def plot_MI(self):
        """ plot label values vs the node with highest MI
        """
        raise NotImplementedError