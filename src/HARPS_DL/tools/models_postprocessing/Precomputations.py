import os
import pickle

import numpy as np
import time

from pdb import set_trace
import pandas as pd

from HARPS_DL.project_config import MODELS_BANK

import torch
from astropy import units as u

from HARPS_DL.datasets.Spectra_real_data_module import Spectra_real_data_module
from HARPS_DL.tools.quick_fix_tools.update_config import update_config

class Precomputations():
    def precompute_labels(self, override=False, folder_name='precomputed_labels'):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        # this precomputes labels for unsupervision
        # because we lack 1-1 correspondence between labels and nodes
        # we learn linear projection from nodes to labels
        self.precompute_unsupervised2labels() 

        for model_dict in self.models:
            model_name = model_dict['name']
            file_name = os.path.join(folder_name, f'{model_name}.pkl')
            if model_dict['type'] == 'unsupervised': # this is solved separately
                continue
            
            if not override and os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    labels_data = pickle.load(f)
                    model_dict['labels_gt_norm'] = labels_data['labels_gt_norm']
                    model_dict['labels_predict_norm'] = labels_data['labels_predict_norm']

                    model_dict['labels_gt'] = labels_data['labels_gt']
                    model_dict['labels_predict'] = labels_data['labels_predict']
            else:
                model = model_dict['model'].eval().to('cuda:0')
                dataset = model_dict['dataset_test']

                # get gt + prediction (normalized)
                model_dict['labels_gt_norm'], model_dict['labels_predict_norm'] = \
                    model.predictions_per_index(dataset, batch_size=32)

                labels = model_dict['labels']
                model_dict['labels_gt_norm'] = model_dict['labels_gt_norm'][:labels.get_vec_length(), :]
                model_dict['labels_predict_norm'] = model_dict['labels_predict_norm'][:labels.get_vec_length(), :]

                # undo normalization
                model_dict['labels_gt'] = labels.inverse_normalization_array(
                    model_dict['labels_gt_norm'])
                model_dict['labels_predict'] = labels.inverse_normalization_array(
                    model_dict['labels_predict_norm'])

            if not os.path.exists(file_name) or override:
                labels_data = {
                    'labels_gt_norm': model_dict['labels_gt_norm'],
                    'labels_predict_norm': model_dict['labels_predict_norm'],
                    'labels_gt': model_dict['labels_gt'],
                    'labels_predict': model_dict['labels_predict'],
                }
                with open(file_name, 'wb') as f:
                    pickle.dump(labels_data, f)

    def precompute_reconstruction_errors(self, override=False, folder_name='precomputed_reconstruction_errors'):
        """
        this function directly computes the reconstruction error
        we ommit precomputing the reconstructions themselves due to memory requirements
        """               
        # if folder does not exists, create it
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for model_dict in self.models:
            if model_dict['type'] == 'supervised':
                continue
            model_name = model_dict['name']
            file_name = os.path.join(folder_name, f'{model_name}.pkl')
           
            if not override and os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    rec_data = pickle.load(f)
                    model_dict['rec_MAE'] = rec_data['rec_MAE']
            else:
                model = model_dict['model'].eval().to('cuda:0')
                dataset = model_dict['dataset_test']
                if model_dict['type'] == 'simulation':
                    model_dict['rec_MAE'] = model.simulations_reconstruction_overview(dataset, batch_size=32)
                else:
                    model_dict['rec_MAE'] = model.reconstruction_overview(dataset, batch_size=32)

            if not os.path.exists(file_name) or override:
                rec_data = {
                    'rec_MAE': model_dict['rec_MAE'],
                }
                with open(file_name, 'wb') as f:
                    pickle.dump(rec_data, f)



    def precompute_labels_errors(self):
        for model_dict in self.models:
            labels = model_dict['labels']
            labels_gt = model_dict['labels_gt']
            labels_predict = model_dict['labels_predict']
            labels_gt_norm = model_dict['labels_gt_norm']
            labels_predict_norm = model_dict['labels_predict_norm']

            print('labels_gt_norm.shape', labels_gt_norm.shape)
            print('labels_predict_norm.shape', labels_predict_norm.shape)
            model_dict['err_normalized'] = pd.DataFrame(np.abs(labels_gt_norm - labels_predict_norm).T, columns=labels.labels)
            model_dict['err'] = pd.DataFrame(np.abs(labels_gt - labels_predict).T, columns=labels.labels)
            model_dict['err_relative'] = pd.DataFrame((np.abs(labels_gt - labels_predict) / labels_gt).T, columns=labels.labels)
            model_dict['residuals'] = pd.DataFrame((labels_gt - labels_predict).T, columns=labels.labels)

            err = model_dict['err_normalized'].to_numpy()
            medians = np.nanmedian(err.ravel())
            model_dict['median_labels_err'] = medians
            model_dict['mad_labels_err'] = np.nanmedian(np.abs(err.ravel() - medians))

            model_dict['mean_labels_err'] = np.nanmean(err.ravel())
            model_dict['std_labels_err'] = np.nanstd(np.abs(err.ravel()))

    def precompute_GIS(self, override=False, folder_name='precomputed_GIS'):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for model in self.models:
            if model['type'] == 'supervised' or model['type'] == 'unsupervised' :
                continue

            model_name = model['name']
            file_name = os.path.join(folder_name, f'{model_name}.pkl')

            if not override and os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    GIS_data = pickle.load(f)
                    model['GIS'] = GIS_data['GIS']
            else:
                model['GIS'] = {}
                for label2investigate in self.get_GIS_labels():        
                    model['GIS'][label2investigate] = model['model'].core_intervention_error(
                                        dataset=model['dataset_gen'],
                                        df_main=model['df_gen'],
                                        labels=model['labels'],
                                        label2investigate=label2investigate,
                                        )

            if not os.path.exists(file_name) or override:
                GIS_data = {
                    'GIS': model['GIS'],
                }
                with open(file_name, 'wb') as f:
                    pickle.dump(GIS_data, f)

    def precompute_RVIS(self, override=False, folder_name='precomputed_RVIS'):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for model in self.models:
            if model['type'] == 'supervised':
                continue

            model_name = model['name']
            file_name = os.path.join(folder_name, f'{model_name}.pkl')

            if not override and os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    RVIS_data = pickle.load(f)
                    model['RVIS'] = RVIS_data['RVIS']
            else:
                rv_shifts = np.linspace(-40, 40, 20)*u.kilometer/u.second
                dataset = model['dataset_test']

                # fix seed
                np.random.seed(0)
                spectra_idxs = np.random.choice(len(dataset), 100)

                model['RVIS'] = model['model'].rv_intervention_error(
                                    dataset,
                                    spectra_idxs,
                                    labels=model['labels'],
                                    rv_shifts=rv_shifts,
                                    plot_all=False,
                                    output_per_rv=False)

                if not os.path.exists(file_name) or override:
                    RVIS_data = {
                        'RVIS': model['RVIS'],
                    }
                    with open(file_name, 'wb') as f:
                        pickle.dump(RVIS_data, f)

    def precompute_nodes(self, override=False, folder_name='precomputed_nodes'):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        for model_dict in self.models:
            model_name = model_dict['name']
            file_name = os.path.join(folder_name, f'{model_name}.pkl')

            if not override and os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    precomputed_nodes_data = pickle.load(f)
                    model_dict['nodes_mu'] = precomputed_nodes_data['nodes_mu'] 
                    model_dict['nodes_std'] = precomputed_nodes_data['nodes_std'] 
            else:
                model = model_dict['model']
                datamodule = model_dict['datamodule']

                original_device = model.device
                if torch.cuda.is_available():
                    model.to('cuda:0')

                dataset_train = datamodule.real_train_dataset
                assert(dataset_train.dataset.dataset_name == 'real')

                dataset_test = datamodule.real_test_dataset
                assert(dataset_test.dataset.dataset_name == 'real')

                # precomputation is left for top level
                nodes_mu, nodes_std = model.get_nodes(dataset_test)    
                model_dict['nodes_mu'] = nodes_mu
                model_dict['nodes_std'] = nodes_std

                model.to(original_device)

                if not os.path.exists(file_name) or override:
                    precomputed_nodes_data = {
                        'nodes_mu': model_dict['nodes_mu'],
                        'nodes_std': model_dict['nodes_std'],
                    }

                    with open(file_name, 'wb') as f:
                        pickle.dump(precomputed_nodes_data, f)

            model_dict['nodes_mu'] = model_dict['nodes_mu'].T
            model_dict['nodes_std'] = model_dict['nodes_std'].T

    def precompute_unique_nodes(self, override=False, folder_name='precomputed_unique_nodes'):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        for model_dict in self.models:
            model_name = model_dict['name']
            file_name = os.path.join(folder_name, f'{model_name}.pkl')

            if not override and os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    precomputed_nodes_data = pickle.load(f)
                    model_dict['nodes_mu_unique'] = precomputed_nodes_data['nodes_mu_unique'] 
                    model_dict['nodes_std_unique'] = precomputed_nodes_data['nodes_std_unique'] 
                    model_dict['labels_gt_unique'] = precomputed_nodes_data['labels_gt_unique'] 
            else:
                model = model_dict['model']
                dataset_unique = model_dict['dataset_unique']

                original_device = model.device
                if torch.cuda.is_available():
                    model.to('cuda:0')

                assert(dataset_unique.dataset.dataset_name == 'real')

                # precomputation is left for top level
                nodes_mu_unique, nodes_std_unique, labels_gt_unique = model.get_nodes_w_custom_labels(dataset_unique,
                                                                                                    batch_size=32,
                                                                                                    labels=model_dict['labels'],
                                                                                                    )
                model_dict['nodes_mu_unique'] = nodes_mu_unique
                model_dict['nodes_std_unique'] = nodes_std_unique
                model_dict['labels_gt_unique'] = labels_gt_unique

                model.to(original_device)

                if not os.path.exists(file_name) or override:
                    precomputed_nodes_data = {
                        'nodes_mu_unique': model_dict['nodes_mu_unique'],
                        'nodes_std_unique': model_dict['nodes_std_unique'],
                        'labels_gt_unique': model_dict['labels_gt_unique'],
                    }

                    with open(file_name, 'wb') as f:
                        pickle.dump(precomputed_nodes_data, f)

            model_dict['nodes_mu_unique'] = model_dict['nodes_mu_unique'].T
            model_dict['nodes_std_unique'] = model_dict['nodes_std_unique'].T
            model_dict['labels_gt_unique'] = model_dict['labels_gt_unique'].T
    
    def precompute_unsupervised2labels(self, metric='MSE', override=False, folder_name='precomputed_downstream'):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        # I use this model to provide the labels
        model_sup = self.load_model({'name':'EN-67', 'desc': 'encoder real', 'type': 'supervised'}, Spectra_real_data_module)
        labels = model_sup['labels']

        for model_dict in self.models:
            if model_dict['type'] != 'unsupervised':
                continue

            model_name = model_dict['name']
            file_name = os.path.join(folder_name, f'{model_name}.pkl')
            model_dict['labels'] = labels

            if not override and os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    precomputed_downstream_data = pickle.load(f)
                    model_dict['labels_gt'] = precomputed_downstream_data['labels_gt'] 
                    model_dict['labels_predict'] = precomputed_downstream_data['labels_predict']
            else:
                if model_dict['name'] == 'reference':
                    # reference model from Sedaghat et al. 2021
                    model_dict['labels_gt'], model_dict['labels_predict'] = model_dict['model'].labels_prediction_reference_test(
                                                                            labels.labels, metric=metric)
                else:
                    model_dict['labels_gt'], model_dict['labels_predict'] = self.downstream_learn(
                                                                                model_dict['model'],
                                                                                model_dict['datamodule'],
                                                                                labels.labels,
                                                                                metric=metric,
                                                                                )
                if not os.path.exists(file_name) or override:
                    precomputed_downstream_data = {
                        'labels_gt': model_dict['labels_gt'],
                        'labels_predict': model_dict['labels_predict'],
                    }

                    with open(file_name, 'wb') as f:
                        pickle.dump(precomputed_downstream_data, f)

            model_dict['labels_gt'] = model_dict['labels_gt'].T
            model_dict['labels_predict'] = model_dict['labels_predict'].T
            model_dict['labels_gt_norm'] = model_dict['labels_gt'].copy()
            model_dict['labels_predict_norm'] = model_dict['labels_predict'].copy()
            for idx, label in enumerate(labels.labels):
                model_dict['labels_gt_norm'][idx, :] = labels.normalize_label(label, model_dict['labels_gt_norm'][idx, :])
                model_dict['labels_predict_norm'][idx, :] = labels.normalize_label(label, model_dict['labels_predict_norm'][idx, :])   


    def precompute_MI(self, override=False, folder_name='precomputed_MI'):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        #get_MIs_by_nearest_neighbor(self, X, Y)
        for model_dict in self.models:
            model_name = model_dict['name']
            file_name = os.path.join(folder_name, f'{model_name}.pkl')


            if not override and os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    precomputed_MI = pickle.load(f)
                    model_dict['MI'] = precomputed_MI['MI'] 
            else:
                model_dict['MI'] =  model_dict['model'].get_MIs_by_nearest_neighbor(
                                        model_dict['nodes_mu_unique'],
                                        model_dict['labels_gt_unique'],
                                        )

                if not os.path.exists(file_name) or override:
                    precomputed_MI = {
                        'MI': model_dict['MI'],
                    }

                    with open(file_name, 'wb') as f:
                        pickle.dump(precomputed_MI, f)

    def precompute_timing(self, override=False, folder_name='precomputed_timing', batch_size=32):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        torch.set_num_threads(1)
        
        #get_MIs_by_nearest_neighbor(self, X, Y)
        for model_dict in self.models:
            model_name = model_dict['name']
            file_name = os.path.join(folder_name, f'{model_name}.pkl')

            if not override and os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    precomputed_timing = pickle.load(f)
                    model_dict['time_GPU'] = precomputed_timing['time_GPU'] 
                    model_dict['time_CPU'] = precomputed_timing['time_CPU'] 
            else:
                for device in ['cpu', 'cuda:0']:
                    model = model_dict['model'].to(device)
                    if model_dict['model type'] == 'simulation':
                        data = torch.randn(batch_size, 1, model_dict['bottleneck'])
                    else:
                        data = torch.randn(batch_size, 1, 327680)
                    data = data.to(device)

                    start_time = time.time()
                    
                    if device == 'cpu':
                        out = model(data)
                        model_dict['time_CPU'] = time.time() - start_time
                        model_dict['time_CPU'] = model_dict['time_CPU']/batch_size
                    else:
                        for i in range(10):
                            out = model(data)
                        model_dict['time_GPU'] = time.time() - start_time
                        model_dict['time_GPU'] = model_dict['time_GPU']/10/batch_size

                if not os.path.exists(file_name) or override:
                    precomputed_timing = {
                        'time_GPU': model_dict['time_GPU'],
                        'time_CPU': model_dict['time_CPU'],
                    }

                    with open(file_name, 'wb') as f:
                        pickle.dump(precomputed_timing, f)

        num_cores = os.cpu_count()
        torch.set_num_threads(num_cores)

    def precompute_params_count(self, override=False, folder_name='precomputed_params_count'):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        #get_MIs_by_nearest_neighbor(self, X, Y)
        for model_dict in self.models:
            model_name = model_dict['name']
            file_name = os.path.join(folder_name, f'{model_name}.pkl')

            if not override and os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    precomputed_params_count = pickle.load(f)
                    model_dict['params_count'] = precomputed_params_count['params_count'] 
            else:
                model_dict['params_count'] = sum(p.numel() for p in model_dict['model'].parameters())

                if not os.path.exists(file_name) or override:
                    precomputed_params_count = {
                        'params_count': model_dict['params_count'],
                    }

                    with open(file_name, 'wb') as f:
                        pickle.dump(precomputed_params_count, f)