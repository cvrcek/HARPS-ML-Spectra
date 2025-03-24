import os
import yaml
from importlib import resources


from HARPS_DL.datasets.Spectra_split_data_module import Spectra_split_data_module
from HARPS_DL.datasets.Labels import Labels

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from HARPS_DL.tools.state_dict_tool import rename_prefix

import pickle


def reference_model_loader(cls, ckpt_path):
    # loading model from Sedaghat et al. 2021
    state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
    model = cls(bottleneck=128)

    # load encoder
    prefix_old = ['conv' + str(i) + '.' for i in range(1,16)]
    prefix_new = ['blocks.' + str(i) + '.0' for i in range(15)]
    state_dict_encoder = rename_prefix(state_dict, prefix_old, prefix_new)
    model.encoder.load_state_dict(state_dict_encoder)

    # load bottleneck
    prefix_old = ['fc11']
    prefix_new = ['']
    state_dict_mu = rename_prefix(state_dict, prefix_old, prefix_new)
    model.fc_mu.load_state_dict(state_dict_mu)

    prefix_old = ['fc12']
    prefix_new = ['']
    state_dict_mu = rename_prefix(state_dict, prefix_old, prefix_new)
    model.fc_logvar.load_state_dict(state_dict_mu)

    prefix_old = ['fc2']
    prefix_new = ['']
    state_dict_mu = rename_prefix(state_dict, prefix_old, prefix_new)
    model.fc_z.load_state_dict(state_dict_mu)

    # load decoder
    prefix_old = ['deconv' + str(i) + '.' for i in range(15,0, -1)] + ['predict0']
    prefix_new = ['blocks.' + str(i) + '.0.0' for i in range(15)] + ['blocks.15.0.']
    state_dict_decoder = rename_prefix(state_dict, prefix_old, prefix_new)
    model.decoder.load_state_dict(state_dict_decoder)

    return model

def reference_model_rv_scaling(model):
    # scaling model for Sedaghat et al. 2021
    yaml_file = resources.files('HARPS_DL.datasets').joinpath('labels_precomputed.yaml')
    if yaml_file.exists():
        with yaml_file.open('r') as f:
            rv_labels = yaml.safe_load(f)
        mean = rv_labels['mean']
        sigma = rv_labels['sigma']
    else:
        # load datamodule
        datamodule = Spectra_split_data_module(
                        radvel_range=(-120., 120.),
                        etc_train_folder='ETC_uniform_dist/memmap_train',
                        etc_val_folder='ETC_crossed_dist/memmap_val',
                        etc_noise_free=False,
                        labels=None,
                        normalize_labels=False,
                        precompute_rv='no',
                    )
        dataset = datamodule.val_dataloader()[1].dataset
        assert(dataset.dataset.dataset_name == 'real')

        bottlenecks = []
        rvs = []
        # for i in range(100):
        for i in range(len(dataset)):
            spec_in, dic = dataset[i]
            _, _, bottleneck, _ = model(spec_in)
            bottlenecks.append(bottleneck.detach().numpy().ravel())
            rvs.append(dic['radvel'])

        rvs_gt = np.array(rvs).ravel()
        bottlenecks = np.array(bottlenecks)
        rvs_bottleneck = bottlenecks[:, 124].ravel() # hard-coded radial velocity node

        # nan ignore
        ind_ignore = np.isnan(rvs)

        rvs_gt = rvs_gt[~ind_ignore]
        rvs_bottleneck = rvs_bottleneck[~ind_ignore]

        # out of range
        ind_ignore = np.logical_or(rvs_gt < -26, rvs_gt > 40)

        rvs_gt = rvs_gt[~ind_ignore]
        rvs_bottleneck = rvs_bottleneck[~ind_ignore]

        from sklearn.linear_model import RANSACRegressor
        reg = RANSACRegressor().fit(rvs_gt.reshape(-1, 1), rvs_bottleneck.reshape(-1, 1))

        # coef_ * rvs + intercept_ = rvs_bottleneck (from linear regression)
        # (rvs - mean)/sigma = rvs_bottleneck (z normalization)
        # leads to =>
        sigma = float(1/reg.estimator_.coef_)
        mean = float(-reg.estimator_.intercept_/reg.estimator_.coef_)

        rv_labels = {'mean': mean, 'sigma': sigma}
        with yaml_file.open('w') as f:
            yaml.safe_dump(rv_labels, f)

    labels = Labels(
                datasets_names=['ETC', 'real'],
                labels=[
                            'radvel',
                        ],
                labels_type=[
                            'shared',
                        ],
                labels_normalization={'radvel': {'median': mean, 'mad': sigma}},
            )
    labels.idxs['radvel'] = {'real': 124, 'ETC': 124}

    return labels

class Legacy_mix_in:
    @classmethod
    def reference_model_loader(cls,
                    ckpt_path='',
                    opt={}):
        # loading model from Sedaghat et al. 2021
        if ckpt_path == '':
            if os.getenv('HOME') == '/home/cv':
                ckpt_path='/home/cv/Dropbox/PHD/Python/ESO/spectra_DL/models/model_128d_e182_i1500000.pth.tar'
            else:
                ckpt_path='/diska/vcvrcek/models/model_128d_e182_i1500000.pth.tar'


        # load model
        model = reference_model_loader(cls, ckpt_path)
        output = {}
        output['model'] = model
        labels = reference_model_rv_scaling(model)

        if 'labels' in opt:
            output['labels'] = labels
        if 'datamodule' in opt:
            # load datamodule
            datamodule = Spectra_split_data_module(
                        radvel_range=(-120., 120.),
                        etc_train_folder='ETC_uniform_dist/memmap_train',
                        etc_val_folder='ETC_crossed_dist/memmap_val',
                        etc_noise_free=False,
                        labels=None,
                        normalize_labels=False,
                        precompute_rv='no',
                    )
            dataset = datamodule.val_dataloader()[1].dataset
            assert(dataset.dataset.dataset_name == 'real')

            datamodule.labels = labels
            output['datamodule'] = datamodule


        return output

    def predictions_per_index_slow(self, dataset):
        labels_predict = np.full((self.bottleneck, len(dataset)), np.nan)
        with torch.no_grad():
            for idx in tqdm(range(len(dataset))):
                spectra_in, labels_in = dataset[idx]
                _, _, mu, _ = self(spectra_in.to(self.device))
                mu = mu.detach().cpu().numpy()
                labels_predict[:, idx] = mu.ravel()
                
        
        return labels_predict
    
    def labels_prediction_reference_test(self, labels_sel, metric='MSE'):
        #
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.preprocessing import StandardScaler

        model, datamodule = self.reference_model_loader()
        dataset_train = datamodule.real_train_dataset
        assert(dataset_train.dataset.dataset_name == 'real')

        dataset_test = datamodule.real_test_dataset
        assert(dataset_test.dataset.dataset_name == 'real')

        # load/compute pickles
        pickle_file = '/home/cv/Dropbox/PHD/Python/ESO/spectra_DL/notebooks/journal_visualisations/labels_cmp_all_methods/nodes_mu.pkl'

        if os.path.exists(pickle_file):
            # If it does, load it
            with open(pickle_file, 'rb') as f:
                nodes_mu = pickle.load(f)
        else:    
            nodes_mu = model.predictions_per_index_slow(dataset_train)    
            with open(pickle_file, 'wb') as f:
                pickle.dump(nodes_mu, f)

        pickle_file = '/home/cv/Dropbox/PHD/Python/ESO/spectra_DL/notebooks/journal_visualisations/labels_cmp_all_methods/nodes_mu_test.pkl'

        if os.path.exists(pickle_file):
            # If it does, load it
            with open(pickle_file, 'rb') as f:
                nodes_mu_test = pickle.load(f)
        else:    
            nodes_mu_test = model.predictions_per_index_slow(dataset_test)    
            with open(pickle_file, 'wb') as f:
                pickle.dump(nodes_mu_test, f)       

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