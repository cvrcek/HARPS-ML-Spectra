import os
import sys
from tqdm import tqdm

import numpy as np
import torch

from HARPS_DL.models_structured.Intervention_mix_in import Intervention_mix_in
import matplotlib.pyplot as plt
from pdb import set_trace
from astropy import units as u

from HARPS_DL.tools.spectra_tools import doppler_shift, DER_SNR
   
from HARPS_DL.datasets.Dataset_mixin import Dataset_mixin
 

class Overview_encoder_mix_in(Intervention_mix_in):
    """
        mixin class with analytic tools for models
    """
    def predictions_per_index(self, dataset, batch_size=32):
        """
            accelareted function to iterate through an input dataset

            returns ground truth labels and respective predictions
            (dataset)
        """
        class Dataset_with_idx(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __getitem__(self, idx):
                return self.dataset[idx], idx

            def __len__(self):
                return len(self.dataset)
        
        _, labels_in = dataset[0]
        labels_num_in_bottleneck = labels_in.shape[1]

        ds = Dataset_with_idx(dataset)
        dl = torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size)
        labels_predict = np.full((self.bottleneck, len(dataset)), np.nan)
        labels_gt = np.full((labels_num_in_bottleneck, len(dataset)), np.nan)
        with torch.no_grad():
            for (spectra_in, labels_in), idxs in tqdm(iter(dl)):
                mu = self(spectra_in.to(self.device))
                mu = mu.detach().cpu().numpy()
                if len(idxs) == 1:
                    labels_predict[:, idxs] = mu.T.ravel()
                    labels_gt[:, idxs] = torch.reshape(labels_in, [labels_in.shape[0], -1]).detach().cpu().numpy().T.ravel()
                else:
                    labels_predict[:, idxs] = mu.T
                    labels_gt[:, idxs] = torch.reshape(labels_in, [labels_in.shape[0], -1]).detach().cpu().numpy().T

        return labels_gt, labels_predict

    def get_nodes(self, dataset, batch_size=32):
        class Dataset_with_idx(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __getitem__(self, idx):
                return self.dataset[idx], idx

            def __len__(self):
                return len(self.dataset)

        ds = Dataset_with_idx(dataset)
        dl = torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size)
        
        nodes_mu = np.full((self.bottleneck, len(dataset)), np.nan)
        nodes_logvar = np.full((self.bottleneck, len(dataset)), np.nan)
        i = 0
        with torch.no_grad():
            for (spectra_in, labels_in), idxs in tqdm(iter(dl)):
                _, _, mu, logvar = self(spectra_in.to(self.device))
                mu = mu.detach().cpu().numpy()
                logvar = logvar.detach().cpu().numpy()

                if len(idxs) == 1:
                    nodes_mu[:, idxs] = mu.T.ravel()
                    nodes_logvar[:, idxs] = logvar.T.ravel()
                else:
                    nodes_mu[:, idxs] = mu.T
                    nodes_logvar[:, idxs] = logvar.T
                
        return nodes_mu, nodes_logvar 
