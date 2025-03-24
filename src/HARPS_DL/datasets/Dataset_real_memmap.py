import torch.utils.data as data
import torch
import os
import os.path
import numpy as np
from astropy.io import fits

import pandas as pd

from HARPS_DL.datasets.Dataset_mixin import Dataset_mixin
from HARPS_DL.datasets.Spectra_info import Harps_spec_info


class Harps_real_memmap(Dataset_mixin, data.Dataset):

    def __init__(self,
                 memmap_filename,
                 dataset_name = 'real',
                 fix_for_artifacts=False,
                 median_norm=True,
                 median_threshold=0,
                 df_catalog=None,
                 labels_norm=None,
                 dictionary_output=False,
                ):

        self.memmap_filename = memmap_filename
        self.dataset_name = dataset_name

        self.fix_for_artifacts = fix_for_artifacts
        self.median_norm = median_norm
        self.median_threshold = median_threshold
        self.labels_norm=labels_norm
        self.dictionary_output = dictionary_output

        # Get info about the number of spectra in the memmap
        self.file_length = os.stat(self.memmap_filename).st_size
        self.num_spectra = self.file_length // Harps_spec_info.spectrum_bytes
        
        # load memmap for workers
        if(data.get_worker_info() is None):
            self.memmap = np.memmap(self.memmap_filename, dtype='float32', mode='r', shape=(self.num_spectra, Harps_spec_info.PaddedLength))
        else:
            self.memmap = None

        # load pandas
        if df_catalog is None:
            self.df = pd.read_csv('/home/vcvrcek/Python/spectra_DL/labels/harps_labels_complete.csv')
            self.df.loc[(self.df.airmass < Harps_spec_info.airmass_min) | 
                        (self.df.airmass > Harps_spec_info.airmass_max), 'airmass'] = np.nan
        else:
            self.df = df_catalog


    def __getitem__(self, index):
        spectrum = self.memmap[index,:].copy()

        if(self.fix_for_artifacts):
            spectrum = self.fix_mid(spectrum)
            spectrum = self.fix_paddings(spectrum)

        if(self.median_norm):
            spectrum = self.norm_by_median(spectrum)

        st = Harps_spec_info.wave_resolution
        emul_wave = np.arange(Harps_spec_info.desiredMinW-Harps_spec_info.padL*st,Harps_spec_info.desiredMaxW+.001+Harps_spec_info.padR*st,step=st)

        labels = self.df.iloc[index].to_dict()
        labels['name'] = self.dataset_name

        spectrum = np.expand_dims(spectrum, axis=0)

        if self.labels_norm is None:
            if self.dictionary_output: # this is for postprocessing, where dic might be useful
                return torch.Tensor(spectrum), labels # spectrum + dictionary
            else: # this for training, when I cant use dictionary anyway
                return torch.Tensor(spectrum), torch.Tensor(0) # spectrum + dummy
        else:
            labels = self.labels_norm.dic2vec(labels)
            labels = np.expand_dims(labels, axis=0)

            return torch.Tensor(spectrum), torch.Tensor(labels)

    def __len__(self):
        return self.num_spectra

    def baryname(self, folder, dp_id):
        # constants

        # load data
        filename = os.path.join(folder,dp_id) + '.fits'
        hdu = fits.open(filename)
        wave = hdu[1].data.field('WAVE').astype(np.float32).T # barycentric
        spectrum = hdu[1].data.field('FLUX').astype(np.float32).T
        spectrum = spectrum.astype(np.float32) # spectrum is topocentric

        #- Trim
        spectrum  = spectrum[np.logical_and(wave>=(Harps_spec_info.desiredMinW-Harps_spec_info.eps),
                                    wave<=(Harps_spec_info.desiredMaxW+Harps_spec_info.eps)) ]
        wave = wave[ np.logical_and(wave>=(Harps_spec_info.desiredMinW-Harps_spec_info.eps),
                                    wave<=(Harps_spec_info.desiredMaxW+Harps_spec_info.eps)) ]
        #- Pad
        spectrum = np.pad(spectrum,pad_width=(Harps_spec_info.padL,Harps_spec_info.padR),
                      mode='constant',constant_values=(0,0))
        wave = np.pad(wave,pad_width=(Harps_spec_info.padL,Harps_spec_info.padR),
                      mode='constant',constant_values=(0,0))

        if(self.fix_for_artifacts):
            spectrum = self.fix_mid(spectrum)
            spectrum = self.fix_paddings(spectrum)

        if(self.median_norm):
            spectrum = self.norm_by_median(spectrum)

        spectrum = np.expand_dims(spectrum,axis=0)
        return torch.Tensor(spectrum)

    def inverse_transform(self, data, label):
        for i, wl in enumerate(self.watched_labels):
            if wl == label:
                mu = self.labels_median[i]
                std = self.labels_mad[i]
                return data*std + mu

# experimental stuff
    def randomize_label(self, label):
        self.df[label] = self.df.loc[:,label].sample(frac=1.0).values

    def add_noise2label(self, label, fraction = 0.1, noise = 0.1, seed=42):
        idxs = np.argwhere(~np.isnan(self.df[label].values))

        rng = np.random.default_rng(seed)
        fraction = round(len(idxs)*fraction)
        idxs = rng.permutation(idxs)[0:fraction].ravel()

        np.random.seed(seed)
        noise_arr = np.random.normal(0, noise, fraction)
        self.df.loc[idxs, label] += self.df.loc[idxs, label] + noise_arr


    def get_k_nearest(self, labels, k):
    # search normalized catalog (self.df)
    # and get k indexes of catalog entries that are closest to the input labels
    # note: labels are supposed to be normalized in the same way
    # labels .. dictionary labels[label] contains value
    # k .. number of indexes to return
        # get subset of label values
        labels_names = []
        labels_values = []
        for label in labels.keys():
            if label in self.df.columns:
                labels_names.append(label)
                labels_values.append(labels[label])
            elif label == 'snr_compute':
                labels_names.append('snr')
                labels_values.append(labels[label])
        print(f'labels_names {labels_names}')
        print(f'labels_values {np.array(labels_values)}')
        
        # compute distances
        dist = self.df[labels_names].values - np.array(labels_values)
        dist = np.power(dist, 2)
        dist = np.power(np.nansum(dist, axis=1), 0.5)
        nanidx = np.any(np.isnan(self.df[labels_names].values), axis=1)
        
        # sort distances
        indexes = np.argsort(dist)
        indexes = indexes[~nanidx[indexes]]
        dist = dist[indexes]
        
        # return k indexes
        return dist[:k], indexes[:k]

    def labels2dic(self, labels_values):
        dic = {}
        for i, label in enumerate(self.watched_labels):
            dic[label] = labels_values[i]
        return dic
