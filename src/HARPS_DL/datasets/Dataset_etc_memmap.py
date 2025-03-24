from pathlib import Path
import torch.utils.data as data
import torch
import os

import os.path
import numpy as np

import pandas as pd




from HARPS_DL.generate_simulations.ETC_processing.etc_tools import runtime_fast_2 as runtime # three versions were tested, runtime_fast_2 is fastest for noise free data


from HARPS_DL.datasets.Spectra_info import Harps_spec_info
from HARPS_DL.datasets.Dataset_sim_memmap import Harps_sim_memmap_layer

from sklearn.neighbors import KernelDensity

class harps_etc_memmap(Harps_sim_memmap_layer):
    def __init__(self,
                 memmaps_folder,
                 dataset_name = 'ETC',
                 radvel_range=[0, 0],
                 fix_for_artifacts=False,
                 median_norm=False,
                 median_threshold=0,
                 noise_free=False,
                 labels_norm=None,
                 precompute_rv='no',
                 dictionary_output=False,
                 ):

        self.memmaps = {}
        self.memmaps['keys'] =  ('target_SED', 'T_A', 'T_B', 'h_A', 'h_B',
                             'B_A', 'B_B', 'n_A', 'n_B')

        self.memmaps_folder = Path(memmaps_folder)
        self.dataset_name = dataset_name
        self.noise_free = noise_free
        self.labels_norm = labels_norm
        self.precompute_rv = precompute_rv

        self.fix_for_artifacts = fix_for_artifacts
        self.median_norm = median_norm
        self.median_threshold = median_threshold

        self.dictionary_output = dictionary_output

        # Get info about the number of spectra in the memmap
        self.file_length = os.stat(self.memmaps_folder.joinpath(self.memmaps['keys'][0])).st_size
        self.num_spectra = self.file_length // Harps_spec_info.spectrum_bytes

        # load memmap for workers
        if(data.get_worker_info() is None):
            for key in self.memmaps['keys']:
                self.memmaps[key] = np.memmap(self.memmaps_folder.joinpath(key),
                                    dtype='float32', mode='r', shape=(self.num_spectra, Harps_spec_info.PaddedLength))
        else:
            self.memmaps = {}

        # load pandas
        self.csv_file = self.memmaps_folder.joinpath('etc_labeled.csv')
        self.df = pd.read_csv(self.csv_file)

        self.fun = runtime
        
        if self.precompute_rv == 'uniform':
            self.rv_values = np.random.default_rng(seed=0).uniform(radvel_range[0], radvel_range[1], len(self))
        if self.precompute_rv == 'gaussian':
            loc = (radvel_range[0] + radvel_range[1])/2
            scale = (radvel_range[1] - radvel_range[0])/6
            self.rv_values = np.random.default_rng(seed=0).normal(loc, scale, len(self))
        elif self.precompute_rv == 'no':
            if np.abs(radvel_range[1] - radvel_range[0]) <= 1e-6:
                self.radvel_kde = None
            else:
                assert(radvel_range[1] > radvel_range[0])
                rv_bandwidthw = (radvel_range[1] - radvel_range[0])/2
                rv_mean = (radvel_range[1] + radvel_range[0])/2
                # by fitting a single point, we achieve an uniform distribution
                # note: we could easily fit here the real data radvel to achieve a more realistic distribution
                # We use uniform distribution to support learning of the shifting mechanism
                self.radvel_kde = KernelDensity(kernel="tophat",
                                                bandwidth=rv_bandwidthw).fit(np.array(rv_mean).reshape(-1, 1))
        else:
            raise Exception(f'unknown precompute_rv {self.precompute_rv}')
            
    def __getitem__(self, index):
        radvel = self.get_radvel(index) # this has radvel units (km/s ?)
        return self.getitem_w_custom_radvel(index, radvel)

        #return torch.Tensor(spectrum)

    def getitem_w_custom_radvel(self, index, radvel):
        # radvel can't be normalized (shift wont work)

        dic_in = {}
        for key in self.memmaps['keys']:
             dic_in[key] = self.memmaps[key][index,:]

#        spectrum = runtime(dic_in, rv=radvel, debug=False)['S_samp']
        output = self.fun(dic_in,
                         rv=radvel,
                         debug=False,
                         noise_free=self.noise_free)
        if self.noise_free:
            spectrum = output['S']
        else:
            spectrum = output['S_samp']

        if(self.fix_for_artifacts):
            spectrum = self.fix_mid(spectrum)
            spectrum = self.fix_paddings(spectrum)

        if(self.median_norm):
            spectrum = self.norm_by_median(spectrum)

        # this simplifies loss function computation
        spectrum[np.isnan(spectrum)] = 0

        spectrum = np.expand_dims(spectrum, axis=0)

        labels = self.df.iloc[index].to_dict()
        labels['name'] = self.dataset_name
        labels['radvel'] = radvel

        if self.labels_norm is None:
            if self.dictionary_output: # this is for postprocessing, where dic might be useful
                return torch.Tensor(spectrum), labels # spectrum + dictionary
            else: # this for training, when I cant use dictionary anyway
                return torch.Tensor(spectrum), torch.Tensor(0) # spectrum + dummy
        else:
            labels = self.labels_norm.dic2vec(labels)
            labels = np.expand_dims(labels, axis=0)

            return torch.Tensor(spectrum), torch.Tensor(labels)

    def get_radvel(self, index):
        if self.precompute_rv == 'uniform' or self.precompute_rv == 'gaussian':
            return self.rv_values[index]            
        elif self.precompute_rv == 'no':
            if self.radvel_kde is None:
                return 0
            else:
                return self.radvel_kde.sample(n_samples=1).ravel()
        else:
            raise Exception(f'unknown precompute_rv {self.precompute_rv }')
    
    
