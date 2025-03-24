import torch.utils.data as data
import torch
import os
import os.path
import numpy as np
from pdb import set_trace
import pandas as pd
from pathlib import Path
from scipy.stats import median_abs_deviation as mad


from HARPS_DL.datasets.Spectra_info import Harps_spec_info 
from HARPS_DL.datasets.Dataset_mixin import Dataset_mixin

from HARPS_DL.project_config import DATASETS_PATH

class Harps_sim_memmap_layer(Dataset_mixin, data.Dataset):
    def __init__(self,
                 memmap_filename,
                 fix_for_artifacts=False,
                 median_norm=False,
                 median_threshold=0,
                 radvel_kde=None, # distribution of radial velocities, None means no doppler shift
                 csv_file='', # must contain watched labels 
                 csv_real_file='', # must contain watched labels 
                 watched_labels=['Teff', 'logg', '[M/H]'],):

        self.fix_for_artifacts = fix_for_artifacts
        self.median_norm = median_norm
        self.median_threshold = median_threshold

        self.memmap_filename = memmap_filename
        
        # Get info about the number of spectra in the memmap
        self.file_length = os.stat(self.memmap_filename).st_size
        self.num_spectra = self.file_length // Harps_spec_info.spectrum_bytes
        
        # load memmap for workers        
        if(data.get_worker_info() is None):
            self.memmap = np.memmap(self.memmap_filename, dtype='float32', mode='r', shape=(self.num_spectra, Harps_spec_info.PaddedLength))
        else:
            self.memmap = None

        if 'radvel' not in watched_labels: # radvel is always watched
            watched_labels = ['radvel'] + watched_labels

        # radvel is on the first position
        assert('radvel' not in watched_labels[1:] and watched_labels[0] == 'radvel') 

        self.watched_labels = watched_labels

        self.radvel_kde = radvel_kde
        
        # load pandas
        self.csv_file = csv_file
        self.df = pd.read_csv(self.csv_file)
        self.df = self.df.loc[:,self.watched_labels[1:]] # subset of labels, excluding radvel

        # robust zscore + remove outliers (explored in Semi supervised learning/data_check.ipynb)
        # save median/mad for reconstruction
        if 1:
            print('labels normalization active')
            if csv_real_file == '':
                data_folder = Path(DATASETS_PATH)            
                csv_real_file = data_folder.joinpath('real_data/harps_artefacts_marked.csv')

            df_real = pd.read_csv(csv_real_file)
            df_real = df_real.loc[:,self.watched_labels]
            self.df_median = np.nanmedian(df_real, axis =0)
            df_real = df_real - self.df_median
            self.df_mad = mad(df_real, axis=0, nan_policy='omit')
            self.df = (self.df - self.df_median[1:])/self.df_mad[1:]

            # real RV = 0 means VAE RV = 0, this is important for restframe 
            self.radvel_mean = 0 # self.df_mean[0] <- this would be restframe shift, NOT DESIRED 
            self.radvel_mad = self.df_mad[0]
        else:
            print('no labels normalization!')



    def __getitem__(self, index):
        radvel = self.get_radvel() # this is normalized (VAE ready)
        #print('radvel' + str(radvel))
        #print('iradvel' + str(self.inverse_transform(radvel, 'radvel')))
#        print('iiradvel' + str(self.normalize_radvel(self.inverse_transform(radvel, 'radvel'))))

        
        return self.getitem_w_custom_radvel(index, radvel)
        
        #return torch.Tensor(spectrum)

    def getitem_w_custom_radvel(self, index, radvel):
        spectrum = self.memmap[index,:]
        spectrum = self.shift_by_radvel(spectrum, radvel)

        if(self.fix_for_artifacts):
            spectrum = self.fix_mid(spectrum)
            spectrum = self.fix_paddings(spectrum)

        if(self.median_norm):
            spectrum = self.norm_by_median(spectrum)

       # set_trace()
        labels = self.index_labels(index).to_numpy().astype(np.float64)
        #print(radvel)
        labels = np.insert(labels, 0, self.normalize_radvel(radvel)) # insert radvel at the beginning of the labels

        spectrum = np.expand_dims(spectrum, axis=0)
        labels = np.expand_dims(labels, axis=0)

        #print(labels)
        return torch.Tensor(spectrum), torch.Tensor(labels)

    def __len__(self):
        return self.num_spectra

    def get_radvel(self):
        if self.radvel_kde is None:
            return 0
        else:
            return self.radvel_kde.sample(n_samples=1).ravel()

    def normalize_radvel(self, radvel):
        return (radvel-self.radvel_mean)/self.radvel_mad

    def shift_by_radvel(self, spectrum, radvel):
        # get implicit wavelength
        st = Harps_spec_info.wave_resolution
        wave = np.arange(Harps_spec_info.desiredMinW-Harps_spec_info.padL*st,Harps_spec_info.desiredMaxW+.001+Harps_spec_info.padR*st,step=st)
        shifted_wave, shifted_spectrum = Harps_sim_memmap_layer.doppler_shift(wave, spectrum, radvel)
        # interpolate back to the implicit wavelength 
        spectrum = np.interp(wave, shifted_wave, shifted_spectrum)

        #- Trim
        spectrum = spectrum[ np.logical_and(wave>=(Harps_spec_info.desiredMinW-Harps_spec_info.eps),
            wave<=(Harps_spec_info.desiredMaxW+Harps_spec_info.eps)) ]
        wave = wave[ np.logical_and(wave>=(Harps_spec_info.desiredMinW-Harps_spec_info.eps),
            wave<=(Harps_spec_info.desiredMaxW+Harps_spec_info.eps)) ]

        #- Pad
        spectrum = np.pad(spectrum,pad_width=(Harps_spec_info.padL,Harps_spec_info.padR),
                mode='constant',constant_values=(0,0))
        wave = np.pad(wave,pad_width=(Harps_spec_info.padL,Harps_spec_info.padR),
                mode='constant',constant_values=(0,0))

        # masking
        spectrum[:Harps_spec_info.left_last_zero+1] = 0
        spectrum[Harps_spec_info.right_first_zero:] = 0
        spectrum[Harps_spec_info.mid_first_zero:Harps_spec_info.mid_last_zero+1] = 0

        return spectrum

    @staticmethod
    def doppler_shift(wave, flux, radial_velocity):
        doppler_factor = (1 + radial_velocity/299792.458)
        new_wave = wave * doppler_factor
        flux_preserved = flux / doppler_factor # increase of bins shouldn't increase flux
        return new_wave, flux_preserved

    def inverse_transform(self, data, label):
        if label == 'radvel':
            return data*self.radvel_mad + self.radvel_mean
        else:
            i = self.watched_labels.index(label)
            return data*self.df_mad[i] + self.df_median[i]

    def estimate_rv_fft(self, flux_1, flux_2):
        raise Exception('problem with always returning 0')
        st = Harps_spec_info.wave_resolution
        wave = np.arange(Harps_spec_info.desiredMinW-Harps_spec_info.padL*st,Harps_spec_info.desiredMaxW+.001+Harps_spec_info.padR*st,step=st)

        log_wave = np.linspace(np.min(np.log(wave)), np.max(np.log(wave)), len(wave))
        st_log = (np.max(log_wave) - np.min(log_wave))/(len(log_wave) - 1)
        log_flux_1 = np.interp(log_wave, np.log(wave), flux_1)
        log_flux_2 = np.interp(log_wave, np.log(wave), flux_2)

        # split fluxes into 2 parts
        slice_A = slice((Harps_spec_info.left_last_zero + 1),
                       Harps_spec_info.mid_first_zero)
        slice_B = slice((Harps_spec_info.mid_last_zero + 1),
                       Harps_spec_info.right_first_zero)
        log_flux_1_A = log_flux_1[slice_A]
        log_flux_1_B = log_flux_1[slice_B]
        
        log_flux_2_A = log_flux_2[slice_A]
        log_flux_2_B = log_flux_2[slice_B]

        set_trace()
        r = self.get_phase_shift(log_flux_1_A, log_flux_2_A)
        idx_max = np.argmax(r)
        idx_shift = np.min([len(r) - idx_max, idx_max])
        rv_shift_A = np.exp(idx_shift*st_log)

        r = self.get_phase_shift(log_flux_1_B, log_flux_2_B)
        idx_max = np.argmax(r)
        idx_shift = np.min([len(r) - idx_max, idx_max])
        rv_shift_B = np.exp(idx_shift*st_log)

        rv_shift = (rv_shift_A + rv_shift_B)/2

        return rv_shift

    def get_phase_shift(self, sig_1, sig_2, noise_amp=1e-6):
        # measure the phase shift between two real signals
        assert(np.all(np.isreal(sig_1)))
        assert(np.all(np.isreal(sig_2)))

        fft_sig1 = np.fft.fft(sig_1 + np.random.randn(len(sig_1))*noise_amp)
        fft_sig2 = np.fft.fft(sig_2 + np.random.randn(len(sig_1))*noise_amp)
        fft_sig2_conj = np.conj(fft_sig2)

        R = (fft_sig1 * fft_sig2_conj) / abs(fft_sig1 * fft_sig2_conj)
        r = np.real(np.fft.ifft(R)) # imaginary part is non-zero only due to numerical errors
        return r
