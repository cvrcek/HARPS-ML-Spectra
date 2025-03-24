import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation as mad
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt
from pdb import set_trace

from HARPS_DL.datasets.Spectra_info import Harps_spec_info
from HARPS_DL.datasets.Dataset_sim_memmap import Harps_sim_memmap_layer
from HARPS_DL.datasets.Dataset_mixin import Dataset_mixin


class Test_sim_memmap_layer:
    def create_mock_data(self, N=3):
        # creates N identical spectra
        # each spectrum has flux values equal to its wavelengths
        # ie spectrum[Dataset_mixin.get_wave()[i]] = Dataset_mixin.get_wave()[i]
        
        from tempfile import mkdtemp
        import os.path as path
        tmp_dir = mkdtemp()
        filename_spectra = path.join(tmp_dir, 'spectra.dat')
        
        fp = np.memmap(filename_spectra, dtype='float32', mode='w+', shape=(N, Harps_spec_info.PaddedLength))        
        for i in range(N):
            fp[i,:] = Dataset_mixin.get_wave()        
        fp.flush()

        filename_labels = path.join(tmp_dir, 'labels.csv')
        data = {'Teff': np.random.normal(loc=1000,scale=100, size=N),
                'logg': np.random.normal(loc=5,scale=5, size=N),
                '[M/H]': np.random.normal(loc=1,scale=1, size=N)}
        pd.DataFrame(data).to_csv(filename_labels)
        
        return filename_spectra, filename_labels, data

    def get_rv_data(self, rv_mean, rv_bw):
        # simulate sampling radial velocity during runtime
        N = 10
        mock_spectra_filename, mock_csv, labels = self.create_mock_data(N=N)
        kde = KernelDensity(kernel="tophat", bandwidth=rv_bw).\
            fit(np.array(rv_mean).reshape(-1, 1))
        dataset = Harps_sim_memmap_layer(
                    mock_spectra_filename,
                    fix_for_artifacts=False,
                    median_norm=False,
                    median_threshold=0,
                    radvel_kde=kde, # distribution of radial velocities
                    csv_file=mock_csv,
                    watched_labels=['Teff', 'logg', '[M/H]'],)
        rvs = np.zeros((N,))
        for i in range(N):
            labels_ = dataset[i][1].numpy().ravel() # data[index] -> (spectrum, label)
            rvs[i] = labels_[0]
        
        return rvs, dataset.inverse_transform(rvs, 'radvel')

    def test_masking(self):
        mock_spectra_filename, mock_csv, _ = self.create_mock_data()
        #set_trace()
        dataset = Harps_sim_memmap_layer(
                    mock_spectra_filename,
                    fix_for_artifacts=False,
                    median_norm=False,
                    median_threshold=0,
                    radvel_kde=None, # distribution of radial velocities
                    csv_file=mock_csv,                    
                    watched_labels=['Teff', 'logg', '[M/H]'],)
        for i in range(3):
            data = dataset[i][0]
            # spectrum has correct size
            assert(data.shape == (1, Harps_spec_info.PaddedLength))
            data = data.ravel().numpy()
            my_slice = slice(Harps_spec_info.left_last_zero + 1,
                            Harps_spec_info.right_first_zero,
                            1)
            # mask test
            # left tail is zero
            assert(np.all(data[:Harps_spec_info.left_last_zero + 1] == 0))
            # first non-zero segment
            assert(np.all(data[(Harps_spec_info.left_last_zero + 1):
                (Harps_spec_info.mid_first_zero)] != 0))
            # zero middle
            assert(np.all(data[(Harps_spec_info.mid_first_zero):
                (Harps_spec_info.mid_last_zero + 1)] == 0))
            # second non-zero segment
            assert(np.all(data[(Harps_spec_info.mid_last_zero + 1):
                (Harps_spec_info.right_first_zero)] != 0))
            # right tail is zero
            assert(np.all(data[Harps_spec_info.right_first_zero:] == 0))
        

    def test_shifting(self):
        def check_shift(dataset_rest, wave_1, shift):
            # -shift wave_1 by shift (units Angstroms)            
            wave = Dataset_mixin.get_wave()
            
            wave_2 = wave_1 + shift
            doppler_factor = wave_2/wave_1
            rv = (doppler_factor - 1)*299792.458
            
            kde = KernelDensity(kernel="tophat", bandwidth=1e-12).\
                fit(np.array(rv).reshape(-1, 1))
            dataset = Harps_sim_memmap_layer(
                        mock_spectra_filename,
                        fix_for_artifacts=False,
                        median_norm=False,
                        median_threshold=0,
                        radvel_kde=kde, # distribution of radial velocities
                        csv_file=mock_csv,
                        watched_labels=['Teff', 'logg', '[M/H]'],)

            data = dataset[0][0].numpy().ravel() # data[index] -> (spectrum, label)
            data_rest = dataset_rest[0][0].numpy().ravel()
            
            ind_1 = closest_index(wave, wave_1)
            ind_2 = closest_index(wave, wave_2)
            #print(f'data[ind] = {data[ind]}, wave_1 = {wave_1}, wave_2 = {wave_2}')
            assert(np.abs(data_rest[ind_1]/doppler_factor - data[ind_2]) < 1e-3)        
                
        def closest_index(arr, val):
            return (np.abs(arr - val)).argmin()
                
        mock_spectra_filename, mock_csv, _ = self.create_mock_data()
                
        dataset_rest = Harps_sim_memmap_layer(
                    mock_spectra_filename,
                    fix_for_artifacts=False,
                    median_norm=False,
                    median_threshold=0,
                    radvel_kde=None, # distribution of radial velocities
                    csv_file=mock_csv,
                    watched_labels=['Teff', 'logg', '[M/H]'],)
        
        # TEST NO SHIFT
        check_shift(dataset_rest, 3785, 0)  
        
        # TEST 1 Angstrom SHIFT at 3785 Angstrom
        check_shift(dataset_rest, 3785, 1)  

        # TEST 100 Angstrom SHIFT at 3785 Angstrom
        check_shift(dataset_rest, 3785, 100)
        
        # TEST -100 Angstrom SHIFT at 5000 Angstrom
        check_shift(dataset_rest, 5000, -100)
        
    def test_radvel_distributions(self):        
        # test that sampled data are distributed as expected
        # uniform(0, 1)
        center = 0
        half_spread = 0.5
        rvs_normalized, rvs = self.get_rv_data(center, half_spread) 
        assert(np.max(rvs) <= center + half_spread)
        assert(np.min(rvs) >= center - half_spread) 

        # uniform(-1, 1)
        center = 0
        half_spread = 1
        rvs_normalized, rvs = self.get_rv_data(center, half_spread)
        assert(np.max(rvs) <= center + half_spread)
        assert(np.min(rvs) >= center - half_spread)

        # uniform(0, 2)
        center = 1
        half_spread = 1
        rvs_normalized, rvs = self.get_rv_data(center, half_spread) 
        assert(np.max(rvs) <= center + half_spread)
        assert(np.min(rvs) >= center - half_spread)

    def test_stellar_distributions(self): # except radvel        
        N = 10
        mock_spectra_filename, mock_csv, labels = self.create_mock_data(N=N)
        watched_labels=['Teff', 'logg', '[M/H]']
        dataset = Harps_sim_memmap_layer(
                    mock_spectra_filename,
                    fix_for_artifacts=False,
                    median_norm=False,
                    median_threshold=0,
                    radvel_kde=None, # distribution of radial velocities
                    csv_file=mock_csv,
                    watched_labels=watched_labels,)
        labels_out = {'Teff': [], 'logg': [], '[M/H]': []}

        for i in range(N):
            labels_ = dataset[i][1].numpy().ravel() # data[index] -> (spectrum, label)
            labels_ = labels_[1:] # remove radvel
            for li, label in enumerate(watched_labels):
                labels_out[label].append(dataset.inverse_transform(labels_[li], label))

        for li, label in enumerate(watched_labels):            
            assert(np.all(np.abs(labels_out[label] - labels[label]) < 1e-3))

    