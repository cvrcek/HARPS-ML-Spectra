import torch.utils.data as data
import torch
import os
import os.path
import numpy as np
from astropy.io import fits
#from downsample_spec import downsample2nd
from pdb import set_trace
import pandas as pd
from scipy.stats import zscore
from scipy.stats import median_abs_deviation as mad
from HARPS_DL.datasets.Spectra_info import Harps_spec_info

class Dataset_mixin():
    def __init__(self,                                                   
                 median_threshold=0,                 
                 ):
        self.median_threshold = median_threshold

    @staticmethod
    def fix_mid(spectrum):
        # Fix middle zeros
        x1 = Harps_spec_info.mid_first_zero
        x2 = Harps_spec_info.mid_last_zero
        f1_ = spectrum[x1-1]
        f2_ = spectrum[x2+1]
        interp = np.linspace(f1_,f2_,num=x2-x1+1)
        spectrum[x1:x2+1] = interp
        return spectrum

    @staticmethod
    def fix_paddings(spectrum):
        # Fix left and right zeros
        x1 = Harps_spec_info.left_last_zero
        x2 = Harps_spec_info.right_first_zero
        exterp1 = spectrum[x1+1:2*x1+2]
        exterp2 = spectrum[2*x2-len(spectrum):x2]
        spectrum[0:x1+1] = exterp1[::-1]
        spectrum[x2:] = exterp2[::-1]
        return spectrum

    @staticmethod
    def get_artifact_mask(batch_size):
        mask = np.ones((1,Harps_spec_info.PaddedLength),dtype=np.float32)
        mask[0,:Harps_spec_info.left_last_zero+1] = 0
        mask[0,Harps_spec_info.right_first_zero:] = 0
        mask[0,Harps_spec_info.mid_first_zero:Harps_spec_info.mid_last_zero+1] = 0

        tiled_mask = np.repeat(mask,batch_size,axis=0)

        return tiled_mask

    @staticmethod
    def static_norm_by_median(spec, median_threshold):
        med = np.nanmedian(spec * Dataset_mixin.get_artifact_mask(1))
        if(med <= median_threshold):
            spec[:] = 0 
            return spec
        if(np.mean(spec) <= 0):
            spec[:] = 0 
            return spec

        return spec / med
    
    def norm_by_median(self, spec):
        return self.static_norm_by_median(spec, self.median_threshold)
        
    @staticmethod    
    def get_wave():
        st = Harps_spec_info.wave_resolution
        emul_wave = np.arange(Harps_spec_info.desiredMinW-Harps_spec_info.padL*st,Harps_spec_info.desiredMaxW+.001+Harps_spec_info.padR*st,step=st)
        return emul_wave
    
    def index_labels(self, index):
        return self.df.iloc[index]
    
