import pandas as pd
import numpy as np
from pdb import set_trace as st
from scipy.stats import zscore
from scipy.stats import median_abs_deviation as mad



class Spectra_search():
    def __init__(self,csv_file='/home/vcvrcek/Python/spectra_DL/labels/harps_labels_complete.csv',
                 watched_labels=['radvel', 'Teff', 'Mass', 'logg', '[M/H]', 'airmass', 'snr']):
        self.df = pd.read_csv(csv_file)
        self.df_complete = self.df
        self.watched_labels = watched_labels
        self.df = self.df.loc[:,self.watched_labels] # subset of labels

        #--- robust zscore + remove outliers (explored in Semi supervised learning/data_check.ipynb)
        # save median/mad for reconstruction
        self.df_median = np.nanmedian(self.df, axis =0)
        self.df = self.df - self.df_median
        self.df_mad = mad(self.df, axis=0, nan_policy='omit')
        self.df = self.df/self.df_mad
        self.df[np.abs(self.df) > 5] = np.nan


    def spectra_distance(self, idx_1, idx_2):
        # compute distance between spectra in the label space
        #
        #
        labels_1 = self.df.iloc[idx_1].values
        labels_2 = self.df.iloc[idx_2].values
        dist = np.power(labels_1 - labels_2,2)
        ind_val = ~np.isnan(dist)
        dist = np.sqrt(np.sum(dist, where=ind_val))
        return dist

    def k_nearest(self, idx, k=3):
        # search k nearest spectra

        # get distances
        dists = np.sqrt(np.sum(np.power(self.df - self.df.iloc[idx], 2), axis=1))
        nan_cnt = np.sum(np.isnan(self.df), axis=1).values
        idx_val = np.argwhere(nan_cnt == np.min(nan_cnt)).ravel()
        idxs_sorted = np.argsort(dists[idx_val])


        idxs = idx_val[idxs_sorted[0:k]]
        dists = dists[idxs]
        return  dists, idxs


    def labels_k_nearest(self, labels, k):
        # get distances
        dists = np.sqrt(np.sum(np.power(self.df - labels, 2), axis=1))
        nan_cnt = np.sum(np.isnan(self.df), axis=1).values
        idx_val = np.argwhere(nan_cnt == np.min(nan_cnt)).ravel()
        idxs_sorted = np.argsort(dists[idx_val])


        idxs = idx_val[idxs_sorted[0:k]]
        dists = dists[idxs]
        return  dists, idxs



    def norm_labels(self, labels):
        labels = labels - self.df_median
        labels = labels/self.df_mad
        return labels





csv_file= "/home/cv/Dropbox/PHD/Python/ESO/spectra_DL/" +\
             "/spectra_collecting/harps_metadata_and_labels.csv"
ss=Spectra_search(csv_file=csv_file)

# test spectra_distance
if 0:
    print(ss.spectra_distance(0, 1))

# test k_nearest
if 0:
    dists, idxs = ss.k_nearest(idx=2,k=10)
    print(dists)
    print(idxs)

# test labels_k_nearest
if 1:
     #['radvel', 'Teff', 'Mass', 'logg', '[M/H]', 'airmass', 'snr']):
    labels=[0.036, 6119.8, 1.15, 4.1479, 0.1241, np.nan, np.nan]
    labels=ss.norm_labels(labels)
    dists, idxs = ss.labels_k_nearest(labels,k=100)
    for idx in idxs:
        print(ss.df_complete.iloc[idx])
    print(labels)
    print(dists.values)
    print(idxs)
