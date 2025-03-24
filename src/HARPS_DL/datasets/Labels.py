import numpy as np
import argparse
import json
import pandas as pd
import os
from HARPS_DL.project_config import NORMALIZATION_PATH

class Labels():
    """
    normalizes dataset labels and groups them into a bottleneck ready vectors
    """
    def __init__(self,
                 labels: list[str]=[],
                 datasets_names: list[str]=['ETC', 'real'],
                 labels_type: list[str]=[],
                 labels_normalization: dict={},
                 fill_bottleneck: list[str]=[],
                 bottleneck: int=0,
                ):
        """
            labels .. set of recognised labels
            datasets_names .. recognised names of datasets (required for indexing)
            labels_type .. 'shared' = label is shared between datasets,
                           'separated' = label is split across datasets
                           dataset_name = label is used exclusively for this dataset
            labels_normalization .. dictionary with mean/std of labels for normalization
            fill_bottleneck .. fill bottleneck for selected datasets by zeros
            bottleneck .. number of elements in bottleneck
        """
        self.labels = labels
        self.dataset_names = datasets_names

        assert(type(labels_normalization) is dict) # I made a mistake and labels normalization is sometimes bool, fix it!
        # False -> {}, True -> normalization dictionary
        self.labels_normalization = labels_normalization

        self.labels_type = labels_type

        self.fill_bottleneck = fill_bottleneck
        self.bottleneck = bottleneck

        self.check_input()

        self.idxs = {} # map label-dataset_name -> vector index
        self.N = 0
        for idx, label in enumerate(self.labels):
            if self.labels_type[idx] == 'shared' or self.labels_type[idx] in self.dataset_names:
                self.idxs[label] = {datasets_names[i]: self.N for i in range(len(datasets_names))}
                self.N += 1 
            elif self.labels_type[idx] == 'separated':
                self.idxs[label] = {datasets_names[i]: (self.N + i) for i in range(len(datasets_names))}
                #self.N += 2 if idx != len(self.labels) - 1 else 1 # correct dimensions
                self.N += 2
            else:
                raise Exception('labels_type can be either ''shared'', ''separated'', or a name of one of the input datasets')
        
        # collect names per vector element into dictionary 
        vec_names = {}
        for label, label_idxs in self.idxs.items():
            for dataset_name, idx in label_idxs.items():
                if idx in vec_names.keys():
                    vec_names[idx] = vec_names[idx] + '_' + dataset_name
                else:
                    vec_names[idx] = label + '_' + dataset_name

        # dictionary to list
        self.vec_names = []
        for i in range(len(vec_names.keys())):
            self.vec_names.append(vec_names[i])

    def df2normalization(self, df_real, df_etc):
        """
            this would be typically called from some datamodule 
        """
        labels_normalization = {}
        for label in self.labels:
            if label == 'snr_compute': # normalize from snr in df_real
                values = df_real['snr']
            elif label == 'mag': # normalize from simulation data (not in df_real)
                values = df_etc[label]
            elif label == 'Texp': # normalize from simulation data (not in df_real)
                values = df_etc[label]
            elif label == 'H2O_pwv': # normalize from simulation data (not in df_real)
                values = df_etc[label]
            else:
                values = df_real[label]
            
            label_median = np.nanmedian(values, axis=0)
            label_mad = np.nanmedian(np.abs(values - label_median), axis=0)
            labels_normalization[label] = {'median': label_median, 'mad': label_mad}
        self.labels_normalization = labels_normalization

    def json2normalization(self):
        """
            this alternative method for setting normalization
        """
        with open(NORMALIZATION_PATH, 'r') as f:
            labels_normalization = json.load(f)
        labels_normalization_reduced = {}
        for label in self.labels:
            labels_normalization_reduced[label] = labels_normalization[label]

        self.labels_normalization = labels_normalization_reduced
        

    @staticmethod
    def df2json(df_real, df_etc):
        """
            this is used to prepare persistent normalization file
        """
        labels_normalization = {}
        labels = ['Teff', 'logg', '[M/H]', 'Mass', 'radvel', 'BERV', 'mag', 'Texp', 'airmass', 'H2O_pwv', 'snr_compute']
        for label in labels:
            if label == 'snr_compute': # normalize from snr in df_real
                values = df_real['snr']
            elif label == 'mag': # normalize from simulation data (not in df_real)
                values = df_etc[label]
            elif label == 'Texp': # normalize from simulation data (not in df_real)
                values = df_etc[label]
            elif label == 'H2O_pwv': # normalize from simulation data (not in df_real)
                values = df_etc[label]
            else:
                values = df_real[label]
            
            label_median = np.nanmedian(values, axis=0)
            label_mad = np.nanmedian(np.abs(values - label_median), axis=0)
            labels_normalization[label] = {'median': label_median, 'mad': label_mad}
        with open(NORMALIZATION_PATH, 'w') as f:
            json.dump(labels_normalization, f)
        
    def check_input(self):
        assert(len(self.labels) == len(self.labels_type))

    def dic2vec(self, dic_in):
        # dictionary of unnormalized labels (label, value) + dataset name ('name', dataset_name)
        assert(dic_in['name'] in self.dataset_names)
        
        if len(self.fill_bottleneck) == 0:
            vec = np.full((self.N,), np.nan)
        else:
            vec = np.full((self.bottleneck,), np.nan)

        for idx, label in enumerate(self.labels):
            if self.labels_type[idx] != 'shared' and \
                self.labels_type[idx] != 'separated' and \
                self.labels_type[idx] != dic_in['name']:
                continue # this label is ignored for this dataset
            if label == 'name' or not (label in dic_in.keys()) :
                continue
            vec[self.idxs[label][dic_in['name']]] = self.normalize_label(label, dic_in[label])

        # assign zero to unlabeled data
        if dic_in['name'] in self.fill_bottleneck:
            vec[self.N:] = 0 

        return vec
    
    def normalize_label(self, label, value):
        if self.labels_normalization == None:
            nvalue = value #no normalization
        else:
            nvalue = (value - self.labels_normalization[label]['median'])\
                    /self.labels_normalization[label]['mad']
        return nvalue

    def scale_label(self, label, value):
        """
            scale labels 

            Important when shifting by label, such as Doppler shift by 5 [km/s]/
            Normalization is incorrect for such case!!!
        """
        if self.labels_normalization == None:
            nvalue = value #no scaling
        else:
            nvalue = value/self.labels_normalization[label]['mad']
                    
        return nvalue

    def inverse_label_normalization(self, value, label:str):
        if self.labels_normalization == None:
            nvalue = value # no normalization took place
        else:
            nvalue = (value*self.labels_normalization[label]['mad'])\
                    + self.labels_normalization[label]['median']
        return nvalue

    def vec2dics(self, vec):
        # return dictionary for every name
        #
        dic_out = {}
        for name in self.dataset_names:
            dic_out[name] = {}
            for idx, label in enumerate(self.labels):
                dic_out[name][label] = vec[self.idxs[label][name]]
        return dic_out
    
    def get_vec_length(self):
        """
        return size of the output label vector
        """
        return self.N

    def label2idx(self, label):
        return self.idxs[label]

    def get_vec_names(self):
        """
        name (label + dataset_name) associated with each position in vector
        """
        return self.vec_names

    def get_label_median(self, label):
        return self.labels_normalization[label]['median']
        
    def get_label_mad(self, label):
        return self.labels_normalization[label]['mad']

    def inverse_normalization_array(self, array):
        assert(array.shape[0] == self.get_vec_length())
        array_inversed = np.copy(array)
        for label in self.labels:
            for _, idx in self.label2idx(label).items():
                array_inversed[idx,:] = self.inverse_label_normalization(array[idx,:], label)
        return array_inversed

def test_output_dimension_Labels():
    print('test output vector dimensions:')
    # 1
    labels = Labels(labels=['radvel', 'Teff'],
                    datasets_names=['ETC', 'real'],
                    labels_normalization={'radvel': {'median': 1, 'mad': 2},
                                          'Teff': {'median': 10, 'mad': 20},
                                          },
                    labels_type=['shared', 'separated'],
                    )
    dic_in = {'radvel':3, 'name':'ETC'}
    vec = labels.dic2vec(dic_in)
    dic_in = {'radvel':3, 'name':'real'}
    vec = labels.dic2vec(dic_in)
    assert(len(vec)==3)
    # 2
    labels = Labels(labels=['radvel', 'Teff', '[M/H]'],
                    datasets_names=['ETC', 'real'],
                    labels_normalization={'radvel': {'median': 1, 'mad': 2},
                                          'Teff': {'median': 10, 'mad': 20},
                                          '[M/H]': {'median': 10, 'mad': 20},
                                          },
                    labels_type=['shared', 'separated', 'shared'],
                    )
    dic_in = {'radvel':3, 'name':'real'}
    vec = labels.dic2vec(dic_in)
    assert(len(vec)==4)
    # 3
    labels = Labels(labels=['radvel', 'Teff', '[M/H]'],
                    datasets_names=['ETC', 'real'],
                    labels_normalization={'radvel': {'median': 1, 'mad': 2},
                                          'Teff': {'median': 10, 'mad': 20},
                                          '[M/H]': {'median': 10, 'mad': 20},
                                          },
                    labels_type=['separated', 'shared', 'separated'],
                    )
    dic_in = {'radvel':3, 'name':'real'}
    vec = labels.dic2vec(dic_in)
    assert(len(vec)==5)
    print(' -> Pass')

def test_values_Labels():
    print('test values in vectors:')
    labels = Labels(labels=['radvel', 'Teff'],
                    datasets_names=['ETC', 'real'],
                    labels_normalization={'radvel': {'median': 1, 'mad': 2},
                                          'Teff': {'median': 10, 'mad': 20},
                                          },
                    labels_type=['shared', 'separated'],
                    )
    # 1
    dic_in = {'radvel':3, 'name':'ETC'}
    vec = labels.dic2vec(dic_in)
    assert(np.allclose(vec, [1, np.nan, np.nan], equal_nan=True))
    # 2 
    dic_in = {'radvel':5, 'name':'real'}
    vec = labels.dic2vec(dic_in)
    assert(np.allclose(vec, [2, np.nan, np.nan], equal_nan=True))
    # 3 
    dic_in = {'radvel':3, 'Teff':10, 'name':'ETC'}
    vec = labels.dic2vec(dic_in)
    assert(np.allclose(vec, [1, 0, np.nan], equal_nan=True))
    # 4 
    dic_in = {'radvel':-1, 'Teff':30, 'name':'real'}
    vec = labels.dic2vec(dic_in)
    assert(np.allclose(vec, [-1, np.nan, 1], equal_nan=True))
    print(' -> Pass')

def test_vec_names():
    print('test get_vec_names:')
    labels = Labels(labels=['radvel', 'Teff'],
                    datasets_names=['ETC', 'real'],
                    labels_normalization={'radvel': {'median': 1, 'mad': 2},
                                          'Teff': {'median': 10, 'mad': 20},
                                          },
                    labels_type=['shared', 'separated'],
                    )
    #print(labels.get_vec_names())
    assert(labels.get_vec_names()[0] == 'radvel_ETC_real')
    assert(labels.get_vec_names()[1] == 'Teff_ETC')
    assert(labels.get_vec_names()[2] == 'Teff_real')
    print(' -> Pass')

def test_real_etc_label_type():
    print('test labels exclusivity')
    # 1
    labels = Labels(labels=['radvel', 'Teff'],
                    datasets_names=['ETC', 'real'],
                    labels_normalization={'radvel': {'median': 1, 'mad': 2},
                                          'Teff': {'median': 10, 'mad': 20},
                                          },
                    labels_type=['shared', 'real'],
                    )

    dic_in = {'radvel':3, 'Teff':2, 'name':'ETC'}
    vec = labels.dic2vec(dic_in)
    assert(np.isnan(vec[1]))
    dic_in = {'radvel':2, 'Teff':1, 'name':'real'}
    vec = labels.dic2vec(dic_in)
    assert(not np.isnan(vec[1]))
    print(' -> Pass')

def run_all_tests():
    test_output_dimension_Labels()
    test_values_Labels()
    test_vec_names()
    test_real_etc_label_type()

#run_all_tests()

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--csv_real', type=str, required=True, help='Path to the real CSV file')
    parser.add_argument('--csv_ETC', type=str, required=True, help='Path to the ETC CSV file')

    args = parser.parse_args()

    run_all_tests()

    # check that both files exists
    if os.path.isfile(args.csv_real) and os.path.isfile(args.csv_ETC):
        print('creating normalizing JSON')
        Labels.df2json(pd.read_csv(args.csv_real), pd.read_csv(args.csv_ETC))
    
    run_all_tests()    
    # check that both files exists
    if os.path.isfile(args.csv_real) and os.path.isfile(args.csv_ETC) :
        print('creating normalizing JSON')
        Labels.df2json( pd.read_csv(args.csv_real),
                        pd.read_csv(args.csv_ETC))
