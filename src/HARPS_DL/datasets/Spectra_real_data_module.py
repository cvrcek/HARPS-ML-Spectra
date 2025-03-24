import os
from pathlib import Path
import sys
from pdb import set_trace
import pandas as pd
from torch.utils.data import WeightedRandomSampler

import numpy as np
import pytorch_lightning as pl
import torch.utils.data

import neptune as neptune

from HARPS_DL.datasets.Dataset_real_memmap import Harps_real_memmap
from HARPS_DL.datasets.Dataset_etc_memmap import harps_etc_memmap
from HARPS_DL.datasets.Labels import Labels
from HARPS_DL.datasets.Datamodule_overview import Datamodule_overview

from HARPS_DL.project_config import DATASETS_PATH

class Spectra_real_data_module(Datamodule_overview, pl.LightningDataModule):
    def __init__(self,
                num_workers: int=8,
                batch_size: int=32,
                etc_train_folder: str='ETC_uniform_new_dist/memmap_train',
                etc_val_folder: str='ETC_uniform_new_dist/memmap_val',
                labels=None,
                normalize_labels: bool=True,
                median_norm: bool=True,
                labels2nan_frac: float=0.0,
                labels_fraction_seed: int=42,
                remove_all_nan_labels: bool=False,
                legacy_split: bool=False,
                ):
        """ETC data module

        Args:
            batch_size: samples per batch (for DataLoader)
            labels: Labels instance
            normalize_labels: flag signaling labels normalization
        """
        super().__init__()
        assert(labels2nan_frac >= 0.0 and labels2nan_frac <= 1.0) # fraction of NaNs in the dataset

        self.batch_size = batch_size
        self.num_workers=num_workers
        self.labels = labels
        self.normalize_labels = normalize_labels
        self.median_norm = median_norm

        data_folder = Path(DATASETS_PATH)
        self.real_memmap_filename = data_folder.joinpath('real_data/harps-nonan-stable.dat')
        csv_real_file = data_folder.joinpath('real_data/harps_artefacts_marked.csv')

        self.etc_train_dataset_folder = data_folder.joinpath(etc_train_folder)
        self.etc_val_dataset_folder = data_folder.joinpath(etc_val_folder)

        self.etc_train_csv_file = self.etc_train_dataset_folder.joinpath('etc_labeled.csv')
        self.etc_val_csv_file = self.etc_val_dataset_folder.joinpath('etc_labeled.csv')

        self.csv_real_file = csv_real_file

        self.labels_fraction_seed = labels_fraction_seed

        if self.normalize_labels:
            self.labels.df2normalization(pd.read_csv(self.csv_real_file),
                                         pd.read_csv(self.etc_train_csv_file),
                                         )


        # (look for old approach for both)
        # I have to split the real dataset into training/validation sets
        # Make sure unique targets aren't spilling between sets!!!
        df = pd.read_csv(self.csv_real_file)

        mask = ~df['is_artefact']
        out = df.loc[mask, 'target_name_fixed'].unique()
        unique_targets_count = out.shape[0]

        rng = np.random.default_rng(42)
        indexes = np.arange(unique_targets_count)
        rng.shuffle(indexes)

        if legacy_split:
            train_val_ratio = 0.95
            unique_indexes_train, unique_indexes_val = np.split(indexes,
                                            [int(train_val_ratio*unique_targets_count)])
            unique_indexes_test = unique_indexes_val
        else:
            train_val_test_ratio = [0.9, 0.05, 0.05]
            unique_indexes_train, unique_indexes_val, unique_indexes_test = np.split(indexes,
            [int(train_val_test_ratio[0]*unique_targets_count),
            int(sum(train_val_test_ratio[:2])*unique_targets_count)])

        targets_train = out[unique_indexes_train]
        targets_val = out[unique_indexes_val]
        targets_test = out[unique_indexes_test]

        # get indexes for training/validating dataset
        index_train = df[df.target_name_fixed.isin(targets_train) & ~df.is_artefact].index
        index_val = df[df.target_name_fixed.isin(targets_val) & ~df.is_artefact].index
        index_test = df[df.target_name_fixed.isin(targets_test) & ~df.is_artefact].index


        # remove a fraction of labels
        if labels is not None:
            np.random.seed(self.labels_fraction_seed)
            df_reduction = pd.DataFrame(columns=['label', 'original non-nan', 'non-nan after reduction'])
            for i, label in enumerate(labels.labels):
                if label not in df.columns:
                    continue
                # Get indices of non-NaN values
                non_nan_indices = df[(df[label].notna()) & (df.index.isin(index_train))].index

                # Randomly select a fraction of these indices
                rand_indices = np.random.choice(non_nan_indices,
                                                int(labels2nan_frac * len(non_nan_indices)),
                                                replace=False)

                # Assign NaN to the selected indices
                df.loc[rand_indices, label] = np.nan
                non_nan_assumed_count = len(non_nan_indices) - len(rand_indices)
                assert(df.loc[non_nan_indices, label].notna().sum() == non_nan_assumed_count)

                df_reduction.loc[i] = [label, len(non_nan_indices), non_nan_assumed_count]
            print(df_reduction)

        if remove_all_nan_labels and labels is not None:
            print('Removing all nan labels')
            all_nan_rows = df[labels.labels].isna().all(axis=1)
            # if index_train contains all nan rows, remove them
            index_train = index_train[~all_nan_rows[index_train]]
            # check that index_train does not contain all nan rows
            assert(not df.loc[index_train, labels.labels].isna().all(axis=1).any())

        self.real_dataset = Harps_real_memmap(
                                memmap_filename=self.real_memmap_filename,
                                dataset_name='real',
                                df_catalog=df,
                                median_threshold=50,
                                labels_norm=self.labels,
                                median_norm = self.median_norm
                           )

        # create training/validation/testing dataset from collected indexes
        self.real_train_dataset = torch.utils.data.Subset(self.real_dataset,
                                                          index_train)
        self.real_val_dataset = torch.utils.data.Subset(self.real_dataset,
                                                        index_val)
        self.real_test_dataset = torch.utils.data.Subset(self.real_dataset,
                                                        index_test)
        # train dataset
        # I have to prepare weighted sampler based on uniqueness
        # every unique sample is equally likely ->
        sample_weights_real = 1/df.iloc[index_train].counts_wout_artefact
        sample_weights_real = sample_weights_real.to_numpy()
        sample_weights_real = sample_weights_real/np.sum(
            sample_weights_real) # sum to one

        self.train_sampler = WeightedRandomSampler(
                            sample_weights_real,
                            len(self.real_train_dataset),
                            replacement=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                    dataset=self.real_train_dataset,
                    sampler=self.train_sampler,
                    num_workers=self.num_workers,
                    batch_size=self.batch_size,
                    )


    def val_dataloader(self):
        return [torch.utils.data.DataLoader(
            self.real_val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )]


    def test_dataloader(self):
        return torch.utils.data.DataLoader(
                            self.real_test_dataset,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                        )



def test_training_sampling():
    """
    test that sampling of training sampling is correct (etc_frac)
    """
    labels = Labels(
                    datasets_names=['ETC', 'real'],
                    labels=[
                            'radvel',
                            'H2O_pwv',
                            'Teff',
                            '[M/H]',
                            'logg',
                            'Mass',
                            'airmass',
                            ],
                    labels_type=[
                                 'shared',
                                 'shared',
                                 'separated',
                                 'separated',
                                 'separated',
                                 'separated',
                                 'separated',
                                 ],
                    )
    # try reaching all dataloaders
    datamodule = Spectra_mixed_data_module(
                            radvel_range=[-30., 30],
                            batch_size=32,
                            etc_frac=1.00,
                            labels=labels,
                            )
    # iterate through dataloader and check that H2O_pwv is fine
    train_dataloader = datamodule.train_dataloader()
    H2O_pwv_idx = labels.label2idx('H2O_pwv')['ETC']
    H2O_pwv_nonnan_count = 0
    total_count = 0
    import time
    start = time.time()
    for idx, batch in enumerate(train_dataloader[0]):
        spectra, y = batch
        H2O_pwv_values = y[:,0, H2O_pwv_idx].numpy()
        H2O_pwv_nonnan_count += np.sum(~np.isnan(H2O_pwv_values))
        total_count += len(H2O_pwv_values)
        if idx % 100 == 0:
            frac = H2O_pwv_nonnan_count/total_count
            print(f'total count {total_count}')
            print(f'H2O_pwv nonan count {H2O_pwv_nonnan_count}')
            print(f'fraction of etc to real data is {frac}')
            print(f'it took {time.time() - start} [s] to load 100 batches')
            start = time.time()


def test_datamodule():
    labels = Labels(
                    datasets_names=['ETC', 'real'],
                    labels=[
                            'radvel',
                            'H2O_pwv',
                            'Teff',
                            '[M/H]',
                            'logg',
                            'Mass',
                            'airmass',
                            ],
                    labels_type=[
                                 'shared',
                                 'shared',
                                 'separated',
                                 'separated',
                                 'separated',
                                 'separated',
                                 'separated',
                                 ],
                    )
    # try reaching all dataloaders
    datamodule = Spectra_mix_data_module(
                            radvel_range=[-30., 30],
                            batch_size=32,
                            labels=labels,
                            )
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    # load samples from each dataset
    sample_etc_train = train_dataloader[0].dataset[0]
    sample_real_train = train_dataloader[1].dataset[0]
    sample_etc_val = val_dataloader[0].dataset[0]
    sample_real_val = val_dataloader[1].dataset[0]
    # TODO: test labels mechanism
    batch_etc_train = next(iter(train_dataloader[0]))
    batch_real_train = next(iter(train_dataloader[1]))
    batch_etc_val = next(iter(val_dataloader[0]))
    batch_real_val = next(iter(val_dataloader[1]))
    # TODO: check dimensions
    # TODO: check that some labels are not nan


def test_artifacts():
    labels = Labels(
                    datasets_names=['ETC', 'real'],
                    labels=[
                            'radvel',
                            'H2O_pwv',
                            'Teff',
                            '[M/H]',
                            'logg',
                            'Mass',
                            'airmass',
                            ],
                    labels_type=[
                                 'shared',
                                 'shared',
                                 'separated',
                                 'separated',
                                 'separated',
                                 'separated',
                                 'separated',
                                 ],
                    )
    # try reaching all dataloaders
    datamodule = Spectra_real_data_module(
                            labels=labels,
                            median_norm=False,
                            )
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    # load samples from each dataset
    from tqdm import tqdm

    for spectra, y in tqdm(train_dataloader):
        spectra = torch.reshape(spectra, (spectra.shape[0], -1))
        # check that there are no artefacts
        assert(np.all(torch.median(spectra, 1)[0].numpy() > 50))
        assert(np.all(torch.mean(spectra, 1)[0].numpy() > 0))

    for spectra, y in tqdm(val_dataloader):
        spectra = torch.reshape(spectra, (spectra.shape[0], -1))
        # check that there are no artefacts
        assert(np.all(torch.median(spectra, 1)[0].numpy() > 50))
        assert(np.all(torch.mean(spectra, 1)[0].numpy() > 0))




#test_artifacts()
