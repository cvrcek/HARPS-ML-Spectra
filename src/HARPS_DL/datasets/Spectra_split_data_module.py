from pathlib import Path
from pdb import set_trace
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler

from HARPS_DL.project_config import DATASETS_PATH

import numpy as np
import pytorch_lightning as pl
from lightning.pytorch.utilities.combined_loader import CombinedLoader
import torch.utils.data

import neptune

from HARPS_DL.datasets.Dataset_real_memmap import Harps_real_memmap
from HARPS_DL.datasets.Dataset_etc_memmap import harps_etc_memmap
from HARPS_DL.datasets.Labels import Labels

class Spectra_split_data_module(pl.LightningDataModule):
    def __init__(self,
                radvel_range: tuple[float, float]=(-120., 120.),
                num_workers: int=8,
                batch_size: int=32,
                etc_train_folder: str='ETC_uniform_new_dist/memmap_train',
                etc_val_folder: str='ETC_uniform_new_dist/memmap_val',
                etc_noise_free: bool=True,
                etc_frac: float=1.,
                labels2nan_frac: float=0.0,
                remove_all_nan_labels: bool=False,
                labels_fraction_seed: int=42,
                labels=None,
                normalize_labels: bool=True,
                precompute_rv: str='no',
                median_norm: bool=True,
                legacy_split: bool=False,
                ):
        """ETC data module

        Args:
            radvel_range: symmetricrange of radial velocities in native units [km/s]
                80% of the real data are in the range +-30 km/s which mean
                num_workers: workers per
            batch_size: samples per batch (for DataLoader)
            etc_noise_free: flag to work with signal without additive noise
            etc_frac: sum of weights for ETC dataset during training.
                Real dataset has always sum equal to one. This can
                increase/decrease relative importance of ETC dataset.
            labels: Labels instance that take
            normalize_labels: flag signaling labels normalization
            precompute_rv .. precompute radial velocity:
                            no - sample online from radvel_range,
                            uniform - each sample has a preset rv value from radvel_range)
        """
        super().__init__()
        assert(labels2nan_frac >= 0.0 and labels2nan_frac <= 1.0) # fraction of NaNs in the dataset

        self.batch_size = batch_size
        self.num_workers=num_workers
        self.etc_noise_free = etc_noise_free
        self.etc_frac = etc_frac
        self.labels = labels
        self.normalize_labels = normalize_labels
        self.precompute_rv = precompute_rv
        self.radvel_range = radvel_range
        self.median_norm = median_norm

        data_folder = Path(DATASETS_PATH)
        self.real_memmap_filename = data_folder.joinpath('real_data/harps-nonan-stable.dat')
        csv_real_file = data_folder.joinpath('real_data/harps_artefacts_marked.csv')

        median_threshold = 0

        self.etc_train_dataset_folder = data_folder.joinpath(etc_train_folder)
        self.etc_val_dataset_folder = data_folder.joinpath(etc_val_folder)
        self.etc_gen_dataset_folder = data_folder.joinpath('ETC_generative_dist/memmap_val')

        self.etc_train_csv_file = self.etc_train_dataset_folder.joinpath('etc_labeled.csv')
        self.etc_val_csv_file = self.etc_val_dataset_folder.joinpath('etc_labeled.csv')
        self.etc_gen_csv_file = self.etc_gen_dataset_folder.joinpath('etc_labeled.csv')

        self.csv_real_file = csv_real_file

        self.labels_fraction_seed = labels_fraction_seed

        self.median_threshold = median_threshold

        if self.normalize_labels:
            self.labels.df2normalization(pd.read_csv(self.csv_real_file),
                                         pd.read_csv(self.etc_train_csv_file),
                                         )

        self.etc_train_dataset = harps_etc_memmap(
                               self.etc_train_dataset_folder,
                               dataset_name='ETC',
                               radvel_range=self.radvel_range,
                               median_norm=True,
                               median_threshold=self.median_threshold,
                               noise_free=self.etc_noise_free,
                               labels_norm=self.labels,
                               precompute_rv=self.precompute_rv,
                           )
        self.etc_val_dataset = harps_etc_memmap(
                               self.etc_val_dataset_folder,
                               dataset_name='ETC',
                               radvel_range=self.radvel_range,
                               median_norm=True,
                               median_threshold=self.median_threshold,
                               noise_free=self.etc_noise_free,
                               labels_norm=self.labels,
                               precompute_rv=self.precompute_rv,
                           )

        self.etc_gen_dataset = harps_etc_memmap(
                               self.etc_gen_dataset_folder,
                               dataset_name='ETC',
                               median_norm=True,
                               median_threshold=self.median_threshold,
                               noise_free=True,
                               labels_norm=self.labels,
                               precompute_rv='no',
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
            print('Removing real samples with all nan labels')
            labels_strings = labels.labels

            # remove labels that are not in df.columns
            labels_strings = [label for label in labels_strings if label in df.columns]

            all_nan_rows = df[labels_strings].isna().all(axis=1)
            # if index_train contains all nan rows, remove them
            index_train = index_train[~all_nan_rows[index_train]]
            # check that index_train does not contain all nan rows
            assert(not df.loc[index_train, labels_strings].isna().all(axis=1).any())

        self.real_dataset = Harps_real_memmap(
                                memmap_filename=self.real_memmap_filename,
                                dataset_name='real',
                                df_catalog=df,
                                median_threshold=50,
                                labels_norm=self.labels,
                           )

        # create training/validation/testing dataset from collected indexes
        self.real_train_dataset = torch.utils.data.Subset(self.real_dataset,
                                                          index_train)
        self.real_val_dataset = torch.utils.data.Subset(self.real_dataset,
                                                        index_val)
        self.real_test_dataset = torch.utils.data.Subset(self.real_dataset,
                                                        index_test)

        # I have to prepare weighted sampler based on uniqueness
        # every unique sample is equally likely ->
        sample_weights_real = 1/df.iloc[index_train].counts_wout_artefact
        sample_weights_real = sample_weights_real.to_numpy()
        sample_weights_real = sample_weights_real/np.sum(
            sample_weights_real) # sum to one

        self.train_real_sampler = WeightedRandomSampler(
                            sample_weights_real,
                            len(self.etc_train_dataset),
                            replacement=True)



    def train_dataloader(self):
        dataloader_etc = torch.utils.data.DataLoader(
                    dataset=self.etc_train_dataset,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    prefetch_factor=1,
                    batch_size=self.batch_size,
                    )

        dataloader_real = torch.utils.data.DataLoader(
                    dataset=self.real_train_dataset,
                    sampler=self.train_real_sampler,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    prefetch_factor=1,
                    batch_size=self.batch_size,
                    )

        return CombinedLoader(
            {"ETC": dataloader_etc, "real": dataloader_real},
            "sequential",
        )

    def val_dataloader(self):
        dataloader_etc = torch.utils.data.DataLoader(
                            self.etc_val_dataset,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            shuffle=False,
                        )
        dataloader_real = torch.utils.data.DataLoader(
                            self.real_val_dataset,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                        )

        return [dataloader_etc, dataloader_real]

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
                            self.real_test_dataset,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                        )




    def real_unique_dataloader(self):
        df = pd.read_csv(self.csv_real_file)
        df_filtered = df[df['is_artefact'] == False]
        # Group by 'column_name' and randomly select one index per group
        random_indices = df_filtered.groupby('target_name_fixed').apply(lambda x: x.sample(n=1).index[0]).tolist()

        # Test 1: Check that all 'target_name_fixed' values for the selected indices are unique
        assert df.loc[random_indices, 'target_name_fixed'].nunique() == len(random_indices), "Not all 'target_name_fixed' values for the selected indices are unique"
    
        # Test 2: Check that all indices in random_indices correspond to 'is_artefact' being False
        assert all(df.loc[random_indices, 'is_artefact'] == False), "Some indices correspond to 'is_artefact' being True"

        real_unique_train_dataset = torch.utils.data.Subset(self.real_dataset,
                                                          random_indices)
        return torch.utils.data.DataLoader(
                            real_unique_train_dataset,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                        )

    def real_unique_and_complete_dataloader(self):
        df = pd.read_csv(self.csv_real_file)
        df_filtered = df[df['is_artefact'] == False]
        # Group by 'column_name' and randomly select one index per group
        random_indices = df_filtered.groupby('target_name_fixed').apply(lambda x: x.sample(n=1).index[0]).tolist()

        # mark indices that have any nan labels
        labels_strings = [label for label in self.labels.labels if label in df.columns]
        any_nan_rows = df[labels_strings].isna().any(axis=1)

        # filter out indices that have any nan labels
        random_indices = [idx for idx in random_indices if not any_nan_rows[idx]]

        # create a dataset where non of the labels are missing
        real_unique_complete_dataset = torch.utils.data.Subset(self.real_dataset,
                                                          random_indices)
        return torch.utils.data.DataLoader(
                            real_unique_complete_dataset,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                        )


    def log_overview(self, logger):
        def log_dataset(dataset, descriptor, logger, label):
            data = dataset.df[label].to_numpy()
            fig = plt.figure()
            plt.hist(data)
            plt.title(label)
            logger.experiment[f'{descriptor}'].log(
                neptune.types.File.as_image(fig))

        shared_labels = ['Teff', 'logg', '[M/H]', 'airmass']
        etc_unique_labels = ['mag', 'Texp', 'H2O_pwv']
        real_unique_labels = []
        for label in self.labels.labels:
            if label in shared_labels:
                log_dataset(self.etc_train_dataset, 'etc_train_data_overview', logger, label)
                log_dataset(self.etc_val_dataset, 'etc_val_data_overview', logger, label)
                log_dataset(self.real_dataset, 'real_data_overview', logger, label)
            if label in etc_unique_labels:
                log_dataset(self.etc_train_dataset, 'etc_train_data_overview', logger, label)
                log_dataset(self.etc_val_dataset, 'etc_val_data_overview', logger, label)
            if label in real_unique_labels:
                log_dataset(self.real_dataset, 'real_data_overview', logger, label)

        dic = {}
        dic['labels'] = self.labels.labels
        dic['medians'] = [self.labels.get_label_median(label)
                          for label in self.labels.labels]
        dic['mads'] = [self.labels.get_label_mad(label)
                       for label in self.labels.labels]
        df = pd.DataFrame(dic)
        logger.experiment['datasets_normalization'].upload(neptune.types.File.as_html(df))

#test_training_sampling()
