from pathlib import Path
from pdb import set_trace
import pandas as pd
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import ConcatDataset


from HARPS_DL.project_config import DATASETS_PATH

import numpy as np
import pytorch_lightning as pl
import torch.utils.data

from HARPS_DL.datasets.Dataset_real_memmap import Harps_real_memmap
from HARPS_DL.datasets.Dataset_etc_memmap import harps_etc_memmap
from HARPS_DL.datasets.Labels import Labels
from HARPS_DL.datasets.Datamodule_overview import Datamodule_overview



class Spectra_mixed_data_module(Datamodule_overview, pl.LightningDataModule):
    def __init__(self,
                radvel_range: tuple[float, float]=(-120., 120.),
                num_workers: int=8,
                batch_size: int=32,
                etc_train_folder: str='ETC_uniform_new_dist/memmap_train',
                etc_val_folder: str='ETC_crossed_dist/memmap_val',
                etc_noise_free: bool=True,
                etc_frac: float=1.,
                labels2nan_frac: float=0.0,
                labels=None,
                normalize_labels: bool=True,
                precompute_rv: str='no',
                median_norm: bool=True,
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

        self.real_dataset = Harps_real_memmap(
                                memmap_filename=self.real_memmap_filename,
                                dataset_name='real',
                                csv_file=self.csv_real_file,
                                median_threshold=50,
                                labels_norm=self.labels,
                           )
        # (look for old approach for both)
        # I have to split the real dataset into training/validation sets
        # Make sure unique targets aren't spilling between sets!!!
        train_val_ratio = 0.95
        df = pd.read_csv(self.csv_real_file)

        mask = ~df['is_artefact']
        out = df.loc[mask, 'target_name_fixed'].unique()
        unique_targets_count = out.shape[0]

        rng = np.random.default_rng(42)
        indexes = np.arange(unique_targets_count)
        rng.shuffle(indexes)

        unique_indexes_train, unique_indexes_val = np.split(indexes,
                                            [int(train_val_ratio*unique_targets_count)])

        targets_train = out[unique_indexes_train]
        targets_val = out[unique_indexes_val]

        # get indexes for training/validating dataset
        index_train = df[df.target_name_fixed.isin(targets_train) & ~df.is_artefact].index
        index_val = df[df.target_name_fixed.isin(targets_val) & ~df.is_artefact].index

        # remove a fraction of labels
        if labels is not None:
            for label in labels.labels:
                if label not in df.columns:
                    continue
                # Get indices of non-NaN values
                non_nan_indices = df[df[label].notna()].index
                print(f'non_nan_indices for {label}: {len(non_nan_indices)} (before reduction)')

                # Randomly select a fraction of these indices
                rand_indices = np.random.choice(non_nan_indices,
                                                int(labels2nan_frac * len(non_nan_indices)),
                                                replace=False)

                # Assign NaN to the selected indices
                df.loc[rand_indices, label] = np.nan
                non_nan_indices = df[df[label].notna()].index
                print(f'non_nan_indices for {label}: {len(non_nan_indices)} (after reduction)')

        # split dataset
        self.real_train_dataset = torch.utils.data.Subset(self.real_dataset,
                                                          index_train)
        self.real_val_dataset = torch.utils.data.Subset(self.real_dataset,
                                                        index_val)

        # train dataset
        self.train_dataset = ConcatDataset([self.real_train_dataset,
                                            self.etc_train_dataset])
        # I have to prepare weighted sampler based on uniqueness
        # every unique sample is equally likely ->
        sample_weights_real = 1/df.iloc[index_train].counts_wout_artefact
        sample_weights_real = sample_weights_real.to_numpy()
        sample_weights_real = sample_weights_real/np.sum(
            sample_weights_real) # sum to one

        sample_weights_etc = self.etc_frac/np.ones(
            (len(self.etc_train_dataset),))/len(self.etc_train_dataset)

        sample_weights = np.append(sample_weights_real,
                                   sample_weights_etc)

        self.train_sampler = WeightedRandomSampler(
                            sample_weights,
                            len(self.etc_train_dataset),
                            replacement=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                    dataset=self.train_dataset,
                    sampler=self.train_sampler,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    prefetch_factor=2,
                    batch_size=self.batch_size,
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

        dataloader = [dataloader_etc, dataloader_real]

        return dataloader

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

#test_training_sampling()
