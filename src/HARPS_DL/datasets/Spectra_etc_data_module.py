import sys
import numpy as np
import torch.utils.data
import pandas as pd

from HARPS_DL.datasets.Spectra_split_data_module import Spectra_split_data_module

class Spectra_data_module(Spectra_split_data_module):
    def __init__(self,
                radvel_range: tuple[float, float]=(-120., 120.),
                num_workers: int=8,
                batch_size: int=32,
                train_folder: str='ETC_uniform_new_dist/memmap_train',
                val_folder: str='ETC_crossed_dist/memmap_val',
                noise_free: bool=True,
                labels=None, # see labels class
                normalize_labels: bool=True,
                precompute_rv: str='no',
                ):
        """ETC data module

        See Spectra_mixed_data_module for parameters documentation.
        This class merely selects ETC datasets

        """
        super().__init__(
		radvel_range = radvel_range,
		num_workers = num_workers,
		batch_size=batch_size,
        etc_train_folder=train_folder ,
        etc_val_folder=val_folder,
		etc_noise_free=noise_free,
		etc_frac=0, # isn't used anyway
		labels=labels,
		normalize_labels=normalize_labels,
		precompute_rv=precompute_rv,
		)

        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.etc_train_dataset,
                                           num_workers=self.num_workers,
                                           batch_size=self.batch_size,
                                           shuffle=True)

    def val_dataloader(self):
        return [torch.utils.data.DataLoader(self.etc_val_dataset,
                                           num_workers=self.num_workers,
                                           batch_size=self.batch_size,
                                           shuffle=False)]



def test():
    # test three versions of the runtime function
    # runtime itself was well test and is considered ground truth
    # runtime_fast, runtime_fast_2 attempt to limit unecessary computation
    sys.path.append(str(project_folder.joinpath('generate_simulations/ETC_processing/')))
    from etc_tools import runtime, runtime_fast, runtime_fast_2
    from Labels import Labels
    import time

    datamodule = Spectra_data_module(noise_free=True, labels=Labels())

    dataloader = datamodule.train_dataloader()
    dataloader.dataset.fun = runtime
    idx = np.random.randint(100)

    trials = 20

    start = time.time() 
    it = iter(dataloader)
    for i in range(trials):
        next(it)
    old_fun = time.time() - start
    spectrum_old = dataloader.dataset.getitem_w_custom_radvel(idx, radvel=0)

    dataloader.dataset.fun = runtime_fast
    start = time.time() 
    it = iter(dataloader)
    for i in range(trials):
        next(it)
    fast_fun = time.time() - start
    spectrum_fast = dataloader.dataset.getitem_w_custom_radvel(idx, radvel=0)

    dataloader.dataset.fun = runtime_fast
    start = time.time() 
    it = iter(dataloader)
    for i in range(trials):
        next(it)
    fast_2_fun = time.time() - start

    spectrum_fast_2 = dataloader.dataset.getitem_w_custom_radvel(idx, radvel=0)

    print(f'old fun took {old_fun/trials} [s] per batch')
    print(f'fast fun took {fast_fun/trials} [s] per batch')
    print(f'fast fun 2 took {fast_2_fun/trials} [s] per batch')

    #print(torch.sum(torch.abs(spectrum_old[0] - spectrum_fast[0])))
    #print(torch.sum(torch.abs(spectrum_fast[0] - spectrum_fast_2[0])))
    assert(torch.all(torch.eq(spectrum_old[0], spectrum_fast[0])))
    assert(torch.all(torch.eq(spectrum_fast[0], spectrum_fast_2[0])))

