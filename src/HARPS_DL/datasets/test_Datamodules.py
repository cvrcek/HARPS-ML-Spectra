import numpy as np
from HARPS_DL.datasets.Labels import Labels
from HARPS_DL.datasets.Spectra_split_data_module import Spectra_split_data_module
import pytest

@pytest.fixture
def bottleneck_fixture():
    return 32

@pytest.fixture
def labels_fixture(bottleneck_fixture):
    labels = Labels(
        datasets_names=['ETC', 'real'],
        labels=["radvel", "BERV", "Teff", "[M/H]", "logg", "airmass", "H2O_pwv"],
        labels_type=["shared", "shared", "shared", "shared", "shared", "shared", "ETC"],
        bottleneck=bottleneck_fixture,
        fill_bottleneck=['ETC'],
    )
    labels.json2normalization()
    return labels

@pytest.fixture
def datamodule_fixture(labels_fixture):
    datamodule = Spectra_split_data_module(
        radvel_range=[-30., 30],
        batch_size=32,        
        labels=labels_fixture,
    )
    return datamodule


class Test_split_data_module:
    # Class for testing the main module
    # note: test_training_sampling should be modified to
    #       - assert properties (it isn't test right now)
    #       - check that the distribution of labels is correct
    #       wrt to the flags (maybe split in several tests)
    @pytest.mark.slow
    def test_training_sampling(self, datamodule_fixture, labels_fixture):        
        """
        test that sampling of training sampling is correct (etc_frac)
        """  
        # TODO: test distribution of labels, based on various flags
        # try reaching all dataloaders
        
        # iterate through dataloader and check that H2O_pwv is fine
        train_dataloader = datamodule_fixture.train_dataloader()
        H2O_pwv_idx = labels_fixture.label2idx('H2O_pwv')['ETC']
        H2O_pwv_nonnan_count = 0
        total_count = 0
        import time
        start = time.time()
        for idx, batch in enumerate(train_dataloader.iterables['ETC']):
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


    def test_datamodule(self, datamodule_fixture, bottleneck_fixture):                 
        def check_sample(dataset):
            # get a sample and check that it has the right shape
            sample = dataset[0]
            assert sample[0].shape == (1, 327680)
            assert sample[1].shape == (1, bottleneck_fixture)  
            
        def check_batch(batch):
            # get a batch and check that it has the right shape
            assert batch[0].shape == (32, 1, 327680)
            assert batch[1].shape == (32, 1, bottleneck_fixture)     
        # try reaching all dataloaders        
        train_dataloader = datamodule_fixture.train_dataloader()
        val_dataloader = datamodule_fixture.val_dataloader()
        
        # load and check sample from each dataset        
        check_sample(train_dataloader.iterables['ETC'].dataset)
        check_sample(train_dataloader.iterables['real'].dataset)
        check_sample(val_dataloader[0].dataset)
        check_sample(val_dataloader[1].dataset)
        
        # load and check batch from each dataset        
        check_batch(next(iter(train_dataloader.iterables['ETC'])))
        check_batch(next(iter(train_dataloader.iterables['real'])))
        check_batch(next(iter(val_dataloader[0])))
        check_batch(next(iter(val_dataloader[1])))
        