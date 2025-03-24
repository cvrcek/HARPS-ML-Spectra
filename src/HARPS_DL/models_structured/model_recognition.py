import pandas as pd
import numpy as np
import torch
from torch import nn
import os
import sys
import neptune as neptune
import matplotlib.pyplot as plt

from HARPS_DL.models_structured.model_basic import ae1d
from HARPS_DL.models_structured.encoders_builder import encoders_builder
from HARPS_DL.models_structured.Overview_encoder_mix_in import Overview_encoder_mix_in

from HARPS_DL.datasets.Dataset_mixin import Dataset_mixin

from pdb import set_trace

class recognition_NN(Overview_encoder_mix_in, ae1d):
    def __init__(self,
                    learning_rate: float=1e-3,
                    input_size: int=327680, # = output size
                    lab_loss_type: str='MAE',
                    lab_delta: float=1.0,
                    encoder_name='CNN_classic',
                    bottleneck_in: int=10240,
                    bottleneck: int=6,
                    overview_epoch: int=50,
                    overview_samples: list[int]=[0, 1, 2, 3, 4], # epochs and samples for overview
                    labels=None,
                    ):
        """
        add documentation!

        """
        super().__init__(learning_rate=learning_rate)
        # architecture choice
        self.input_size = input_size

        self.lab_loss_type = lab_loss_type
        self.lab_delta = lab_delta

        self.bottleneck = bottleneck
        self.bottleneck_in = bottleneck_in

        #self.tiled_mask = torch.from_numpy(Dataset_mixin.get_artifact_mask(batch_size = 1))
        self.register_buffer("tiled_mask", torch.from_numpy(Dataset_mixin.get_artifact_mask(batch_size = 1)))

        self.fc = nn.Linear(bottleneck_in, bottleneck)

        self.encoder_name = encoder_name
        self.encoder = encoders_builder(self.encoder_name)

        # epochs and samples for overview on validation epoch end
        self.overview_epoch = overview_epoch
        self.overview_samples = overview_samples

        self.labels = labels

        self._check_architecture()

        self.validation_step_outputs = [] 


    def _check_architecture(self):
        # decoder/encoder can't be None
        if self.encoder is None:
            raise('class requires encoder')
        # check dimensions for encoder/bottleneck
        x_mock = torch.zeros((2, 1, self.input_size))
        x_encoded = self.encoder(x_mock)
        if x_encoded.shape != torch.Size([2, self.bottleneck_in]):
            raise Exception(f'''Error encoder output dimension is {x_encoded.shape},
                  but expected bottleneck input dimension is {self.bottleneck_in}.
                  Check encoder architecture or change bottleneck_in.
                  ''')

    def forward(self, x):
        x_encoded = self.encoder(x)
        z = self.fc(x_encoded)

        return  z

    def loss(self, batch):
        """loss function

        Args:
            self: self reference
            batch: spectrum, labels
            objective: if true, ignore weights (objective value for validation/tuning)
        """
        x, y_gt = batch
        y_pred = self.forward(x)
        labels_loss, mask = self.labels_loss(y_pred,
                                            y_gt,
                                            loss_type=self.lab_loss_type,
                                            delta=self.lab_delta
                                            )

        return torch.mean(labels_loss), torch.mean(labels_loss, axis=0), mask
        
    def training_step(self, batch, batch_idx):
        #self.memory_print()
        loss_label, _, mask = self.loss(batch)
        active_values = (~mask).sum()
        num_values = mask.numel()

        if active_values != 0:
            # modify the mean value based on number of missing values (that are set to 0)
            # missing values should be ok for the training
            self.log('train_loss_label', loss_label.detach()*num_values/active_values)
        
        return loss_label
        #return None if torch.isnan(loss_label) else loss_label
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        dataset = self.trainer.val_dataloaders[dataloader_idx].dataset

        if isinstance(dataset, torch.utils.data.dataset.Subset):
            dataset_name = dataset.dataset.dataset_name
        else:
            dataset_name = dataset.dataset_name

        x, y_gt = batch
        y_pred = self.forward(x)
        labels_loss, mask = self.labels_loss(y_pred,
                                            y_gt,
                                            loss_type=self.lab_loss_type,
                                            delta=self.lab_delta
                                            )

        total_labels_loss = torch.mean(labels_loss)
        per_label_loss = torch.mean(labels_loss, axis=0)

        active_values = (~mask).sum()
        num_values = mask.numel()
        
        self.log(dataset_name + '/val_loss_label',
                 total_labels_loss.detach()*num_values/active_values,
                 sync_dist=True)

        #labels_loss = labels_loss.detach()
        active_valus_per_columns = (~mask).sum(axis=0)
        num_valus_per_columns = mask.size(dim=0)
        for idx, label_str in enumerate(self.labels.get_vec_names()):
            if active_valus_per_columns[idx] != 0:
                self.log(dataset_name + '/val_loss_' + label_str,
                         per_label_loss[idx]*num_valus_per_columns/active_valus_per_columns[idx],
                         )

        labels_loss_MAE, _ = self.labels_loss(y_pred,
                                            y_gt,
                                            loss_type='MAE',
                                            )
        total_labels_loss = torch.mean(labels_loss_MAE)

        self.log(dataset_name + '/val_loss_label_MAE',
                 total_labels_loss.detach()*num_values/active_values,
                 sync_dist=True)

        median_abs_diff = torch.median(labels_loss_MAE[~mask])
        self.log(dataset_name + '/val_loss_label_median',
                 median_abs_diff,
                 sync_dist=True)

import unittest
class test_loss(unittest.TestCase):
    def test_labels_loss(self):
        raise Exception('update to labels class first')
        print('testing label loss function')
        watched_labels = ['a', 'b']
        labels_gt = [[[0, 1]], [[2, 3]], [[np.nan, 4]]]
        z = [[1, 1, 10, 20], [2, 2.5, 20, 500], [-1, 4, 30, 400]]
        
        labels_gt = torch.Tensor(labels_gt)
        z = torch.Tensor(z)

        model = composite_VAE(watched_labels=watched_labels)

        loss = model.labels_loss(z, labels_gt)
        loss = loss.detach().numpy()
        assert(loss.shape == (3, len(watched_labels)))
        assert(np.array_equal(loss, np.array([[1.0, 0.], [0., 0.5],[np.nan, 0.]]), equal_nan=True))

if __name__ == '__main__':
        unittest.main()
