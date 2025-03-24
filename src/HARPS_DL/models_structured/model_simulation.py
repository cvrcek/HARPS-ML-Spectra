import time
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from HARPS_DL.models_structured.decoders_builder import decoders_builder
from HARPS_DL.models_structured.model_basic import ae1d
from HARPS_DL.models_structured.Intervention_mix_in import Intervention_mix_in
from HARPS_DL.models_structured.Overview_mix_in import Overview_mix_in

from HARPS_DL.datasets.Dataset_mixin import Dataset_mixin

from pdb import set_trace
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger





class simulator_NN(Overview_mix_in, Intervention_mix_in, ae1d):
    def __init__(self, learning_rate: float=1e-3,
                       bottleneck: int=6,
                       decoder_name: str='CNN_classic',
                       overview_epoch: int=50,
                       overview_samples: list[int]=[0, 50, 100, 200, 300, 400, 500], # epochs and samples for overview
                       ): 
        super().__init__(learning_rate=learning_rate)
        self.save_hyperparameters()
        # architecture choice
        self.bottleneck = bottleneck

        # epochs and samples for overview on validation epoch end
        self.overview_epoch = overview_epoch
        self.overview_samples = overview_samples

        #self.tiled_mask = torch.from_numpy(Dataset_mixin.get_artifact_mask(batch_size = 1))
        self.register_buffer("tiled_mask", torch.from_numpy(Dataset_mixin.get_artifact_mask(batch_size = 1)))

        # profiling
        self.step_last = time.time()

        self.fc = nn.Linear(self.bottleneck, 512*20)

        self.decoder_name = decoder_name

        self.decoder = decoders_builder(self.decoder_name)


    def forward_labels(self, labels):
        return self.forward(labels)

    def forward(self, labels):
        assert(self.decoder is not None) # decoder must be specified
        x = self.fc(labels)
        return  self.decoder(x)

    def loss(self, batch):
        x, y = batch
        recon_x = self.forward(y)
        #not_nan_mask = torch.nonzero(torch.logical_or(torch.isnan(recon_x),torch.isnan(x)))
        #recon_x[not_nan_mask] = 0
        #x[not_nan_mask] = 0
        reconst_loss = self.EPE_my(recon_x, x, mean=True, tiled_mask = self.tiled_mask)
      #  assert(not torch.isnan(reconst_loss))
        return reconst_loss

    def training_step(self, batch):
        loss = self.loss(batch)
        self.log('train_loss', loss.detach())
        return loss

    def validation_step(self, batch , batch_idx):
        loss = self.loss(batch)
        self.log('val_loss', loss.detach(), sync_dist=True)

    def test_step(self, batch , batch_idx):
        loss = self.loss(batch)
        self.log('test_loss', loss.detach(), sync_dist=True)

    def on_validation_epoch_end(self):
        if self.current_epoch  % self.overview_epoch == 0:
            self.high_verbose_inspection()

    def high_verbose_inspection(self):
        wave_resolution = 0.01
        desiredMinW = 3785
        desiredMaxW = 6910
        L = 327680
        padL = 7589
        padR = 7590

        st = wave_resolution
        WAVE = np.arange(desiredMinW-padL*st,desiredMaxW+.001+padR*st,step=st)


        # RECONSTRUCTION OVERVIEW
        idx = 0
        val_dataset =  self.trainer.val_dataloaders.dataset
        #val_dataset =  next(iter(self.trainer.val_dataloaders[0]))
        #for sample_id in np.random.choice(len(val_dataset), size=5, replace=False):
        #for sample_id in [10, 19, 2000, 3000, 4000]:
        #for sample_id in [0, 130, 543, 633, 1007]:
        for sample_id in self.overview_samples:
            data, y = val_dataset[sample_id]
            data = torch.reshape(data, (1,1,data.shape[1])).cuda()
            y = torch.reshape(y, (1,1,y.shape[1])).cuda()
            loss = self.loss((data, y))
            out0 = self.forward(y)

            out0 = out0.cpu().squeeze()
            data = data.cpu().squeeze()
            # Generate chart - overview
            fig = plt.figure(figsize=(7, 9))
            plt.plot(WAVE, data)
            plt.plot(WAVE, out0)
            plt.title(str(sample_id))

            desc_str = f'validate/epoch_{self.current_epoch}_{sample_id}'
            if isinstance(self.logger, NeptuneLogger):
                self.logger.experiment[desc_str].log(fig)
            elif isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(desc_str, fig)
            else:
                raise NotImplementedError

            # Generate chart - detail
            fig1 = plt.figure(figsize=(7, 9))
            wave_det = np.logical_and(WAVE >= 4836, WAVE <= 4845)
            plt.plot(WAVE[wave_det], data[wave_det])
            plt.plot(WAVE[wave_det], out0[wave_det])
            plt.title(str(sample_id))

            desc_str = f'validate/epoch_{self.current_epoch}_{sample_id}_zoomin'
            if isinstance(self.logger, NeptuneLogger):
                self.logger.experiment[desc_str].log(fig)
            elif isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(desc_str, fig)
            else:
                raise NotImplementedError
            plt.close('all')
