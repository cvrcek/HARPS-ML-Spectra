import numpy as np
import torch
from torch import nn
import os
import sys
import neptune as neptune
from astropy import units as u

from HARPS_DL.models_structured.model_basic import ae1d
from HARPS_DL.models_structured.decoders_builder import decoders_builder
from HARPS_DL.models_structured.encoders_builder import encoders_builder

from HARPS_DL.datasets.Dataset_mixin import Dataset_mixin

from HARPS_DL.tools.MI_support import get_MIs_by_nearest_neighbor
from HARPS_DL.tools.IRS_support import irs_dic

from pdb import set_trace

from HARPS_DL.models_structured.legacy_mix_in import Legacy_mix_in
from HARPS_DL.models_structured.Overview_mix_in import Overview_mix_in
from HARPS_DL.models_structured.Intervention_mix_in import Intervention_mix_in

from typing import List

class composite_VAE(Legacy_mix_in, Overview_mix_in, Intervention_mix_in, ae1d):
    def __init__(self,
                    learning_rate: float=1e-3,
                    input_size: int=327680, # = output size
                    lambd_labels: float=1,
                    anneal_epoch_start: int=0,
                    anneal_epoch_stop: int=0,
                    lambd_KL_list: List[dict]=[{'interval': (0,0), 'weight': 0}],
                    encoder_name='CNN_classic',
                    decoder_name='CNN_classic',
                    bottleneck_in: int=10240,
                    bottleneck: int=4,
                    bottleneck_out: int=10240,
                    overview_epoch: int=50,
                    overview_samples: list[int]=[0, 1, 2, 3, 4], # epochs and samples for overview
                    labels=None,
                    unfreeze_epoch: int=0,
                    ):
        """
        add documentation!

        lambd_KL ..  list of dictionaries lambd_KL[idx] = {'interval': tuple, 'weight': float},
                     where interval is the bottleneck interval and weight is the respective KL weight.
                     Class checks that intervals are not overlapping and produces an array lambd_KL
                     
        unfreeze_epoch .. unfreeze on this epoch (before training starts).

        """
        super().__init__(learning_rate=learning_rate)
        # architecture choice
        self.input_size = input_size
        self.lambd_labels = lambd_labels


        self.anneal_epoch_start = anneal_epoch_start 
        self.anneal_epoch_stop = anneal_epoch_stop 

        self.bottleneck = bottleneck
        self.bottleneck_in = bottleneck_in
        self.bottleneck_out = bottleneck_out

        self.lambd_KL = self.KL_list2vec(lambd_KL_list)

        # encoder/decoder are freezed until this epoch
        self.unfreeze_epoch = unfreeze_epoch

        #self.tiled_mask = torch.from_numpy(Dataset_mixin.get_artifact_mask(batch_size = 1))
        self.register_buffer("tiled_mask", torch.from_numpy(Dataset_mixin.get_artifact_mask(batch_size = 1)))

        self.fc_mu = nn.Linear(bottleneck_in, bottleneck)
        self.fc_logvar = nn.Linear(bottleneck_in, bottleneck)
        self.fc_z = nn.Linear(bottleneck, bottleneck_out)

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name

        self.decoder = decoders_builder(self.decoder_name)
        self.decoder.freeze()

        self.encoder = encoders_builder(self.encoder_name)
        self.encoder.freeze()

        # epochs and samples for overview on validation epoch end
        self.overview_epoch = overview_epoch
        self.overview_samples = overview_samples

        self.labels = labels


    def _check_architecture(self):
        # decoder/encoder can't be None
        if self.decoder is None:
            raise('class requires decoder')
        if self.encoder is None:
            raise('class requires encoder')
        # check dimensions for encoder/bottleneck
        x_mock = torch.zeros((1, 1, self.input_size))
        x_encoded = self.encoder(x_mock)
        if x_encoded.shape[-1] != self.bottleneck_in:
            raise(f'''Error encoder output dimension is {x_encoded.shape[-1]},
                  but expected bottleneck input dimension is {self.bottleneck_in}.
                  Check encoder architecture or change bottleneck_in.
                  ''')

        # check dimensions for bottleneck/decoder
        x_mock = torch.zeros((1, 1, self.bottleneck_out))
        x_decoded = self.decoder(x_mock) # failure here could mean bottlneck_out is different from decoder assumed input size
        if x_decoded.shape[-1] != self.input_size:
            raise(f'''Error decoder output dimension is {x_decoded.shape[-1]},
                  but expected input/output size is {self.input_size}.
                  Decoder architecture might be wrong.
                  ''')

    def KL_list2vec(self, KL_list):
        lambd_KL = np.zeros((self.bottleneck,))
        overlap_check = np.zeros((self.bottleneck,))
        for i in range(len(KL_list)):
            interval = KL_list[i]['interval']
            assert(len(interval) == 2) # interval is defined as a tuple (or list) of two numbers
            assert(interval[0] <= interval[1]) # interval must be non-decreasing
            interval_range = range(interval[0], interval[1] + 1) # interval is inclusive

            weight = KL_list[i]['weight']
            assert(weight >= 0) # weight is non-negative

            lambd_KL[interval_range] = weight
            overlap_check[interval_range] += 1

        assert(np.all(overlap_check < 2)) # intervals shouldn't overlap

        return torch.Tensor(lambd_KL)
    
    def on_train_start(self):
        self.lambd_KL = self.lambd_KL.to(self.device)


    def forward(self, x):
        x_encoded = self.encoder(x)

        # bottleneck distribution
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        # clip logvar to avoid inf or nan loss
        logvar = torch.clamp(logvar, -20, 20)

        # reparametrization trick
        z = torch.randn_like(mu) # same size as mu, samples from N(0, 1)
        z = mu + z*torch.exp(0.5*logvar) # samples from N(mu, exp(0.5*logvar))

        # decoder ready input
        x_decoded = self.fc_z(z)

        return  self.decoder(x_decoded), z, mu, logvar

    def forward_labels(self, mu):
        x_decoded = self.fc_z(mu)
        return  self.decoder(x_decoded)



    def labels_loss(self, z, y_gt):
        """ labels loss
            return  loss for each label

        """
        if len(y_gt.shape) == 2:
            yh = y_gt.clone()
            z_supervised = z[:,0:yh.shape[1]].clone()
        else:
            yh = y_gt[:,0,:].clone()
            z_supervised = z[:,0:yh.shape[1]].clone()

        labels_loss = torch.abs(yh-z_supervised) # per label loss

        return labels_loss


    def loss(self, batch, objective=False):
        """loss function

        Args:
            self: self reference
            batch: spectrum, labels
            objective: if true, ignore weights (objective value for validation/tuning)
        """
        x, y = batch
        recon_x, sample, mu, logvar = self.forward(x)
        #print('forward duration: ' + str(time.time() - self.loss_start))

        # KL divergence (Kingma and Welling, https://arxiv.org/abs/1312.6114, Appendix B)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        wkl = self.get_KL_weight().to(self.device)
        KLD = -0.5 * torch.sum(wkl*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=0))
#        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #<- bug repaired

        reconst_loss = self.EPE_my(recon_x, x, mean=True, tiled_mask = self.tiled_mask)

#        self.log('global_step', self.global_step)

        labels_loss = self.labels_loss(sample, y)
        total_labels_loss = torch.nanmean(labels_loss)
        labels_loss = torch.nanmean(labels_loss, axis=0)

        # logvar_numpy = logvar.detach().cpu().numpy()
        # df = pd.DataFrame(logvar_numpy)
        # self.logger.experiment['logvar/' + str(self.global_step)].upload(neptune.types.File.as_html(df))

        if objective:
            # this good for validation/hyperparameter tuning
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
            wkl = torch.Tensor(0) # doesn't really matter
            return reconst_loss + total_labels_loss,\
                   reconst_loss, KLD, wkl, total_labels_loss, labels_loss
        else:
            return reconst_loss + KLD + self.lambd_labels*total_labels_loss,\
                   reconst_loss, KLD, wkl, total_labels_loss, labels_loss

    def get_KL_weight(self):
        if self.current_epoch < self.anneal_epoch_start: # no KL penalty
            weight = 0.
        elif self.current_epoch >= self.anneal_epoch_stop: # full KL penalty
            weight = 1.
        else: # annealing
            weight = (self.current_epoch - self.anneal_epoch_start)\
                    /(self.anneal_epoch_stop - self.anneal_epoch_start)

        return weight*self.lambd_KL



    def on_train_epoch_start(self):
        if self.current_epoch == self.unfreeze_epoch:
            self.decoder.unfreeze()
            self.encoder.unfreeze()

    def training_step(self, batch, batch_idx):
        #self.memory_print()
        loss, loss_rec, kld, _, loss_label, _ = self.loss(batch)
        self.log('train_loss', loss.detach())
        self.log('train_loss_rec', loss_rec.detach())
        self.log('train_loss_kld', kld.detach())
        self.log('train_loss_label', loss_label.detach())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        dataset = self.trainer.val_dataloaders[dataloader_idx].dataset
        dataset_name = self.get_dataset_name(dataset)
        loss, loss_rec, kld, _, loss_label, loss_per_label  = self.loss(
            batch, objective=True
            )
        
        self.log(dataset_name + '_val_loss', loss.detach(), sync_dist=True)
        self.log(dataset_name + '_val_loss_rec', loss_rec.detach(), sync_dist=True)
        self.log(dataset_name + '_val_loss_kld', kld.detach(), sync_dist=True)
        self.log(dataset_name + '_val_loss_label', loss_label.detach(), sync_dist=True)

        loss_per_label = loss_per_label.detach()
        for idx, label_str in enumerate(self.labels.get_vec_names()):
            # log only if label isn't nan
            if not torch.isnan(loss_per_label[idx]):
                self.log(dataset_name + '_val_loss_' + label_str, loss_per_label[idx])



    # def test_step(self, batch , batch_idx):
        # loss, _, _, _, _ = self.loss(batch)
        # self.log('test_loss', loss.detach(), sync_dist=True)

    def on_validation_epoch_end(self):
        if (self.current_epoch) != 0 and (self.current_epoch  % self.overview_epoch == 0):
            # ETC dataset
            for i in range(len(self.trainer.val_dataloaders)):
                dataset = self.trainer.val_dataloaders[i].dataset
                dataset_name = self.get_dataset_name(dataset)

                self.high_verbose_inspection(dataset, dataset_name)

                self.labels_inspection(dataset, dataset_name)

                if dataset_name == 'ETC':
                    spectra_idxs = [1, 10, 44, 60, 100, 200, 301, 352, 405, 406, 440, 445, 600]
                else:
                    spectra_idxs = [1, 10, 44, 60, 100, 200, 301, 352, 405, 406, 440, 445, 600]
                err = self.rv_intervention_error(
                            dataset,
                            spectra_idxs,
                            self.labels,
                            wave_region=[6050, 6250],
                            rv_shifts=np.linspace(-40, 40, 40)*u.kilometer/u.second,
                        )
                self.log(dataset_name + '/RVIS', np.mean(err))

                if dataset_name == 'ETC':
                    spectra_idxs = [1, 10, 44, 60, 100]
                else:
                    spectra_idxs = [1, 10, 44, 60, 100]

                self.radvel_overview(dataset, dataset_name, spectra_idxs, self.labels, logging=True)

    def memory_print(self):
        import gc
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
            except:
                pass

    def get_dataset_name(self, dataset):
        if isinstance(dataset, torch.utils.data.dataset.Subset):
            dataset_name = dataset.dataset.dataset_name
        else:
            dataset_name = dataset.dataset_name
        return dataset_name
    
    @staticmethod
    def load_reference_model(checkpoint_path):
        model = composite_VAE()
        return model



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
