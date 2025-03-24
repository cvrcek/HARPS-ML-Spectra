import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from typing import List

from HARPS_DL.datasets.Dataset_mixin import Dataset_mixin

from HARPS_DL.models_structured.model_basic import ae1d
from HARPS_DL.models_structured.decoders_builder import decoders_builder
from HARPS_DL.models_structured.encoders_builder import encoders_builder

from HARPS_DL.models_structured.legacy_mix_in import Legacy_mix_in
from HARPS_DL.models_structured.Overview_mix_in import Overview_mix_in
from HARPS_DL.models_structured.Intervention_mix_in import Intervention_mix_in


class info_VAE(Legacy_mix_in, Overview_mix_in, Intervention_mix_in, ae1d):
    def __init__(self,
                    learning_rate: float=1e-3,
                    input_size: int=327680, # = output size
                    lambd_labels: float=1,
                    rec_loss_type: str='MAE',
                    rec_delta: float=1.0,
                    lab_loss_type: str='MAE',
                    lab_delta: float=1.0,
                    anneal_epoch_start: int=0,
                    anneal_epoch_stop: int=0,
                    lambd_KL_list: List[dict]=[{'interval': (0,0), 'weight': 0}],
                    alpha_KL_list: List[dict]=[{'interval': (0,0), 'weight': 0}],
                    encoder_name='CNN_classic',
                    decoder_name='ResNet_small',
                    bottleneck_in: int=10240,
                    bottleneck: int=32,
                    bottleneck_out: int=10240,
                    ETC_AE: bool=False,
                    overview_epoch: int=50,
                    overview_samples: List[int]=[0, 1, 2, 3, 4], # epochs and samples for overview
                    labels=None,
                    unfreeze_epoch: int=0,
                    ):
        """
        add documentation!

        lambd_KL ..  list of dictionaries lambd_KL[idx] = {'interval': tuple, 'weight': float},
                     where interval is the bottleneck interval and weight is the respective KL weight.
                     Class checks that intervals are not overlapping and produces an array lambd_KL
        ETC_AE .. turn off KL or MMD just for ETC dataset
        unfreeze_epoch .. unfreeze on this epoch (before training starts).

        """
        super().__init__(learning_rate=learning_rate)
        # architecture choice
        self.input_size = input_size
        self.lambd_labels = lambd_labels

        self.rec_loss_type = rec_loss_type
        self.rec_delta = rec_delta
        self.lab_loss_type = lab_loss_type
        self.lab_delta = lab_delta

        self.anneal_epoch_start = anneal_epoch_start 
        self.anneal_epoch_stop = anneal_epoch_stop 

        self.bottleneck = bottleneck
        self.bottleneck_in = bottleneck_in
        self.bottleneck_out = bottleneck_out

        self.ETC_AE = ETC_AE

        self.alpha_KL = self.KL_list2vec(alpha_KL_list)
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

    def self_AE_classification(self):
        """ decide if I am AE, VAE, infoVAE, or mistake """
        if torch.any(self.alpha_KL + self.lambd_KL < 1):
            raise Exception('MMD term cannot be negative')
        if torch.any(self.alpha_KL + self.lambd_KL > 1):
            return 'infoVAE'
        elif torch.any(self.alpha_KL < 1) and torch.all(self.alpha_KL + self.lambd_KL == 1):
            return 'betaVAE'
        elif torch.all(self.alpha_KL == 1) and torch.all(self.alpha_KL + self.lambd_KL == 1):
            return 'AE'
        else:
            raise Exception('unknown classification, investigate!')

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
        self.alpha_KL = self.alpha_KL.to(self.device)


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


    def compute_kernel(self, x, y):
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]

        tiled_x = x.view(x_size,1,dim).repeat(1, y_size,1)
        tiled_y = y.view(1,y_size,dim).repeat(x_size, 1,1)

        # return torch.exp(-torch.mean((tiled_x - tiled_y)**2,dim=2)/dim*1.0)
        # modification to output kernels per variables
        return torch.exp(-(tiled_x - tiled_y)**2/dim*1.0)


    def compute_mmd(self, y):
        # true sample
        x = Variable(torch.randn(y.shape[0], self.bottleneck), requires_grad=False).to(self.device)

        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        # return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)
        # modification to return mmd per variables (nodes)
        return torch.mean(x_kernel,axis=[0,1]) + torch.mean(y_kernel,axis=[0,1]) - 2*torch.mean(xy_kernel,axis=[0,1])

    def loss(self, batch, loader_type, objective=False):
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
        # (enhanced by per node custom wkl)
        KL_weights = self.get_KL_weights().to(self.device)
        MMD_weights = self.get_MMD_weights().to(self.device)
        KLD = -0.5 * torch.sum(KL_weights*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=0))
        MMD = torch.mean(MMD_weights*self.compute_mmd(sample))

        if loader_type == 'ETC' and self.ETC_AE:
            KLD = 0*KLD
            MMD = 0*MMD

        reconst_loss = self.reconstruction_loss(recon_x,
                                                x,
                                                tiled_mask = self.tiled_mask,
                                                loss_type = self.rec_loss_type,
                                                delta=self.rec_delta
                                                )

        if self.labels is not None:
            labels_loss, mask = self.labels_loss(sample,
                                                y,
                                                loss_type=self.lab_loss_type,
                                                delta=self.lab_delta
                                                )

            if labels_loss[~mask].numel() > 0: # Check that the tensor is not empty
                labels_loss = torch.mean(labels_loss[~mask])
            else:
                # this shouldn't be later logged!
                labels_loss = torch.tensor(0.0, device=labels_loss.device)
        else:
            labels_loss = torch.tensor(0.0, device=reconst_loss.device)

        
        if objective:
            # this good for validation/hyperparameter tuning
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
            MMD = torch.mean(self.compute_mmd(sample))
            return reconst_loss + KLD + MMD + labels_loss,\
                   reconst_loss, KLD, MMD, labels_loss
        else:
            if loader_type == 'ETC': # ETC does not use lambd_labels (hack, could be done better)
                return reconst_loss + KLD + MMD + labels_loss,\
                    reconst_loss, KLD, MMD, labels_loss
            else:
                return reconst_loss + KLD + MMD + self.lambd_labels*labels_loss,\
                    reconst_loss, KLD, MMD, labels_loss

    def get_KL_weights(self):
        return 1 - self.alpha_KL

    def get_MMD_weights(self):
        return self.alpha_KL + self.lambd_KL - 1

    def on_train_epoch_start(self):
        if self.current_epoch == self.unfreeze_epoch:
            self.decoder.unfreeze()
            self.encoder.unfreeze()

    def training_step(self, batch, batch_idx):
        #self.memory_print()
        (x, y), _, dataloader_idx = batch
        dataset = self.trainer.train_dataloader.flattened[dataloader_idx].dataset
        dataset_name = self.get_dataset_name(dataset)
        loss, loss_rec, kld, mmd, loss_label = self.loss((x, y), dataset_name)

        # Check for nan values with descriptive messages
        assert torch.sum(torch.isnan(loss_rec)) == 0, "Nan values found in 'loss_rec'"
        assert torch.sum(torch.isnan(kld)) == 0, "Nan values found in 'kld'"
        assert torch.sum(torch.isnan(mmd)) == 0, "Nan values found in 'mmd'"
        assert torch.sum(torch.isnan(loss_label)) == 0, "Nan values found in 'loss_label'"
        assert torch.sum(torch.isnan(loss)) == 0, "Nan values found in 'loss'"

        # Check for inf values with descriptive messages
        assert torch.sum(torch.isinf(loss_rec)) == 0, "Inf values found in 'loss_rec'"
        assert torch.sum(torch.isinf(kld)) == 0, "Inf values found in 'kld'"
        assert torch.sum(torch.isinf(mmd)) == 0, "Inf values found in 'mmd'"
        assert torch.sum(torch.isinf(loss_label)) == 0, "Inf values found in 'loss_label'"
        assert torch.sum(torch.isinf(loss)) == 0, "Inf values found in 'loss'"

        self.log('train_loss', loss.detach())
        self.log('train_loss_rec', loss_rec.detach())
        self.log('train_loss_kld', kld.detach())
        self.log('train_loss_mmd', mmd.detach())

        # loss_label is zero if all nan, and logging is meaningless
        if not torch.all(torch.isnan(y)):
            self.log('train_loss_label', loss_label.detach())

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        dataset = self.trainer.val_dataloaders[dataloader_idx].dataset
        dataset_name = self.get_dataset_name(dataset)
        loss, loss_rec, kld, mmd, loss_label = self.loss(
            batch, dataset_name, objective=True
            )
        
        self.log(dataset_name + '/val_loss', loss.detach(), sync_dist=True)
        self.log(dataset_name + '/val_loss_rec', loss_rec.detach(), sync_dist=True)
        self.log(dataset_name + '/val_loss_kld', kld.detach(), sync_dist=True)
        self.log(dataset_name + '/val_loss_mmd', mmd.detach(), sync_dist=True)

        # loss_label is zero if all nan, and logging is meaningless
        if (self.labels is not None) and (not torch.all(torch.isnan(batch[1]))):
            self.log(dataset_name + '/val_loss_label', loss_label.detach(), sync_dist=True)

    # def test_step(self, batch , batch_idx):
        # loss, _, _, _, _ = self.loss(batch)
        # self.log('test_loss', loss.detach(), sync_dist=True)

    def memory_print(self):
        import gc
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
            except:
                pass

   
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
