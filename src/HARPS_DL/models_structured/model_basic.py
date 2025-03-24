import torch
import torch.nn as nn

import pytorch_lightning as pl
import numpy as np

from pdb import set_trace
import psutil


class ae1d(pl.LightningModule):
    def __init__(self, learning_rate=1e-3,
                       lambd_KL=0.3, # weight of KL divergence ("distribution penalty")
                       anneal_KL=True,
                       ):
        super().__init__()
        self.learning_rate = learning_rate
        self.lambd_KL = lambd_KL
        self.anneal_KL = anneal_KL

    def get_KL_weight(self):
        if self.anneal_KL:
            n_iter = self.global_step
            I1 = 50000 # since we use four GPUs
            I2 = 100000 # since we use four GPUs

            I = (n_iter-I1)/(I2-I1)
            if I < 0:
                wkl = 0.
            elif I > 1:
                wkl = self.lambd_KL
            else:
                wkl = self.lambd_KL*I
            return wkl
        else:
            return self.lambd_KL

    def conv(self, in_planes, out_planes, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv1d(int(in_planes), int(out_planes), kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

    def predict(self, in_planes):
        return nn.Conv1d(in_planes,1,kernel_size=3,stride=1,padding=1,bias=True)

    def deconv(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1, output_padding=0):
        return nn.Sequential(
            nn.ConvTranspose1d(int(in_planes), int(out_planes), kernel_size=kernel_size, stride=stride, padding=1, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

    def in_channels(self, layer):
        return layer._modules['0'].in_channels
    
    def EPE_my(self, output, target, mean, tiled_mask):
        assert(output.shape == target.shape)
        EPE_map = torch.norm(target-output,1,1)

        EPE_map = EPE_map * tiled_mask

        if mean:
            return EPE_map.mean()
        else:
            batch_size = output.shape[0]
            return EPE_map.sum()/batch_size



    def reconstruction_loss(self, output, target, tiled_mask, loss_type='MAE', delta=1.0):
        assert(output.shape == target.shape)
        if loss_type == 'MAE':
            loss = torch.norm(target-output,1,1)
            loss = loss * tiled_mask
            return loss.mean()
        elif loss_type == 'Cauchy':
            loss = torch.norm(target-output, p=2, dim=1)
            loss = loss * tiled_mask
            cauchy_loss = delta**2 * torch.log1p((loss / delta)**2)
            return cauchy_loss.mean()

    def labels_loss(self, z, y_gt, loss_type='MAE', delta=1.0):
        """ labels loss
            return  loss for each label

        """
        if len(y_gt.shape) == 2:
            yh = y_gt.clone()
            z_supervised = z[:,0:yh.shape[1]].clone()
        else:
            yh = y_gt[:,0,:].clone()
            z_supervised = z[:,0:yh.shape[1]].clone()

        # Create a mask for valid entries
        mask = torch.isnan(yh)
        yh[mask] = 0.0  # NaNs will not contribute to the loss
        z_supervised[mask] = 0.0  # corresponding predictions are also made zero

        if loss_type == 'MAE':
            labels_loss = torch.abs(yh-z_supervised) # per label loss
            labels_loss[mask] = 0.0  # ensure no contribution from NaN labels
        elif loss_type == 'Cauchy':
            diff = yh-z_supervised
            # print('diff:')
            # print(diff)
            labels_loss = delta**2 * torch.log1p((diff / delta)**2) # per label loss
            labels_loss[mask] = 0.0  # ensure no contribution from NaN labels
        # print('loss:')
        # print(labels_loss)
        # set_trace()

        return labels_loss, mask

    def encode(self, x):
        raise NotImplementedError

    def decode(self, latent_vector):
        raise NotImplementedError

    def forward(self, x):
        #set_trace()
        mu, logvar = self.encode(x)

        # sampling
        sample = torch.randn_like(mu) # same size as mu, samples from N(0, 1)
        sample = mu + sample*torch.exp(0.5*logvar) # samples from N(mu, exp(0.5*logvar))

        out0 = self.decode(sample)
        return out0, sample, mu, logvar

    # ELBO loss
    #
    def loss(self, batch):
        raise NotImplementedError


    def validation_step(self, batch , batch_idx):
        raise NotImplementedError

    def mem_logging(self):
        if 0: # deep logs, it requires importing gubby!
            if self.current_epoch == 50:
                set_trace()
            hp_str= str(hp.heap()).split('\n')
            self.logger.experiment.add_text(str(self.current_epoch) + '_hp_total',
                    hp_str[0])
            self.logger.experiment.add_text(str(self.current_epoch) + '_hp_legend',
                    hp_str[1])
            for i in range(2,7):
                self.logger.experiment.add_text(str(self.current_epoch) + '_hp_idx_' + str(i-2),
                    hp_str[i])
            if self.current_epoch % 10 == 0:
                # various memory experiments
                if 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                elif 0:
                    self.logger.save() # does not seem to help
                elif 0:
                    self.logger.experiment.flush()
                elif 0:
                    self.logger.flush()
                elif 0:
                    self.logger.experiment.save()
        # standard psutils logs (MB used, and percentage of SWAP used)
        self.log('mem_total', float(psutil.swap_memory().used + psutil.virtual_memory().used)/1024./1024.)
        self.log('swap_prct', float(psutil.swap_memory().percent))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_mask(self):
        L = 327680
        left_last_zero = 7588
        right_first_zero = 320090
        mid_first_zero = 159453
        mid_last_zero = 162893

        mask = np.ones((1,L),dtype=np.float32)
        mask[0,:left_last_zero+1] = 0
        mask[0,right_first_zero:] = 0
        mask[0,mid_first_zero:mid_last_zero+1] = 0
        self.mask = torch.tensor(mask)
        return self.mask

    def get_dataset_name(self, dataset):
        if isinstance(dataset, torch.utils.data.dataset.Subset):
            dataset_name = dataset.dataset.dataset_name
        else:
            dataset_name = dataset.dataset_name
        return dataset_name
    