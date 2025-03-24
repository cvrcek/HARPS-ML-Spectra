from types import SimpleNamespace
import torch
from torch import nn
from pdb import set_trace
import pytorch_lightning as pl
from collections import OrderedDict


class PreActTransResNetBlock(nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F
        self.net = nn.Sequential(
            nn.InstanceNorm1d(c_in),
            act_fn(),
            nn.ConvTranspose1d(c_in, c_out, kernel_size=5, padding=2,
                               output_padding=0 if not subsample else 1,
                               stride=1 if not subsample else 2, bias=True),
            nn.InstanceNorm1d(c_in),
            act_fn(),
            nn.ConvTranspose1d(c_out, c_out, kernel_size=5, padding=2, bias=True)
        )

        # 1x1 convolution needs to apply non-linearity as well as not done on skip connection
        self.upsample = nn.Sequential(
            nn.InstanceNorm1d(c_in),
            act_fn(),
            nn.ConvTranspose1d(c_in, c_out, kernel_size=1, stride=2, output_padding=1, bias=True)
        ) if subsample else None

    def forward(self, x):
        #print(x.shape)
        z = self.net(x)
        if self.upsample is not None:
            x = self.upsample(x)
        out = z + x
        return out

class PreActTransResNetBlock_nobatchnorm(nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F
        self.net = nn.Sequential(
            nn.BatchNorm1d(c_in),
            act_fn(),
            nn.ConvTranspose1d(c_in, c_out, kernel_size=5, padding=2,
                               output_padding=0 if not subsample else 1,
                               stride=1 if not subsample else 2, bias=True),
            act_fn(),
            nn.ConvTranspose1d(c_out, c_out, kernel_size=5, padding=2, bias=True)
        )

        # 1x1 convolution needs to apply non-linearity as well as not done on skip connection
        self.upsample = nn.Sequential(
            act_fn(),
            nn.ConvTranspose1d(c_in, c_out, kernel_size=1, stride=2, output_padding=1, bias=True)
        ) if subsample else None

    def forward(self, x):
        #print(x.shape)
        z = self.net(x)
        if self.upsample is not None:
            x = self.upsample(x)
        out = z + x
        return out



resnet_blocks_by_name = {
    #"ResNetBlock": ResNetBlock,
    "PreActTransResNetBlock": PreActTransResNetBlock,
    "PreActTransResNetBlock_nobatchnorm": PreActTransResNetBlock_nobatchnorm
    #"PreActResNetBlock": PreActResNetBlock
}

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}

class Decoder(pl.LightningModule):
    def __init__(self,
                 num_blocks: list[int]=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
                 c_hidden: list[int]=[512,512,512,256,256,256,128,128,64,64,32,32,16,16,16],
                 act_fn_name: str="relu",
                 block_name: str="PreActTransResNetBlock",
                 **kwargs):
        """ResNet based decoder

        Args:
            num_blocks:  number of resnet blocks per layer
            c_hidden: hidden layers
            act_fn_name: activation function
            block_name: name of resnet block
        """
        super().__init__()
        assert block_name in resnet_blocks_by_name
        #set_trace()
        self.my_params = SimpleNamespace(
                                       c_hidden=c_hidden,
                                       num_blocks=num_blocks,
                                       act_fn_name=act_fn_name,
                                       act_fn=act_fn_by_name[act_fn_name],
                                       block_class=resnet_blocks_by_name[block_name])
        self._create_network()
        self._init_params()
    
    @classmethod
    def from_ckpt(cls, ckpt_path: str):
        print(f'loading from ckpt {ckpt_path}')
        def fix_state_dict(state_dict):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k[:7] == 'decoder':
                    name = k[8:] # remove "decoder."
                    new_state_dict[name] = v
            return new_state_dict

        state_dict = fix_state_dict(torch.load(ckpt_path)['state_dict'])
        decoder = cls()
        decoder.load_state_dict(state_dict)
        return decoder



    def _create_network(self):
        c_hidden = self.my_params.c_hidden


        # Creating the ResNet blocks
        blocks = []
        for block_idx, block_count in enumerate(self.my_params.num_blocks):
            for bc in range(block_count):
                subsample = (bc == 0 and block_idx > 0) # Subsample the first block of each group, except the very first one.
                blocks.append(
                    self.my_params.block_class(c_in=c_hidden[block_idx if not subsample else (block_idx-1)],
                                             act_fn=self.my_params.act_fn,
                                             subsample=subsample,
                                             c_out=c_hidden[block_idx])
                )
        self.blocks = nn.Sequential(*blocks)

        # mapping to output spectrum
        self.output_net =  nn.Conv1d(c_hidden[-1],1,kernel_size=3,stride=1,padding=1,bias=True)



    def _init_params(self):
        #initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.my_params.act_fn_name)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.my_params.act_fn_name)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #set_trace()
        x = x.view(x.size(0),self.my_params.c_hidden[0],-1)
        x = self.blocks(x)
        x = self.output_net(x)
        return x

