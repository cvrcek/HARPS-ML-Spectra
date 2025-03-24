from types import SimpleNamespace
import torch
from torch import nn
import os
from pdb import set_trace
import pytorch_lightning as pl
from collections import OrderedDict


class PreActTransResNetBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation, stride, act_fn):
        """
        basic building block of the ResNet architecture,
        it is build on the previous implementation in decoder.py, but
        it allows more combinations of kernel size, dilation, and stride
        
        Inputs:
            c_in - Number of input features
            c_out - Number of output features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            upsample - If True, we want to apply a stride inside the block and increase the output size by 2
        """
        super().__init__()
        assert(stride == 2 or stride == 1)

        if kernel_size == 3 and dilation == 1:
            if stride == 2:
                padding_1 = 1
                output_padding_1 = 1
            else:
                padding_1 = 1
                output_padding_1 = 0

            padding_2 = 1
            output_padding_2 = 0
        else:
            raise Exception(f'combination of kernel size={kernel_size} and dilation={dilation} isn''t solved')

        # Network representing F
        self.net = nn.Sequential(
            nn.InstanceNorm1d(c_in),
            act_fn(),
            nn.ConvTranspose1d(c_in, c_in, kernel_size=kernel_size, padding=padding_1,
                               output_padding=output_padding_1,
                               stride=stride, bias=True),
            nn.InstanceNorm1d(c_in),
            act_fn(),
            nn.ConvTranspose1d(c_in, c_out, kernel_size=kernel_size, padding=padding_2,
                               output_padding=output_padding_2, bias=True)
        )

        # 1x1 convolution needs to apply non-linearity as well as not done on skip connection
        self.upsample = nn.Sequential(
            nn.InstanceNorm1d(c_in),
            act_fn(),
            nn.ConvTranspose1d(c_in, c_out, kernel_size=1, stride=2,
                               output_padding=1, bias=True)
        ) if stride==2 else None

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
                 c_hidden: list[int]=[512,512,512,256,256,256,128,128,64,64,32,32,16,16,16],
                 kernel_size: list[int]=[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
                 dilation: list[int]=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                 stride: list[int]=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
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

        assert(len(c_hidden) == len(kernel_size) + 1)
        assert(len(kernel_size) == len(dilation))
        assert(len(dilation) == len(stride))

        self.my_params = SimpleNamespace(
                                       c_hidden=c_hidden,
                                       kernel_size=kernel_size,
                                       dilation=dilation,
                                       stride=stride,
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
        # Creating the ResNet blocks
        layers = []
        for c_idx in range(len(self.my_params.c_hidden) - 1):
            c_in = self.my_params.c_hidden[c_idx]
            c_out = self.my_params.c_hidden[c_idx + 1]

            kernel_size = self.my_params.kernel_size[c_idx]
            dilation = self.my_params.dilation[c_idx]
            stride = self.my_params.stride[c_idx]

            if stride == 1:
                assert(c_in == c_out) # changing c_in->c_out only when upscaling (stride = 2)

            layers.append(
                self.my_params.block_class(c_in=c_in,
                                         c_out=c_out,
                                         kernel_size=kernel_size,
                                         dilation=dilation,
                                         stride=stride,
                                         act_fn=self.my_params.act_fn,
                )
            )
        self.layers = nn.Sequential(*layers)

        # mapping to output spectrum
        self.output_net =  nn.Conv1d(
                self.my_params.c_hidden[-1],
                1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
                )

    def _init_params(self):
        #initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.my_params.act_fn_name)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.my_params.act_fn_name)
            # elif isinstance(m, nn.InstanceNorm1d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #set_trace()
        x = x.view(x.size(0),self.my_params.c_hidden[0],-1)
        x = self.layers(x)
        x = self.output_net(x)
        return x


def test_PreActTransResNetBlock():
    import numpy as np
    import torch
    x = np.ones((1, 10240))
    x = torch.tensor(x).type(torch.FloatTensor)

    decoder = Decoder(
        c_hidden=[512, 256],
        kernel_size=[3],
        dilation=[1],
        stride=[2],
    )
    print(decoder(x).shape)
    assert(decoder(x).shape[2]==40)

    decoder = Decoder(
        c_hidden=[512, 256, 256, 128],
        kernel_size=[3, 3, 3],
        dilation=[1, 1, 1],
        stride=[2, 1, 2],
    )
    print(decoder(x).shape)

    decoder = Decoder(
        c_hidden=[512, 512, 256, 256, 128, 128, 64, 64, 64, 32, 32, 32, 16, 16, 16],
        kernel_size=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        dilation=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        stride=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    )
    print(decoder(x).shape)



#test_PreActTransResoetBlock()