from types import SimpleNamespace
from torch import nn
import os
from pdb import set_trace
import pytorch_lightning as pl

negative_slope = 0.1 # for leaky relu

class PreActResNetBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation, stride, act_fn):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        padding_1 = ((kernel_size - 1) * dilation) // 2
        padding_2 = ((kernel_size - 1) * dilation) // 2

        # Network representing F
        self.net = nn.Sequential(
            nn.InstanceNorm1d(c_in),
            act_fn(),
            nn.Conv1d(c_in, c_out, kernel_size=kernel_size,
                      padding=padding_1, stride=stride,
                      dilation=dilation, bias=True),
            nn.InstanceNorm1d(c_in),
            act_fn(),
            nn.Conv1d(c_out, c_out, kernel_size=kernel_size,
                      padding=padding_2, dilation=dilation, bias=True)
        )

        # 1x1 convolution needs to apply non-linearity as well as not done on skip connection
        self.downsample = nn.Sequential(
            nn.InstanceNorm1d(c_in),
            act_fn(),
            nn.Conv1d(c_in, c_out, kernel_size=1, stride=2, padding=0, bias=True)
        ) if stride==2 else None

    def forward(self, x):
        #print(x.shape)
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return out

resnet_blocks_by_name = {
    #"ResNetBlock": ResNetBlock,
    "PreActResNetBlock": PreActResNetBlock,
    #"PreActResNetBlock": PreActResNetBlock
}

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": lambda: nn.LeakyReLU(negative_slope=negative_slope),
    "gelu": nn.GELU
}

class Encoder(pl.LightningModule):
    def __init__(self,
                 c_hidden: list[int]=[1, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128, 256, 512],
                 kernel_size: list[int]=[7, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                 dilation: list[int]=[1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                 stride: list[int]=[2,2,2,2,2,2,2,2,2,2,2,2,2,2],
                 act_fn_name: str="relu",
                 block_name: str="PreActResNetBlock",
                 **kwargs):
        """ResNet based encoder

        Args:
            num_blocks:  number of resnet blocks per layer
            c_hidden: hidden layers
            act_fn_name: activation function
            block_name: name of resnet block
        """
        super().__init__()
        assert block_name in resnet_blocks_by_name
        #set_trace()
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
        layers.append(nn.Sequential(nn.Flatten()))
        self.layers = nn.Sequential(*layers)




    def _init_params(self):
        #initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Conv1d):
                if self.my_params.act_fn_name == "leakyrelu":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=negative_slope)  # Change 0.2 to your actual negative slope value
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.my_params.act_fn_name)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        #set_trace()
        x = x.view(x.size(0),1,-1)
        x = self.layers(x)
        return x

def test_PreActResNetBlock():
    import numpy as np
    import torch
    x = np.ones((1, 327680))
    x = torch.tensor(x).type(torch.FloatTensor)

    # decoder = Encoder(
    #     c_hidden=[1, 16, 16],
    #     kernel_size=[3, 3],
    #     dilation=[1, 1],
    #     stride=[2, 2],
    # )
    decoder = Encoder()
    print(decoder(x).shape)


#test_PreActResNetBlock()