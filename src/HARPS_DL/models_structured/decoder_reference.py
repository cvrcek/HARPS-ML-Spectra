from types import SimpleNamespace
from torch import nn
import os
import sys
from pdb import set_trace

from HARPS_DL.models_structured.model_basic import ae1d

class Decoder(ae1d):
    def __init__(self,
                c_hidden: list[int]=[512, 512, 256, 128, 128, 128, 64, 64, 64, 32, 32, 32, 16, 16, 16, 1],
                kernel_size: list[int]=[3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                stride:      list[int]=[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                 **kwargs):
        """
            c_hidden: hidden chanels
        """
        super().__init__()
        self.save_hyperparameters()
        assert(len(c_hidden) == len(kernel_size) + 1)
        assert(len(kernel_size) == len(stride))

        self.my_params = SimpleNamespace(
                        c_hidden=c_hidden,
                        kernel_size=kernel_size,
                        stride=stride,
                        )

        self._create_network()
        self._init_params()

    def _create_network(self):
        blocks = []
        for layer_idx in range(len(self.my_params.c_hidden) - 1):
            channels_in = self.my_params.c_hidden[layer_idx]
            channels_out = self.my_params.c_hidden[layer_idx + 1]
            kernel_size = self.my_params.kernel_size[layer_idx]
            stride = self.my_params.stride[layer_idx]
            blocks.append(
                nn.Sequential(
                        self.deconv(int(channels_in),
                                    int(channels_out),
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    ),
                    )
            )
        blocks.append(nn.Sequential(self.predict(1)))

        self.blocks = nn.Sequential(*blocks)

    def _init_params(self):
        #initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.view(x.size(0),512,-1)
        x = self.blocks(x)
        return x
