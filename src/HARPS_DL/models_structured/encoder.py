from torch import nn
import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from types import SimpleNamespace

from HARPS_DL.models_structured.model_basic import ae1d

#from HARPS_DL.datasets.Dataset_mixin import Dataset_mixin

from HARPS_DL.tools.MI_support import get_MIs_by_nearest_neighbor
from HARPS_DL.tools.IRS_support import irs_dic

from pdb import set_trace

# designed for long sequences
# pooling to shorten the sequence
class SelfAttentionLocal(nn.Module):
    def __init__(self, in_dim, window_size, stride, pooling_factor):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.window_size = window_size
        self.stride = stride
        self.pool = nn.MaxPool1d(pooling_factor)

    def forward(self,x):
        """
            inputs :
                x : input feature maps(B X C X N)
            returns :
                out : self attention value + input feature 
        """
        B, C, N = x.size()
        out = torch.zeros_like(x)
        count_overlap = torch.zeros_like(x)
        
        for i in range(0, N-self.window_size+1, self.stride):
            j = i + self.window_size
            x_local = x[:, :, i:j]
            
            B_local, C_local, N_local = x_local.size()
            proj_query = self.query_conv(x_local).view(B_local, -1, N_local).permute(0, 2, 1) 
            proj_key = self.key_conv(x_local).view(B_local, -1, N_local)
            energy = torch.bmm(proj_query, proj_key)
            
            attention = F.softmax(energy, dim=-1)
            proj_value = self.value_conv(x_local).view(B_local, -1, N_local)
            out_local = torch.bmm(proj_value, attention.permute(0, 2, 1))
            
            out[:, :, i:j] += self.gamma*out_local + x_local
            count_overlap[:, :, i:j] += 1
        
        out /= count_overlap  # Average the overlapped parts

        out = self.pool(out)  # Downsample the output sequence using max pooling

        return out



class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        """
            inputs :
                x : input feature maps(B X C X N)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        B, C, N = x.size()
        proj_query = self.query_conv(x).view(B, -1, N).permute(0, 2, 1) # B X N X C
        proj_key =  self.key_conv(x).view(B, -1, N) # B X C x N
        energy =  torch.bmm(proj_query, proj_key) # transpose check
        attention = F.softmax(energy, dim=-1) # B X N X N
        proj_value = self.value_conv(x).view(B, -1, N) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = self.gamma*out + x
        return out#, attention



class Encoder(pl.LightningModule):
    def __init__(self,
                input_size: int=327680,
                output_size: int=10240,
                c_hidden: list[int]=[16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128, 256, 512, 512],
                kernel_size: list[int]=[7, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                stride: list[int]=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
                self_attention: bool=False,
                **kwargs):
        """CNN based encoder

        Args:
            input_size: number of pixels in the input spectrum
            c_hidden: hidden layers dimension
            kernel_size: hidden layer kernel size
            stride: hidden layer stride
        """
        super().__init__()
        self.save_hyperparameters()

        self.my_params = SimpleNamespace(
                        c_hidden=c_hidden,
                        kernel_size=kernel_size,
                        stride=stride,
                        self_attention=self_attention,
                        )

        # these are merely checked for dimensionality consistency
        self.input_size = input_size
        self.output_size = output_size

        self._create_network()
        self._init_params()



    def _create_network(self):
        blocks = []
        layer_size_prev = 1
        for layer_idx, layer_size in enumerate(self.my_params.c_hidden):
            layer_size = self.my_params.c_hidden[layer_idx]
            kernel_size = self.my_params.kernel_size[layer_idx]
            stride = self.my_params.stride[layer_idx]
            blocks.append(
                nn.Sequential(
                        nn.Conv1d(  int(layer_size_prev),
                                    int(layer_size),
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=(kernel_size-1)//2,
                                    bias=True),
                        nn.LeakyReLU(0.1,inplace=True)
                    )
            )
            layer_size_prev = layer_size



        if self.my_params.self_attention:
            blocks.append(nn.Sequential(SelfAttention(self.my_params.c_hidden[-1])))

        blocks.append(nn.Sequential(nn.Flatten()))
        self.blocks = nn.Sequential(*blocks)
    
        # Dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, self.input_size)
            dummy_output = self.forward(dummy_input)
            assert dummy_output.shape[1:] == (self.output_size,), \
                f"Output shape {dummy_output.shape} doesn't match expected shape {self.output_size}"


    def _init_params(self):
        #initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):        
        #x = x.view(x.size(0),self.my_params.c_hidden[0],-1)
        x = x.view(x.size(0),1,-1)
        x = self.blocks(x)

        # if self.my_params.self_attention:
        #     x, _ = self.selfAttention(x) # add attention

        return x

