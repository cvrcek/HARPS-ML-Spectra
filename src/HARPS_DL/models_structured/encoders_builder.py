from HARPS_DL.models_structured.encoder import Encoder as Encoder_CNN
from HARPS_DL.models_structured.encoder_ResNet import Encoder as Encoder_ResNet
from pdb import set_trace


def encoders_builder(encoder_name):
    # match known decoder names to their definitions
    if encoder_name == 'CNN_classic':
        # Classic/reference CNN encoder from Sedaghat et al. 2021
        encoder = Encoder_CNN()
    elif encoder_name == 'CNN_650K':
        encoder = Encoder_CNN(
                c_hidden=[16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128, 256, 512],
                kernel_size=[7, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                stride=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                )
    elif encoder_name == 'CNN_1000K':
        encoder = Encoder_CNN(
                c_hidden=[16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512],
                kernel_size=[7, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                stride=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
                )
    elif encoder_name == 'CNN_attention':
        encoder = Encoder_CNN(
                c_hidden=[16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128, 256, 512],
                kernel_size=[7, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                stride=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                self_attention=True,
                )
    elif encoder_name == 'ResNet_small':
        encoder = Encoder_ResNet(
            c_hidden=[1, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128, 256, 512],
            kernel_size=[7, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            dilation=[1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            stride=[2,2,2,2,2,2,2,2,2,2,2,2,2,2],
            act_fn_name="relu",
            block_name="PreActResNetBlock",
        )
    elif encoder_name == 'ResNet_1':
        encoder = experimental_architectures(encoder_name)
    elif encoder_name == 'ResNet_dilated':
        encoder = experimental_architectures(encoder_name)
    elif encoder_name == 'Starnet':
        encoder = experimental_architectures(encoder_name)
    else:
        raise Exception('unknown encoder name: {encoder_name}')
    return encoder

def experimental_architectures(name):
    # TRY NOT TO MODIFY OLD NAMES
    if name == 'ResNet_1':
        # correction to reach 
        encoder = Encoder_ResNet(
            c_hidden=[1, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128, 256, 512, 512],
            kernel_size=[7, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            dilation=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            stride=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
            act_fn_name="relu",
            block_name="PreActResNetBlock",
        )
    elif name == 'Starnet':
        num_filters = 32
        filter_length = 3
        pool_length = 2
        num_hidden = [64, 32]
        num_labels = 6
        encoder = Starnet(input_size=327680,
                          num_filters=32,
                          filter_length=3,
                          pool_length=2,
                          num_hidden=[64, 32],
                          num_labels=6)
    elif name == 'ResNet_dilated':
        encoder = Encoder_ResNet(
            c_hidden=[1, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128, 256, 512],
            kernel_size=[7, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            dilation=[2,2,2,2,2,2,2,2,2,2,2,2,2,2],
            stride=[2,2,2,2,2,2,2,2,2,2,2,2,2,2],
            act_fn_name="relu",
            block_name="PreActResNetBlock",
        )
    else:
        raise Exception('unknown experimental encoder name: {encoder_name}')
    return encoder

def test():
    import numpy as np
    import torch
    from torchinfo import summary
    x = np.ones((1, 327680))
    x = torch.tensor(x).type(torch.FloatTensor)


#    encoder = encoders_builder('ResNet_small')
    encoder = encoders_builder('CNN_attention')
    #encoder = encoders_builder('CNN_classic')
    #encoder = encoders_builder('ResNet_1')
  #  encoder = encoders_builder('ResNet_dilated')
    summary(encoder, input_size=(1, 1, 327680), device='cpu')
    print(encoder(x).shape)
    

#test()
