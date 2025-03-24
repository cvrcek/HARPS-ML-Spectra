from HARPS_DL.models_structured.decoder_reference import Decoder as Decoder_CNN
from HARPS_DL.models_structured.decoder import Decoder as Decoder_ResNet
from HARPS_DL.models_structured.decoder_ResNet_new import Decoder as Decoder_ResNet_new



def decoders_builder(decoder_name):
    # match known decoder names to their definitions
    if decoder_name == 'CNN_classic':
        # classic/reference CNN decoder from Sedaghat et al. 2021
        decoder = Decoder_CNN()
    elif decoder_name == 'CNN_big':
        decoder = Decoder_CNN(
                c_hidden=[512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 128, 128, 128, 64, 64, 64, 32, 32, 32, 16, 16, 16, 1],
                kernel_size=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                stride=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        )
    elif decoder_name == 'ResNet_big':
        # Border-line (memory-wise) ResNet decoder
        num_blocks = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        c_hidden = [
                    512, 512, 512,
                    256, 256, 256,
                    128, 128,
                    64, 64,
                    32, 32,
                    16, 16, 16,
                    ]
        act_fn_name = 'relu'
        block_name = 'PreActTransResNetBlock'
        decoder = Decoder_ResNet(
                       num_blocks=num_blocks,
                       c_hidden=c_hidden,
                       act_fn_name=act_fn_name,
                       block_name=block_name,
                       )
    elif decoder_name == 'ResNet_small':        
        num_blocks = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        c_hidden = [
                    512, 512, 512,
                    256, 256, 256,
                    128, 128,
                    64, 64,
                    32, 32,
                    16, 16, 16,
                    ]
        act_fn_name = 'relu'
        block_name = 'PreActTransResNetBlock'
        decoder = Decoder_ResNet(num_blocks=num_blocks,
                       c_hidden=c_hidden,
                       act_fn_name=act_fn_name,
                       block_name=block_name,
                       )
    elif decoder_name == 'ResNet_1':
        decoder = experimental_architectures(decoder_name)
    elif decoder_name == 'ResNet_2':
        decoder = experimental_architectures(decoder_name)
    else:
        raise Exception('unknown decoder name: {decoder_name}')
    
    return decoder

def experimental_architectures(name):
    # TRY NOT TO MODIFY OLD NAMES
    if name == 'ResNet_1':
        # tiny architecture
        c_hidden=[512, 512, 256, 256, 128, 128, 64, 64, 64, 32, 32, 32, 16, 16, 16]
        kernel_size=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        dilation=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        stride=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

        decoder = Decoder_ResNet_new(
                    c_hidden=c_hidden,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=stride,
                    )
    if name == 'ResNet_2':
        # tiny architecture
        c_hidden=[512, 256, 256, 128, 128, 64, 64, 64, 32, 32, 32, 16, 16, 16, 16]
        kernel_size=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        dilation=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        stride=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

        decoder = Decoder_ResNet_new(
                    c_hidden=c_hidden,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=stride,
                    )
    else:
        raise Exception('unknown experimental decoder name: {decoder_name}')
    return decoder
