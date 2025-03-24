#!/usr/bin/env python

from pdb import set_trace


import sys
if 0:
    print("Python version")
    print (sys.version)
    print("Version info.")
    print (sys.version_info)

from HARPS_DL.project_config import MODELS_BANK

from HARPS_DL.models_structured.model_VAE_intervals import composite_VAE
from HARPS_DL.datasets.Spectra_mixed_data_module import Spectra_mixed_data_module as Spectra_data_module
from HARPS_DL.tools.state_dict_tool import collect_prefix

import torch
import torch.utils.data


from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.cli import LightningCLI

from collections import OrderedDict

# ad hoc fix
def fix_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'decoder':
            name = k[8:] # remove "decoder."
            new_state_dict[name] = v
    return new_state_dict

class MyLightningCLI(LightningCLI):
#   def __init__(self):
#       if os.environ['HOME'] == '/home/vcvrcek':
#           gpu_idx = get_empty_gpu()
#           print('using GPU: ' + str(gpu_idx))
#       elif os.environ['HOME'] == '/home/cv':
#           gpu_idx = 0 # this should be RTX 3090
#           print('using GPU: ' + str(gpu_idx))
#       super().__init__({'devices': [gpu_idx]})

    def add_arguments_to_parser(self, parser) -> None:
        default_labels = {
            "class_path": "Labels.Labels",
            "init_args": {
                "datasets_names": [
                                   'ETC',
                                   'real',
                ],
                "labels": [
                            'radvel',
                            'H2O_pwv',
                            'Teff',
                            '[M/H]',
                            'logg',
                            'Mass',
                            'airmass',
                ],
                "labels_type": [
                                'shared',
                                'shared',
                                'shared',
                                'shared',
                                'shared',
                                'shared',
                                'shared',
                ],
                "labels_normalization": {},
                "fill_bottleneck":[
                                'ETC'
                ]
            },
        }
        parser.set_defaults({"data.labels": default_labels})

        parser.add_argument("--encoder_checkpoint", default="")
        parser.add_argument("--bottleneck_checkpoint", default="")
        parser.add_argument("--decoder_checkpoint", default="")

        parser.add_argument("--project_neptune", default="cvrcekv/developing")

def cli_main():
    cli = MyLightningCLI(composite_VAE,
                         datamodule_class=Spectra_data_module,
                         parser_kwargs={"error_handler": None},
                         save_config_callback=None,
                         run=False,
                         )

    tags=['composite']
    tags += [cli.config['model']['decoder_name']]
    tags += [cli.config['model']['encoder_name']]
    tags+=['rv_range=' + str(cli.config['data']['radvel_range'])]
    if cli.config['data']['etc_noise_free']:
        tags+= ['noise_free']
    else:
        tags+= ['noise_ETC']

    if cli.config['project_neptune'] != '':        
        cli.trainer.logger = NeptuneLogger(
                            project=cli.config['project_neptune'],
                            tags=tags,  # optional
                            )
        assert(len(cli.parser.parse_args().config) == 1) # when writting this, I assume a single config file (modification is simple though)
        config_path = cli.parser.parse_args().config[0].abs_path
        cli.trainer.logger.experiment['config'].upload(config_path)

    if len(cli.config['encoder_checkpoint']) != 0:
        checkpoint_path = MODELS_BANK  + cli.config['encoder_checkpoint']
        state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        state_dict = collect_prefix(state_dict, 'encoder')
        cli.model.encoder.load_state_dict(state_dict)
    #    cli.model.encoder.freeze()

    if len(cli.config['bottleneck_checkpoint']) != 0:
        checkpoint_path = MODELS_BANK  + cli.config['bottleneck_checkpoint']
        state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        state_dict_bottleneck = {}
        for key in ['fc_mu', 'fc_logvar', 'fc_z']:
            state_dict_bottleneck[key] = collect_prefix(state_dict, key)

        cli.model.fc_mu.load_state_dict(state_dict_bottleneck['fc_mu'])
        cli.model.fc_logvar.load_state_dict(state_dict_bottleneck['fc_logvar'])
        cli.model.fc_z.load_state_dict(state_dict_bottleneck['fc_z'])

    if len(cli.config['decoder_checkpoint']) != 0:
        checkpoint_path = MODELS_BANK  + cli.config['decoder_checkpoint']
        state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        state_dict = collect_prefix(state_dict, 'decoder')
        cli.model.decoder.load_state_dict(state_dict)
        cli.model.decoder.freeze()
    
    cli.datamodule.log_overview(cli.trainer.logger)

    cli.model.labels = cli.datamodule.labels
    assert(cli.model.bottleneck >= cli.datamodule.labels.get_vec_length()) # labels must be covered by bottleneck!
    assert(cli.model.bottleneck == cli.datamodule.labels.bottleneck) # 

    cli.trainer.fit(cli.model, cli.datamodule)

if __name__ == '__main__':
    cli_main()
