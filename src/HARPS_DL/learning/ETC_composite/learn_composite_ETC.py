#!/usr/bin/env python

#from pytorch_model_summary import summary
import os

import sys
if 0:
    print("Python version")
    print (sys.version)
    print("Version info.")
    print (sys.version_info)

from HARPS_DL.models_structured.model_composite import composite_VAE
from HARPS_DL.datasets.Spectra_etc_data_module import Spectra_data_module

from HARPS_DL.tools.state_dict_tool import collect_prefix

import torch
import torch.utils.data

from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.cli import LightningCLI


# Hyperparameter optimization library

project_neptune="cvrcekv/ETC-VAE"

class MyLogger(NeptuneLogger):
    def __init__(self):
        super().__init__(
            project=project_neptune,
            tags=['ResNet', 'VAE', 'noise-free'],  # optional
        )


from jsonargparse import class_from_function

class MyLightningCLI(LightningCLI):
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
                "labels_normalization": True,
            },
            
        }
        parser.set_defaults({"data.labels": default_labels})

        parser.add_argument("--encoder_checkpoint", default="")
        parser.add_argument("--decoder_checkpoint", default="")

        parser.add_argument("--project_neptune", default="cvrcekv/developing")

def cli_main():
   # from pytorch_lightning.profiler import PyTorchProfiler


    cli = MyLightningCLI(composite_VAE,
                         datamodule_class=Spectra_data_module,
                         parser_kwargs={"error_handler": None},
                         save_config_callback=None,
                         run=False)
    cli.model.labels = cli.datamodule.labels    

    tags=['composite']
    tags += ['decoder='+cli.config['model']['decoder_name']]
    tags += ['encoder='+cli.config['model']['encoder_name']]
    tags+=['rv_range=' + str(cli.config['data']['radvel_range'])]
    if cli.config['data']['noise_free']:
        tags+= ['noise_free']
    else:
        tags+= ['noise_ETC']



    cli.trainer.logger = NeptuneLogger(
                        project=cli.config['project_neptune'],
                        tags=tags,  # optional
                        )
    assert(len(cli.parser.parse_args().config) == 1) # when writting this, I assume a single config file (modification is simple though)
    config_path = cli.parser.parse_args().config[0].abs_path
    cli.trainer.logger.experiment['config'].upload(config_path)

    # state_dict = fix_state_dict(torch.load('checkpoint.ckpt')['state_dict'])
    # decoder = Decoder_ResNet()
    # decoder.load_state_dict(state_dict)
    # decoder = simulator_NN.load_state_dict(state_dict)
    if len(cli.config['decoder_checkpoint']) != 0:
        checkpoint_path = models_bank  + cli.config['decoder_checkpoint']
        state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        state_dict = collect_prefix(state_dict, 'decoder')
        cli.model.decoder.load_state_dict(state_dict)
        cli.model.decoder.freeze()
    else:
        cli.model.decoder.unfreeze()

    if len(cli.config['encoder_checkpoint']) != 0:
        raise('not tested')
        cli.model.encoder = cli.model.encoder.from_ckpt(cli.config['encoder_checkpoint']).eval()
        cli.model.encoder.freeze()
    else:
        cli.model.encoder.unfreeze()

    cli.datamodule.log_overview(cli.trainer.logger)
    cli.trainer.fit(cli.model, cli.datamodule)

if __name__ == '__main__':
    cli_main()
