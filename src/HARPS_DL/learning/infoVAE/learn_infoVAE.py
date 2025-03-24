#!/usr/bin/env python
import os
from pdb import set_trace
#from pytorch_model_summary import summary

from HARPS_DL.project_config import MODELS_BANK, MODELS_TMP
from HARPS_DL.models_structured.model_infoVAE import info_VAE
#from model_VAE_intervals import composite_VAE
from HARPS_DL.datasets.Spectra_split_data_module import Spectra_split_data_module
from HARPS_DL.tools.state_dict_tool import collect_prefix


# mem debug tools


import torch
import torch.utils.data

from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint


from collections import OrderedDict
import numpy as np


class MyLogger(NeptuneLogger):
    def __init__(self):
        super().__init__(
            project=project_neptune,
            tags=['ResNet', 'VAE', 'noise-free'],  # optional
        )

def recall_checkpoint(ckpt_path):
    if os.path.isfile(MODELS_BANK  + ckpt_path):
        return MODELS_BANK  + ckpt_path
    elif os.path.isfile(MODELS_TMP + ckpt_path):
        return MODELS_TMP  + ckpt_path
    else:
        raise FileNotFoundError(f"The file '{ckpt_path}' does not exist.")


# ad hoc fix
def fix_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'decoder':
            name = k[8:] # remove "decoder."
            new_state_dict[name] = v
    return new_state_dict

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

        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")

def cli_main():
   # from pytorch_lightning.profiler import PyTorchProfiler
    cli = MyLightningCLI(info_VAE,
                        datamodule_class=Spectra_split_data_module,
                        parser_kwargs={"error_handler": None},
                        save_config_callback=None,
                        run=False,
                        )
    using_neptune = cli.config['project_neptune'] != ''
    if not using_neptune:
        cli = MyLightningCLI(info_VAE,
                            datamodule_class=Spectra_split_data_module,
                            parser_kwargs={"error_handler": None},
                            run=False,
                            )

    tags = [cli.model.self_AE_classification()]

    tags += [cli.config['model']['decoder_name']]
    tags += [cli.config['model']['encoder_name']]
    tags+=['rv_range=' + str(cli.config['data']['radvel_range'])]
    if cli.config['data']['etc_noise_free']:
        tags+= ['noise_free']
    else:
        tags+= ['noise_ETC']

    if using_neptune:        
        cli.trainer.logger = NeptuneLogger(                     
                            project=cli.config['project_neptune'],
                            tags=tags,  # optional
                            )
        assert(len(cli.parser.parse_args().config) == 1) # when writting this, I assume a single config file (modification is simple though)
        config_path = cli.parser.parse_args().config[0].absolute
        cli.trainer.logger.experiment['config'].upload(config_path)

    if len(cli.config['encoder_checkpoint']) != 0:
        checkpoint_path = recall_checkpoint(cli.config['encoder_checkpoint'])
        state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        state_dict = collect_prefix(state_dict, 'encoder')
        cli.model.encoder.load_state_dict(state_dict)
    #    cli.model.encoder.freeze()

    if len(cli.config['bottleneck_checkpoint']) != 0:
        checkpoint_path = recall_checkpoint(cli.config['bottleneck_checkpoint'])
        state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        state_dict_bottleneck = {}
        for key in ['fc_mu', 'fc_logvar', 'fc_z']:
            state_dict_bottleneck[key] = collect_prefix(state_dict, key)

        cli.model.fc_mu.load_state_dict(state_dict_bottleneck['fc_mu'])
        cli.model.fc_logvar.load_state_dict(state_dict_bottleneck['fc_logvar'])
        cli.model.fc_z.load_state_dict(state_dict_bottleneck['fc_z'])

    if len(cli.config['decoder_checkpoint']) != 0:
        checkpoint_path = recall_checkpoint(cli.config['decoder_checkpoint'])
        state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        state_dict = collect_prefix(state_dict, 'decoder')
        cli.model.decoder.load_state_dict(state_dict)
        cli.model.decoder.freeze()

    cli.model.labels = cli.datamodule.labels

    if cli.datamodule.labels is not None:
        cli.datamodule.log_overview(cli.trainer.logger)

        assert(cli.model.bottleneck >= cli.datamodule.labels.get_vec_length()) # labels must be covered by bottleneck!
        assert(cli.model.bottleneck == cli.datamodule.labels.bottleneck) #

    cli.trainer.fit(cli.model, cli.datamodule)

if __name__ == '__main__':
    cli_main()
