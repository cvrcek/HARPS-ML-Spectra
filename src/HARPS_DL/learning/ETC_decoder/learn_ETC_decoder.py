#!/usr/bin/env python

from pdb import set_trace
#from pytorch_model_summary import summary

from HARPS_DL.project_config import MODELS_BANK

from HARPS_DL.models_structured.model_simulation import simulator_NN
from HARPS_DL.datasets.Spectra_etc_data_module import Spectra_data_module
from HARPS_DL.tools.state_dict_tool import collect_prefix

import torch
import torch.utils.data

from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.cli import LightningCLI

class MyLogger(NeptuneLogger):
    def __init__(self):
        super().__init__(
            project=project_neptune,
            tags=['ResNet', 'Decoder'],  # optional
        )

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        default_labels = {
            "class_path": "datasets.Labels.Labels",
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
                            'airmass',
                ],
                "labels_type": [
                                'shared',
                                'shared',
                                'shared',
                                'shared',
                                'shared',
                                'shared',
                ],
                "labels_normalization": {},
            },
            
        }
        parser.set_defaults({"data.labels": default_labels})

        parser.add_argument("--decoder_checkpoint", default="")
        parser.add_argument("--project_neptune", default="cvrcekv/developing")

def cli_main():
    cli = MyLightningCLI(simulator_NN,
                        datamodule_class=Spectra_data_module,
                        parser_kwargs={"error_handler": None},
                        save_config_callback=None,
                        run=False)
    using_neptune = cli.config['project_neptune'] != ''
    if not using_neptune:
        cli = MyLightningCLI(info_VAE,
                            datamodule_class=Spectra_split_data_module,
                            parser_kwargs={"error_handler": None},
                            run=False,
                            )

    tags = ['Decoder_only', cli.config['model']['decoder_name']]
    tags += ['rv_range=' + str(cli.config['data']['radvel_range'])]
    tags += ['noise_free'] if cli.config['data']['noise_free'] else ['noise_ETC']

    if using_neptune:
        cli.trainer.logger = NeptuneLogger(

                            project=cli.config['project_neptune'],
                            tags=tags,  # optional
                            )
        assert(len(cli.parser.parse_args().config) == 1) # when writting this, I assume a single config file (modification is simple though)
        config_path = cli.parser.parse_args().config[0].absolute
        cli.trainer.logger.experiment['config'].upload(config_path)


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
