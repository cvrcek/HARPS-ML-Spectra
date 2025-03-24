#!/usr/bin/env python

from pdb import set_trace
#from pytorch_model_summary import summary

from HARPS_DL.models_structured.model_recognition_mix import recognition_NN_mix
from HARPS_DL.datasets.Spectra_split_data_module import Spectra_split_data_module as Spectra_data_module

from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint


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
                    'BERV',
                    'Teff',
                    '[M/H]',
                    'logg',
                    'airmass',
                    'H2O_pwv',
                ],
                "labels_type": [
                                'shared',
                                'shared',
                                'shared',
                                'shared',
                                'shared',
                                'shared',
                                'ETC',
                ],
                "labels_normalization": {},
                "fill_bottleneck":[]
            },
        }
        parser.set_defaults({"data.labels": default_labels})

        parser.add_argument("--encoder_checkpoint", default="")
        parser.add_argument("--project_neptune", default="cvrcekv/developing")

        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")

def cli_main():
   # from pytorch_lightning.profiler import PyTorchProfiler
    cli = MyLightningCLI(recognition_NN_mix,
                         datamodule_class=Spectra_data_module,
                         parser_kwargs={"error_handler": None},
                         save_config_callback=None,
                         run=False,
                         )
    using_neptune = cli.config['project_neptune'] != ''
    if not using_neptune:
        cli = MyLightningCLI(recognition_NN_mix,
                            datamodule_class=Spectra_data_module,
                            parser_kwargs={"error_handler": None},
                            run=False,
                            )


    tags=['real encoder']
    tags += [cli.config['model']['encoder_name']]

    if using_neptune:        
        cli.trainer.logger = NeptuneLogger(
                            project=cli.config['project_neptune'],
                            tags=tags,  # optional
                            )
        assert(len(cli.parser.parse_args().config) == 1) # when writting this, I assume a single config file (modification is simple though)
        config_path = cli.parser.parse_args().config[0].abs_path
        cli.trainer.logger.experiment['config'].upload(config_path)

#    cli.datamodule.log_overview(cli.trainer.logger)

    cli.model.labels = cli.datamodule.labels

    assert(cli.model.bottleneck >= cli.datamodule.labels.get_vec_length()) # labels must be covered by bottleneck!
    assert(cli.model.bottleneck == cli.datamodule.labels.bottleneck) # 

    # x, y = cli.datamodule.val_loa
    cli.trainer.fit(cli.model, cli.datamodule)

if __name__ == '__main__':
    cli_main()
