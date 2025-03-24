from pytorch_lightning.callbacks import Callback
import numpy as np
from HARPS_DL.models_structured.Overview_encoder_mix_in import Overview_encoder_mix_in

import matplotlib.pyplot as plt
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from astropy import units as u
import pandas as pd
from neptune.types import File

from pdb import set_trace



class Labels_inspection_callback(Overview_encoder_mix_in, Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch != 0 and pl_module.current_epoch % pl_module.overview_epoch == 0:
            for i in range(len(self.trainer.val_dataloaders)):
                dataset = self.trainer.val_dataloaders[i].dataset
                dataset_name = self.get_dataset_name(dataset)

                self.high_verbose_inspection(dataset, dataset_name)

                self.labels_inspection(dataset, dataset_name)

                if dataset_name == 'ETC':
                    spectra_idxs = [1, 10, 44, 60, 100, 200, 301, 352, 405, 406, 440, 445, 600]
                else:
                    spectra_idxs = [1, 10, 44, 60, 100, 200, 301, 352, 405, 406, 440, 445, 600]
                err = self.rv_intervention_error(
                            dataset,
                            spectra_idxs,
                            self.labels,
                            wave_region=[6050, 6250],
                            rv_shifts=np.linspace(-40, 40, 40)*u.kilometer/u.second,
                        )
                self.log(dataset_name + '/RVIS', np.mean(err))

                if dataset_name == 'ETC':
                    spectra_idxs = [1, 10, 44, 60, 100]
                else:
                    spectra_idxs = [1, 10, 44, 60, 100]

                self.radvel_overview(dataset, dataset_name, spectra_idxs, self.labels, logging=True)


class Radvel_callback(Overview_encoder_mix_in, Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch != 0 and pl_module.current_epoch % pl_module.overview_epoch == 0:
            for i in range(len(self.trainer.val_dataloaders)):
                dataset = self.trainer.val_dataloaders[i].dataset
                dataset_name = self.get_dataset_name(dataset)

                if dataset_name == 'ETC':
                    spectra_idxs = [1, 10, 44, 60, 100, 200, 301, 352, 405, 406, 440, 445, 600]
                else:
                    spectra_idxs = [1, 10, 44, 60, 100, 200, 301, 352, 405, 406, 440, 445, 600]
                err = self.rv_intervention_error(
                            dataset,
                            spectra_idxs,
                            self.labels,
                            wave_region=[6050, 6250],
                            rv_shifts=np.linspace(-40, 40, 40)*u.kilometer/u.second,
                        )
                self.log(dataset_name + '/RVIS', np.mean(err))

                if dataset_name == 'ETC':
                    spectra_idxs = [1, 10, 44, 60, 100]
                else:
                    spectra_idxs = [1, 10, 44, 60, 100]

                self.radvel_overview(dataset, dataset_name, spectra_idxs, self.labels, logging=True)

class Visual_inspection_callback(Overview_encoder_mix_in, Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch != 0 and pl_module.current_epoch % pl_module.overview_epoch == 0:
            for i in range(len(self.trainer.val_dataloaders)):
                dataset = self.trainer.val_dataloaders[i].dataset
                dataset_name = self.get_dataset_name(dataset)
                self.visual_inspection(dataset, dataset_name)

class Nodes_inspection_callback(Overview_encoder_mix_in, Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch != 0 and pl_module.current_epoch % pl_module.overview_epoch == 0:
            for i in range(len(self.trainer.val_dataloaders)):
                dataset = self.trainer.val_dataloaders[i].dataset
                dataset_name = self.get_dataset_name(dataset)

                nodes_mu, nodes_logvar = self.get_nodes(dataset)

                fig = plt.figure()
                medians = np.median(nodes_mu, axis=1)[:, np.newaxis]

                plt.plot(np.median(np.abs(nodes_mu-medians), axis=1).ravel())
                plt.title('MAD of nodes (~0 = collaps)')
                desc_str = f'validate/epoch_{self.current_epoch}/nodes_mu'
                if isinstance(self.logger, NeptuneLogger):
                    self.logger.experiment[desc_str].log(fig)
                elif isinstance(self.logger, TensorBoardLogger):
                    self.logger.experiment.add_figure(desc_str, fig)
                else:
                    raise NotImplementedError

                fig = plt.figure()
                plt.plot(np.median(nodes_logvar, axis=1).ravel())
                plt.title('median of logvar of nodes (~0 = collaps)')
                desc_str = f'validate/epoch_{self.current_epoch}/nodes_logvar'
                if isinstance(self.logger, NeptuneLogger):
                    self.logger.experiment[desc_str].log(fig)
                elif isinstance(self.logger, TensorBoardLogger):
                    self.logger.experiment.add_figure(desc_str, fig)
                else:
                    raise NotImplementedError


class Labels_table_callback(Overview_encoder_mix_in, Callback):
    def __init__(self, overview_epoch=1):
        self.table = {}
        self.overview_epoch = overview_epoch

    def on_validation_epoch_end(self, trainer, pl_module):
        #set_trace()
        if pl_module.current_epoch != 0 and pl_module.current_epoch % self.overview_epoch == 0:
            for i in range(len(trainer.val_dataloaders)):
                dataset = trainer.val_dataloaders[i].dataset
                dataset_name = pl_module.get_dataset_name(dataset)

                if dataset_name not in self.table:
                    self.table[dataset_name] = {'catalog': pd.DataFrame(), 'normalized': pd.DataFrame()}

                labels_gt, labels_pred = pl_module.predictions_per_index(dataset, batch_size=32)


                row = {}
                row_ori = {}
                for label_str in pl_module.labels.labels:
                    idx = pl_module.labels.label2idx(label_str)[dataset_name]

                    label_gt = labels_gt[idx, :]
                    label_pred = labels_pred[idx, :]

                    row[label_str] = np.nanmean(np.abs(label_gt - label_pred))

                    label_gt_ori = pl_module.labels.inverse_label_normalization(label_gt, label_str)
                    label_pred_ori = pl_module.labels.inverse_label_normalization(label_pred, label_str)

                    row_ori[label_str] = np.nanmean(np.abs(label_gt_ori - label_pred_ori))

                row['all'] = np.nanmean(np.abs(labels_gt - labels_pred))
                row['epoch'] = pl_module.current_epoch

                row_ori['epoch'] = pl_module.current_epoch

                row_df = pd.DataFrame(row, index=[pl_module.current_epoch])
                self.table[dataset_name]['normalized'] = pd.concat([self.table[dataset_name]['normalized'], row_df], axis=0)

                row_ori_df = pd.DataFrame(row_ori, index=[pl_module.current_epoch])
                self.table[dataset_name]['catalog'] = pd.concat([self.table[dataset_name]['catalog'], row_ori_df], axis=0)

                trainer.logger.experiment["data/labels_table_normalized_"
                                        + dataset_name].upload(File.as_html(self.table[dataset_name]['normalized']))

                trainer.logger.experiment["data/labels_table_catalog_"
                                        + dataset_name].upload(File.as_html(self.table[dataset_name]['catalog']))

                
