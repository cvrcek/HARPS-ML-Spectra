from matplotlib import pyplot as plt

import neptune
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
import pandas as pd

class Datamodule_overview():
    def log_overview(self, logger):
        def log_dataset(dataset, descriptor, logger, label):
            data = dataset.df[label].to_numpy()
            fig = plt.figure()
            plt.hist(data)
            plt.title(label)
            if isinstance(logger, NeptuneLogger):
                logger.experiment[f'{descriptor}'].log(
                    neptune.types.File.as_image(fig))
            elif isinstance(logger, TensorBoardLogger):
                logger.experiment.add_figure(f'{descriptor}/{label}', fig)
            else:
                raise NotImplementedError

        shared_labels = ['Teff', 'logg', '[M/H]', 'airmass']
        etc_unique_labels = ['mag', 'Texp', 'H2O_pwv']
        real_unique_labels = []
        for label in self.labels.labels:
            if label in shared_labels:
                log_dataset(self.real_dataset, 'real_data_overview', logger, label)
            if label in real_unique_labels:
                log_dataset(self.real_dataset, 'real_data_overview', logger, label)

        dic = {}
        dic['labels'] = self.labels.labels
        dic['medians'] = [self.labels.get_label_median(label)
                          for label in self.labels.labels]
        dic['mads'] = [self.labels.get_label_mad(label)
                       for label in self.labels.labels]
        df = pd.DataFrame(dic)
        if isinstance(logger, NeptuneLogger):
            logger.experiment['datasets_normalization'].upload(neptune.types.File.as_html(df))
        elif isinstance(logger, TensorBoardLogger):
            logger.experiment.add_text('datasets_normalization', df.to_html())
        else:
            raise NotImplementedError

