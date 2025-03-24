import pandas as pd
import numpy as np
import torch
from torch import nn
import os
import sys
import neptune as neptune
import matplotlib.pyplot as plt

from HARPS_DL.models_structured.model_basic import ae1d
from HARPS_DL.models_structured.encoders_builder import encoders_builder
from HARPS_DL.models_structured.Overview_encoder_mix_in import Overview_encoder_mix_in
from HARPS_DL.models_structured.model_recognition import recognition_NN


from pdb import set_trace

class recognition_NN_mix(recognition_NN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss(self, batch):
        """loss function

        Args:
            self: self reference
            batch: spectrum, labels
            objective: if true, ignore weights (objective value for validation/tuning)
        """

        (x, y_gt), _, dataloader_idx = batch
        y_pred = self.forward(x)
        labels_loss, mask = self.labels_loss(y_pred,
                                            y_gt,
                                            loss_type=self.lab_loss_type,
                                            delta=self.lab_delta
                                            )

        return torch.mean(labels_loss), torch.mean(labels_loss, axis=0), mask
     
