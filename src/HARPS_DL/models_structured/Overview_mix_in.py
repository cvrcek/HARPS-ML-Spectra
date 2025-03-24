from tqdm import tqdm

import numpy as np
import torch

from HARPS_DL.models_structured.Intervention_mix_in import Intervention_mix_in
import matplotlib.pyplot as plt
import seaborn as sns
from pdb import set_trace
from astropy import units as u

from HARPS_DL.tools.spectra_tools import doppler_shift
   
from HARPS_DL.datasets.Dataset_mixin import Dataset_mixin
 
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.feature_selection import mutual_info_regression

from sklearn.neighbors import KernelDensity

class Overview_mix_in(Intervention_mix_in):
    """
        mixin class with analytic tools for models
    """
    def labels_inspection(self, dataset, dataset_name, batch_size=32, original_units=True, error_plot=True):

        """
            original_units: apply inverse transform for visualization
            error_plot: plot ground truth X (prediction-ground truth)
        """
        class Dataset_with_idx(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __getitem__(self, idx):
                return self.dataset[idx], idx

            def __len__(self):
                return len(self.dataset)
        
        _, labels_in = dataset[0]
        labels_num_in_bottleneck = labels_in.shape[1]

        ds = Dataset_with_idx(dataset)
        dl = torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size)

        labels_gt = np.full((labels_num_in_bottleneck, len(dataset)), np.nan)
        labels_predict = np.full((self.bottleneck, len(dataset)), np.nan)

        for (spectra_in, labels_in), idxs in iter(dl):
            z = self.forward(spectra_in.to(self.device))
            z = z.detach().cpu().numpy()
            if len(idxs) == 1:
                labels_predict[:, idxs] = z.T.ravel()
                labels_gt[:, idxs] = torch.reshape(labels_in, [labels_in.shape[0], -1]).detach().cpu().numpy().T.ravel()
            else:
                labels_predict[:, idxs] = z.T
                labels_gt[:, idxs] = torch.reshape(labels_in, [labels_in.shape[0], -1]).detach().cpu().numpy().T
                
        labels_gt = torch.cat(labels_gt)
        labels_pred = torch.cat(labels_pred)

        for label in self.labels.labels:
            idx = self.labels.label2idx(label)[dataset_name]

            label_gt = labels_gt[idx, :].ravel()
            label_pred = labels_predict[idx, :].ravel()

            fig = plt.figure()
            if len(label_gt) == 0:
                plt.plot()
                plt.title(f'label: {label} has no data for dataset {dataset_name} ')
            else:
                if original_units:
                    label_gt = self.labels.inverse_label_normalization(label_gt, label)
                    label_pred = self.labels.inverse_label_normalization(label_pred, label)
                
                label_med_loss = np.nanmedian(np.abs(label_gt - label_pred))
                label_mean_loss = np.nanmean(np.abs(label_gt - label_pred))

                if error_plot:
                    plt.plot(label_gt, label_pred - label_gt, '.')
                else:
                    plt.plot(label_gt, label_pred, '.')
                    plt.axis('square')
                plt.title(f'{label} median(|gt-pred|)= {label_med_loss} mean(|gt-pred|)= {label_mean_loss}')

            plt.xlabel('ground truth')
            plt.ylabel('prediction - ground truth')

            desc_str = f'validate/epoch_{self.current_epoch}/labels'
            if isinstance(self.logger, NeptuneLogger):
                self.logger.experiment[desc_str].log(fig)
            elif isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(desc_str, fig)
            else:
                raise NotImplementedError

        plt.close('all')
    
    def GIS_inspection(self, dataset_gen, df_gen):
        # compute and show GIS
        labels2investigate = ['Teff', '[M/H]', 'logg', 'H2O_pwv']
        desc_str = f'validate/epoch_{self.current_epoch}/GIS'
        for label2investigate in labels2investigate:
            errors = self.core_intervention_error(
                                dataset=dataset_gen,
                                df_main=df_gen,
                                labels=self.labels,
                                label2investigate=label2investigate,
                                )
            fig = plt.figure()
            plt.boxplot(errors.ravel())
            plt.title('GIS for ' + label2investigate)

            if isinstance(self.logger, NeptuneLogger):
                self.logger.experiment[desc_str].log(fig)
            elif isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(desc_str, fig)
            else:
                raise NotImplementedError
        plt.close('all')

    def high_verbose_inspection(self, dataset, name):
        self.visual_inspection(dataset, name, logging=True)
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

        if 0: # compute and show MI
            fig = plt.figure()
            MI = self.get_MIs_by_nearest_neighbor(nodes_mu.T, nodes_mu.T)
            MI_average = (np.sum(MI) - np.trace(MI))/MI.shape[0]/(MI.shape[0] - 1)
            sns.heatmap(MI, cmap='viridis', cbar=True, square=True)
            plt.title(f'average non-diagonal MI = {MI_average}')
            desc_str = f'validate/epoch_{self.current_epoch}/MI'
            if isinstance(self.logger, NeptuneLogger):
                self.logger.experiment[desc_str].log(fig)
            elif isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(desc_str, fig)
            else:
                raise NotImplementedError

    def visual_inspection(self, dataset, name, logging=False):
        """ see reconstruction for selected spectra """
        wave_resolution = 0.01
        desiredMinW = 3785
        desiredMaxW = 6910
        L = 327680
        padL = 7589
        padR = 7590

        mask = Dataset_mixin.get_artifact_mask(batch_size = 1)
        mask = mask.ravel() == 1 # to get boolean array

        st = wave_resolution
        WAVE = np.arange(desiredMinW-padL*st,desiredMaxW+.001+padR*st,step=st)

        # RECONSTRUCTION OVERVIEW
        idx = 0
        #val_dataset =  next(iter(self.trainer.val_dataloaders[0]))
        #for sample_id in np.random.choice(len(val_dataset), size=5, replace=False):
        #for sample_id in [10, 19, 2000, 3000, 4000]:
        #for sample_id in [0, 130, 543, 633, 1007]:
        for sample_id in self.overview_samples:
            data, y = dataset[sample_id]

            if logging:
                data = torch.reshape(data, (1, 1, data.shape[1])).cuda()
                # y = torch.reshape(y, (1, 1, y.shape[1])).cuda()
            else:
                data = torch.reshape(data, (1, 1, data.shape[1]))
                # y = torch.reshape(y, (1, 1, y.shape[1]))

            # loss = self.loss((data, y))
            out0, _, _, _ = self.forward(data)

            out0 = out0.detach().cpu().numpy().squeeze()
            data = data.cpu().numpy().squeeze()
            # Generate chart - overview
            fig = plt.figure(figsize=(7, 9))
            plt.plot(WAVE[mask], data[mask])
            plt.plot(WAVE[mask], out0[mask])
            plt.title(str(sample_id))

            if logging:
                desc_str = f'validate/{name}/epoch_{self.current_epoch}/reconstructions'
                if isinstance(self.logger, NeptuneLogger):
                    self.logger.experiment[desc_str].log(fig)
                elif isinstance(self.logger, TensorBoardLogger):
                    self.logger.experiment.add_figure(desc_str, fig)
                else:
                    raise NotImplementedError



            # Generate chart - detail
            fig1 = plt.figure(figsize=(7, 9))
            wave_det = np.logical_and(WAVE >= 4836, WAVE <= 4845)
            plt.plot(WAVE[wave_det], data[wave_det])
            plt.plot(WAVE[wave_det], out0[wave_det])
            plt.title(str(sample_id))

            if logging:
                desc_str = f'validate/{name}/epoch_{self.current_epoch}/reconstructions'
                if isinstance(self.logger, NeptuneLogger):
                    self.logger.experiment[desc_str].log(fig1)
                elif isinstance(self.logger, TensorBoardLogger):
                    self.logger.experiment.add_figure(desc_str, fig1)
                else:
                    raise NotImplementedError
                plt.close('all')

    def radvel_inspection(self, dataset):
        wave_resolution = 0.01
        desiredMinW = 3785
        desiredMaxW = 6910
        L = 327680
        padL = 7589
        padR = 7590

        mask = Dataset_mixin.get_artifact_mask(batch_size = 1)
        mask = mask.ravel() == 1 # to get boolean array

        st = wave_resolution
        WAVE = np.arange(desiredMinW-padL*st,desiredMaxW+.001+padR*st,step=st)

        # 
        wave_ind =  np.logical_and(WAVE >= 4836, WAVE <= 4845)
        WAVE_sub = WAVE[wave_ind]

        # RECONSTRUCTION OVERVIEW
        for sample_id in self.overview_samples:
            spectrum_0, labels_0 = dataset.getitem_w_custom_radvel(sample_id, 0)
            spectrum_0 = torch.reshape(spectrum_0, (1, 1, spectrum_0.shape[1])).cuda()
            labels_0 = torch.reshape(labels_0, (1, 1, labels_0.shape[1])).cuda()
            #loss = self.loss((spectrum_0, labels_0))
            spectrum_0, _, z_0, _ = self.forward(spectrum_0)

            rv_idx = self.labels.label2idx('radvel')['ETC']
            rv_intervention_losses = []
            for idx, radvel in enumerate(np.linspace(-30, 30, 5)):
                z_intervened = z_0.clone().detach()
                z_intervened[rv_idx] = self.labels.normalize_label('radvel', radvel)

                spectrum_gt, _ = dataset.getitem_w_custom_radvel(sample_id, radvel)

                spectrum_intervened = self.forward_labels(z_intervened)

                spectrum_gt = spectrum_gt.cpu().squeeze().numpy()
                spectrum_intervened = spectrum_intervened.cpu().squeeze().numpy()

                # compute differences
                rv_intervention_loss = np.mean(np.abs(spectrum_gt[wave_ind] - spectrum_intervened[wave_ind]))
                rv_intervention_losses.append(rv_intervention_loss)

                # plot and log figure
                fig = plt.figure(figsize=(7, 9))
                plt.plot(WAVE_sub, spectrum_gt[wave_ind], label='ground truth shift')
                plt.plot(WAVE_sub, spectrum_intervened[wave_ind], label='shift by bottleneck intervention')
                plt.legend()
                plt.title(f'sample_id: {sample_id}, RV intervention loss: {rv_intervention_loss}')

                desc_str = f'validate/epoch_{self.current_epoch}/{sample_id}/{idx}/RV_intervention'
                self.logger.experiment[desc_str].log(fig)

                plt.close('all')
    
    def radvel_overview(
                self,
                dataset,
                dataset_name,
                idxs,
                labels,
                wave_region=[6050, 6250],
                rv_shifts=np.linspace(-120, 120, 120)*u.kilometer/u.second,
                logging=False,
        ):
        _, err_per_idx_rv = self.rv_intervention_error(dataset,
                                   idxs,
                                   labels,
                                   wave_region,
                                   rv_shifts,
                                   output_per_rv=True,
                                   )

        err_per_idx_rv_ = np.array(err_per_idx_rv)
        fig = plt.figure(figsize=(9,6))
        plt.plot(rv_shifts, err_per_idx_rv_.T, '.-', label=idxs)
        plt.xlabel('rv shift [km/s]')
        plt.ylabel('MAE [-]')
        plt.title('rv intervention induced error')
        plt.legend()

        if logging:
            desc_str = f'validate/epoch_{self.current_epoch}/{dataset_name}/RV_intervention_err'
            if isinstance(self.logger, NeptuneLogger):
                self.logger.experiment[desc_str].log(fig)
            elif isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(desc_str, fig)
            else:
                raise NotImplementedError
            plt.close('all')

   

    def rv_intervention_error(
            self,
            dataset,
            idxs,
            labels,
            wave_region=[6050, 6250],
            rv_shifts=[-20, -10, 0, 10, 20]*u.kilometer/u.second,
            plot_all=False,
            output_per_rv=False,
        ):
        """
            RVIS = radial velocity intervention score
            compute average distance per pixel over several spectra and rv interventions.
            I.e:
                err[spec_idx] = E_{px, rv}[spec_rv_intervention - spec_rv_ground_truth]

            the error is computed over specific wave region to mitigate tellurics
        """
        err = []
        mask = Dataset_mixin.get_artifact_mask(1).ravel()

        if type(dataset) is torch.utils.data.dataset.Subset:
            wave = dataset.dataset.get_wave()
        else:
            wave = dataset.get_wave()

        wave_mask = np.logical_and(wave > wave_region[0], wave < wave_region[1])
        if output_per_rv:
            err_per_idx = []

        for idx in idxs:
            err_idx = 0.

            spec_in, _ = dataset[idx]
            spec_gt, _, mu, _ = self.forward(spec_in.to(self.device))
            spec_gt = spec_gt.detach().cpu().numpy().ravel()
            spec_gt[mask == 0] = 0

            # print(f'bottleneck_gt = {mu}')
            output = self.rv_intervention_with_labels(mu, rv_shifts, labels)

            if output_per_rv:
                err_per_rv = []

            for rv_idx, rv_shift in enumerate(rv_shifts):
                # not that this is rv shift on top of natural shift:
                wave_gt, spectrum_shift_gt = doppler_shift(wave, spec_gt, rv_shift)# rv shift - ground truth
                spectrum_shift_gt = np.interp(wave, wave_gt, spectrum_shift_gt)

                err_idx += np.mean(np.abs(output[rv_idx]['spectrum_intervened'][wave_mask] - spectrum_shift_gt[wave_mask]))

                if output_per_rv:
                    err_per_rv.append(np.mean(np.abs(output[rv_idx]['spectrum_intervened'][wave_mask] - spectrum_shift_gt[wave_mask])))

                if plot_all:
                    plt.figure()
                    plt.plot(wave[wave_mask], spectrum_shift_gt[wave_mask], label='ground truth')
                    plt.plot(wave[wave_mask], output[rv_idx]['spectrum_intervened'][wave_mask], label='intervention')
                    # plt.plot(wave[wave_mask], spec_gt.detach().numpy().ravel()[wave_mask])
                    # plt.plot(wave, spec_gt.detach().numpy().ravel())
                    # plt.plot(wave[wave_mask], spec_in.ravel()[wave_mask])
                    plt.legend()
                    plt.savefig('/tmp/intervention_' + str(idx) + "_" + str(rv_idx) + '.png')
            err.append(err_idx/len(rv_shifts))

            if output_per_rv:
                err_per_idx.append(err_per_rv)

        if output_per_rv:
            return np.array(err), err_per_idx
        else:
            return np.array(err)

    def rv_intervention_error_for_simulation(
            self,
            dataset,
            idxs,
            labels,
            wave_region=[6050, 6250],
            rv_shifts=[-20, -10, 0, 10, 20]*u.kilometer/u.second,
            plot_all=False,
        ):
        """
            compute average distance per pixel over several spectra and rv interventions.
            I.e:
                err = E_{px, rv, spec_idx}[spec_rv_intervention - spec_rv_ground_truth]

            the error is computed over specific wave region to mitigate tellurics
        """

        err = []
        mask = Dataset_mixin.get_artifact_mask(1).ravel()

        if type(dataset) is torch.utils.data.dataset.Subset:
            wave = dataset.dataset.get_wave()
        else:
            wave = dataset.get_wave()

        wave_mask = np.logical_and(wave > wave_region[0], wave < wave_region[1])
        for idx in idxs:
            err_idx = 0.

            _, bottleneck = dataset[idx]
            spec_gt = self.forward_labels(bottleneck)
            spec_gt = spec_gt.detach().numpy().ravel()
            spec_gt[mask == 0] = 0

            # print(f'bottleneck_gt = {mu}')
            output = self.rv_intervention_with_labels(bottleneck, rv_shifts, labels)
            for rv_idx, rv_shift in enumerate(rv_shifts):
                # not that this is rv shift on top of natural shift:
                wave_gt, spectrum_shift_gt = doppler_shift(wave, spec_gt, rv_shift)# rv shift - ground truth
                spectrum_shift_gt = np.interp(wave, wave_gt, spectrum_shift_gt)

                err_idx += np.mean(np.abs(output[rv_idx]['spectrum_intervened'][wave_mask] - spectrum_shift_gt[wave_mask]))
                if plot_all:
                    plt.figure()
                    plt.plot(wave[wave_mask], spectrum_shift_gt[wave_mask], label='ground truth')
                    plt.plot(wave[wave_mask], output[rv_idx]['spectrum_intervened'][wave_mask], label='intervention')
                    # plt.plot(wave[wave_mask], spec_gt.detach().numpy().ravel()[wave_mask])
                    # plt.plot(wave, spec_gt.detach().numpy().ravel())
                    # plt.plot(wave[wave_mask], spec_in.ravel()[wave_mask])
                    plt.legend()
                    plt.title(output[rv_idx]['rv'])
                    plt.savefig('/tmp/intervention_' + str(idx) + "_" + str(rv_idx) + '.png')
            err.append(err_idx/len(rv_shifts))
        return np.array(err)

    def reconstruction_overview(self, dataset, batch_size=32):
        # rec_err = [] 
        # # for i in tqdm(range(100)):
        # for i in tqdm(range(len(dataset))):
        #     spectrum_in, spectrum_out, _ = self.plot_preprocess_idx(dataset, i)
        #     rec_err.append(np.median(np.abs(spectrum_in - spectrum_out)))
        # return np.array(rec_err)
        """
            accelareted computation of reconstruction error (mean absolute error)
        """
        class Dataset_with_idx(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __getitem__(self, idx):
                return self.dataset[idx][0], idx

            def __len__(self):
                return len(self.dataset)
        
        ds = Dataset_with_idx(dataset)
        dl = torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size)
        rec_err = np.full((len(dataset),), np.nan)
        with torch.no_grad():
            for (spectra_in), idxs in tqdm(iter(dl)):
                spectra_in = spectra_in.to(self.device)
                spec_out, _, _, _ = self(spectra_in)
                rec_err_batch = self.EPE_my(spec_out, spectra_in, mean=True, tiled_mask = self.tiled_mask)
                rec_err_batch = rec_err_batch.detach().cpu().numpy().ravel()
                rec_err[idxs] = rec_err_batch
                        
        return rec_err

    def simulations_reconstruction_overview(self, dataset, batch_size=32):
        """
            accelareted computation of reconstruction error (mean absolute error)
            (for simulations only)
        """
        class Dataset_with_idx(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __getitem__(self, idx):
                return self.dataset[idx], idx

            def __len__(self):
                return len(self.dataset)
        
        ds = Dataset_with_idx(dataset)
        dl = torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size)
        rec_err = np.full((len(dataset),), np.nan)
        with torch.no_grad():
            for (spectra_in, labels), idxs in tqdm(iter(dl)):
                spectra_in = spectra_in.to(self.device)
                labels = labels.to(self.device)
                spec_out = self(labels)
                rec_err_batch = self.EPE_my(spec_out, spectra_in, mean=True, tiled_mask = self.tiled_mask)
                rec_err_batch = rec_err_batch.detach().cpu().numpy().ravel()
                rec_err[idxs] = rec_err_batch
                        
        return rec_err



    def predictions_per_index(self, dataset, batch_size=32):
        """
            accelareted function to iterate through an input dataset

            returns ground truth labels and respective predictions (normalized, as seen by bottleneck)
        """
        class Dataset_with_idx(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __getitem__(self, idx):
                return self.dataset[idx], idx

            def __len__(self):
                return len(self.dataset)
        
        _, labels_in = dataset[0]
        labels_num_in_bottleneck = labels_in.shape[1]

        ds = Dataset_with_idx(dataset)
        dl = torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size)
        labels_predict = np.full((self.bottleneck, len(dataset)), np.nan)
        labels_gt = np.full((labels_num_in_bottleneck, len(dataset)), np.nan)
        with torch.no_grad():
            for (spectra_in, labels_in), idxs in tqdm(iter(dl)):
                _, _, mu, _ = self(spectra_in.to(self.device))
                mu = mu.detach().cpu().numpy()
                if len(idxs) == 1:
                    labels_predict[:, idxs] = mu.T.ravel()
                    labels_gt[:, idxs] = torch.reshape(labels_in, [labels_in.shape[0], -1]).detach().cpu().numpy().T.ravel()
                else:
                    labels_predict[:, idxs] = mu.T
                    labels_gt[:, idxs] = torch.reshape(labels_in, [labels_in.shape[0], -1]).detach().cpu().numpy().T
                
        
        return labels_gt, labels_predict


    def get_nodes(self, dataset, batch_size=32):
        nodes_mu, nodes_logvar, _ = self.get_nodes_w_custom_labels(dataset, batch_size=batch_size, labels=None)
        return nodes_mu, nodes_logvar

    def get_nodes_w_custom_labels(self, dataset, batch_size=32, labels=None):
        class Dataset_with_idx(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __getitem__(self, idx):
                return self.dataset[idx], idx

            def __len__(self):
                return len(self.dataset)
        
        if labels is not None: # set custom labels
            # check if dataset is subset
            if type(dataset) is torch.utils.data.dataset.Subset:
                if hasattr(dataset.dataset, 'labels'):
                    original_labels_cls = dataset.dataset.labels_norm
                else:
                    original_labels_cls = None
                dataset.dataset.labels_norm = labels
            else:
                if hasattr(dataset, 'labels'):
                    original_labels_cls = dataset.labels_norm
                else:
                    original_labels_cls = None
                dataset.labels_norm = labels

        ds = Dataset_with_idx(dataset)
        dl = torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size)
        
        nodes_mu = np.full((self.bottleneck, len(dataset)), np.nan)
        nodes_logvar = np.full((self.bottleneck, len(dataset)), np.nan)

        if labels is not None:
            labels_gt = np.full((len(labels.labels), len(dataset)), np.nan)
        with torch.no_grad():
            for (spectra_in, labels_in), idxs in tqdm(iter(dl)):
                _, _, mu, logvar = self(spectra_in.to(self.device))
                mu = mu.detach().cpu().numpy()
                logvar = logvar.detach().cpu().numpy()
                if labels is not None:
                    labels_in = labels_in.detach().cpu().numpy()[:,0,:]

                if len(idxs) == 1:
                    nodes_mu[:, idxs] = mu.T.ravel()
                    nodes_logvar[:, idxs] = logvar.T.ravel()
                    if labels is not None:
                        labels_gt[:, idxs] = labels_in.T.ravel()
                else:
                    nodes_mu[:, idxs] = mu.T
                    nodes_logvar[:, idxs] = logvar.T
                    if labels is not None:
                        labels_gt[:, idxs] = labels_in.T

        if labels is not None: # unset custom labels
            if original_labels_cls is not None:
                if type(dataset) is torch.utils.data.dataset.Subset:
                    dataset.dataset.labels_norm = original_labels_cls 
                else:
                    dataset.labels_norm = original_labels_cls 

            return nodes_mu, nodes_logvar, labels_gt
        else:
            return nodes_mu, nodes_logvar, None 

    def get_active_nodes(self, nodes_mu, active_thrsh):
        # check if the nodes are active
        return np.abs(np.median(nodes_mu, axis=1)) > active_thrsh

    def get_MIs_by_nearest_neighbor(self, X, Y):
        # compute mutual information table
        # 
        # I can use it to: 
        #   MI between ground truth (X) and nodes (Y)
        #   MI between supervised nodes (X) and unsupervised nodes (Y)
        #   MI between nodes (X = Y)
        assert(X.shape[0] == Y.shape[0]) # same number of observations
        if X.shape[0] < X.shape[1] and X.shape[1] == Y.shape[1]:
            print('Warning: there are fewer observations than features. Are you sure the matrix is not transposed?')

        mutual_info = np.zeros((X.shape[1], Y.shape[1]))
        for j in range(Y.shape[1]):        
            Y_j = Y[:, j]
            not_nan_indices = ~np.isnan(Y_j)
            if np.sum(not_nan_indices) != 0:        
                Y_j = Y_j[not_nan_indices]
                X_ = X[not_nan_indices, :]
                mutual_info[:, j] = mutual_info_regression(X_, Y_j, n_neighbors=5)

        return mutual_info

    def compute_total_correlation_kde(self, data):
        if data.shape[0] < data.shape[1]:
            print('there is less observations than features, are you sure the matrix is not transposed?')

        # Fit a Gaussian KDE to the joint data
        joint_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data)

        # Compute the joint log-probability density for each observation
        joint_log_probs = joint_kde.score_samples(data)

        # Fit Gaussian KDEs for each feature
        marginal_kdes = [KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data[:, i].reshape(-1, 1)) for i in range(data.shape[1])]
        
        # Compute the sum of marginal log-probability densities for each observation
        marginal_log_probs_sum = np.sum([marginal_kde.score_samples(data[:, i].reshape(-1, 1)) for i, marginal_kde in enumerate(marginal_kdes)], axis=0)

        # Compute the KL divergence between the joint and product of marginal distributions
        kl_divergence = np.mean(joint_log_probs - marginal_log_probs_sum)

        # Convert KL divergence to total correlation (nats to bits)
        total_correlation = kl_divergence * np.log2(np.e)

        return total_correlation
    
    def core_intervention_error(self,
                                dataset,
                                df_main,
                                labels,
                                label2investigate,
                                ):
        """
            summarize label intervention for generative datasets (simulated only)

            generative dataset contains few core samples with uniformly distributed parameters.
            For each core sample, we precompute modification in several parameter.

            Ie. if there is C cores, P parameters we are modifying, and V values for each parameter,
            we have C*P*V spectra.

            input:
            dataset .. generative dataset (core samples + ground truth interventions)
            df_main .. dataframe associated with the dataset
            labels .. class handling labels, it is responsible for correctly indexing bottleneck and
                    transformations between bottleneck and catalog
            label2investigate .. label that is controlled by the slider

            output:
            error .. error of the intervention

        """
        assert('core' in df_main.columns)
        cores_idxs = self.cores_idxs(df_main)
        errs = []
        for core_idx in cores_idxs:
            assert(df_main.iloc[core_idx].core)
            if 'simulator_NN' in str(type(self)):
                # decoders
                bottleneck = dataset[core_idx][1]
            else:
                # autoencoders
                _, _, bottleneck, _ = self.forward(dataset[core_idx][0].to(self.device))

            if type(dataset) is torch.utils.data.dataset.Subset:
                wave = dataset.dataset.get_wave()
            else:
                wave = dataset.get_wave()

            mask = Dataset_mixin.get_artifact_mask(1).ravel()

            idxs = self.core_and_label2idxs(df_main, core_idx, label2investigate)
            label_shifts = df_main.iloc[idxs][label2investigate]
            core_value = df_main.iloc[core_idx][label2investigate]
            print(f'core {label2investigate} ground truth value: {core_value}')

            for label_shift in label_shifts:
                output = self.label_intervention(bottleneck, label2investigate, [label_shift - core_value], labels)[0]
                spec_int = output['spectrum_intervened']
                spec_int[mask == 0] = 0

                idx = np.flatnonzero(df_main[label2investigate] == label_shift)[0]

                if 'simulator_NN' in str(type(self)):
                    # decoders
                    bottleneck_gt = dataset[idx][1]
                else:
                    # autoencoders
                    _, _, bottleneck_gt, _ = self.forward(dataset[idx][0].to(self.device))

                spectrum_gt = self.forward_labels(bottleneck_gt).detach().cpu().numpy().ravel() # shift by intervention
                spectrum_gt[mask == 0] = 0

                errs.append(np.mean(np.abs(spectrum_gt - spec_int)))

                if 0:
                    set_trace()
                    plt.figure()
                    plt.plot(wave, spectrum_gt, label='ground truth spectrum')
                    plt.plot(wave, spec_int, label='intervened spectrum')
                    plt.legend() 
        
        return np.array(errs)