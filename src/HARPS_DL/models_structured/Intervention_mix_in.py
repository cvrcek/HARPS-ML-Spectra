from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import torch

from pdb import set_trace
from HARPS_DL.datasets.Dataset_mixin import Dataset_mixin

from HARPS_DL.tools.spectra_tools import doppler_shift

from ipywidgets import interact

class Intervention_mix_in():
    """
        mixin class with qualitative tools for models
    """
    def __rv_intervention(self, bottleneck, rv_values, rv_index):
        """
            given a bottleneck tensor, set the values for the bottleneck[index] and return the resulting spectra.
        """
        intervention_data = []
        for rv_value in rv_values:
            bottleneck_intervention = bottleneck.detach().clone()
            bottleneck_intervention[0,rv_index] = rv_value 
            spectrum_shift_intervention = self.forward_labels(bottleneck_intervention) # shift by intervention
            intervention_data.append(
                {'rv': rv_value.detach().cpu().numpy(),
                'spectrum_intervened': spectrum_shift_intervention.detach().cpu().numpy().ravel(),
                }
            )
        return intervention_data

    def rv_intervention_with_labels(self, bottleneck, rv_shifts, labels):
        """
            given a bottleneck tensor, shift the values for the bottleneck[index] and return the resulting spectra.

            Input is assumed to be in km/s (astropy units class).
            Normalization and rv index in bottleneck is deduced from the labels class

            If the model wasn't trained using the labels class, results will be bad!
            Model loaded using the config.yaml should be safe

            input:
            bottleneck .. autoencoder (or VAE) bottleneck
            rv_shifts .. desired radial velocity shifts
            labels .. class handling labels, it is responsible for correctly indexing bottleneck and transformation
                      between bottleneck (z-score) and radial velocity units [km/s]

        """
        assert(labels.label2idx('radvel')['real'] == labels.label2idx('radvel')['ETC']) # typical situation
        assert(bottleneck.shape[0] == 1) # assumption about the shape

        rv_index = labels.label2idx('radvel')['ETC']
        rv_values_norm = []
        bottleneck_rv = bottleneck[0,rv_index].detach().clone()

        for i in range(len(rv_shifts)):
            assert(rv_shifts[i].unit == u.kilometer/u.second)
            rv_values_norm.append(bottleneck_rv + labels.scale_label('radvel', rv_shifts[i].value))

        intervention_data = self.__rv_intervention(bottleneck, rv_values_norm, rv_index)
        return intervention_data
    
    def rv_intervention_interactive_bottleneck(self, bottleneck, spectrum_gt, labels, wave, wave_range,
                                               rv_shifts=np.linspace(-10, 10, 30)*u.kilometer/u.second):
        """
            NOTEBOOK FUNCTION!!!
            visually investigate rv intervention
        """
        mask = Dataset_mixin.get_artifact_mask(1).ravel()
        wave_sel = np.logical_and(wave > wave_range[0], wave < wave_range[1])

        def fun(rv_shift):
            output = self.rv_intervention_with_labels(bottleneck, [rv_shift], labels)[0]
            spec_int = output['spectrum_intervened']
            spec_int[mask == 0] = 0
            rv_val = output['rv']

            wave_gt, spectrum_shift_gt = doppler_shift(wave, spectrum_gt, rv_shift)# rv shift - ground truth
            spectrum_shift_gt = np.interp(wave, wave_gt, spectrum_shift_gt.detach().numpy().ravel())
            spectrum_shift_gt[mask == 0] = 0

            plt.figure()
            plt.plot(wave[wave_sel], spectrum_shift_gt[wave_sel], label='gt')
            plt.plot(wave[wave_sel], spec_int[wave_sel], label='inter')
            plt.title(f'rv = {rv_val}')
            plt.legend()
        
        interact(fun, rv_shift=rv_shifts)

    def rv_intervention_interactive(self,
                                    dataset,
                                    idx,
                                    labels,
                                    wave_range=(6120, 6140),
                                    rv_shifts=np.linspace(-10, 10, 30)*u.kilometer/u.second):
        """
            NOTEBOOK FUNCTION!!!
            visually investigate rv intervention

            intervention means modification of the bottleneck (radial velocity node),
            and observing the resulting spectrum (spectrum = modified bottleneck | decoder )

            The function is used in notebook to determine how well does the (V)AE generalizes RV.
            

            input:
            dataset .. input dataset
            idx .. spectrum index
            labels .. class handling labels, it is responsible for correctly indexing bottleneck and transformation
                      between bottleneck (z-score) and radial velocity units [km/s]
            wave_range .. visualization range
        """

        spectrum_gt, _, bottleneck, _ = self.forward(dataset[idx][0])


        if type(dataset) is torch.utils.data.dataset.Subset:
            wave = dataset.dataset.get_wave()
        else:
            wave = dataset.get_wave()

        self.rv_intervention_interactive_bottleneck(bottleneck, spectrum_gt, labels, wave, wave_range, rv_shifts=rv_shifts)

    def __label_intervention(self, bottleneck, values, labels, label2slide):
        """
            given a bottleneck tensor, set the values for the bottleneck[index] and return the resulting spectra.
        """
        label_index = labels.label2idx(label2slide)['real']
        intervention_data = []
        for value in values:
            bottleneck_intervention = bottleneck.detach().clone()
            bottleneck_intervention[0,label_index] = value 
            spectrum_shift_intervention = self.forward_labels(bottleneck_intervention) # shift by intervention
            val = value.detach().cpu().numpy()
            intervention_data.append(
                {'label_val': val,
                'spectrum_intervened': spectrum_shift_intervention.detach().cpu().numpy().ravel(),
                'label_val_ori': labels.inverse_label_normalization(val, label2slide)
                }
            )
        return intervention_data

    def label_intervention(self, bottleneck, label2slide, label_shifts, labels):
        """
            given a bottleneck tensor, shift the values for the bottleneck[index] and return the resulting spectra.


            Normalization and label index in bottleneck is deduced from the labels class

            If the model wasn't trained using the labels class, results will be bad!
            Model loaded using the config.yaml should be safe

            i.e.:
            bottleneck[label2slide] += scale(label_shifts, label2slide)
            where scale searches for the correct scale in labels class
            not that normalizing shift is incorrect for shifting!

            input:
            bottleneck .. autoencoder (or VAE) bottleneck
            rv_shifts .. desired radial velocity shifts
            labels .. class handling labels, it is responsible for correctly indexing bottleneck and transformation
                      between bottleneck (z-score) and radial velocity units [km/s]
            output:
            intervention_data .. list of dictionaries with keys
                label_val .. value of the label in the bottleneck
                spectrum_intervened .. spectrum after the intervention
                label_val_ori .. original value of the label (catalog units)

        """
        assert(bottleneck.shape[0] == 1) # assumption about the shape

        label_index = labels.label2idx(label2slide)['real']
        label_values_norm = []
        bottleneck_label = bottleneck[0,label_index].detach().clone()

        for i in range(len(label_shifts)):
            label_values_norm.append(bottleneck_label +
                                     labels.scale_label(label2slide,
                                                        label_shifts[i]))

        intervention_data = self.__label_intervention(bottleneck,
                                                      label_values_norm,
                                                      labels,
                                                      label2slide,
                                                      )
        return intervention_data
 
    
    def label_intervention_interactive_bottleneck(self,
                                               bottleneck,
                                               spectrum_rest,
                                               labels,
                                               label2slide,
                                               wave,
                                               wave_range,
                                               slide_values):
        """
            NOTEBOOK FUNCTION!!!
            visually investigate label intervention
        """
        mask = Dataset_mixin.get_artifact_mask(1).ravel()
        wave_sel = np.logical_and(wave > wave_range[0], wave < wave_range[1])

        def fun(label_shift):
            output = self.label_intervention(bottleneck, label2slide, [label_shift], labels)[0]
            spec_int = output['spectrum_intervened']
            spec_int[mask == 0] = 0
            label_val = output['label_val']
            label_val_ori = output['label_val_ori']


            plt.figure()
            plt.plot(wave[wave_sel], spectrum_rest[wave_sel], label='rest spectrum')
            plt.plot(wave[wave_sel], spec_int[wave_sel], label='intervened spectrum')
            plt.title(f'{label2slide} = {label_val:.5f} | {label_val_ori:.5f} (Z-score | catalog value)')
            plt.legend()
        
        interact(fun, label_shift=slide_values)

    def label_intervention_interactive(self,
                                    dataset,
                                    idx,
                                    labels,
                                    label2slide,
                                    wave_range=(6120, 6140),
                                    slide_values=np.linspace(-10, 10, 30),
                                    ):
        """
            NOTEBOOK FUNCTION!!!
            visually investigate some label intervention

            intervention means modification of the bottleneck (associated label node),
            and observing the resulting spectrum (spectrum = modified bottleneck | decoder )

            The function is used in notebook to observe sliding.
            

            input:
            dataset .. input dataset
            idx .. spectrum index
            labels .. class handling labels, it is responsible for correctly indexing bottleneck and transformation
                      between bottleneck (z-score) and radial velocity units [km/s]
            label2slide .. label to slide
            wave_range .. visualization range
            slide_values .. which values does slider contain
        """

        spectrum_gt, _, bottleneck, _ = self.forward(dataset[idx][0])
        spectrum_gt = spectrum_gt.detach().cpu().numpy().ravel()


        if type(dataset) is torch.utils.data.dataset.Subset:
            wave = dataset.dataset.get_wave()
        else:
            wave = dataset.get_wave()

        self.label_intervention_interactive_bottleneck(bottleneck,
                                                       spectrum_gt,
                                                       labels,
                                                       label2slide,
                                                       wave,
                                                       wave_range,
                                                       slide_values=slide_values)
    
    
    def plot_preprocess_idx(self, dataset, idx):
        """
            utility function to provide input and output with applied mask and wave
            can be used for direct plotting
        """
        mask = self.get_mask().numpy().ravel()
        spectrum_in = dataset[idx][0].numpy().ravel()
        spectrum = self(dataset[idx][0])[0].detach().numpy().ravel() 
        spectrum[mask == 0] = 0
        spectrum_in[mask == 0] = 0

        if type(dataset) is torch.utils.data.dataset.Subset:
            wave = dataset.dataset.get_wave()
        else:
            wave = dataset.get_wave()

        return spectrum_in, spectrum, wave

    def cores_idxs(self, df):
        assert('core' in df.columns)
        return df[df['core'] == True].index
    
    def core_and_label2idxs(self, df, core_idx, label):
        # given core idx, and target label obtain idxs of samples with the same core,
        # where the target label is different
        assert(df.iloc[core_idx].core)
        assert('core' in df.columns)
        assert(label in df.columns)

        core = df.iloc[core_idx]

        df_without_label = df.drop(label, axis=1)
        core_without_label = core.drop(label)

        # get samples based on core
        mask_1 = (df != core).sum(axis=1) == 3 # select samples sharing the same core
        mask_2 = (df_without_label != core_without_label).sum(axis=1) == 2 # select samples that differ only in the input label 
        mask = np.logical_and(mask_1, mask_2)
        mask[core_idx] = 1 # include the core itself

        return np.flatnonzero(mask)

    def core_intervention_interactive_bottleneck(self,
                                                dataset,
                                                df_main,
                                                core_idx,
                                                labels,
                                                label2slide,
                                                wave_range,
                                                ):
        """
            NOTEBOOK FUNCTION!!!
            visually investigate label intervention for generative datasets

            generative dataset contains few core samples with uniformly distributed parameters.
            For each core sample, we precompute modification in several parameter.

            Ie. if there is C cores, P parameters we are modifying, and V values for each parameter,
            we have C*P*V spectra.

            input:
            dataset .. generative dataset (core samples + ground truth interventions)
            df_main .. dataframe associated with the dataset
            core_idx .. index of the core sample in df_main
            labels .. class handling labels, it is responsible for correctly indexing bottleneck and
                    transformations between bottleneck and catalog
            label2slide .. label that is controlled by the slider
            wave_range .. visualization range

        """
        assert('core' in df_main.columns)
        assert(df_main.iloc[core_idx].core)

#        set_trace()
        if 'simulator_NN' in str(type(self)):
            # decoders
            bottleneck = dataset[core_idx][1]
        else:
            # autoencoders
            _, _, bottleneck, _ = self.forward(dataset[core_idx][0])

        if type(dataset) is torch.utils.data.dataset.Subset:
            wave = dataset.dataset.get_wave()
        else:
            wave = dataset.get_wave()

        mask = Dataset_mixin.get_artifact_mask(1).ravel()
        wave_sel = np.logical_and(wave > wave_range[0], wave < wave_range[1])

        idxs = self.core_and_label2idxs(df_main, core_idx, label2slide)
        slide_values = df_main.iloc[idxs][label2slide].sort_values()
        core_value = df_main.iloc[core_idx][label2slide]
        print(f'core {label2slide} ground truth value: {core_value}')

        label_index = labels.label2idx(label2slide)['real']
        bottleneck_val = labels.inverse_label_normalization(
            bottleneck.detach().cpu().numpy().ravel()[label_index],
            label2slide)
        print(f'core {label2slide} predicted value: {bottleneck_val}')

        print('intervention: label_shift - core_value_gt')
        print('title label: bottleneck value (prediction + intervention)')
        def fun(label_shift):
            print(f'intervention is {label_shift - core_value}')
            output = self.label_intervention(bottleneck, label2slide, [label_shift - core_value], labels)[0]
            spec_int = output['spectrum_intervened']
            spec_int[mask == 0] = 0
            label_val = output['label_val']
            label_val_ori = output['label_val_ori']

            idx = np.flatnonzero(df_main[label2slide] == label_shift)[0]

            if 'simulator_NN' in str(type(self)):
                # decoders
                bottleneck_gt = dataset[idx][1]
            else:
                # autoencoders
                _, _, bottleneck_gt, _ = self.forward(dataset[idx][0])

            spectrum_gt = self.forward_labels(bottleneck_gt).detach().cpu().numpy().ravel() # shift by intervention

            plt.figure()
            plt.plot(wave[wave_sel], spectrum_gt[wave_sel], label='ground truth spectrum')
            plt.plot(wave[wave_sel], spec_int[wave_sel], label='intervened spectrum')
            plt.title(f'{label2slide} = {label_val:.5f} | {label_val_ori:.5f} (Z-score | catalog value)')
            plt.legend()
        
        interact(fun, label_shift=slide_values)