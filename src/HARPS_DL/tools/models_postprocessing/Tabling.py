from sys import displayhook
import numpy as np
import math
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from pdb import set_trace

from HARPS_DL.project_config import MODELS_BANK

class Tabling():
    def get_table(self,
                aggregation='mean',
                spread_measure='mad',
                labels_sel=None,
                summary_err=False,
                counts=[]):
        descs = self.get_descs()
        df_err = pd.DataFrame(index=descs)
        df_spread = pd.DataFrame(index=descs)

        if summary_err:
            df_all_err = pd.DataFrame(index=descs)
            df_all_spread = pd.DataFrame(index=descs)

        labels_all = []
        for model_dict in self.models:
            labels_all += model_dict['labels'].labels
        labels_all = np.unique(labels_all)
        labels_all = labels_all[labels_all != 'H2O_pwv']

        if labels_sel is not None:
            labels_all = labels_sel

        for label in labels_all:
            for model_dict in self.models:
                if aggregation == 'mean':
                    label_err = np.nanmean(model_dict['err'].get(label, np.nan))
                elif aggregation == 'median':
                    label_err = np.nanmedian(model_dict['err'].get(label, np.nan))
                else:
                    raise ValueError('aggregation must be mean or median')

                if spread_measure == 'std':
                    label_spread = np.nanstd(model_dict['err'].get(label, np.nan))
                elif spread_measure == 'mad':
                    label_median = np.nanmedian(model_dict['err'].get(label, np.nan))
                    label_spread = np.nanmedian(np.abs(model_dict['err'].get(label, np.nan).ravel() - label_median))
                else:
                    raise ValueError('spread_measure must be std or mad')

                df_err.loc[model_dict['desc'], label] = label_err
                df_spread.loc[model_dict['desc'], label] = label_spread

                if summary_err:
                    if aggregation == 'mean':
                        label_err = np.nanmean(model_dict['err_normalized'].get(label, np.nan))  # Changed 'err_norm' to 'err_normalized'
                    elif aggregation == 'median':
                        label_err = np.nanmedian(model_dict['err_normalized'].get(label, np.nan))
                    else:
                        raise ValueError('aggregation must be mean or median')

                    if spread_measure == 'std':
                        label_spread = np.nanstd(model_dict['err_normalized'].get(label, np.nan))
                    elif spread_measure == 'mad':
                        label_median = np.nanmedian(model_dict['err_normalized'].get(label, np.nan))
                        label_spread = np.nanmedian(np.abs(model_dict['err_normalized'].get(label, np.nan).ravel() - label_median))
                    else:
                        raise ValueError('spread_measure must be std or mad')

                    df_all_err.loc[model_dict['desc'], label] = label_err
                    df_all_spread.loc[model_dict['desc'], label] = label_spread

        if summary_err:
            labels_all = df_all_err.mean(axis=1)
            labels_all_spread = df_all_spread.mean(axis=1)
            df_err['all'] = labels_all
            df_spread['all'] = labels_all_spread
            #df_err['relative'] = (labels_all - np.min(labels_all)) / np.min(labels_all)
        if len(counts) > 0:
            sum = 0
            for label in labels_sel:
                df_spread.loc[:, label] = df_spread.loc[:, label] / np.sqrt(counts[label])
                sum += counts[label]
            df_spread.loc[:, 'all'] = df_spread.loc[:, 'all'] / np.sqrt(sum)


        return df_err, df_spread

    def get_table_norm(self, aggregation='mean', spread_measure='mad'):
        descs = self.get_descs()
        df_err = pd.DataFrame(index=descs)
        df_spread = pd.DataFrame(index=descs)

        labels_all = self.get_labels_all()

        for label in labels_all:
            for model_dict in self.models:
                if aggregation == 'mean':
                    label_err = np.nanmean(model_dict['err_normalized'].get(label, np.nan))  # Changed 'err_norm' to 'err_normalized'
                elif aggregation == 'median':
                    label_err = np.nanmedian(model_dict['err_normalized'].get(label, np.nan))
                else:
                    raise ValueError('aggregation must be mean or median')

                if spread_measure == 'std':
                    label_spread = np.nanstd(model_dict['err_normalized'].get(label, np.nan))
                elif spread_measure == 'mad':
                    label_median = np.nanmedian(model_dict['err_normalized'].get(label, np.nan))
                    if np.isnan(label_median):
                        label_spread = np.nan
                    else:
                        label_spread = np.nanmedian(np.abs(model_dict['err_normalized'].get(label, np.nan).ravel() - label_median))
                else:
                    raise ValueError('spread_measure must be std or mad')

                df_err.loc[model_dict['desc'], label] = label_err
                df_spread.loc[model_dict['desc'], label] = label_spread

        labels_all = df_err.mean(axis=1)
        labels_subset = df_err[['Teff', '[M/H]', 'logg']].mean(axis=1)

        df_err['all'] = labels_all
        df_err['subset'] = labels_subset
        df_err['relative'] = (labels_all - np.min(labels_all)) / np.min(labels_all)

        return df_err, df_spread


    def get_label_significance(self, errors, model_names, alpha=0.05, verbose=False):
        # Determine the best model based on mean error (assuming you meant mean since you calculate means)
        means = [np.mean(error) for error in errors]
        best_model_index = np.argmin(means)
        best_model_name = model_names[best_model_index]

        # Calculate the number of comparisons
        num_models = len(errors)
        num_comparisons = num_models - 1  # Comparisons of the best model with each other model

        # Store p-values and corresponding model pairs for comparisons
        p_values = []
        for i, model_error in enumerate(errors):
            if i == best_model_index:
                continue
            stat, p = stats.mannwhitneyu(errors[best_model_index], model_error, alternative='two-sided')
            p_values.append((p, i))

        # Sort p-values in ascending order
        p_values.sort()

        # Apply Holm-Bonferroni correction
        significant_differences = []
        for rank, (p, i) in enumerate(p_values, start=1):
            threshold = alpha / (num_comparisons - rank + 1)
            if p < threshold:
                significant_differences.append((best_model_name, model_names[i], p))
            else:
                # Stop checking further as subsequent p-values are larger
                significant_differences = None
                break

        # Reporting
        if verbose:
            if significant_differences:
                print(f"Best model for the label: {best_model_name}. Significantly better than:")
                for diff in significant_differences:
                    print(f"- {model_names[diff[1]]} (p = {diff[2]:.4f})")
            else:
                print(f"Best model for the label: {best_model_name}, but not significantly better than all others based on Holm-Bonferroni correction.")

        if significant_differences:
            return best_model_name, significant_differences
        else:
            return best_model_name, None

    def get_table_significance(self, labels_sel=None, alpha=0.05, verbose=False):
        displayhook()
        descs = self.get_descs()
        labels_all = self.get_labels_all()

        if labels_sel is not None:
            labels_all = labels_sel

        df_sig = pd.DataFrame(index=descs)

        # init to empty list with descs as keys
        label_all_err = {desc: [] for desc in descs} 
        for label in labels_all:
            label_err_all_models = []
            model_names = []
            for model_dict in self.models:
                label_err = model_dict['err_normalized'].get(label, np.nan)
                if isinstance(label_err, pd.core.series.Series):
                    label_err = label_err.dropna()
                    if not label_err.empty:
                        label_err_all_models.append(label_err)
                        model_names.append(model_dict['desc'])
                        label_all_err[model_dict['desc']].extend(label_err)

            if len(label_err_all_models) <= 1:
                continue

            best_model_name, significant_differences = self.get_label_significance(
                label_err_all_models, model_names, alpha=alpha, verbose=verbose)

            # Reporting
            df_sig.loc[:, label] = 0
            if significant_differences:
                if verbose:
                    print(f"Best model for {label}: {best_model_name}. Significantly better than:")
                    for diff in significant_differences:
                        print(f"- {diff[1]} (p = {diff[2]:.4f})")
                df_sig.loc[best_model_name, label] = 1
            else:
                if verbose:
                    print(f"Best model for {label}: {best_model_name}, but not significantly better than all others.")

        # label_all_err to list of lists
        label_all_err = [label_all_err[model_name] for model_name in model_names]

        best_model_name, significant_differences = self.get_label_significance(
            label_all_err, model_names, alpha=alpha, verbose=verbose)

        # Reporting
        df_sig.loc[:, 'all'] = 0
        if significant_differences:
            if verbose:
                print(f"Best model for all: {best_model_name}. Significantly better than:")
                for diff in significant_differences:
                    print(f"- {diff[1]} (p = {diff[2]:.4f})")
            df_sig.loc[best_model_name, 'all'] = 1
        else:
            if verbose:
                print(f"Best model for all: {best_model_name}, but not significantly better than all others.")

        return df_sig



    def combine_tables(self, df_mean, df_spread, df_sig=None):
        df_combined = pd.DataFrame(index=df_mean.index)
        for column in df_mean.columns:
            combined_values = []
            for index in df_mean.index:
                # Get the significant digits for rounding
                if np.isnan(df_spread.loc[index, column]):
                    combined_values.append(f"NaN ± NaN")
                else:
                    round_digits = -int(np.floor(np.log10(abs(df_spread.loc[index, column])))) + 1

                    # Round the mean and std according to the significant digits
                    mean_rounded = round(df_mean.loc[index, column], round_digits)
                    std_rounded = round(df_spread.loc[index, column], round_digits)

                    # Combine them into a single string
                    combined_values.append(f"{mean_rounded} ± {std_rounded}")

            df_combined[column] = combined_values
        #apply modification where df_sig is 1
        if df_sig is not None:
            df_combined = df_combined.where(df_sig == 0,
                                            '\\textbf{' + df_combined + '}')
        return df_combined

    def add_units2table(self, df):
        df = df.rename(columns={
            'Teff': 'Teff (K)',
            '[M/H]': '[M/H] (dex)',
            'logg': 'logg (dex)',
            'radvel': 'Radvel (\si{\kilo\meter\per\second})',
            'BERV': 'BERV (\si{\kilo\meter\per\second})',
            'airmass': 'Airmass (-)',
            'all': 'All (-)',
            })
        return df
        
    def timing_table(self):
        descs = self.get_descs()
        df_timing = pd.DataFrame(index=descs)

        for model_dict in self.models:
            cpu_time = model_dict['time_CPU'] * 100
            gpu_time = model_dict['time_GPU'] * 100

            # Function to round to the second most significant digit
            def round_second_significant_digit(value):
                if value == 0:
                    return 0
                else:
                    # Determine the number of digits to the left of the decimal point
                    digits = int(math.floor(math.log10(abs(value)))) + 1
                    # Round to the second most significant digit
                    return round(value, -digits + 2)

            df_timing.loc[model_dict['desc'], 'CPU [ms]'] = round_second_significant_digit(cpu_time)
            df_timing.loc[model_dict['desc'], 'GPU [ms]'] = round_second_significant_digit(gpu_time)

        return df_timing