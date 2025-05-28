import json
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

class InferenceWrapper:
    
    @staticmethod
    def default_postprocessing(results):
        """
        Default postprocessing layer that can be used to modify the results of the inference.
        The postprocessing layer will modify the results dataframe and return it.

        The default postprocessing layer will:
            - enforce no negative g_u values : g_u is clipped to 0

        Parameters
        ----------
        results : pandas.DataFrame
            The results of the inference.

        Returns
        -------
        pandas.DataFrame
            The postprocessed results.
        """
        results['g_u'] = results['g_u'].clip(lower=0)
        return results
    
    @staticmethod
    def cleaner_default_postprocessing(results, rounding=True):
        """
        Cleaner default postprocessing layer that can be used to modify the results of the inference.
        The postprocessing layer will modify the results dataframe and return it.
        
        The cleaner default postprocessing layer will:
            - enforce no negative g_u values : g_u is clipped to 0
            - put 0 in f_spiking for silent cells, weird cells and bursting cells
            - put 0 in f_intra_bursting, f_inter_bursting, duration_bursting and nbr_spikes_bursting for silent cells, weird cells and tonic spiking cells
            - round nbr_spikes_bursting to the nearest integer if rounding is True

        Parameters
        ----------
        results : pandas.DataFrame
            The results of the inference.
        rounding : bool, optional
            Whether to round nbr_spikes_bursting to the nearest integer. Defaults to True.

        Returns
        -------
        pandas.DataFrame
            The postprocessed results.
        """
        results['g_u'] = results['g_u'].clip(lower=0)
        results.loc[results['label'] == 0, 'f_spiking'] = 0
        results.loc[results['label'] == 1, 'f_spiking'] = 0
        results.loc[results['label'] == 3, 'f_spiking'] = 0
        results.loc[results['label'] == 0, 'f_intra_bursting'] = 0
        results.loc[results['label'] == 1, 'f_intra_bursting'] = 0
        results.loc[results['label'] == 2, 'f_intra_bursting'] = 0
        results.loc[results['label'] == 0, 'f_inter_bursting'] = 0
        results.loc[results['label'] == 1, 'f_inter_bursting'] = 0
        results.loc[results['label'] == 2, 'f_inter_bursting'] = 0
        results.loc[results['label'] == 0, 'duration_bursting'] = 0
        results.loc[results['label'] == 1, 'duration_bursting'] = 0
        results.loc[results['label'] == 2, 'duration_bursting'] = 0
        results.loc[results['label'] == 0, 'nbr_spikes_bursting'] = 0
        results.loc[results['label'] == 1, 'nbr_spikes_bursting'] = 0
        results.loc[results['label'] == 2, 'nbr_spikes_bursting'] = 0
        if rounding:
            results['nbr_spikes_bursting'] = results['nbr_spikes_bursting'].apply(lambda x: round(x))
        return results
    
    @staticmethod
    def std_postprocessing(results):
        """
        Std postprocessing layer that can be used to modify the results of the inference.
        The postprocessing layer will modify the results dataframe and return it.

        The std postprocessing layer will:
            - Transform the logvar_g_s and logvar_g_u to std_g_s and std_g_u

        Parameters
        ----------
        results : pandas.DataFrame
            The results of the inference.

        Returns
        -------
        pandas.DataFrame
            The postprocessed results.
        """
        results['std_g_s'] = results['logvar_g_s'].apply(lambda x: np.sqrt(np.exp(x)))
        results['std_g_u'] = results['logvar_g_u'].apply(lambda x: np.sqrt(np.exp(x)))
        results.drop(columns=['logvar_g_s', 'logvar_g_u'], inplace=True)

        return results
            
    @staticmethod
    def identity_postprocessing(results):
        """
        Identity postprocessing layer that can be used to modify the results of the inference.
        The postprocessing layer will modify the results dataframe and return it.

        The identity postprocessing layer will:
            - return the results unchanged

        Parameters
        ----------
        results : pandas.DataFrame
            The results of the inference.

        Returns
        -------
        pandas.DataFrame
            The unchanged results.
        """
        return results
    
    @staticmethod
    def cleaner_and_std_postprocessing(results, rounding=True):
        """
        Cleaner and std postprocessing layer that can be used to modify the results of the inference.
        The postprocessing layer will modify the results dataframe and return it.

        The cleaner and std postprocessing layer will:
            - enforce no negative g_u values : g_u is clipped to 0
            - put 0 in f_spiking for silent cells, weird cells and bursting cells
            - put 0 in f_intra_bursting, f_inter_bursting, duration_bursting and nbr_spikes_bursting for silent cells, weird cells and tonic spiking cells
            - round nbr_spikes_bursting to the nearest integer
            - Transform the logvar_g_s and logvar_g_u to std_g_s and std_g_u

        Parameters
        ----------
        results : pandas.DataFrame
            The results of the inference.

        Returns
        -------
        pandas.DataFrame
            The postprocessed results.
        """
        r = InferenceWrapper.cleaner_default_postprocessing(results, rounding=rounding)
        r = InferenceWrapper.std_postprocessing(r)
        return r
    
class SpikeFeatureExtractor:
    """
    A SpikeFeatureExtractor can be instantiated to extract features from spiking data.
    The extractor provides methods to extract features from CSV files or pandas DataFrames.
    """

    def __init__(self, postprocessing=None, model="stg", truncate_burst=False):
        """
        Initialize the SpikeFeatureExtractor with an optional postprocessing layer.
        The postprocessing layer is a callable that will be applied to the extracted features.

        Parameters
        ----------
        postprocessing : callable, optional
            A callable to process the extracted features. Defaults to None.
        model : str, optional
            The model to use for the extraction. Defaults to "stg".
            Available models are "stg" and "da".
            The type of model will determine the available labels.
        """
        self.postprocessing = postprocessing
        if postprocessing is None:
            self.postprocessing = InferenceWrapper.identity_postprocessing
        self.model = model
        self.truncate_burst = truncate_burst

    def extract_from_csv(self, csv_path, save_path=None, verbose=False, num_workers=8):
        """
        Extract features from a CSV file containing spiking data.
        The data is loaded from the CSV file, and features are extracted from the data.
        The results are saved to the save_path if provided.
        The results are also returned.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing the data.
        save_path : str, optional
            Path to save the results. Defaults to None.
        verbose : bool, optional
            Whether to print verbose output. Defaults to False.
        num_workers : int, optional
            Number of workers to use for parallel processing. Defaults to 8.

        Returns
        -------
        pandas.DataFrame
            The extracted features.
        """
        if verbose:
            print(f"Loading data from {csv_path}", flush=True)
        
        column_to_load = ['spiking_times']
        data = pd.read_csv(csv_path, usecols=column_to_load)
        
        if verbose:
            print(f"Data loaded, number of samples: {len(data)}", flush=True)
        
        results = self.extract_from_dataframe(data, verbose=verbose, num_workers=num_workers)

        if save_path is not None:
            if verbose:
                print(f"Saving results to {save_path}", flush=True)
            results.to_csv(save_path, index=False)
        else:
            if verbose:
                print("No save path provided, returning the results.", flush=True)
        return results
    
    def extract_from_dataframe(self, data, verbose=False, num_workers=16, should_preprocess=True):
        """
        Extract features from a pandas DataFrame containing spiking data.
        The data is expected to have a column named 'spiking_times' that contains the input data as
        a string that encodes the list with comma-separated values. e.g. [1, 2, 3] -> "[1, 2, 3]"

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing the data.
        verbose : bool, optional
            Whether to print verbose output. Defaults to False.
        num_workers : int, optional
            Number of workers to use for parallel processing. Defaults to 8.
        should_preprocess : bool, optional
            Whether to preprocess the data. Defaults to True.

        Returns
        -------
        pandas.DataFrame
            The extracted features.
        """
        # Create an empty results DataFrame
        results = pd.DataFrame(columns=[
            'spiking_times', 'label', 'f_spiking', 'f_intra_bursting', 'f_inter_bursting', 'duration_bursting', 'nbr_spikes_bursting'
        ], index=data.index)
        
        results["spiking_times"] = data["spiking_times"].copy()

        if should_preprocess:
            if verbose:
                print("Preprocessing data", flush=True)
            
            results['spiking_times'] = results['spiking_times'].apply(
                lambda x: np.fromstring(x[1:-1], sep=',').astype(np.float32) 
                if pd.notna(x) and x != '[]' 
                else (np.array([]) if x == '[]' else np.nan)
            )
            # shift the spiking times to be relative to the first spike
            results['spiking_times'] = results['spiking_times'].apply(
                lambda x: x - x[0] if len(x) > 0 else x
            )   
            
            if verbose:
                print("Data preprocessed, extracting features", flush=True)

        # Cut into chunks
        chunks = []
        for i in range(0, num_workers):
            start = i * len(results) // num_workers
            end = (i + 1) * len(results) // num_workers
            end = min(end, len(results))
            chunks.append(results.iloc[start:end])

        if verbose:
            print(f"Extracting features from {len(results)} samples using {num_workers} workers", flush=True)
            tqdm.pandas()
        
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(self.parallel_extract_features, chunks), total=len(chunks), disable=not verbose))

        # Concatenate the processed chunks back together
        return pd.concat(results)
    
    def parallel_extract_features(self, data):
        """
        Extract features from a chunk of data in parallel.
        The chunk is a pandas DataFrame containing the data to extract features from.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing the chunk data.

        Returns
        -------
        pandas.DataFrame
            The extracted features.
        """
        return data.apply(self.extract_from_row, axis=1)

    def extract_from_row(self, row):
        """
        Extract features from a single row of data.
        The row is a pandas Series containing the data to extract features from.

        Parameters
        ----------
        row : pandas.Series
            Series containing the row data.

        Returns
        -------
        pandas.Series
            The row with extracted features.
        """
        # Initialize features with 0 (or NaN if necessary)
        row['f_spiking'] = 0
        row['f_intra_bursting'] = 0
        row['f_inter_bursting'] = 0
        row['duration_bursting'] = 0
        row['nbr_spikes_bursting'] = 0

        spiking_times = row['spiking_times']

        label = -1
        if len(spiking_times) < 3:
            label = 0  # Silent
    
        cv_th = 0.15 if self.model == "stg" else 0.15

        if label == -1:
            # Check for regular spiking versus bursting
            ISIs = np.diff(spiking_times)
            CV_ISI = np.std(ISIs) / np.mean(ISIs)
            if CV_ISI <= cv_th:
                label = 1  # Regular spiking
            else:
                label = 2  # Bursting

        row['label'] = label

        if row['label'] == 1:
            # Tonic spiking
            if len(spiking_times) > 0:
                ISIs = np.diff(spiking_times)
                if len(ISIs) > 0:
                    row['f_spiking'] = (1000. / ISIs).mean()

        elif row['label'] == 2:
            # Bursting
            if len(spiking_times) > 1:
                ISIs = np.diff(spiking_times)
                threshold = (ISIs.max() + ISIs.min()) / 2
                burst_starts = np.where(ISIs > threshold)[0] + 1
                bursts = np.split(spiking_times, burst_starts)
                bursts = [b for b in bursts if len(b) > 1]  # keep only valid bursts
                bursts = bursts[1:-1]  # remove first and last as per definition

               
                if len(bursts) > 0:
                    if self.truncate_burst:
                        # we replace the spiking_times to only have the keeped bursts
                        row['spiking_times'] = np.concatenate(bursts)
                        row['spiking_times'] = row['spiking_times'] - row['spiking_times'][0]

                    row['nbr_spikes_bursting'] = np.nanmean([len(b) for b in bursts])
                    row['duration_bursting'] = np.nanmean([b[-1] - b[0] for b in bursts])
                    row['f_intra_bursting'] = np.nanmean([len(b) / (b[-1] - b[0]) if (b[-1] - b[0]) > 0 else np.nan for b in bursts]) * 1000.0
                    
                    burst_onsets = [b[0] for b in bursts]
                    if len(burst_onsets) > 1:
                        inter_burst_ISIs = np.diff(burst_onsets)
                        row['f_inter_bursting'] = 1000. / np.mean(inter_burst_ISIs)
                    else:
                        row['f_inter_bursting'] = np.nan
                else:
                    row['nbr_spikes_bursting'] = np.nan
                    row['duration_bursting'] = np.nan
                    row['f_intra_bursting'] = np.nan
                    row['f_inter_bursting'] = np.nan

        return row