import json
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import model
from model import MultiTaskModel
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

class InferenceWrapper:
    """
    An InferenceWrapper can be instantiated to perform easy inference from a model. 
    The Wrapper will load and initialize the model, and provide a method to perform inference along with some utility methods and
    a possibility to save the results and add a postprocessing layer.
    """

    def __init__(self, model_path, model_args_path, postprocessing=None, device='cpu'):
        """
        Initialize the InferenceWrapper with the model and the postprocessing layer.
        The model is loaded from the model_path, and the postprocessing layer is a callable that will be applied to the output of the model.
        The model_args_path is a path to a json file that contains the arguments to be passed to the model.
        
        The postprocessing layer should take the output of the model and return the processed output.

        Parameters
        ----------
        model_path : str
            Path to the model file.
        model_args_path : str
            Path to the JSON file containing model arguments.
        postprocessing : callable, optional
            A callable to process the model output. Defaults to None.
        device : str, optional
            The device to run the model on. Defaults to 'cpu'.
        """
        self.model_path = model_path
        self.postprocessing = postprocessing
        if postprocessing is None:
            self.postprocessing = self.default_postprocessing
        self.model_args_path = model_args_path
        self.model_args = None
        self.device = device

        self.load_model()

    def load_model(self):
        """
        Load the model from the model_path and initialize it.
        """
        self.parse_model_args()

        hyperparameters = self.model_args
        self.model = MultiTaskModel(input_size=hyperparameters['input_size'],
                                    hidden_size=hyperparameters['hidden_size'],
                                    hidden_size_classification=hyperparameters['hidden_size_classification'],
                                    hidden_size_regression=hyperparameters['hidden_size_regression'],
                                    hidden_size_main=hyperparameters['hidden_size_main'],
                                    output_size=hyperparameters['output_size'],
                                    num_classes=hyperparameters['num_classes'],
                                    num_regression=hyperparameters['num_regression'],
                                    num_layers_encoder=hyperparameters['num_layers_encoder'],
                                    num_layers_classification=hyperparameters['num_layers_classification'],
                                    num_layers_regression=hyperparameters['num_layers_regression'],
                                    num_layers_main=hyperparameters['num_layers_main'],
                                    num_layers_logvar=hyperparameters['num_layers_logvar'],
                                    dropout=hyperparameters['dropout'],
                                    activation=hyperparameters['activation'],
                                    input_scaler=hyperparameters['input_scaler'],
                                    regression_scaler=hyperparameters['regression_scaler'],
                                    main_scaler=hyperparameters['main_scaler']).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def parse_model_args(self):
        """
        Parse the model arguments from the model_args json file.
        """
        with open(self.model_args_path) as f:
            self.model_args = json.load(f)

        # handling the str arguments that should be classes
        self.model_args['activation'] = getattr(torch.nn.functional, self.model_args['activation'])
        self.model_args['input_scaler'] = getattr(model, self.model_args['input_scaler'])
        self.model_args['regression_scaler'] = getattr(model, self.model_args['regression_scaler'])
        self.model_args['main_scaler'] = getattr(model, self.model_args['main_scaler'])

    def inference_from_csv(self, csv_path, save_path=None, verbose=False, batch_size=1024):
        """
        Perform inference from a CSV file containing the data.
        The data is loaded from the CSV file, and the model is used to perform inference on the data.
        The results are saved to the save_path if provided.
        The results are also returned.

        In the data, the model is looking for a column named 'spiking_times' that contains the input data as
        a string that encodes the list with comma-separated values. e.g. [1, 2, 3] -> "[1, 2, 3]"

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing the data.
        save_path : str, optional
            Path to save the results. Defaults to None.
        verbose : bool, optional
            Whether to print verbose output. Defaults to False.
        batch_size : int, optional
            Batch size for inference. Defaults to 1024.

        Returns
        -------
        pandas.DataFrame
            The results of the inference.
        """
        
        if verbose:
            print(f"Loading data from {csv_path}", flush=True)
        
        column_to_load = ['spiking_times']
        data = pd.read_csv(csv_path, usecols=column_to_load)
        
        if verbose:
            print(f"Data loaded, number of samples: {len(data)}", flush=True)
        
        results = self.inference_from_dataframe(data, verbose=verbose, batch_size=batch_size)

        if save_path is not None:
            if verbose:
                print(f"Saving results to {save_path}", flush=True)
            results.to_csv(save_path, index=False)
        else:
            if verbose:
                print("No save path provided, returning the results.", flush=True)
        return results

    
    def inference_from_dataframe(self, data, verbose=False, batch_size=-1, should_preprocess=True):
        """
        Perform inference from a pandas DataFrame containing the data.
        The data is expected to have a column named 'spiking_times' that contains the input data as
        a string that encodes the list with comma-separated values. e.g. [1, 2, 3] -> "[1, 2, 3]"

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing the data.
        verbose : bool, optional
            Whether to print verbose output. Defaults to False.
        batch_size : int, optional
            Batch size for inference. Defaults to -1.
        should_preprocess : bool, optional
            Whether to preprocess the data. Defaults to True.

        Returns
        -------
        pandas.DataFrame
            The results of the inference.
        """
        if should_preprocess:
            if verbose:
                print("Preprocessing data", flush=True)
            
            data['spiking_times'] = data['spiking_times'].apply(
                lambda x: np.fromstring(x[1:-1], sep=',').astype(np.float32) 
                if pd.notna(x) and x != '[]' 
                else (np.array([]) if x == '[]' else np.nan)
            )
    
        # Create an empty results DataFrame
        results = pd.DataFrame(columns=[
            'spiking_times', 'label', 'g_s', 'g_u', 'logvar_g_s', 'logvar_g_u',
            'f_spiking', 'f_intra_bursting', 'f_inter_bursting', 'duration_bursting', 'nbr_spikes_bursting'
        ], index=data.index)
        
        if verbose:
            print("Data preprocessed, performing inference", flush=True)

        # Step 1: Handle NaN values
        nan_values = data['spiking_times'].isna()

        if verbose:
            print(f"Found {nan_values.sum()} NaN values in the data. Output will be NaN for these samples.", flush=True)

        # Create a DataFrame for NaN rows with appropriate shape and assign it to `results`
        nan_results = pd.DataFrame({
            'spiking_times': data.loc[nan_values, 'spiking_times'],
            'label': np.nan, 'g_s': np.nan, 'g_u': np.nan,
            'logvar_g_s': np.nan, 'logvar_g_u': np.nan, 'f_spiking': np.nan,
            'f_intra_bursting': np.nan, 'f_inter_bursting': np.nan,
            'duration_bursting': np.nan, 'nbr_spikes_bursting': np.nan
        }, index=data.loc[nan_values].index)

        # the index of the results DataFrame is the same as the data DataFrame, we can assign the NaN results to the results DataFrame
        results.iloc[nan_values] = nan_results.values

        # Step 2: Handle silent cells
        valid_indices = ~nan_values
        silent_cells = data.loc[valid_indices, 'spiking_times'].apply(lambda x: len(x) < 3)
        
        if verbose:
            print(f"Found {silent_cells.sum()} silent [less than 3 spikes] cells in the data. Output will be 0 for these samples.", flush=True)

        # Create a DataFrame for silent cell rows
        silent_results = pd.DataFrame({
            'spiking_times': data.loc[valid_indices & silent_cells, 'spiking_times'].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x),
            'label': 0, 'g_s': 0, 'g_u': 0,
            'logvar_g_s': 0, 'logvar_g_u': 0, 'f_spiking': 0,
            'f_intra_bursting': 0, 'f_inter_bursting': 0,
            'duration_bursting': 0, 'nbr_spikes_bursting': 0
        }, index=data.loc[valid_indices & silent_cells].index)

        # Assign the silent results to the results DataFrame
        results.iloc[valid_indices & silent_cells] = silent_results.values

        # Step 3: Perform inference on non-silent, valid data
        valid_non_silent_indices = valid_indices & ~silent_cells
        
        if batch_size is None or batch_size <= 0 or batch_size > len(valid_non_silent_indices):
            batch_size = len(valid_non_silent_indices)

        if verbose:
            print(f"Performing inference on {valid_non_silent_indices.sum()} valid samples using batch size of {batch_size}", flush=True)
        
        for i in range(0, valid_non_silent_indices.sum(), batch_size):
            if verbose:
                print(f"Processing batch {i//batch_size}...", flush=True, end='\r')
            
            if i + batch_size > valid_non_silent_indices.sum():
                batch_size = valid_non_silent_indices.sum() - i

            batch_indices = data.loc[valid_non_silent_indices].iloc[i:i + batch_size].index
            batch = data.loc[batch_indices]
            batch_results = self.inference_on_batch(batch)
            
            results.iloc[batch_indices] = batch_results.values

        if verbose:
            print("Inference processing done.", flush=True)

        # Apply the postprocessing layer if available
        if self.postprocessing is not None:
            if verbose:
                print("Applying postprocessing layer.", flush=True)
            results = self.postprocessing(results)

        return results

    def inference_from_raw_data(self, data, verbose=False, batch_size=-1):
        """
        Perform inference on raw data.
        The data is expected to be a list of lists of floats, where each list of floats represents the spiking times of a single cell.
        Data can also be a single list of floats, in which case it is assumed to be a single cell.
        The results are returned as a pandas dataframe.

        Parameters
        ----------
        data : list of lists of floats or list of floats
            The raw data to perform inference on.
        verbose : bool, optional
            Whether to print verbose output. Defaults to False.
        batch_size : int, optional
            Batch size for inference. Defaults to -1.

        Returns
        -------
        pandas.DataFrame
            The results of the inference.
        """
        # Handle the case where data is a single list of floats
        if isinstance(data[0], (int, float)):
            data = [data]

        if batch_size is None or batch_size <= 0:
            batch_size = len(data)

        if verbose:
            print(f"Performing inference on {len(data)} samples.", flush=True)

        # Create a DataFrame from the data
        data = pd.DataFrame({'spiking_times': data})

        results = self.inference_from_dataframe(data, verbose=verbose, batch_size=batch_size, should_preprocess=False)

        return results
    
    
    # Building block for inference.
    def inference_on_batch(self, batch):
        """
        Perform inference on a batch of data.
        The batch is a pandas dataframe containing the data to perform inference on.
        The data is expected to have a column named 'spiking_times' that contains the input data as
        a list of floats.
        The results are returned as a pandas dataframe with the same index as the input batch.

        Parameters
        ----------
        batch : pandas.DataFrame
            DataFrame containing the batch data.

        Returns
        -------
        pandas.DataFrame
            The results of the inference.
        """
        # from spiking times to ISI
        batch['ISI'] = batch['spiking_times'].apply(lambda x: np.diff(x).astype(np.float32))
        
        X = [torch.tensor(isi, dtype=torch.float32) for isi in batch['ISI']]
        L = torch.tensor([len(x) for x in X], dtype=torch.int32)
        X = pad_sequence(X, batch_first=True, padding_value=0.)
        X = X.to(self.device).unsqueeze(-1)
        L = L.to("cpu")

        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device, enabled=True if self.device != 'cpu' else False):
                output = self.model(X, L) # output is : batch_size x [y_c, y_r, y_m] with y_c the classification output (logits), y_r the regression output (f_spiking, f_intra_bursting, f_inter_bursting, duration_bursting, nbr_spikes_bursting), y_m the main output (g_s, g_u, logvar_g_s, logvar_g_u)
                y_c, y_r, y_m = output
                label = torch.argmax(y_c, dim=1)
                f_spiking, f_intra_bursting, f_inter_bursting, duration_bursting, nbr_spikes_bursting = y_r[:, 0], y_r[:, 1], y_r[:, 2], y_r[:, 3], y_r[:, 4]
                g_s, g_u, logvar_g_s, logvar_g_u = y_m[:, 0], y_m[:, 1], y_m[:, 2], y_m[:, 3]

        results = pd.DataFrame({
            'spiking_times': batch['spiking_times'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x),
            'label': (label.cpu().numpy() + 1),  # Add 1 to match original labels
            'g_s': g_s.cpu().numpy(),
            'g_u': g_u.cpu().numpy(),
            'logvar_g_s': logvar_g_s.cpu().numpy(),
            'logvar_g_u': logvar_g_u.cpu().numpy(),
            'f_spiking': f_spiking.cpu().numpy(),
            'f_intra_bursting': f_intra_bursting.cpu().numpy(),
            'f_inter_bursting': f_inter_bursting.cpu().numpy(),
            'duration_bursting': duration_bursting.cpu().numpy(),
            'nbr_spikes_bursting': nbr_spikes_bursting.cpu().numpy()
        }, index=batch.index)

        return results
    
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

    def __init__(self, postprocessing=None, model="stg"):
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

        results['len'] = results['spiking_times'].apply(lambda x: len(x) if isinstance(x, np.ndarray) else 0)
        # print the min, max and mean and median of the len column
        # drop label 0 in the computation
        if verbose:
            print("min, max, mean and median of the len column (excluding label 0):", flush=True)
            print(f"min: {results.loc[results['label'] != 0, 'len'].min()}", flush=True)
            print(f"max: {results.loc[results['label'] != 0, 'len'].max()}", flush=True)
            print(f"mean: {results.loc[results['label'] != 0, 'len'].mean()}", flush=True)
            print(f"median: {results.loc[results['label'] != 0, 'len'].median()}", flush=True)

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