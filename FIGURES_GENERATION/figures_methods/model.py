"""
This file contains the model i.e. the building blocks of the neural network.
Some of the building blocks are not used in the final model but are kept for future reference and experimentation.
"""

"""TODO:
Everywhere:
- Review what is saved in the class attributes, exemple : dropout and hiddensize are useless in the ResidualFeedForward class.
Scalers:
- Add the inverse_transform method to the DiffLogStandardScaler and CumsumDiffLogStandardScaler classes.
- Add the inverse_transform method to the MinMaxScaler class.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# == Scalers ==
# The scalers are used to standardize the input data or the output data.
# Some scalers also augment the data directly within the model.
# Note : the fit methods rely on numpy arrays and not torch tensors whereas the transform methods rely on torch tensors. This strange behavior is due to the fact that the fit method is called before the model is trained and the data is converted to torch tensors. If time permits, this behavior should be changed to be consistent with the rest of the code.

class Scaler(nn.Module):
    """
    Base class for scalers.

    Attributes
    ----------
    dim : int
        Dimension of the input data.
    a : torch.Tensor
        Mean or minimum value of the training data.
    b : torch.Tensor
        Standard deviation or range of the training data.

    Methods
    -------
    fit(X)
        Compute the mean and standard deviation or min and max of the training data. To be implemented in the child classes.
    transform(X)
        Standardize the input data. By default, it is a linear transformation.
    fit_transform(X)
        Fit the scaler and then transform the input data.
    inverse_transform(X)
        Inverse the transformation of the input data. By default, it is a linear transformation.
    forward(X)
        Alias for the transform method.
    """

    def __init__(self, dim):
        super(Scaler, self).__init__()
        self.dim = dim
        # Register non-trainable tensors
        self.register_buffer('a', torch.zeros(dim))  # Non-trainable
        self.register_buffer('b', torch.ones(dim))   # Non-trainable

    def fit(self, X):
        raise NotImplementedError

    def transform(self, X):
        return (X - self.a.view(1, -1)) / self.b.view(1, -1)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * self.b.view(1, -1) + self.a.view(1, -1)
    
    def forward(self, X):
        return self.transform(X)

class StandardScaler(Scaler):
    """
    Standardize the input data to have zero mean and unit variance from the mean and the standard deviation
    of the training data.

    Attributes
    ----------
    dim : int
        Dimension of the input data.
    a : torch.Tensor
        Mean value of the training data.
    b : torch.Tensor
        Standard deviation of the training data.

    Methods
    -------
    fit(X)
        Compute the mean and standard deviation of the training data.
    transform(X)
        Standardize the input data.
    inverse_transform(X)
        Inverse the transformation of the input data.
    forward(X)
        Alias for the transform method.
    """

    def __init__(self, dim):
        super(StandardScaler, self).__init__(dim)
        # Register non-trainable tensors
        self.register_buffer('a', torch.zeros(dim))
        self.register_buffer('b', torch.ones(dim))
        
    def fit(self, X):
        self.a.data = torch.tensor(
            X.mean(
                axis=0),
            dtype=torch.float32,
            device=self.a.device).view_as(
            self.a)
        self.b.data = torch.tensor(
            X.std(
                axis=0),
            dtype=torch.float32,
            device=self.b.device).view_as(
            self.b)
        
    def transform(self, X):
        return (X - self.a.view(1, -1)) / self.b.view(1, -1)
    
    def inverse_transform(self, X):
        return X * self.b.view(1, -1) + self.a.view(1, -1)
    
    def forward(self, X):
        return self.transform(X)

class LogStandardScaler(Scaler):
    """
    Standardize the input data to have zero mean and unit variance from the mean and the standard deviation
    of the training data after applying a log transformation.

    Attributes
    ----------
    dim : int
        Dimension of the input data.
    eps : float
        Small value to avoid 0 in the log transformation.
    a : torch.Tensor
        Mean value of the log-transformed training data.
    b : torch.Tensor
        Standard deviation of the log-transformed training data.

    Methods
    -------
    fit(X)
        Compute the mean and standard deviation of the log-transformed training data.
    transform(X)
        Standardize the log-transformed input data.
    inverse_transform(X)
        Inverse the transformation of the log-transformed input data.
    forward(X)
        Alias for the transform method.
    """

    def __init__(self, dim, eps=1e-6):
        super(LogStandardScaler, self).__init__(dim)
        self.eps = eps
        # Register non-trainable tensors
        self.register_buffer('a', torch.zeros(dim))
        self.register_buffer('b', torch.ones(dim))

    def fit(self, X):
        if X.ndim == 1:
            X = X[:, None]
        X = X.copy()
        X = np.log(X + self.eps)
        self.a.data = torch.tensor(
            X.mean(
                axis=0),
            dtype=torch.float32,
            device=self.a.device).view_as(
            self.a)
        self.b.data = torch.tensor(
            X.std(
                axis=0),
            dtype=torch.float32,
            device=self.b.device).view_as(
            self.b)

    def transform(self, X):
        return (torch.log(X + self.eps) - self.a.view(1, -1)) / self.b.view(1, -1)
    
    def inverse_transform(self, X):
        return torch.exp(X * self.b.view(1, -1) + self.a.view(1, -1))
    
    def forward(self, X):
        return self.transform(X)

class DiffLogStandardScaler(Scaler):
    """
    Standardize the input data to have zero mean and unit variance from the mean and the standard deviation
    of the training data and then augment the data with:
    - torch.diff(x, dim=1) (padded)

    The augmented data is concatenated to the original data in the last dimension.
    
    Note : dim is the dimension of the input data AFTER the augmentation (so times 2); A placeholder is added to the
    a and b parameters for the augmented data when calling the fit method (a = 0 and b = 1).

    Attributes
    ----------
    dim : int
        Dimension of the input data.
    eps : float
        Small value to avoid 0 in the log transformation.
    a : torch.Tensor
        Mean value of the log-transformed training data.
    b : torch.Tensor
        Standard deviation of the log-transformed training data.

    Methods
    -------
    fit(X)
        Compute the mean and standard deviation of the log-transformed training data.
    transform(X)
        Standardize the log-transformed input data and augment it with the difference.
    inverse_transform(X)
        Inverse the transformation of the log-transformed input data.
    forward(X)
        Alias for the transform method.
    """

    def __init__(self, dim, eps=1e-6):
        super(DiffLogStandardScaler, self).__init__(dim)
        self.eps = eps
        # Register non-trainable tensors
        self.register_buffer('a', torch.zeros(dim))
        self.register_buffer('b', torch.ones(dim))

    def fit(self, X):
        if X.ndim == 1:
            X = X[:, None]
        X = np.concatenate([X, X], axis=-1)
        X = np.log(X + self.eps)
        self.a.data = torch.tensor(
            X.mean(
                axis=0),
            dtype=torch.float32,
            device=self.a.device).view_as(
            self.a)
        self.b.data = torch.tensor(
            X.std(
                axis=0),
            dtype=torch.float32,
            device=self.b.device).view_as(
            self.b)
        
    def transform(self, X):
        # X : [batch_size, seq_len, dim//2] -> X : [batch_size, seq_len, dim]
        X = torch.cat([X, X], dim=-1)

        # X : [batch_size, seq_len, dim] -> X : [batch_size, seq_len, dim]
        X = torch.log(X + self.eps)
        
        # Extract the original data
        # X : [batch_size, seq_len, dim] -> X : [batch_size, seq_len, dim//2]
        X = X[:, :, :X.size(-1) // 2]

        # X diff:
        # X : [batch_size, seq_len, dim//2] -> X : [batch_size, seq_len - 1, dim//2]
        X_diff = torch.diff(X, dim=1)
        # pad the diff
        # X_diff : [batch_size, seq_len - 1, dim//2] -> X_diff : [batch_size, seq_len, dim//2]
        X_diff = torch.cat([torch.zeros(X_diff.size(0), 1, X_diff.size(-1), device=X_diff.device), X_diff], dim=1)

        # Concatenate the original data and the diff
        # X : [batch_size, seq_len, dim//2], X_diff : [batch_size, seq_len, dim//2] -> X : [batch_size, seq_len, dim]
        X = torch.cat([X, X_diff], dim=-1)

        return X
    
    def inverse_transform(self, X):
        X = X[:, :, :X.size(-1) // 2]
        return torch.exp(X * self.b.view(1, -1) + self.a.view(1, -1))
    
class CumsumDiffStandardScaler(Scaler):
    """
    Standardize the input data to have zero mean and unit variance from the mean and the standard deviation
    of the training data and then augment the data with:
    - torch.diff(x, dim=1) (padded)
    - torch.cumsum(x, dim=1)

    The augmented data is concatenated to the original data in the last dimension.
    
    Note : dim is the dimension of the input data AFTER the augmentation (so times 3); A placeholder is added to the
    a and b parameters for the augmented data when calling the fit method (a = 0 and b = 1).

    Same as 'CumsumDiffLogStandardScaler' but no log transformation is applied.
    
    Attributes
    ----------
    dim : int
        Dimension of the input data.
    
        TODO : CONTINUE

    """

    def __init__(self, dim):
        super(CumsumDiffStandardScaler, self).__init__(dim)
        # Register non-trainable tensors
        self.register_buffer('a', torch.zeros(dim))
        self.register_buffer('b', torch.ones(dim))

    def fit(self, X):
        if X.ndim == 1:
            X = X[:, None]

        X = np.concatenate([X, X, X], axis=-1)
        self.a.data = torch.tensor(
            X.mean(
                axis=0),
            dtype=torch.float32,
            device=self.a.device).view_as(
            self.a)
        self.b.data = torch.tensor(
            X.std(
                axis=0),
            dtype=torch.float32,
            device=self.b.device).view_as(
            self.b)
        
    def transform(self, X):
        # X diff:
        # X : [batch_size, seq_len, dim//3] -> X : [batch_size, seq_len - 1, dim//3]
        X_diff = torch.diff(X, dim=1)
        # pad the diff
        # X_diff : [batch_size, seq_len - 1, dim//3] -> X_diff : [batch_size, seq_len, dim//3]
        X_diff = torch.cat([torch.zeros(X_diff.size(0), 1, X_diff.size(-1), device=X_diff.device), X_diff], dim=1)

        #X_cumsum:
        # X : [batch_size, seq_len, dim//3] -> X_cumsum : [batch_size, seq_len, dim//3]
        X_cumsum = torch.cumsum(X, dim=1)

        # We scale the cumsum to get a relative value
        # X_cumsum : [batch_size, seq_len, dim//3] -> X_cumsum : [batch_size, seq_len, dim//3]
        X_cumsum = X_cumsum / (torch.max(X_cumsum, dim=1, keepdim=True).values + 1e-6)

        # X : [batch_size, seq_len, dim//3] -> X : [batch_size, seq_len, dim]
        X = torch.cat([X, X, X], dim=-1)

        # Extract the original data
        # X : [batch_size, seq_len, dim] -> X : [batch_size, seq_len, dim//3]
        X = X[:, :, :X.size(-1) // 3]

        # Concatenate the original data, the diff, and the cumsum
        # X : [batch_size, seq_len, dim//3], X_diff : [batch_size, seq_len, dim//3], X_cumsum : [batch_size, seq_len, dim//3] -> X : [batch_size, seq_len, dim]
        X = torch.cat([X, X_diff, X_cumsum], dim=-1)

        return X
    
    def inverse_transform(self, X):
        X = X[:, :, :X.size(-1) // 3]
        return X * self.b.view(1, -1) + self.a.view(1, -1)

class CumsumDiffLogStandardScaler(Scaler):
    """
    Standardize the input data to have zero mean and unit variance from the mean and the standard deviation
    of the training data and then augment the data with:
    - torch.diff(x, dim=1) (padded)
    - torch.cumsum(x, dim=1)

    The augmented data is concatenated to the original data in the last dimension.
    
    Note : dim is the dimension of the input data AFTER the augmentation (so times 3); A placeholder is added to the
    a and b parameters for the augmented data when calling the fit method (a = 0 and b = 1).

    Attributes
    ----------
    dim : int
        Dimension of the input data.
    eps : float
        Small value to avoid 0 in the log transformation.
    a : torch.Tensor
        Mean value of the log-transformed training data.
    b : torch.Tensor
        Standard deviation of the log-transformed training data.

    Methods
    -------
    fit(X)
        Compute the mean and standard deviation of the log-transformed training data.
    transform(X)
        Standardize the log-transformed input data and augment it with the difference and cumulative sum.
    inverse_transform(X)
        Inverse the transformation of the log-transformed input data.
    forward(X)
        Alias for the transform method.
    """

    def __init__(self, dim, eps=1):
        super(CumsumDiffLogStandardScaler, self).__init__(dim)
        # Register non-trainable tensors
        self.register_buffer('a', torch.zeros(dim))
        self.register_buffer('b', torch.ones(dim))
        # register eps as a buffer
        self.register_buffer('eps', torch.tensor(eps))

    def fit(self, X):
        if X.ndim == 1:
            X = X[:, None]

        X = np.concatenate([X, X, X], axis=-1)
        X = np.log(X + self.eps.item())
        self.a.data = torch.tensor(
            X.mean(
                axis=0),
            dtype=torch.float32,
            device=self.a.device).view_as(
            self.a)
        self.b.data = torch.tensor(
            X.std(
                axis=0),
            dtype=torch.float32,
            device=self.b.device).view_as(
            self.b)
        
    def transform(self, X):
        # X diff:
        # X : [batch_size, seq_len, dim//3] -> X : [batch_size, seq_len - 1, dim//3]
        X_diff = torch.diff(X, dim=1)
        # pad the diff
        # X_diff : [batch_size, seq_len - 1, dim//3] -> X_diff : [batch_size, seq_len, dim//3]
        X_diff = torch.cat([torch.zeros(X_diff.size(0), 1, X_diff.size(-1), device=X_diff.device), X_diff], dim=1)

        # We sign(X_diff)log(|X_diff| + 1) the diff to rescale the data
        # X_diff : [batch_size, seq_len, dim//3] -> X_diff : [batch_size, seq_len, dim//3]
        X_diff = torch.sign(X_diff) * torch.log(torch.abs(X_diff) + 1)

        #X_cumsum:
        # X : [batch_size, seq_len, dim//3] -> X_cumsum : [batch_size, seq_len, dim//3]
        X_cumsum = torch.cumsum(X, dim=1)

        # We scale the cumsum to get a relative value
        # X_cumsum : [batch_size, seq_len, dim//3] -> X_cumsum : [batch_size, seq_len, dim//3]
        X_cumsum = X_cumsum / (torch.max(X_cumsum, dim=1, keepdim=True).values + 1e-6)

        # X : [batch_size, seq_len, dim//3] -> X : [batch_size, seq_len, dim]
        X = torch.cat([X, X, X], dim=-1)

        # X : [batch_size, seq_len, dim] -> X : [batch_size, seq_len, dim]
        X = torch.log(X + self.eps)

        # Extract the original data
        # X : [batch_size, seq_len, dim] -> X : [batch_size, seq_len, dim//3]
        X = X[:, :, :X.size(-1) // 3]

        # Concatenate the original data, the diff, and the cumsum
        # X : [batch_size, seq_len, dim//3], X_diff : [batch_size, seq_len, dim//3], X_cumsum : [batch_size, seq_len, dim//3] -> X : [batch_size, seq_len, dim]
        X = torch.cat([X, X_diff, X_cumsum], dim=-1)

        return X
    
    def inverse_transform(self, X):
        X = X[:, :, :X.size(-1) // 3]
        return torch.exp(X * self.b.view(1, -1) + self.a.view(1, -1))
    
class MinMaxScaler(Scaler):
    """
    Standardize the input data from the min and the max of the training data.

    Attributes
    ----------
    dim : int
        Dimension of the input data.
    a : torch.Tensor
        Minimum value of the training data.
    b : torch.Tensor
        Range (max - min) of the training data.

    Methods
    -------
    fit(X)
        Compute the min and max of the training data.
    transform(X)
        Standardize the input data.
    inverse_transform(X)
        Inverse the transformation of the input data.
    forward(X)
        Alias for the transform method.
    """
    def __init__(self, dim):
        super(MinMaxScaler, self).__init__(dim)
        # Register non-trainable tensors
        self.register_buffer('a', torch.zeros(dim))
        self.register_buffer('b', torch.ones(dim))

    def fit(self, X):
        self.a.data = torch.tensor(
            X.min(
                axis=0),
            dtype=torch.float32,
            device=self.a.device).view_as(
            self.a)
        self.b.data = torch.tensor(
            X.max(
                axis=0) -
            X.min(
                axis=0),
            dtype=torch.float32,
            device=self.b.device).view_as(
                self.b)
        
    def transform(self, X):
        return (X - self.a.view(1, -1) - self.b.view(1, -1)/2) / (self.b.view(1, -1)/2)
    
    def inverse_transform(self, X):
        return X * (self.b.view(1, -1) / 2) + self.a.view(1, -1) + self.b.view(1, -1) / 2
    
    def forward(self, X):
        return self.transform(X)

class HalfMinMaxHalfNothingScaler(Scaler):
    """
    Standardize the input data from the min and the max of the training data. This is for the first half of the data features.
    The second half of the data features are not standardized (a = 0 and b = 1).

    Attributes
    ----------
    dim : int
        Dimension of the input data.
    a : torch.Tensor
        Minimum value of the training data for the first half of the features.
    b : torch.Tensor
        Range (max - min) of the training data for the first half of the features.

    Methods
    -------
    fit(X)
        Compute the min and max of the training data for the first half of the features.
    transform(X)
        Standardize the input data for the first half of the features.
    inverse_transform(X)
        Inverse the transformation of the input data for the first half of the features.
    forward(X)
        Alias for the transform method.
    """
    def __init__(self, dim):
        super(HalfMinMaxHalfNothingScaler, self).__init__(dim)
        # Register non-trainable tensors
        self.register_buffer('a', torch.zeros(dim))
        self.register_buffer('b', torch.ones(dim))

    def fit(self, X):
        half_dim = self.dim // 2
        self.a.data[:half_dim] = torch.tensor(
            X.min(axis=0),
            dtype=torch.float32,
            device=self.a.device).view_as(self.a[:half_dim])
        self.b.data[:half_dim] = torch.tensor(
            X.max(axis=0) - X.min(axis=0),
            dtype=torch.float32,
            device=self.b.device).view_as(self.b[:half_dim])
        self.a.data[half_dim:] = 0
        self.b.data[half_dim:] = 1
        
    def transform(self, X):
        return (X - self.a.view(1, -1) - self.b.view(1, -1)/2) / (self.b.view(1, -1)/2)
    
    def inverse_transform(self, X):
        return X * (self.b.view(1, -1) / 2) + self.a.view(1, -1) + self.b.view(1, -1) / 2
    
    def forward(self, X):
        return self.transform(X)

# == Elementary blocks ==
# The elementary blocks are the building blocks of the neural network.
# They are used to build bigger blocks such as the encoder, ...
class ResidualFeedForward(nn.Module):
    """
    Feedforward layer with configurable activation function, dropout, and a residual connection.
    The feedforward layer is defined as a FFN with a hidden layer of size hidden_size and an output layer of size hidden_size.

    Attributes
    ----------
    hidden_size : int
        Size of the hidden layer.
    dropout : float
        Dropout rate.
    activation : callable
        Activation function to use.

    Methods
    -------
    forward(x)
        Perform the forward pass of the feedforward layer with a residual connection.
    """

    def __init__(self, hidden_size, dropout, activation=F.relu):
        """
        Initialize the ResidualFeedForward layer.

        Parameters
        ----------
        hidden_size : int
            Size of the hidden layer.
        dropout : float
            Dropout rate.
        activation : callable, optional
            Activation function to use. Defaults to F.relu.
        """
        super(ResidualFeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation

        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Perform the forward pass of the feedforward layer with a residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, *, hidden_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, *, hidden_size) after applying the feedforward layer and adding the residual connection.
        """
        residual = x
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x += residual

        return x
    
class FeedForward(nn.Module):
    """
    Feedforward layer with configurable activation function and dropout.
    The feedforward layer is defined as a FFN with a hidden layer of size hidden_size and an output layer of size hidden_size.

    Attributes
    ----------
    hidden_size : int
        Size of the hidden layer.
    dropout : float
        Dropout rate.
    activation : callable
        Activation function to use.

    Methods
    -------
    forward(x)
        Perform the forward pass of the feedforward layer.
    """

    def __init__(self, hidden_size, dropout, activation=F.relu):
        """
        Initialize the FeedForward layer.

        Parameters
        ----------
        hidden_size : int
            Size of the hidden layer.
        dropout : float
            Dropout rate.
        activation : callable, optional
            Activation function to use. Defaults to F.relu.
        """
        super(FeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation

        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Perform the forward pass of the feedforward layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, *, hidden_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, *, hidden_size) after applying the feedforward layer.
        """
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x
    
    
class MeanAggregator(nn.Module):
    """
    The mean aggregator is a simple aggregation layer that computes the mean of the input.
    The input is a tensor of shape (batch_size, seq_len, hidden_size) and the output is a tensor of shape
    (batch_size, hidden_size).
    """

    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, x, L=None):
        """
        The forward pass of the mean aggregator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, hidden_size), a batch of padded sequences, on device.
        L : torch.Tensor, optional
            Length of the sequences of shape (batch_size), on cpu. If None, the mean is computed over the whole sequence.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, hidden_size) after computing the mean over the sequence.
        """

        # Aggregation of the input
        # x : [batch_size, seq_len, hidden_size] -> x : [batch_size, hidden_size]
        if L is None:
            return x.mean(dim=1)
        else:
            mask = torch.arange(x.size(1), device=x.device)[
                None, :] < L[:, None].to(x.device)
            return (x * mask.unsqueeze(-1)).sum(dim=1) / \
                mask.sum(dim=1, keepdim=True)


class AttentionAggregator(nn.Module):
    """
    The attention aggregator is a self-attention aggregation layer that computes the attention of the input.

    Parameters
    ----------
    hidden_size : int
        Size of the hidden layer.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.

    Methods
    -------
    forward(x, L=None)
        Compute the forward pass of the attention aggregator.
    """

    def __init__(self, hidden_size, num_heads, dropout):
        super(AttentionAggregator, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        self.self_attention = MultiHeadAttention(hidden_size, num_heads, dropout)

    def forward(self, x, L=None):
        """
        The forward pass of the attention aggregator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, hidden_size), a batch of padded sequences, on device.
        L : torch.Tensor, optional
            Length of the sequences of shape (batch_size), on cpu. If None, the attention is computed over the whole sequence.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, hidden_size) after computing the attention over the sequence.
        """

        # Compute the self-attention
        # x : [batch_size, seq_len, hidden_size] -> attn_output : [batch_size, seq_len, hidden_size]
        attn_output = self.self_attention(x, L)

        # Compute the weighted sum of the input
        # attn_output : [batch_size, seq_len, hidden_size] -> output : [batch_size, hidden_size]
        if L is None:
            output = attn_output.mean(dim=1)
        else:
            mask = torch.arange(x.size(1), device=x.device)[None, :] < L[:, None].to(x.device)
            output = (attn_output * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)

        return output
    
class PositionalEncoding(nn.Module):
    """
    The positional encoding layer adds positional information to the input.
    The positional encoding is a sinusoidal function of the position of the token in the sequence.
    This positional encoding is NOT learned.

    Attributes
    ----------
    hidden_size : int
        Size of the hidden layer.
    max_length : int
        Maximum length of the input sequence.
    pe : torch.Tensor
        Positional encoding tensor of shape (max_length, hidden_size).

    Methods
    -------
    forward(x)
        Add positional encoding to the input tensor.
    """

    def __init__(self, hidden_size, max_length=500):
        """
        Initialize the PositionalEncoding layer.

        Parameters
        ----------
        hidden_size : int
            Size of the hidden layer.
        max_length : int, optional
            Maximum length of the input sequence. Defaults to 500.
        """
        super(PositionalEncoding, self).__init__()
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.register_buffer('pe', self._get_positional_encoding(hidden_size, max_length))

    def _get_positional_encoding(self, hidden_size, max_length):
        """
        Compute the positional encoding tensor.

        Parameters
        ----------
        hidden_size : int
            Size of the hidden layer.
        max_length : int
            Maximum length of the input sequence.

        Returns
        -------
        torch.Tensor
            Positional encoding tensor of shape (max_length, hidden_size).
        """
        pe = torch.zeros(max_length, hidden_size)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(np.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe

    def forward(self, x):
        """
        The forward pass of the positional encoding layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, hidden_size), a batch of padded sequences, on device.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, hidden_size) with positional encoding added.
        """
        pe = self.pe[:x.size(1)].to(x.device)
        x = x + pe
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer with scaled dot-product attention.

    Attributes
    ----------
    hidden_size : int
        Size of the hidden layer.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    activation : callable
        Activation function to use.

    Methods
    -------
    forward(x, L)
        Compute the forward pass of the multi-head attention layer.
    """

    def __init__(self, hidden_size, num_heads, dropout):
        """
        Initialize the MultiHeadAttention layer.

        Parameters
        ----------
        hidden_size : int
            Size of the hidden layer.
        num_heads : int
            Number of attention heads.
        dropout : float
            Dropout rate.
        """
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = F.softmax

        self.q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.d_k = torch.tensor(self.hidden_size // self.num_heads, dtype=torch.float32)

    def forward(self, x, L):
        """
        The forward pass of the multi-head attention layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, hidden_size), a batch of padded sequences, on device.
        L : torch.Tensor
            Length of the sequences of shape (batch_size), on cpu.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, hidden_size) after applying multi-head attention.
        """
        batch_size = x.size(0)

        # Create a mask to ignore padding
        # mask : [batch_size, seq_len] -> mask : [batch_size, num_heads, seq_len]
        mask = torch.arange(x.size(1), device=x.device)[
            None, :] >= L[:, None].to(x.device)
        mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1)
        mask = ~mask

        # Linear transformation
        # x : [batch_size, seq_len, hidden_size] -> q, k, v : [batch_size, seq_len, hidden_size]
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Reshape
        # q, k, v : [batch_size, seq_len, hidden_size] -> q, k, v : [batch_size, num_heads, seq_len, hidden_size // num_heads]
        q = q.view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        # q, k : [batch_size, num_heads, seq_len, hidden_size // num_heads] -> scores : [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(self.d_k.to(x.device))

        # Masking
        # scores : [batch_size, num_heads, seq_len, seq_len] -> scores : [batch_size, num_heads, seq_len, seq_len]
        mask = mask.unsqueeze(2).expand(-1, -1, scores.size(2), -1)
        scores = scores.masked_fill(mask == 0, -1e4)

        # Activation
        # scores : [batch_size, num_heads, seq_len, seq_len] -> scores : [batch_size, num_heads, seq_len, seq_len]
        scores = self.activation(scores, dim=-1)
        scores = self.dropout(scores)

        # Weighted sum
        # scores, v : [batch_size, num_heads, seq_len, seq_len], [batch_size, num_heads, seq_len, hidden_size // num_heads] -> output : [batch_size, num_heads, seq_len, hidden_size // num_heads]
        output = torch.matmul(scores, v)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]

        # Linear transformation
        # output : [batch_size, seq_len, hidden_size] -> output : [batch_size, seq_len, hidden_size]
        output = self.out(output)

        return output
    
class SelfAttentionLayer(nn.Module):
    """
    Self-attention layer with a single attention head based on scaled dot-product attention.

    Attributes
    ----------
    hidden_size : int
        Size of the hidden layer.
    dropout : float
        Dropout rate.
    activation : callable
        Activation function to use.
    linear_q : nn.Linear
        Linear transformation for the query.
    linear_k : nn.Linear
        Linear transformation for the key.
    linear_v : nn.Linear
        Linear transformation for the value.
    out : nn.Linear
        Linear transformation for the output.
    dropout : nn.Dropout
        Dropout layer.
    d_k : torch.Tensor
        Scaling factor for the dot-product attention.

    Methods
    -------
    forward(x)
        Compute the forward pass of the self-attention layer.
    """
    def __init__(self, hidden_size, dropout, output_size=None):
        """
        Initialize the SelfAttentionLayer layer.

        Parameters
        ----------
        hidden_size : int
            Size of the hidden layer.
        dropout : float
            Dropout rate.
        """
        super(SelfAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.activation = F.softmax

        self.linear_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, hidden_size)

        if self.output_size is not None:
            self.out = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout)

        self.d_k = torch.tensor(self.hidden_size, dtype=torch.float32)

    def forward(self, x):
        """
        The forward pass of the self-attention layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, hidden_size), a batch of vectors, on device.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, hidden_size) after applying self-attention.
        """
        # Linear transformation
        # x : [batch_size, hidden_size] -> q, k, v : [batch_size, hidden_size]
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # Scaled dot-product attention
        # q, k : [batch_size, hidden_size] -> scores : [batch_size, hidden_size]
        scores = torch.matmul(q.unsqueeze(2), k.unsqueeze(1))  # Outer product
        scores = scores / torch.sqrt(self.d_k.to(x.device))  # Scale scores

        # Activation
        # scores : [batch_size, hidden_size] -> scores : [batch_size, hidden_size]
        scores = self.activation(scores, dim=-1)
        scores = self.dropout(scores)

        # Weighted sum
        # scores, v : [batch_size, hidden_size], [batch_size, hidden_size] -> output : [batch_size, hidden_size]
        output = torch.matmul(scores, v.unsqueeze(2)).squeeze(2)
        
        if self.output_size is not None:
            # Linear transformation
            # output : [batch_size, hidden_size] -> output : [batch_size, hidden_size]
            output = self.out(output)

        return output

# == More complex blocks ==
# The more complex blocks are the building blocks of the neural network.
# They are used to build the encoder, the decoder, the model, ...
# They are built using the elementary blocks.

class HighwayBlock(nn.Module):
    """
    The highway block is a combination of a multi-head attention layer and a feedforward layer with a highway connection.
    The highway connection is a weighted sum of the input and the output of the feedforward layer. The weights are learned but are input-independent.

    Attributes
    ----------
    hidden_size : int
        Size of the hidden layer.
    dropout : float
        Dropout rate.
    activation : callable
        Activation function to use.
    multiheadattention : MultiHeadAttention
        Multi-head attention layer.
    resfeedforward : ResidualFeedForward
        Residual feedforward layer.
    norm1 : nn.LayerNorm
        Layer normalization after the multi-head attention layer.
    norm2 : nn.LayerNorm
        Layer normalization after the residual feedforward layer.
    alphas : nn.Parameter
        Transformation gate parameter.
    """

    def __init__(self, hidden_size, dropout, activation=F.relu):
        """
        Initialize the HighwayBlock layer.

        Parameters
        ----------
        hidden_size : int
            Size of the hidden layer.
        dropout : float
            Dropout rate.
        activation : callable, optional
            Activation function to use. Defaults to F.relu.
        """
        super(HighwayBlock, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation

        self.multiheadattention = MultiHeadAttention(hidden_size, 4, dropout)
        self.resfeedforward = ResidualFeedForward(hidden_size, dropout, activation=activation)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # For the moment the transformation gate is a constant but learned parameter
        # we initialize it to have a highway connection that emphasizes carrying
        self.alphas = nn.Parameter(torch.full((hidden_size,), -1., requires_grad=True))

    def forward(self, x, L):
        """
        The forward pass of the highway block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, hidden_size), a batch of padded sequences, on device.
        L : torch.Tensor
            Length of the sequences of shape (batch_size), on cpu.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, hidden_size) after applying the highway block.
        """

        # == Non-linear block ==

        # Process the hidden state with a multi-head attention layer
        # x : [batch_size, seq_len, hidden_size] -> y : [batch_size, seq_len, hidden_size]
        y = self.multiheadattention(x, L)

        # == Transformation gate ==

        # Compute the transformation gate
        # alphas : [hidden_size] -> z : [1, 1, hidden_size]
        z = torch.sigmoid(self.alphas).unsqueeze(0).unsqueeze(0)

        # Highway connection
        # x : [batch_size, seq_len, hidden_size], y : [batch_size, seq_len, hidden_size], z : [1, 1, hidden_size] -> x : [batch_size, seq_len, hidden_size]
        y = z * y + (1 - z) * x

        # Normalize the output highway block
        # y : [batch_size, seq_len, hidden_size] -> y : [batch_size, seq_len, hidden_size]
        y = self.norm1(y)

        # Shared residual feedforward layer with dropout applied on each hidden state
        # y : [batch_size, seq_len, hidden_size] -> y : [batch_size, seq_len, hidden_size]
        y = self.resfeedforward(y)

        # Normalize the output of the residual feedforward layer
        # y : [batch_size, seq_len, hidden_size] -> y : [batch_size, seq_len, hidden_size]
        y = self.norm2(y)

        # Apply dropout
        # y : [batch_size, seq_len, hidden_size] -> y : [batch_size, seq_len, hidden_size]
        y = self.dropout(y)

        return y
    
# == Model blocks ==
# The model blocks are the building blocks of the neural network.
# Those part are made to be trained and used in the final model.

class Encoder(nn.Module):
    """
    The encoder is a stack of highway blocks followed by an aggregation layer.
    The encoder is used to encode the input data into a fixed-size representation.

    Parameters
    ----------
    input_size : int
        Size of the input features.
    hidden_size : int
        Size of the hidden layer.
    num_layers : int
        Number of highway blocks in the encoder.
    dropout : float
        Dropout rate.
    activation : callable, optional
        Activation function to use. Defaults to F.relu.
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            dropout,
            activation=F.relu):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation

        self.embedding = nn.Sequential(
            nn.Linear(
                input_size,
                hidden_size,
                bias=False))
        self.norm = nn.LayerNorm(hidden_size)

        self.highway_blocks = nn.ModuleList(
            [HighwayBlock(hidden_size, dropout, activation) for _ in range(num_layers)])
        self.aggregation = AttentionAggregator(hidden_size, 1, dropout)

    def forward(self, x, L):
        """
        The forward pass of the encoder model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size), a batch of padded sequences, on device.
        L : torch.Tensor
            Length of the sequences of shape (batch_size), on cpu.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, hidden_size) after encoding the input sequence.
        """

        # Embedding of the input
        # x : [batch_size, seq_len, input_size] -> x : [batch_size, seq_len, hidden_size]
        x = self.embedding(x)

        # Normalize the embedding
        # x : [batch_size, seq_len, hidden_size] -> x : [batch_size, seq_len, hidden_size]
        x = self.norm(x)

        # Sequence of highway blocks
        # x : [batch_size, seq_len, hidden_size] -> x : [batch_size, seq_len, hidden_size]
        for highway_block in self.highway_blocks:
            x = highway_block(x, L)

        # Aggregation of the output of the highway blocks
        # x : [batch_size, seq_len, hidden_size] -> x : [batch_size, hidden_size]
        x = self.aggregation(x, L)

        return x
    
class MultiTaskModel(nn.Module):
    """
    The multi-task model is a combination of an encoder and multiple heads. The encoder is a stack of highway blocks
    and the heads are a classification head, a regression head, and a main head.

    Attributes
    ----------
    input_size : int
        Size of the input features.
    hidden_size : int
        Size of the hidden layer.
    hidden_size_classification : int
        Size of the hidden layer for the classification head.
    hidden_size_regression : int
        Size of the hidden layer for the regression head.
    hidden_size_main : int
        Size of the hidden layer for the main head.
    output_size : int
        Size of the output features.
    num_classes : int
        Number of classes for the classification task.
    num_regression : int
        Number of regression targets.
    num_layers_encoder : int
        Number of highway blocks in the encoder.
    num_layers_classification : int
        Number of layers in the classification head.
    num_layers_regression : int
        Number of layers in the regression head.
    num_layers_main : int
        Number of layers in the main head.
    num_layers_logvar : int
        Number of layers in the logvar head.
    dropout : float
        Dropout rate.
    activation : callable
        Activation function to use.
    input_scaler : callable
        Scaler class for input normalization.
    regression_scaler : callable
        Scaler class for regression output normalization.
    main_scaler : callable
        Scaler class for main output normalization.

    Methods
    -------
    fit_regression_scaler(X)
        Fit the regression scaler on the data.
    fit_main_scaler(X)
        Fit the main scaler on the data.
    fit_input_scaler(X)
        Fit the input scaler on the data.
    initialize_logvar()
        Initialize the logvar head.
    state_dict(*args, **kwargs)
        Return the state dictionary of the model including scalers.
    load_state_dict(state_dict, strict=True)
        Load the state dictionary of the model including scalers.
    forward(x, L)
        Perform the forward pass of the model.
    freeze_encoder()
        Freeze the encoder parameters.
    unfreeze_encoder()
        Unfreeze the encoder parameters.
    freeze_classification_head()
        Freeze the classification head parameters.
    unfreeze_classification_head()
        Unfreeze the classification head parameters.
    freeze_regression_head()
        Freeze the regression head parameters.
    unfreeze_regression_head()
        Unfreeze the regression head parameters.
    freeze_main_head()
        Freeze the main head parameters.
    unfreeze_main_head()
        Unfreeze the main head parameters.
    freeze_logvar_head()
        Freeze the logvar head parameters.
    unfreeze_logvar_head()
        Unfreeze the logvar head parameters.
    load_model(path)
        Load the model from a file.
    save_model(path)
        Save the model to a file.
    training_step(batch, critera)
        Perform a training step.
    validation_step(batch, critera)
        Perform a validation step.
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            hidden_size_classification,
            hidden_size_regression,
            hidden_size_main,
            output_size,
            num_classes,
            num_regression,
            num_layers_encoder,
            num_layers_classification,
            num_layers_regression,
            num_layers_main,
            num_layers_logvar,
            dropout,
            activation=F.relu,
            input_scaler=StandardScaler,
            regression_scaler=MinMaxScaler,
            main_scaler=MinMaxScaler
            ):
        super(MultiTaskModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size_classification = hidden_size_classification
        self.hidden_size_regression = hidden_size_regression
        self.hidden_size_main = hidden_size_main
        self.num_classes = num_classes
        self.num_regression = num_regression

        output_size = 2 * output_size # mu and logvar

        self.output_size = output_size
        
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_classification = num_layers_classification
        self.num_layers_regression = num_layers_regression
        self.num_layers_main = num_layers_main
        self.num_layers_logvar = num_layers_logvar
        
        self.dropout = dropout
        self.activation = activation

        # == Encoder ==

        self.encoder = Encoder(
            input_size,
            hidden_size,
            num_layers_encoder,
            dropout,
            activation)
        self.norm_encoder = nn.LayerNorm(hidden_size)

        # == Classification head ==

        self.classification_linear1 = nn.Linear(
            hidden_size, hidden_size_classification)
        self.classification_linear2 = nn.Linear(
            hidden_size_classification, num_classes)
        self.classification_head = nn.Sequential(
            *
            [
                nn.Sequential(
                    ResidualFeedForward(
                        hidden_size_classification,
                        dropout,
                        activation),
                    nn.LayerNorm(hidden_size_classification)) for _ in range(num_layers_classification)])

        # == Regression head ==

        self.regression_linear1 = nn.Linear(
            hidden_size,
            hidden_size_regression)
        self.regression_linear2 = nn.Linear(
            hidden_size_regression, num_regression)
        self.regression_head = nn.Sequential(
            *
            [
                nn.Sequential(
                    ResidualFeedForward(
                        hidden_size_regression,
                        dropout,
                        activation),
                    nn.LayerNorm(hidden_size_regression)) for _ in range(num_layers_regression)])

        # == Main head ==


        self.dropout = nn.Dropout(dropout)
        self.main_norm = nn.LayerNorm(hidden_size + hidden_size_classification + hidden_size_regression)
        self.attention_concat = SelfAttentionLayer(hidden_size + hidden_size_classification + hidden_size_regression, dropout, output_size=hidden_size_main)
        self.main_linear2 = nn.Linear(hidden_size_main, output_size//2) # mu
        self.main_linear3 = nn.Linear(hidden_size_main, output_size//2) # logvar
        self.main_head = nn.Sequential(
            *
            [
                nn.Sequential(
                    ResidualFeedForward(
                        hidden_size_main,
                        dropout,
                        activation),
                    nn.LayerNorm(hidden_size_main)) for _ in range(num_layers_main)])
        self.main_logvar = nn.Sequential(
            *
            [
                nn.Sequential(
                    ResidualFeedForward(
                        hidden_size_main,
                        dropout,
                        activation),
                    nn.LayerNorm(hidden_size_main)) for _ in range(num_layers_logvar)])

        self.initialize_logvar()

        # == Scalers ==

        self.input_scaler = input_scaler(input_size)
        # no need to scale the output of the classification head
        self.regression_scaler = regression_scaler(num_regression)
        self.main_scaler = main_scaler(output_size)

        # revert output_size to its original value
        self.output_size = output_size // 2

    def fit_regression_scaler(self, X):
        """
        Fit the regression scaler on the data.

        Parameters
        ----------
        X : array-like
            Data to fit the scaler on.
        """
        self.regression_scaler.fit(X)

    def fit_main_scaler(self, X):
        """
        Fit the main scaler on the data.

        Parameters
        ----------
        X : array-like
            Data to fit the scaler on.
        """
        self.main_scaler.fit(X)

    def fit_input_scaler(self, X):
        """
        Fit the input scaler on the data.

        Parameters
        ----------
        X : array-like
            Data to fit the scaler on.
        """
        self.input_scaler.fit(np.concatenate(X))

    def initialize_logvar(self):
        """
        Initialize the logvar head such that the initial value of the variance is 1.
        In practice, the main_scaler will shift the logvar to 1 if the head output is 0.
        """
        self.main_linear3.bias.data.fill_(0)
        self.main_linear3.weight.data.fill_(0)

    def state_dict(self, *args, **kwargs):
        """
        Return the state dictionary of the model including scalers.

        Returns
        -------
        dict
            State dictionary of the model including scalers.
        """
        state = super().state_dict(*args, **kwargs)
        # Add scaler states
        state['input_scaler'] = self.input_scaler.state_dict()
        state['regression_scaler'] = self.regression_scaler.state_dict()
        state['main_scaler'] = self.main_scaler.state_dict()
        return state

    def load_state_dict(self, state_dict, strict=True):
        """
        Load the state dictionary of the model including scalers.

        Parameters
        ----------
        state_dict : dict
            State dictionary of the model including scalers.
        strict : bool, optional
            Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict() function. Defaults to True.
        """
        # Load scaler states
        self.input_scaler.load_state_dict(state_dict.pop('input_scaler'))
        self.regression_scaler.load_state_dict(state_dict.pop('regression_scaler'))
        self.main_scaler.load_state_dict(state_dict.pop('main_scaler'))
        # Load remaining model states
        super().load_state_dict(state_dict, strict=strict)

    def forward(self, x, L):
        """
        Perform the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size), a batch of padded sequences, on device.
        L : torch.Tensor
            Length of the sequences of shape (batch_size), on cpu.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            - y_c : Output tensor of the classification head of shape (batch_size, num_classes).
            - z_r : Output tensor of the regression head of shape (batch_size, num_regression).
            - x : Output tensor of the main head of shape (batch_size, 2 * output_size).
        """
        # == Encoder ==

        # Normalize the input
        # x : [batch_size, seq_len, input_size] -> x : [batch_size, seq_len, input_size]
        x = self.input_scaler(x)

        # Encode the input
        # x : [batch_size, seq_len, input_size] -> x : [batch_size, hidden_size]
        x = self.encoder(x, L)

        # Normalize the output of the encoder
        # x : [batch_size, hidden_size] -> x : [batch_size, hidden_size]
        x = self.norm_encoder(x)

        # == Classification head and regression head ==

        # Classification head
        # x : [batch_size, hidden_size] -> y : [batch_size, hidden_size_classification], y_c : [batch_size, num_classes]
        y = self.classification_linear1(x)
        y = self.dropout(y)
        y = self.classification_head(y)
        y_c = self.classification_linear2(y)

        # Regression head
        # x : [batch_size, hidden_size] -> z : [batch_size, num_regression]
        z = self.regression_linear1(x)
        z = self.dropout(z)
        z = self.regression_head(z)
        z_r = self.regression_linear2(z)

        # Inverse the scaling of the regression output
        # z : [batch_size, num_regression] -> z : [batch_size, num_regression]
        z_r = self.regression_scaler.inverse_transform(z_r)

        # == Main head ==

        # Concatenate the output of the encoder, the output of the classification head and the output of the regression head through a multi-head attention layer
        # x : [batch_size, hidden_size], y : [batch_size, hidden_size_classification], z : [batch_size, hidden_size_regression] -> x : [batch_size, hidden_size_main]
        x = torch.cat([x, y, z], dim=-1)
        x = self.main_norm(x)
        x = self.attention_concat(x)
        x = self.dropout(x)

        # Main head
        # x : [batch_size, hidden_size_main] -> x : [batch_size, hidden_size_main]
        x = self.main_head(x)

        # mu
        # x : [batch_size, hidden_size_main] -> x_mu : [batch_size, output_size]
        x_mu = self.main_linear2(x)

        # logvar
        # x : [batch_size, hidden_size_main] -> x_logvar : [batch_size, output_size]
        x = self.main_logvar(x)
        x_logvar = self.main_linear3(x)

        # Concatenate mu and logvar to have x = [mu, logvar]
        # x_mu : [batch_size, output_size], x_logvar : [batch_size, output_size] -> x : [batch_size, 2 * output_size]
        x = torch.cat([x_mu, x_logvar], dim=-1)

        # Inverse the scaling of the main output
        # x : [batch_size, 2 * output_size] -> x : [batch_size, 2 * output_size]
        x = self.main_scaler.inverse_transform(x)

        # Return the output of the classification head and the regression head
        # y_c : [batch_size, num_classes], z_r : [batch_size, num_regression], x : [batch_size, 2 * output_size]
        return y_c, z_r, x
    
    def predict(self, x, L):
        """
        Perform the forward pass of the model and return the main output.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size), a batch of padded sequences, on device.
        L : torch.Tensor
            Length of the sequences of shape (batch_size), on cpu.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_size) after applying the model.
        """
        self.eval()
        with torch.no_grad():
            _, _, x = self.forward(x, L)
            return x[:, :self.output_size]
        
    def freeze_encoder(self):
        """
        Freeze the encoder parameters.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.norm_encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen", flush=True)

    def unfreeze_encoder(self):
        """
        Unfreeze the encoder parameters.
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.norm_encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen", flush=True)

    def freeze_classification_head(self):
        """
        Freeze the classification head parameters.
        """
        for param in self.classification_linear1.parameters():
            param.requires_grad = False
        for param in self.classification_head.parameters():
            param.requires_grad = False
        for param in self.classification_linear2.parameters():
            param.requires_grad = False
        print("Classification head frozen", flush=True)

    def unfreeze_classification_head(self):
        """
        Unfreeze the classification head parameters.
        """
        for param in self.classification_linear1.parameters():
            param.requires_grad = True
        for param in self.classification_head.parameters():
            param.requires_grad = True
        for param in self.classification_linear2.parameters():
            param.requires_grad = True
        print("Classification head unfrozen", flush=True)

    def freeze_regression_head(self):
        """
        Freeze the regression head parameters.
        """
        for param in self.regression_linear1.parameters():
            param.requires_grad = False
        for param in self.regression_head.parameters():
            param.requires_grad = False
        for param in self.regression_linear2.parameters():
            param.requires_grad = False
        print("Regression head frozen", flush=True)

    def unfreeze_regression_head(self):
        """
        Unfreeze the regression head parameters.
        """
        for param in self.regression_linear1.parameters():
            param.requires_grad = True
        for param in self.regression_head.parameters():
            param.requires_grad = True
        for param in self.regression_linear2.parameters():
            param.requires_grad = True
        print("Regression head unfrozen", flush=True)

    def freeze_main_head(self):
        """
        Freeze the main head parameters.
        """
        for param in self.main_linear2.parameters():
            param.requires_grad = False
        for param in self.main_linear3.parameters():
            param.requires_grad = False
        for param in self.main_head.parameters():
            param.requires_grad = False
        for param in self.main_logvar.parameters():
            param.requires_grad = False
        print("Main head frozen", flush=True)

    def unfreeze_main_head(self):
        """
        Unfreeze the main head parameters.
        """
        for param in self.main_linear2.parameters():
            param.requires_grad = True
        for param in self.main_linear3.parameters():
            param.requires_grad = True
        for param in self.main_head.parameters():
            param.requires_grad = True
        for param in self.main_logvar.parameters():
            param.requires_grad = True
        print("Main head unfrozen", flush=True)
        

    def freeze_logvar_head(self):
        """
        Freeze the logvar head parameters.
        """
        for param in self.main_logvar.parameters():
            param.requires_grad = False
        for param in self.main_linear3.parameters():
            param.requires_grad = False
        print("Logvar head frozen", flush=True)

    def unfreeze_logvar_head(self):
        """
        Unfreeze the logvar head parameters.
        """
        for param in self.main_logvar.parameters():
            param.requires_grad = True
        for param in self.main_linear3.parameters():
            param.requires_grad = True
        print("Logvar head unfrozen", flush=True)

    def load_model(self, path):
        """
        Load the model from a file.

        Parameters
        ----------
        path : str
            Path to the file containing the model state dictionary.
        """
        try:
            self.load_state_dict(torch.load(path))
        except:
            print("Error while loading the model")

    def save_model(self, path):
        """
        Save the model to a file.

        Parameters
        ----------
        path : str
            Path to the file where the model state dictionary will be saved.
        """
        torch.save(self.state_dict(), path)

    def training_step(self, batch, critera):
        """
        Perform a training step.

        Parameters
        ----------
        batch : tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            - x : Input tensor of shape (batch_size, seq_len, input_size).
            - y_c : Target tensor for the classification task of shape (batch_size).
            - y_r : Target tensor for the regression task of shape (batch_size, self.output_size + self.num_regression).
        critera : dict
            Dictionary containing the criteria (loss functions and metrics) for the model.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor, dict, dict, dict)
            - loss_c : Classification loss.
            - loss_r : Regression loss.
            - loss_m : Main loss.
            - classification_metrics : Classification metrics.
            - regression_metrics : Regression metrics.
            - main_metrics : Main metrics.
        """
        self.train()
        x, y_c, y_r = batch
        y_c_pred, z_r_pred, x_pred = self(*x)

        y_c = y_c.long()

        loss_c = critera['classification_loss'](y_c_pred, y_c)
        loss_r = critera['regression_loss'](z_r_pred, y_r[:, self.output_size:])
        loss_m = critera['main_loss'](x_pred, y_r[:, :self.output_size])

        classification_metrics = critera['classification_metrics'](y_c_pred, y_c)
        regression_metrics = critera['regression_metrics'](z_r_pred, y_r[:, self.output_size:])
        main_metrics = critera['main_metrics'](x_pred, y_r[:, :self.output_size])

        return loss_c, loss_r, loss_m, classification_metrics, regression_metrics, main_metrics

    def validation_step(self, batch, critera):
        """
        Perform a validation step.

        Parameters
        ----------
        batch : tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            - x : Input tensor of shape (batch_size, seq_len, input_size).
            - y_c : Target tensor for the classification task of shape (batch_size).
            - y_r : Target tensor for the regression task of shape (batch_size, self.output_size + self.num_regression).
        critera : dict
            Dictionary containing the criteria (loss functions and metrics) for the model.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor, dict, dict, dict)
            - loss_c : Classification loss.
            - loss_r : Regression loss.
            - loss_m : Main loss.
            - classification_metrics : Classification metrics.
            - regression_metrics : Regression metrics.
            - main_metrics : Main metrics.
        """
        self.eval()
        with torch.no_grad():
            x, y_c, y_r = batch
            y_c_pred, z_r_pred, x_pred = self(*x)

            y_c = y_c.long()

            loss_c = critera['classification_loss'](y_c_pred, y_c)
            loss_r = critera['regression_loss'](z_r_pred, y_r[:, self.output_size:])
            loss_m = critera['main_loss'](x_pred, y_r[:, :self.output_size])

            classification_metrics = critera['classification_metrics'](y_c_pred, y_c)
            regression_metrics = critera['regression_metrics'](z_r_pred, y_r[:, self.output_size:])
            main_metrics = critera['main_metrics'](x_pred, y_r[:, :self.output_size])

            return loss_c, loss_r, loss_m, classification_metrics, regression_metrics, main_metrics

# == Adapters == 
# I implement wrapper that will enable to easily add adapter to the model and to save separately the adapter and the model
# I implement a AdapterModel that take a model and add the set of adapters to it

class AdapterWrapper(nn.Module):
    """
    AdapterWrapper is a wrapper that adds an adapter layer to a base layer.
    The output of the AdapterWrapper is the sum of the outputs of the base layer and the adapter layer.

    Attributes
    ----------
    base_layer : nn.Module
        The base layer to which the adapter layer is added.
    adapter_layer : nn.Module
        The adapter layer that is added to the base layer.

    Methods
    -------
    forward(*x)
        Perform the forward pass of the AdapterWrapper.
    """
    def __init__(self, base_layer, adapter_layer):
        super(AdapterWrapper, self).__init__()
        self.base_layer = base_layer
        self.adapter_layer = adapter_layer

    def forward(self, *x):
        return self.base_layer(*x) + self.adapter_layer(*x)
    
    def fit(self, *x):
        self.adapter_layer.fit(*x)

    def inverse_transform(self, *x):
        return self.adapter_layer.inverse_transform(*x)

class ReplacerWrapper(nn.Module): # TODO: in the long term, the replacer doesn't need to store the base_layer, it can be removed because only the adapter_layer is forwarded
    """
    ReplacerWrapper is a wrapper that replaces the output of a base layer with the output of an adapter layer.
    The output of the ReplacerWrapper is the output of the adapter layer.

    This wrapper is useful to retrain a specific part of a model without affecting the rest of the model and easily save the adapter separately.

    Attributes
    ----------
    base_layer : nn.Module
        The base layer to be replaced by the adapter layer.
    adapter_layer : nn.Module
        The adapter layer that replaces the base layer.

    Methods
    -------
    forward(*x)
        Perform the forward pass of the ReplacerWrapper.
    """
    def __init__(self, base_layer, adapter_layer):
        super(ReplacerWrapper, self).__init__()
        self.base_layer = base_layer
        self.adapter_layer = adapter_layer

    def forward(self, *x):
        return self.adapter_layer(*x)
    
    def fit(self, *x):
        self.adapter_layer.fit(*x)

    def inverse_transform(self, *x):
        return self.adapter_layer.inverse_transform(*x)
    
class AdaptedModel(nn.Module):
    """
    AdaptedModel is a wrapper that adds adapters to a base model.
    The adapters are applied to specific layers of the base model.

    Attributes
    ----------
    model : nn.Module
        The base model to which the adapters are added.
    adapters : nn.ModuleDict
        A dictionary of adapters to be added to the base model.

    Methods
    -------
    forward(*x)
        Perform the forward pass of the AdaptedModel.
    save_adapters(path)
        Save the adapters to a file.
    load_adapters(path)
        Load the adapters from a file.
    """

    def __init__(self, model, device='cpu'):
        super(AdaptedModel, self).__init__()
        self.model = model
        self.adapters = nn.ModuleDict()    
        self.device = device
        self.build_adapted_model()

    def forward(self, *x):
        """
        Perform the forward pass of the AdaptedModel.

        Parameters
        ----------
        *x : tuple
            Input tensors for the model.

        Returns
        -------
        torch.Tensor
            Output tensor of the model after applying the adapters.
        """
        return self.model(*x)
    
    def predict(self, *x):
        """
        Perform the forward pass of the model and return the main output.

        Parameters
        ----------
        *x : tuple
            Input tensors for the model.

        Returns
        -------
        torch.Tensor
            Output tensor of the model after applying the model.
        """
        self.eval()
        with torch.no_grad():
            return self.model.predict(*x)

    def training_step(self, batch, critera):
        """
        Perform a training step.

        Parameters
        ----------
        batch : tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            - x : Input tensor of shape (batch_size, seq_len, input_size).
            - y_c : Target tensor for the classification task of shape (batch_size).
            - y_r : Target tensor for the regression task of shape (batch_size, self.output_size + self.num_regression).
        critera : dict
            Dictionary containing the criteria (loss functions and metrics) for the model.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor, dict, dict, dict)
            - loss_c : Classification loss.
            - loss_r : Regression loss.
            - loss_m : Main loss.
            - classification_metrics : Classification metrics.
            - regression_metrics : Regression metrics.
            - main_metrics : Main metrics.
        """
        return self.model.training_step(batch, critera)
    
    def validation_step(self, batch, critera):
        """
        Perform a validation step.

        Parameters
        ----------
        batch : tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            - x : Input tensor of shape (batch_size, seq_len, input_size).
            - y_c : Target tensor for the classification task of shape (batch_size).
            - y_r : Target tensor for the regression task of shape (batch_size, self.output_size + self.num_regression).
        critera : dict
            Dictionary containing the criteria (loss functions and metrics) for the model.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor, dict, dict, dict)
            - loss_c : Classification loss.
            - loss_r : Regression loss.
            - loss_m : Main loss.
            - classification_metrics : Classification metrics.
            - regression_metrics : Regression metrics.
            - main_metrics : Main metrics.
        """
        return self.model.validation_step(batch, critera)

    def save_adapters(self, path):
        """
        Save the adapters to a file.

        Parameters
        ----------
        path : str
            Path to the file where the adapters will be saved.
        """
        adapter_state_dict = {name: adapter.adapter_layer.state_dict() for name, adapter in self.adapters.items()}
        torch.save(adapter_state_dict, path)

    def load_adapters(self, path):
        """
        Load the adapters from a file.

        Parameters
        ----------
        path : str
            Path to the file containing the adapters state dictionary.
        """
        adapter_state_dict = torch.load(path)
        for name, state_dict in adapter_state_dict.items():
            self.adapters[name].adapter_layer.load_state_dict(state_dict)

    def freeze_adapters(self):
        """
        Freeze the adapters parameters (only the adapter layers).
        """
        for adapter in self.adapters.values():
            for param in adapter.adapter_layer.parameters():
                param.requires_grad = False

    def unfreeze_adapters(self):
        """
        Unfreeze the adapters parameters (only the adapter layers).
        """
        for adapter in self.adapters.values():
            for param in adapter.adapter_layer.parameters():
                param.requires_grad = True

    def freeze_base(self):
        """
        Freeze the base model parameters.
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        """
        Unfreeze the base model parameters.
        """
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_base_unfreeze_adapters(self):
        """
        Freeze the base model parameters and unfreeze the adapters parameters.
        """
        self.freeze_base()
        self.unfreeze_adapters()
        
    def build_adapted_model(self):
        """
        Build the adapted model by adding adapters to the base model.
        This method should be implemented by child classes to specify the adapters to be added to the base model.

        Returns
        -------
        AdaptedModel
            The adapted model with the specified adapters.
        """
        raise NotImplementedError("This method should be implemented by child classes.")

class NaiveOnlyScaledAdaptedModel(AdaptedModel):
    """
    NaiveOnlyScaledAdaptedModel is a class that adds naive adapters to the model.
    Adapters are just new fitted scaler. In theory, even the number of classes of the forward can be wrong.
    This Naive approach is used to show that simple scaling is not enough to adapt the model.

    #TODO finish the doc
    """
    
    def build_adapted_model(self):
        """
        Build the adapted model by adding naive replacers to the base model.
        The naive replacers are just new fitted scalers.

        Returns
        -------
        AdaptedModel
            The adapted model with the naive replacers.
        """
        # Add naive replacers
        self.adapters['input_scaler'] = ReplacerWrapper(self.model.input_scaler, CumsumDiffStandardScaler(self.model.input_size))
        self.adapters['regression_scaler'] = ReplacerWrapper(self.model.regression_scaler, StandardScaler(self.model.num_regression))
        self.adapters['main_scaler'] = ReplacerWrapper(self.model.main_scaler, HalfMinMaxHalfNothingScaler(self.model.output_size * 2))

        self.model.input_scaler = self.adapters['input_scaler']
        self.model.regression_scaler = self.adapters['regression_scaler']
        self.model.main_scaler = self.adapters['main_scaler']

        self.to(self.device)

class ScaledEmbeddingAdaptedOutputAdaptedModel(AdaptedModel):
    """
    TODO finish the doc
    """

    def build_adapted_model(self):
        """
        TODO
        """

        # Add naive replacers
        self.adapters['input_scaler'] = ReplacerWrapper(self.model.input_scaler,
                                                        CumsumDiffStandardScaler(self.model.input_size))
        self.adapters['regression_scaler'] = ReplacerWrapper(self.model.regression_scaler,
                                                             StandardScaler(self.model.num_regression))
        self.adapters['main_scaler'] = ReplacerWrapper(self.model.main_scaler,
                                                       HalfMinMaxHalfNothingScaler(self.model.output_size * 2))

        self.model.input_scaler = self.adapters['input_scaler']
        self.model.regression_scaler = self.adapters['regression_scaler']
        self.model.main_scaler = self.adapters['main_scaler']

        # add adapter to the embedding layer
        self.adapters['embedding'] = AdapterWrapper(self.model.encoder.embedding, 
                                                    nn.Sequential(nn.Linear(self.model.encoder.input_size, self.model.encoder.hidden_size, bias=False),
                                                                                                nn.LayerNorm(self.model.encoder.hidden_size),
                                                                                                ResidualFeedForward(self.model.encoder.hidden_size, self.model.encoder.dropout, activation=self.model.encoder.activation)))

        self.model.encoder.embedding = self.adapters['embedding']

        # add adapter to the output layers
        # We need to add an adapterwrapper on the 'heads' of the model
        # We need to add a replacerwrapper on the '_linear2' of the model for the classification head
        # We need to add a replacerwrapper on the '_linear2' of the model for the regression head
        # We need to add a replacerwrapper on the '_linear2' of the model for the main head
        # We need to add a replacerwrapper on the '_linear3' of the model for the main head
        self.adapters['classification_linear2'] = ReplacerWrapper(self.model.classification_linear2, nn.Linear(self.model.hidden_size_classification, self.model.num_classes))
        self.adapters['regression_linear2'] = ReplacerWrapper(self.model.regression_linear2, nn.Linear(self.model.hidden_size_regression, self.model.num_regression))
        self.adapters['main_linear2'] = ReplacerWrapper(self.model.main_linear2, nn.Linear(self.model.hidden_size_main, self.model.output_size))
        self.adapters['main_linear3'] = ReplacerWrapper(self.model.main_linear3, nn.Linear(self.model.hidden_size_main, self.model.output_size))

        self.model.classification_linear2 = self.adapters['classification_linear2']
        self.model.regression_linear2 = self.adapters['regression_linear2']
        self.model.main_linear2 = self.adapters['main_linear2']
        self.model.main_linear3 = self.adapters['main_linear3']
        
        self.to(self.device)
