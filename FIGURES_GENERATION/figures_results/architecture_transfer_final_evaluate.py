import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset, DataLoader
import numpy as np
from inference import SpikeFeatureExtractor
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import os

def check_for_nans(name, tensor):
    if torch.isnan(tensor).any():
        print(f"🚨 NaNs detected in {name}")
        return True
    return False

class LoRA_Adapter(nn.Module):
    def __init__(self, d_model, r=4, output_dim=None):
        super(LoRA_Adapter, self).__init__()
        if output_dim is None:
            output_dim = d_model
        self.output_dim = output_dim
        self.d_model = d_model
        self.r = r
        self.lora_A = nn.Parameter(torch.randn(d_model, r))
        self.lora_B = nn.Parameter(torch.randn(r, output_dim))
        self.scaling = 1.0 / r

    def forward(self, x):
        return self.scaling * (x @ self.lora_A @ self.lora_B)
    
class AdaptedLayer(nn.Module):
    def __init__(self, layer, d_model, r=4, output_dim=None):
        super(AdaptedLayer, self).__init__()
        self.enable_lora = True
        self.lora_layer = layer
        self.lora_adapter = LoRA_Adapter(d_model, r, output_dim)

    def forward(self, x):
        if self.enable_lora:
            x = self.lora_layer(x) + self.lora_adapter(x)
        else:
            x = self.lora_layer(x)
        return x
    
    def enable(self):
        self.enable_lora = True
    
    def disable(self):
        self.enable_lora = False

    def unfreeze_lora_parameters(self):
        """
        Unfreeze the LoRA parameters
        """
        for param in self.lora_adapter.parameters():
            param.requires_grad = True

    def set_lora_weights(self, lora_A, lora_B):
        """
        :param lora_A: LoRA A matrix
        :param lora_B: LoRA B matrix
        """
        if lora_A.size() != self.lora_adapter.lora_A.size():
            raise ValueError(f"LoRA A matrix should be of size {self.lora_adapter.lora_A.size()}, but got {lora_A.size()}")
        if lora_B.size() != self.lora_adapter.lora_B.size():
            raise ValueError(f"LoRA B matrix should be of size {self.lora_adapter.lora_B.size()}, but got {lora_B.size()}")
        
        self.lora_adapter.lora_A.data = lora_A
        self.lora_adapter.lora_B.data = lora_B

    def get_lora_weights(self):
        """
        :return: LoRA A and B matrices
        """
        return self.lora_adapter.lora_A.data, self.lora_adapter.lora_B.data
    
    def get_lora_parameters(self):
        """
        :return: LoRA A and B parameters
        """
        return self.lora_adapter.lora_A, self.lora_adapter.lora_B
    
    def freeze_base_parameters(self):
        """
        Freeze the base layer parameters
        """
        for param in self.lora_layer.parameters():
            param.requires_grad = False

    def unfreeze_base_parameters(self):
        """
        Unfreeze the base layer parameters
        """
        for param in self.lora_layer.parameters():
            param.requires_grad = True

class Embedder(nn.Module):
    def __init__(self, d_encoder, dropout, should_log, max_len=1000, r = 4):
        super(Embedder, self).__init__()
        self.d_encoder = d_encoder
        self.should_log = should_log
        self.max_len = max_len
        self.linear = nn.Linear(2, d_encoder)
        self.linear = AdaptedLayer(self.linear, 2, r=r, output_dim=d_encoder)
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = self.get_sin_cos_positional_encoding(max_len, d_encoder)
        self.register_buffer('pe', self.positional_encoding)
        self.register_buffer('mu', torch.zeros(2))
        self.register_buffer('sigma', torch.ones(2))

    def init_mu_sigma(self, mu, sigma):
        """
        :param mu: mean of the training set
        :param sigma: standard deviation of the training set
        """

        # on the right device ?
        mu = mu.to(self.mu.device)
        sigma = sigma.to(self.sigma.device)
        # check the size
        if mu.size() != (2,):
            raise ValueError(f"mu should be of size (2,), but got {mu.size()}")
        if sigma.size() != (2,):
            raise ValueError(f"sigma should be of size (2,), but got {sigma.size()}")
        
        self.mu.copy_(mu)
        self.sigma.copy_(sigma)

    def get_sin_cos_positional_encoding(self, max_len, d_model):
        """
        :param max_len: maximum length of the sequence
        :param d_model: dimension of the model
        :return: positional encoding matrix of shape (max_len, d_model)
        """
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x, L):
        """
        :param x: input sequence of shape (batch_size, seq_len)
        :param L: length of the sequences before padding of shape (batch_size,)
        :return: embedded sequence of shape (batch_size, seq_len, d_encoder) and mask of shape (batch_size, seq_len - 1)
        """

        # == from x to x_features ==
        # x is the sequence of spike times
        # L is the length of the sequences before padding
        # we compute the ISIs
        if not self.should_log:
            x_ISI = torch.diff(x, dim=1)
        else:
            x_ISI = torch.log1p(torch.abs(torch.diff(x, dim=1)))
        L = L - 1
        # we compute the delta ISIs
        delta_ISI = torch.diff(x_ISI, dim=1)
        # we pad delta_ISI with one zero at the beginning
        delta_ISI = F.pad(delta_ISI, (1, 0), value=0)
        # we stack x_ISI and delta_ISI
        x_features = torch.stack((x_ISI, delta_ISI), dim=-1)
        # we normalize the features
        x_features = (x_features - self.mu) / self.sigma

        # == from x_features to z_emb ==

        # we apply the linear layer
        x_features = self.linear(x_features)
        # we add the positional encoding
        x_features = x_features + self.pe[:x_features.size(1), :]
        # we apply the dropout
        x_features = self.dropout(x_features)
        # we create the mask
        mask = torch.arange(x_features.size(1)).unsqueeze(0) < L.unsqueeze(1)
        mask = mask.to(x_features.device)
        # we apply the mask
        x_features = x_features * mask.unsqueeze(-1).float()
        # we return the embedded sequence and the mask

        # we check for NaNs
        if check_for_nans("x_features", x_features):
            raise ValueError("NaNs detected in x_features")

        return x_features, mask
    
class InteractionCore(nn.Module):
    def __init__(self, d_encoder, n_heads, dropout, n_blocks, activation, r =4):
        super(InteractionCore, self).__init__()

        if activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'gelu':
            activation = nn.GELU()
        elif activation == 'silu':
            activation = nn.SiLU()
        elif activation == 'tanh':
            activation = nn.Tanh()

        self.d_encoder = d_encoder
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.activation = activation

        # Define the multi-head attention and feed-forward layers
        self.attention_layers = nn.ModuleList([nn.MultiheadAttention(d_encoder, n_heads, dropout=dropout, batch_first=True) for _ in range(n_blocks)])
        self.feed_forward_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(d_encoder, d_encoder),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_encoder, d_encoder)
        ) for _ in range(n_blocks)])
        # Define layer normalization
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(d_encoder) for _ in range(n_blocks)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(d_encoder) for _ in range(n_blocks)])
        # Define dropout layers
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_blocks)])
        # Define activation function
        self.activation_fn = activation
        # Define the final linear layer
        self.final_linear = nn.Linear(d_encoder, d_encoder)
        # Define the final layer normalization
        self.final_layer_norm = nn.LayerNorm(d_encoder)
        # Define the final dropout layer
        self.final_dropout = nn.Dropout(dropout)
        # Define the final activation function
        self.final_activation = activation

        # == MODIFY TO PUT LORA ==
        # Define the LoRA adapters for the attention and feed-forward layers
        #self.attention_layers = nn.ModuleList([AdaptedLayer(layer, d_encoder, r=r) for layer in self.attention_layers])
        self.feed_forward_layers = nn.ModuleList([AdaptedLayer(layer, d_encoder, r=r) for layer in self.feed_forward_layers])
        self.final_linear = AdaptedLayer(self.final_linear, d_encoder, r=r)

    def forward(self, x, mask):
        """
        :param x: input sequence of shape (batch_size, seq_len, d_encoder)
        :param mask: attention mask of shape (batch_size, seq_len)
        :return: processed sequence of shape (batch_size, seq_len, d_encoder) and mask of shape (batch_size, seq_len)
        """

        # == from z_emb to z_core ==
        # x is the embedded sequence
        # mask is the attention mask

        for i in range(self.n_blocks):
            # Multi-head attention
            attn_output, _ = self.attention_layers[i](x, x, x, key_padding_mask=~mask)
                
            # check for NaNs
            # it comes from here. Check why by inspecting the mask
            if check_for_nans("attn_output", attn_output):
                raise ValueError("NaNs detected in attn_output")

            x = x + self.dropout_layers[i](attn_output)
            x = self.layer_norms1[i](x)

            # Feed-forward network
            ff_output = self.feed_forward_layers[i](x)
            ff_output = self.activation_fn(ff_output)

            x = x + self.dropout_layers[i](ff_output)
            x = self.layer_norms2[i](x)
            # check for NaNs
            if check_for_nans("ff_output", ff_output):
                raise ValueError("NaNs detected in ff_output")
            if check_for_nans("x", x):
                raise ValueError("NaNs detected in x attention core")

        # Final linear layer
        x = self.final_linear(x)
        # Final layer normalization
        x = self.final_layer_norm(x)
        # Final dropout layer
        x = self.final_dropout(x)
        # Final activation function
        x = self.final_activation(x)
        # Return the processed sequence and mask

        #if check_for_nans("x", x):
        #    raise ValueError("NaNs detected in x after interaction core")

        return x, mask
    
class Pooler(nn.Module):
    def __init__(self, d_encoder, d_latent, dropout, activation, r=4):
        super(Pooler, self).__init__()

        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'gelu':
            activation_fn = nn.GELU()
        elif activation == 'silu':
            activation_fn = nn.SiLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.d_encoder = d_encoder
        self.d_latent = d_latent
        self.dropout = dropout
        self.activation_fn = activation_fn

        self.W_K = nn.Linear(d_encoder, d_encoder)
        self.W_pool = nn.Linear(d_encoder, d_latent)

        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_encoder)
        self.softmax = nn.Softmax(dim=-1)

        # Learnable query vector (shared across all batches)
        self.query_vector = nn.Parameter(torch.randn(1, 1, d_encoder))

        # == MODIFY TO PUT LORA ==
        # Define the LoRA adapter for the pooling layer
        self.W_K = AdaptedLayer(self.W_K, d_encoder, r=r)
        self.W_pool = AdaptedLayer(self.W_pool, d_encoder, r=r, output_dim=d_latent)

    def forward(self, x, mask):
        """
        :param x: Tensor of shape (batch_size, seq_len, d_encoder)
        :param mask: Bool tensor of shape (batch_size, seq_len), where True means valid token
        :return: Tensor of shape (batch_size, d_latent), and original mask
        """

        batch_size, seq_len, _ = x.size()

        # Compute keys from input
        K = self.W_K(x)  # (B, T, d_encoder)

        # Expand query vector to match batch size
        Q = self.query_vector.expand(batch_size, -1, -1)  # (B, 1, d_encoder)

        # Attention scores (B, 1, T)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_encoder ** 0.5)
        with torch.cuda.amp.autocast(enabled=False):
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
            attn_weights = self.softmax(attn_scores.float()).to(attn_scores.dtype)

        # Check for NaNs
        #if check_for_nans("attn_weights", attn_weights):
        #    raise ValueError("NaNs detected in attn_weights")

        # Weighted sum of values (pooled representation)
        pooled_representation = torch.bmm(attn_weights, x)  # (B, 1, d_encoder)
        pooled_representation = pooled_representation.squeeze(1)  # (B, d_encoder)

        # Normalize, project, and activate
        pooled_representation = self.layer_norm(pooled_representation)
        pooled_representation = self.W_pool(pooled_representation)
        pooled_representation = self.dropout_layer(pooled_representation)
        pooled_representation = self.activation_fn(pooled_representation)

        # Check for NaNs
        #if check_for_nans("pooled_representation", pooled_representation):
        #    raise ValueError("NaNs detected in pooled_representation")

        return pooled_representation, mask
    
class Encoder(nn.Module):
    def __init__(self, d_encoder, n_heads, dropout, n_blocks, d_latent, activation, should_log, r=4):
        super(Encoder, self).__init__()
        self.embedder = Embedder(d_encoder, dropout, should_log, r=r)
        self.interaction_core = InteractionCore(d_encoder, n_heads, dropout, n_blocks, activation, r=r)
        self.pooler = Pooler(d_encoder, d_latent, dropout, activation, r=r)
        
    def init_mu_sigma(self, mu, sigma):
        """
        :param mu: mean of the training set
        :param sigma: standard deviation of the training set
        """
        self.embedder.init_mu_sigma(mu, sigma)

    def forward(self, x, L):
        """
        :param x: input sequence of shape (batch_size, seq_len)
        :param L: length of the sequences before padding of shape (batch_size,)
        :return: pooled sequence of shape (batch_size, d_latent)
        """
        # Embed the input sequence
        x_emb, mask = self.embedder(x, L)

        # Process the embedded sequence through the interaction core
        x_core, mask = self.interaction_core(x_emb, mask)

        # Pool the processed sequence to obtain a fixed-size representation
        x_latent, mask = self.pooler(x_core, mask)

        return x_latent
    
class Decoder(nn.Module):
    def __init__(self, d_latent, dropout, n_blocks, activation, inference_only=False, r=4):
        super(Decoder, self).__init__()

        if activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'gelu':
            activation = nn.GELU()
        elif activation == 'silu':
            activation = nn.SiLU()
        elif activation == 'tanh':
            activation = nn.Tanh()

        self.d_latent = d_latent
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.activation = activation

        # Define the residual feedforward networks
        self.ff_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(d_latent, d_latent),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(d_latent, d_latent)
        ) for _ in range(n_blocks)])
        # Define layer normalization
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(d_latent) for _ in range(n_blocks)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(d_latent) for _ in range(n_blocks)])
        # Define dropout layers
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_blocks)])
        # Define the final linear layer
        self.W_final = nn.Linear(d_latent, 2)


        # Define the auxiliary head for classification
        self.classification_head = nn.ModuleList([nn.Sequential(
            nn.Linear(d_latent, d_latent),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(d_latent, d_latent)
        ) for _ in range(2)])
        # Define layer normalization for the classification head
        self.classification_layer_norms = nn.ModuleList([nn.LayerNorm(d_latent) for _ in range(2)])
        # Define dropout layers for the classification head
        self.classification_dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        # Define the final linear layer for the classification head
        self.classification_final = nn.Linear(d_latent, 2)

        # Define the auxiliary head for metrics
        self.metrics_head = nn.ModuleList([nn.Sequential(
            nn.Linear(d_latent, d_latent),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(d_latent, d_latent)
        ) for _ in range(2)])
        # Define layer normalization for the metrics head
        self.metrics_layer_norms = nn.ModuleList([nn.LayerNorm(d_latent) for _ in range(2)])
        # Define dropout layers for the metrics head
        self.metrics_dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        # Define the final linear layer for the metrics head
        self.metrics_final = nn.Linear(d_latent, 5)
        # Define the auxiliary head for uncertainty
        self.uncertainty_head = nn.ModuleList([nn.Sequential(
            nn.Linear(d_latent, d_latent),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(d_latent, d_latent)
        ) for _ in range(2)])
        # Define layer normalization for the uncertainty head
        self.uncertainty_layer_norms = nn.ModuleList([nn.LayerNorm(d_latent) for _ in range(2)])
        # Define dropout layers for the uncertainty head
        self.uncertainty_dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        # Define the final linear layer for the uncertainty head
        self.uncertainty_final = nn.Linear(d_latent, 2)

        # INITIALIZE THE UNCERTAINTY HEAD WITH VERY SMALL VALUES
        self.uncertainty_final.weight.data *= 0.01

        # two vectors of parameters of size d_latent
        self.blending_vector_c = nn.Parameter(torch.randn(d_latent))
        self.blending_vector_m = nn.Parameter(torch.randn(d_latent))

        # == MODIFY TO PUT LORA ==
        # Define the LoRA adapters for the feedforward layers
        self.ff_layers = nn.ModuleList([AdaptedLayer(layer, d_latent, r=r) for layer in self.ff_layers])
        self.W_final = AdaptedLayer(self.W_final, d_latent, r=r, output_dim=2)
        # Define the LoRA adapters for the classification head
        self.classification_head = nn.ModuleList([AdaptedLayer(layer, d_latent, r=r) for layer in self.classification_head])
        self.classification_final = AdaptedLayer(self.classification_final, d_latent, r=r, output_dim=2)
        # Define the LoRA adapters for the metrics head
        self.metrics_head = nn.ModuleList([AdaptedLayer(layer, d_latent, r=r) for layer in self.metrics_head])
        self.metrics_final = AdaptedLayer(self.metrics_final, d_latent, r=r, output_dim=5)
        # Define the LoRA adapters for the uncertainty head
        self.uncertainty_head = nn.ModuleList([AdaptedLayer(layer, d_latent, r=r) for layer in self.uncertainty_head])
        self.uncertainty_final = AdaptedLayer(self.uncertainty_final, d_latent, r=r, output_dim=2)

    def predict(self, x):
        raise NotImplementedError("Please use forward_auxilliary method for training and predict for inference")
    
    def forward(self, x):
        #raise an error, we should use either predict or forward_auxilliary
        raise NotImplementedError("Please use predict method for inference and forward_auxilliary for training")

    def forward_auxilliary(self, x):
        """
        :param x: input sequence of shape (batch_size, d_latent)
        :return: output sequence of shape (batch_size, 2) and auxiliary outputs
        """

        x_temp = x

        # == from z_latent to y_hat_aux ==
        # we compute the classification head
        x_aux_c = x_temp
        for i in range(2):
            ff_output = self.classification_head[i](x_aux_c)
            ff_output = self.activation(ff_output)
            x_aux_c = x_aux_c + self.classification_dropout_layers[i](ff_output)
            x_aux_c = self.classification_layer_norms[i](x_aux_c)

        y_hat_aux_c = self.classification_final(x_aux_c)

        x_aux_m = x_temp
        # we compute the metrics head
        for i in range(2):
            ff_output = self.metrics_head[i](x_aux_m)
            ff_output = self.activation(ff_output)
            x_aux_m = x_aux_m + self.metrics_dropout_layers[i](ff_output)
            x_aux_m = self.metrics_layer_norms[i](x_aux_m)

        y_hat_aux_m = self.metrics_final(x_aux_m)

        # == from z_latent to y_hat ==
        # x is the fixed-size representation
        x_aux = x_temp + self.blending_vector_c * x_aux_c + self.blending_vector_m * x_aux_m
        for i in range(self.n_blocks):
            # Feed-forward network
            ff_output = self.ff_layers[i](x_aux)
            ff_output = self.activation(ff_output)
            x_aux = x_aux + self.dropout_layers[i](ff_output)
            x_aux = self.layer_norms1[i](x_aux)

        # check
        #if check_for_nans("x", x):
        #    raise ValueError("NaNs detected in x in the decoder")

        # Final linear layer
        y_hat = self.W_final(x_aux)

        # we compute the uncertainty head
        for i in range(2):
            ff_output = self.uncertainty_head[i](x_aux)
            ff_output = self.activation(ff_output)
            x_aux = x_aux + self.uncertainty_dropout_layers[i](ff_output)
            x_aux = self.uncertainty_layer_norms[i](x_aux)
        y_hat_aux_s = self.uncertainty_final(x_aux)

        # check
        #if check_for_nans("y_hat_aux_c", y_hat_aux_c):
        #    raise ValueError("NaNs detected in y_hat_aux_c in the decoder")
        #if check_for_nans("y_hat_aux_m", y_hat_aux_m):
        #    raise ValueError("NaNs detected in y_hat_aux_m in the decoder")
        #if check_for_nans("y_hat_aux_s", y_hat_aux_s):
        #    raise ValueError("NaNs detected in y_hat_aux_s in the decoder")
        # check
        #if check_for_nans("y_hat", y_hat):
        #    raise ValueError("NaNs detected in y_hat in the decoder")

        return y_hat, y_hat_aux_c, y_hat_aux_m, y_hat_aux_s

        return y_hat
        
class DICsNet(nn.Module):
    def __init__(self, d_encoder, n_heads, dropout, n_blocks_encoder, n_blocks_decoder, d_latent, activation, inference_only=False, should_log=False, r=4):
        super(DICsNet, self).__init__()
        self.encoder = Encoder(d_encoder, n_heads, dropout, n_blocks_encoder, d_latent, activation, should_log, r)
        self.decoder = Decoder(d_latent, dropout, n_blocks_decoder, activation, inference_only=inference_only, r=r)

    def init_mu_sigma(self, mu, sigma):
        """
        :param mu: mean of the training set
        :param sigma: standard deviation of the training set
        """
        self.encoder.embedder.init_mu_sigma(mu, sigma)

    def predict(self, x, L):
        """
        :param x: input sequence of shape (batch_size, seq_len)
        :param L: length of the sequences before padding of shape (batch_size,)
        :return: output sequence of shape (batch_size, 2)
        """
        # Forward pass through the encoder
        x_latent = self.encoder(x, L)
        # Forward pass through the decoder
        y_hat = self.decoder.predict(x_latent)
        return y_hat
    def forward(self, x):
        raise NotImplementedError("Please use predict method for inference and forward_auxilliary for training")
    def forward_auxilliary(self, x, L):
        """
        :param x: input sequence of shape (batch_size, seq_len)
        :param L: length of the sequences before padding of shape (batch_size,)
        :return: output sequence of shape (batch_size, 2) and auxiliary outputs
        """
        # Forward pass through the encoder
        x_latent = self.encoder(x, L)

        # check for NaNs
        if check_for_nans("x_latent", x_latent):
            raise ValueError("NaNs detected in x_latent")

        # Forward pass through the decoder
        y_hat, y_hat_aux_c, y_hat_aux_m, y_hat_aux_s = self.decoder.forward_auxilliary(x_latent)
        
        return y_hat, y_hat_aux_c, y_hat_aux_m, y_hat_aux_s
    
    def save_lora_adapter(self, path):
        """
        Save the LoRA adapter weights to a file; also save the mu and sigma because it's task dependent
        """
        #Loop through the full model and save the LoRA weights 
        lora_weights = {}
        for name, module in self.named_modules():
            if isinstance(module, AdaptedLayer):
                lora_A, lora_B = module.get_lora_weights()
                lora_weights[name + ".lora_A"] = lora_A.cpu()
                lora_weights[name + ".lora_B"] = lora_B.cpu()

        # add the mu and sigma
        lora_weights["mu"] = self.encoder.embedder.mu.cpu()
        lora_weights["sigma"] = self.encoder.embedder.sigma.cpu()

        # Save the weights to a file
        torch.save(lora_weights, path)

    def load_lora_adapter(self, path):
        """
        Load the LoRA adapter weights from a file
        """
        # Load the weights from the file
        lora_weights = torch.load(path)['lora']

        print(lora_weights)

        # Loop through the full model and load the LoRA weights
        for name, module in self.named_modules():
            if isinstance(module, AdaptedLayer):
                lora_A = lora_weights[name + ".lora_A"]
                lora_B = lora_weights[name + ".lora_B"]
                # print if any error
                try:
                    module.set_lora_weights(lora_A, lora_B)
                except Exception as e:
                    print(f"Error loading LoRA weights for {name}: {e}")
                    raise

        # load mu and sigma
        self.encoder.embedder.mu.data = lora_weights["mu"]
        self.encoder.embedder.sigma.data = lora_weights["sigma"]

    def freeze_base_parameters(self):
        """
        Freeze the base layer parameters
        """
        # loop over ALL the modules and freeze the base parameters, then loop over the adapted layers and unfreeze them; without using the .freeze_base_parameters method
        for name, module in self.named_modules():
            for param in module.parameters():
                param.requires_grad = False

        for name, module in self.named_modules():
            if isinstance(module, AdaptedLayer):
                module.unfreeze_lora_parameters()
                module.enable()

    def get_lora_parameters(self):
        """
        Get the LoRA parameters with their corresponding module names
        :return: dict {module_name: (lora_A, lora_B)}
        """
        lora_parameters = {}
        for name, module in self.named_modules():
            if isinstance(module, AdaptedLayer):
                lora_parameters[name] = module.get_lora_parameters()
        return lora_parameters
    
    @staticmethod
    def HeteroscedasticHuberLoss(y, y_hat, log_sigma):
        """
        :param y: ground truth values
        :param y_hat: predicted values
        :param log_sigma: log of predicted uncertainties
        :return: heteroscedastic Huber loss
        """
        delta = 1.0

        #clamp
        log_sigma = torch.clamp(log_sigma, min=-5, max=2)

        # Compute sigma from log_sigma
        sigma = torch.exp(log_sigma)

        if torch.any(torch.isnan(sigma)):
            raise ValueError("NaN values detected in sigma")

        # Compute the Huber loss
        huber_loss = torch.where(torch.abs(y - y_hat) < delta,
                                0.5 * (y - y_hat) ** 2,
                                delta * (torch.abs(y - y_hat) - 0.5 * delta))
        # Compute the heteroscedastic loss
        heteroscedastic_loss = huber_loss / (sigma ** 2 + 1e-6) + 2 * log_sigma
        return heteroscedastic_loss.mean()

class SpikeTrainDataset(Dataset):
    def __init__(self, data_path, noise_level=2.0, should_log=False, frac=1.0, cherry_pick=False, nl=0):
        
        self.data_path = data_path
        self.noise_level = noise_level
        self.should_log = should_log
        self.spike_feature_extractor = SpikeFeatureExtractor(model="stg")
        self.cherry_pick = cherry_pick
        self.nl = nl

        with np.errstate(all='ignore'):  # Suppress warnings for empty slices

            f = frac

            data_csv = pd.read_csv(data_path, usecols=["g_s", "g_u", "spiking_times"]).sample(frac=f, random_state=42)
            #num_workers_slurm = os.environ.get("SLURM_CPUS_PER_TASK")
            #num_workers_slurm = int(num_workers_slurm)

            num_workers_slurm = 16

            self.data = self.spike_feature_extractor.extract_from_dataframe(data_csv, num_workers=num_workers_slurm, verbose=True)
            # put back g_s and g_u in the data
            self.data["g_s"] = data_csv["g_s"].values
            self.data["g_u"] = data_csv["g_u"].values

            # if 'ID' in data_csv.columns:
            #     self.data["ID"] = data_csv["ID"].values
            if "ID" in data_csv.columns:
                self.data["ID"] = data_csv["ID"].values

            # data_csv not needed anymore
            del data_csv

        self.data = self.load_data()
        # we group by (g_s, g_u) and pick one sample per group
        if cherry_pick:
            self.data = self.data.groupby(["g_s", "g_u"]).apply(lambda x: x.sample(1)).reset_index(drop=True)
            print(f"Number of samples after cherry picking: {len(self.data)}")


    def load_data(self):
        # clean and process the data
        # Print the number of saples with each label
        data = self.data.dropna()
        silent = data[data["label"] == 0]
        spiking = data[data["label"] == 1]
        bursting = data[data["label"] == 2]
        print(f"Number of samples with label 0: {len(silent)}")
        print(f"Number of samples with label 1: {len(spiking)}")
        print(f"Number of samples with label 2: {len(bursting)}")
        # drop silent
        data = data[data["label"] != 0]

        # spiking_times is x
        # (g_s, g_u) is main y
        # (f_spiking, f_intra_bursting, f_inter_bursting, duration_bursting, nbr_spikes_bursting) is aux metrics y
        # label is aux classification y        

        # we need to compute mu and sigma of ISI and delta ISI
        if not self.should_log:
            data["mean_ISI"] = data["spiking_times"].apply(lambda x: np.mean(np.diff(x)))
            data["std_ISI"] = data["spiking_times"].apply(lambda x: np.std(np.diff(x)))
            data["mean_delta_ISI"] = data["spiking_times"].apply(lambda x: np.mean(np.diff(np.diff(x))))
            data["std_delta_ISI"] = data["spiking_times"].apply(lambda x: np.std(np.diff(np.diff(x))))
        else:
            data["mean_ISI"] = data["spiking_times"].apply(lambda x: np.mean(np.log1p(np.diff(x))))
            data["std_ISI"] = data["spiking_times"].apply(lambda x: np.std(np.log1p(np.diff(x))))
            data["mean_delta_ISI"] = data["spiking_times"].apply(lambda x: np.mean(np.diff(np.log1p(np.diff(x)))))
            data["std_delta_ISI"] = data["spiking_times"].apply(lambda x: np.std(np.diff(np.log1p(np.diff(x)))))


        mean_ISI = data["mean_ISI"].mean()
        std_ISI = data["std_ISI"].mean()
        mean_delta_ISI = data["mean_delta_ISI"].mean()
        std_delta_ISI = data["std_delta_ISI"].mean()

        print(f"mean_ISI: {mean_ISI}, std_ISI: {std_ISI}", flush=True)
        print(f"mean_delta_ISI: {mean_delta_ISI}, std_delta_ISI: {std_delta_ISI}", flush=True)

        # into torch tensor
        self.mu = torch.tensor([mean_ISI, mean_delta_ISI])
        self.sigma = torch.tensor([std_ISI, std_delta_ISI])

        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # get the sample
        sample = self.data.iloc[idx]
        # get the spike times
        spike_times = sample["spiking_times"]
        # get the label
        label = sample["label"] - 1  # 0 for spiking, 1 for bursting
        # get the g_s and g_u
        g_s = sample["g_s"]
        g_u = sample["g_u"]
        # get the metrics
        f_spiking = sample["f_spiking"]
        f_intra_bursting = sample["f_intra_bursting"]
        f_inter_bursting = sample["f_inter_bursting"]
        duration_bursting = sample["duration_bursting"]
        nbr_spikes_bursting = sample["nbr_spikes_bursting"]

        # add noise to the spike times
        if self.noise_level is not None:
            # add Gaussian noise

            spike_times = [t + np.random.normal(0, self.noise_level) for t in spike_times]
            
            # apply dropout
            l_spike_times = [t for t in spike_times if np.random.rand() > 0.05]
            if len(l_spike_times) < 5:
                l_spike_times = spike_times
            spike_times = l_spike_times
            # apply windowing only if the number of spikes is greater than 10
            num_spikes = len(spike_times)
            if num_spikes > 10:
                # sample the number of spikes for the recording window
                num_window_spikes = np.random.randint(num_spikes // 2, num_spikes)
                # sample the start index
                start_idx = np.random.randint(0, num_spikes - num_window_spikes)
                end_idx = start_idx + num_window_spikes
                # extract the window
                spike_times = spike_times[start_idx:end_idx]

        # nl is the noise level that is used to add noise to the spike times
        if self.nl > 0:
            spike_times = [t + np.random.normal(0, self.nl) for t in spike_times]
            spike_times = spike_times

        if len(spike_times) > 250:
            spike_times = spike_times[:250]

        return spike_times, label, g_s, g_u, f_spiking, f_intra_bursting, f_inter_bursting, duration_bursting, nbr_spikes_bursting
    
    @staticmethod
    def collate_fn(batch):
        # collate the batch
        spike_times = [torch.tensor(sample[0], dtype=torch.float32) for sample in batch]
        labels = torch.tensor([sample[1] for sample in batch], dtype=torch.int64)
        g_s = torch.tensor([sample[2] for sample in batch], dtype=torch.float32)
        g_u = torch.tensor([sample[3] for sample in batch], dtype=torch.float32)
        f_spiking = torch.tensor([sample[4] for sample in batch], dtype=torch.float32)
        f_intra_bursting = torch.tensor([sample[5] for sample in batch], dtype=torch.float32)
        f_inter_bursting = torch.tensor([sample[6] for sample in batch], dtype=torch.float32)
        duration_bursting = torch.tensor([sample[7] for sample in batch], dtype=torch.float32)
        nbr_spikes_bursting = torch.tensor([sample[8] for sample in batch], dtype=torch.float32)

        # pad the spike times
        spike_times_padded = pad_sequence(spike_times, batch_first=True, padding_value=0)
        L = torch.tensor([len(sample) for sample in spike_times], dtype=torch.float32)

        return spike_times_padded, L, labels, g_s, g_u, f_spiking, f_intra_bursting, f_inter_bursting, duration_bursting, nbr_spikes_bursting
    

def map_to_lora_model(model, dict_before_lora):
    """This function maps the new architecture that includes AdaptedLayer to the old architecture that does not include it.
    It is used to load the weights of the old architecture into the new one.
    """

    model_dict = model.state_dict()
    new_state_dict = {}

    # Load everything that can be loaded
    for key in model_dict.keys():
        if key in dict_before_lora.keys():
            new_state_dict[key] = dict_before_lora[key]
        elif 'lora_layer' in key:
            # Remove the prefix 'lora_layer.'
            new_key = key.replace('lora_layer.', '')
            if new_key in dict_before_lora.keys():
                new_state_dict[key] = dict_before_lora[new_key]
            else:
                # Keep the original weight (random initialization)
                new_state_dict[key] = model_dict[key]
        else:
            # Keep the original weight (random initialization)
            new_state_dict[key] = model_dict[key]

    # Load the updated weights into the model
    model.load_state_dict(new_state_dict)

def main():
   
    # INIT WANDB
    wandb.init(project="DICsNet")
    config = wandb.config

    # CONFIGURATION from the HYPERPARAMETERS TUNING
    
    config.fraction_of_data = 0.1

    config.d_encoder = 64
    config.n_heads = 8
    config.dropout = 0.03406899732322084
    config.n_blocks_encoder = 4
    config.n_blocks_decoder = 2
    config.d_latent = 16
    config.activation = 'gelu'
    config.learning_rate = 2.1045121655425276e-05
    config.should_log = True
    config.lambda_m_factor = 9.18911907136668
    config.lambda_c_factor = 5.443193801814643

    # I increase 
    config.num_epochs = 200
    config.batch_size = 32*4*3

    config.inference_only = False
    config.beta2 = 0.98
    config.weight_decay = 1e-4
    config.val_every = 5
    config.noise_level = 2.0
    config.max_grad_norm = 5.0
    config.T0 = 10


    # == TOTAL

    
    # SETUP DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", str(device))

    # SETUP DATASET
    test_set = SpikeTrainDataset(data_path="./tmp/test_set_da.csv", noise_level=0, should_log=config.should_log, frac=config.fraction_of_data, cherry_pick=False)
    print("test_set_size", len(test_set))

    # SETUP DATALOADER
    #num_worker_slurm = int(os.environ.get("SLURM_CPUS_PER_TASK", 16))
    num_worker_slurm = 16
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True,
        collate_fn=SpikeTrainDataset.collate_fn, prefetch_factor=4, num_workers=num_worker_slurm, pin_memory=True, persistent_workers=True)    
    print("test_loader_size", len(test_loader))

    # SETUP MODEL
    r_value = 32
    model = DICsNet(
        d_encoder=config.d_encoder,
        n_heads=config.n_heads,
        dropout=config.dropout,
        n_blocks_encoder=config.n_blocks_encoder,
        n_blocks_decoder=config.n_blocks_decoder,
        d_latent=config.d_latent,
        activation=config.activation,
        inference_only=config.inference_only,
        should_log=config.should_log,
        r=r_value
        ).to(device)

    m = torch.load("./tmp/best_model_stg.pth", map_location=device)
    map_to_lora_model(model, m['model_state_dict'])

    lora_path = "./tmp/lora_best.pth"
    model.load_lora_adapter(lora_path)

    model = model.to(device)
    # freeze the base parameters
    model.freeze_base_parameters()
    # set the model to evaluation mode
    model.eval()

    # we save the cherry picked value of (g_s, g_u)
    df = pd.DataFrame(columns=["g_s", "g_u", "g_s_hat", "g_u_hat", "f_spiking", "f_intra_bursting", "f_inter_bursting", "duration_bursting", "nbr_spikes_bursting", "label", "f_spiking_hat", "f_intra_bursting_hat", "f_inter_bursting_hat", "duration_bursting_hat", "nbr_spikes_bursting_hat", "label_hat", "L_dics", "sigma_s", "sigma_u"])

    # for each batch in the test loader,
    # we pass the batch to the model
    # and we get the output
    # we save the initial g_s and g_u and
    # the output g_s and g_u
    with torch.no_grad():
        for i, (x, L, y, g_s, g_u, f_spiking, f_intra_bursting, f_inter_bursting, duration_bursting, nbr_spikes_bursting) in enumerate(test_loader):
            # move to device
            x = x.to(device)
            #L = L.to(device)
            y = y.to(device)
            g_s = g_s.to(device)
            g_u = g_u.to(device)
            f_spiking = f_spiking.to(device)
            f_intra_bursting = f_intra_bursting.to(device)
            f_inter_bursting = f_inter_bursting.to(device)
            duration_bursting = duration_bursting.to(device)
            nbr_spikes_bursting = nbr_spikes_bursting.to(device)

            # pass the batch to the model
            with torch.no_grad():
                y_hat, y_hat_aux_c, y_hat_aux_m, y_hat_aux_s = model.forward_auxilliary(x, L)

                # get the output g_s and g_u
                g_s_hat = y_hat[:, 0]
                g_u_hat = y_hat[:, 1]

                f_spiking_hat = y_hat_aux_m[:, 0]
                f_intra_bursting_hat = y_hat_aux_m[:, 1]
                f_inter_bursting_hat = y_hat_aux_m[:, 2]
                duration_bursting_hat = y_hat_aux_m[:, 3]
                nbr_spikes_bursting_hat = y_hat_aux_m[:, 4]
                label_hat = y_hat_aux_c.argmax(dim=1)

                # L_dics is loss_uncertainty
                L_dics = model.HeteroscedasticHuberLoss(g_s, g_s_hat, y_hat_aux_s[:, 0]) + model.HeteroscedasticHuberLoss(g_u, g_u_hat, y_hat_aux_s[:, 1])

                # get the sigma
                sigma_s = y_hat_aux_s[:, 0]
                sigma_u = y_hat_aux_s[:, 1]

                # get the g_s and g_u
                L_dics = L_dics.cpu().numpy()
                sigma_s = sigma_s.cpu().numpy()
                sigma_u = sigma_u.cpu().numpy()
                g_s = g_s.cpu().numpy()
                g_u = g_u.cpu().numpy()
                g_s_hat = g_s_hat.cpu().numpy()
                g_u_hat = g_u_hat.cpu().numpy()
                f_spiking = f_spiking.cpu().numpy()
                f_intra_bursting = f_intra_bursting.cpu().numpy()
                f_inter_bursting = f_inter_bursting.cpu().numpy()
                duration_bursting = duration_bursting.cpu().numpy()
                nbr_spikes_bursting = nbr_spikes_bursting.cpu().numpy()
                label = y.cpu().numpy()
                f_spiking_hat = f_spiking_hat.cpu().numpy()
                f_intra_bursting_hat = f_intra_bursting_hat.cpu().numpy()
                f_inter_bursting_hat = f_inter_bursting_hat.cpu().numpy()
                duration_bursting_hat = duration_bursting_hat.cpu().numpy()
                nbr_spikes_bursting_hat = nbr_spikes_bursting_hat.cpu().numpy()
                label_hat = label_hat.cpu().numpy()
                # append to the dataframe
                df = pd.concat([df, pd.DataFrame({
                    "g_s": g_s,
                    "g_u": g_u,
                    "g_s_hat": g_s_hat,
                    "g_u_hat": g_u_hat,
                    "f_spiking": f_spiking,
                    "f_intra_bursting": f_intra_bursting,
                    "f_inter_bursting": f_inter_bursting,
                    "duration_bursting": duration_bursting,
                    "nbr_spikes_bursting": nbr_spikes_bursting,
                    "label": label,
                    "f_spiking_hat": f_spiking_hat,
                    "f_intra_bursting_hat": f_intra_bursting_hat,
                    "f_inter_bursting_hat": f_inter_bursting_hat,
                    "duration_bursting_hat": duration_bursting_hat,
                    "nbr_spikes_bursting_hat": nbr_spikes_bursting_hat,
                    "label_hat": label_hat,
                    "L_dics": L_dics,
                    "sigma_s": sigma_s,
                    "sigma_u": sigma_u
                })], ignore_index=True)
            # print the progress
            if i % 10 == 0:
                print(f"Batch {i}/{len(test_loader)}", flush=True)      

    df.to_csv("./tmp/cherry_picked_g_s_g_u_total_da.csv", index=False)

    # SETUP DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", str(device))

    # SETUP DATASET
    test_set = SpikeTrainDataset(data_path="./tmp/test_set_da.csv", noise_level=0, should_log=config.should_log, frac=config.fraction_of_data, cherry_pick=True)
    print("test_set_size", len(test_set))

    # SETUP DATALOADER
    #num_worker_slurm = int(os.environ.get("SLURM_CPUS_PER_TASK", 16))
    num_worker_slurm = 16
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True,
        collate_fn=SpikeTrainDataset.collate_fn, prefetch_factor=4, num_workers=num_worker_slurm, pin_memory=True, persistent_workers=True)    
    print("test_loader_size", len(test_loader))


    # we save the cherry picked value of (g_s, g_u)
    df = pd.DataFrame(columns=["g_s", "g_u", "g_s_hat", "g_u_hat", "f_spiking", "f_intra_bursting", "f_inter_bursting", "duration_bursting", "nbr_spikes_bursting", "label", "f_spiking_hat", "f_intra_bursting_hat", "f_inter_bursting_hat", "duration_bursting_hat", "nbr_spikes_bursting_hat", "label_hat", "L_dics", "sigma_s", "sigma_u"])

    # for each batch in the test loader,
    # we pass the batch to the model
    # and we get the output
    # we save the initial g_s and g_u and
    # the output g_s and g_u
    with torch.no_grad():
        for i, (x, L, y, g_s, g_u, f_spiking, f_intra_bursting, f_inter_bursting, duration_bursting, nbr_spikes_bursting) in enumerate(test_loader):
            # move to device
            x = x.to(device)
            #L = L.to(device)
            y = y.to(device)
            g_s = g_s.to(device)
            g_u = g_u.to(device)
            f_spiking = f_spiking.to(device)
            f_intra_bursting = f_intra_bursting.to(device)
            f_inter_bursting = f_inter_bursting.to(device)
            duration_bursting = duration_bursting.to(device)
            nbr_spikes_bursting = nbr_spikes_bursting.to(device)

            # pass the batch to the model
            with torch.no_grad():
                y_hat, y_hat_aux_c, y_hat_aux_m, y_hat_aux_s = model.forward_auxilliary(x, L)

                # get the output g_s and g_u
                g_s_hat = y_hat[:, 0]
                g_u_hat = y_hat[:, 1]

                f_spiking_hat = y_hat_aux_m[:, 0]
                f_intra_bursting_hat = y_hat_aux_m[:, 1]
                f_inter_bursting_hat = y_hat_aux_m[:, 2]
                duration_bursting_hat = y_hat_aux_m[:, 3]
                nbr_spikes_bursting_hat = y_hat_aux_m[:, 4]
                label_hat = y_hat_aux_c.argmax(dim=1)

                # L_dics is loss_uncertainty
                L_dics = model.HeteroscedasticHuberLoss(g_s, g_s_hat, y_hat_aux_s[:, 0]) + model.HeteroscedasticHuberLoss(g_u, g_u_hat, y_hat_aux_s[:, 1])

                # get the sigma
                sigma_s = y_hat_aux_s[:, 0]
                sigma_u = y_hat_aux_s[:, 1]

                # get the g_s and g_u
                L_dics = L_dics.cpu().numpy()
                sigma_s = sigma_s.cpu().numpy()
                sigma_u = sigma_u.cpu().numpy()
                g_s = g_s.cpu().numpy()
                g_u = g_u.cpu().numpy()
                g_s_hat = g_s_hat.cpu().numpy()
                g_u_hat = g_u_hat.cpu().numpy()
                f_spiking = f_spiking.cpu().numpy()
                f_intra_bursting = f_intra_bursting.cpu().numpy()
                f_inter_bursting = f_inter_bursting.cpu().numpy()
                duration_bursting = duration_bursting.cpu().numpy()
                nbr_spikes_bursting = nbr_spikes_bursting.cpu().numpy()
                label = y.cpu().numpy()
                f_spiking_hat = f_spiking_hat.cpu().numpy()
                f_intra_bursting_hat = f_intra_bursting_hat.cpu().numpy()
                f_inter_bursting_hat = f_inter_bursting_hat.cpu().numpy()
                duration_bursting_hat = duration_bursting_hat.cpu().numpy()
                nbr_spikes_bursting_hat = nbr_spikes_bursting_hat.cpu().numpy()
                label_hat = label_hat.cpu().numpy()
                # append to the dataframe
                df = pd.concat([df, pd.DataFrame({
                    "g_s": g_s,
                    "g_u": g_u,
                    "g_s_hat": g_s_hat,
                    "g_u_hat": g_u_hat,
                    "f_spiking": f_spiking,
                    "f_intra_bursting": f_intra_bursting,
                    "f_inter_bursting": f_inter_bursting,
                    "duration_bursting": duration_bursting,
                    "nbr_spikes_bursting": nbr_spikes_bursting,
                    "label": label,
                    "f_spiking_hat": f_spiking_hat,
                    "f_intra_bursting_hat": f_intra_bursting_hat,
                    "f_inter_bursting_hat": f_inter_bursting_hat,
                    "duration_bursting_hat": duration_bursting_hat,
                    "nbr_spikes_bursting_hat": nbr_spikes_bursting_hat,
                    "label_hat": label_hat,
                    "L_dics": L_dics,
                    "sigma_s": sigma_s,
                    "sigma_u": sigma_u
                })], ignore_index=True)
            # print the progress
            if i % 10 == 0:
                print(f"Batch {i}/{len(test_loader)}", flush=True)      

    df.to_csv("./tmp/cherry_picked_g_s_g_u_da.csv", index=False)


    # == WE BUILD THE PREDICTION DATASET == 
    # we separate in cherry picked and not cherry picked;
    # We associate a common ID based on the g_s and g_u values
    # we forward the cherry picked dataset through the model and save the g_s_hat and g_u_hat
    # we save in separate files the cherry picked (and the predictions and the ID) and the not cherry picked (and the ID)

    # we read the csv again, forward through the SpikeFeatureExtractor ; remove the silent ones ; cherry pick the data and and save in two csv files
    sfe = SpikeFeatureExtractor(model="da")
    data_csv = pd.read_csv("./tmp/test_set_da.csv", usecols=["g_s", "g_u", "spiking_times"]).sample(frac=1.0, random_state=42)
    data = sfe.extract_from_dataframe(data_csv, num_workers=16, verbose=True)
    data["g_s"] = data_csv["g_s"].values
    data["g_u"] = data_csv["g_u"].values
    data = data.dropna()
    data = data[data["label"] != 0]

    # add the ID but shouuld use float precision of torch before converting to string
    data["ID"] = torch.tensor(data["g_s"].values).float().cpu().numpy().astype(str) + "_" + torch.tensor(data["g_u"].values).float().cpu().numpy().astype(str)

    cherry_picked = data.groupby(["g_s", "g_u"], group_keys=False).sample(n=1, random_state=42)
    not_cherry_picked = data.drop(cherry_picked.index)

    # spiking time should be saved as a list comma separated string and with '[' and ']'
    cherry_picked["spiking_times"] = cherry_picked["spiking_times"].apply(lambda x: "[" + ",".join(map(str, x)) + "]")
    not_cherry_picked["spiking_times"] = not_cherry_picked["spiking_times"].apply(lambda x: "[" + ",".join(map(str, x)) + "]")

    # save the data
    cherry_picked.to_csv("./tmp/cherry_picked_da.csv", index=False)
    not_cherry_picked.to_csv("./tmp/not_cherry_picked_da.csv", index=False)

    # forward the cherry picked dataset through the model and save the g_s_hat and g_u_hat
    cherry_picked_set = SpikeTrainDataset(data_path="./tmp/cherry_picked_da.csv", noise_level=0, should_log=config.should_log, frac=config.fraction_of_data, cherry_pick=False)
    cherry_picked_loader = DataLoader(cherry_picked_set, batch_size=config.batch_size, shuffle=True,
        collate_fn=SpikeTrainDataset.collate_fn, prefetch_factor=4, num_workers=num_worker_slurm, pin_memory=True, persistent_workers=True)
    print("cherry_picked_loader_size", len(cherry_picked_loader))
    # for each batch in the cherry picked loader,
    # we pass the batch to the model
    # and we get the output
    # we save the initial g_s and g_u and
    # the output g_s and g_u along with the ID
    df_cherry_picked = pd.DataFrame(columns=["g_s", "g_u", "g_s_hat", "g_u_hat", "ID"])
    with torch.no_grad():
        for i, (x, L, y, g_s, g_u, f_spiking, f_intra_bursting, f_inter_bursting, duration_bursting, nbr_spikes_bursting) in enumerate(cherry_picked_loader):
            # move to device
            x = x.to(device)
            #L = L.to(device)
            y = y.to(device)
            g_s = g_s.to(device)
            g_u = g_u.to(device)
            f_spiking = f_spiking.to(device)
            f_intra_bursting = f_intra_bursting.to(device)
            f_inter_bursting = f_inter_bursting.to(device)
            duration_bursting = duration_bursting.to(device)
            nbr_spikes_bursting = nbr_spikes_bursting.to(device)

            # pass the batch to the model
            with torch.no_grad():
                y_hat, y_hat_aux_c, y_hat_aux_m, y_hat_aux_s = model.forward_auxilliary(x, L)

                # get the output g_s and g_u
                g_s_hat = y_hat[:, 0]
                g_u_hat = y_hat[:, 1]

                # build back the ID
                ID = g_s.cpu().numpy().astype(str) + "_" + g_u.cpu().numpy().astype(str)

                g_s_hat = g_s_hat.cpu().numpy()
                g_u_hat = g_u_hat.cpu().numpy()
                g_s = g_s.cpu().numpy()
                g_u = g_u.cpu().numpy()
                
                # append to the dataframe
                df_cherry_picked = pd.concat([df_cherry_picked, pd.DataFrame({
                    "g_s": g_s,
                    "g_u": g_u,
                    "g_s_hat": g_s_hat,
                    "g_u_hat": g_u_hat,
                    "ID": ID
                })], ignore_index=True)
            # print the progress
            if i % 10 == 0:
                print(f"Batch {i}/{len(cherry_picked_loader)}", flush=True)
    # save the dataframe
    df_cherry_picked.to_csv("./tmp/cherry_picked_predictions_da.csv", index=False)

if __name__ == "__main__":
    main()
    wandb.finish()