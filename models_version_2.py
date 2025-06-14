import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr

########## GRU Models#################
class GRU(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param input_dim: number of input features
    :param hidden_dim: hidden size of the GRU
    :param num_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(GRU, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = 0.0 if num_layers == 1 else dropout
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=self.dropout)
        # self.fc = nn.Linear(hidden_dim, input_dim)

        # Residual connection for GRU
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()



    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, h = self.gru(x) #self.gru(x, h0)
        # out, h = out[:, -1, :], h[-1, :, :]  # Extracting from last layer

        residual = self.residual_proj(x)
        out = out + residual
        h = h[-1, :, :] # Last layer and last timestep hidden state
        # out = self.fc(out[:, -1, :])

        return out, h

class StatefulGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StatefulGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden = None  # Initialize hidden state to None

    def forward(self, x):
        if self.hidden is None:
            # No hidden state, perform first call
            out, self.hidden = self.gru(x)
        else:
            out, self.hidden = self.gru(x, self.hidden)  # Pass the maintained hidden state
        self.hidden = self.hidden.detach()  # Detach hidden state from the graph to prevent backprop through the entire sequence history
        out = self.fc(out[:, -1, :])  # Only take the output from the last timestep
        return out

    def reset_hidden_state(self):
        self.hidden = None  
###########################

########## TCN Implementation ########

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, dropout=0.2):
        super(TemporalConvBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        
        # Layer Normalization
        self.ln1 = nn.LayerNorm(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        
        # Layer Normalization
        self.ln2 = nn.LayerNorm(out_channels)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Residual path
        residual = x if self.downsample is None else self.downsample(x)

        # First convolutional block
        out = self.conv1(x)  # Shape: (batch_size, out_channels, seq_len)
        
        # Permute to apply LayerNorm over channels
        out = out.permute(0, 2, 1)  # Shape: (batch_size, seq_len, out_channels)
        out = self.ln1(out)
        out = out.permute(0, 2, 1)  # Shape: (batch_size, out_channels, seq_len)
        
        out = self.relu(out)
        out = self.dropout(out)

        # Second convolutional block
        out = self.conv2(out)  # Shape: (batch_size, out_channels, seq_len)
        
        # Permute to apply LayerNorm over channels
        out = out.permute(0, 2, 1)  # Shape: (batch_size, seq_len, out_channels)
        out = self.ln2(out)
        out = out.permute(0, 2, 1)  # Shape: (batch_size, out_channels, seq_len)
        
        out = self.relu(out)
        out = self.dropout(out)

        # Add residual connection
        out = out + residual

        # Clamp to prevent extreme values
        out = torch.clamp(out, min=-1e5, max=1e5)
        
        return out
    
class TCNFeatureAggregator(nn.Module):
    def __init__(self, num_features, seq_len, different_len, kernel_size=3, dropout=0.2):
        super(TCNFeatureAggregator, self).__init__()
        self.num_features = num_features
        self.channels = [64, 64, 64, num_features]  ## Define three layers: two intermediate layers (64 channels) + final layer (num_features)
        self.init_weights()

        self.temporal_blocks = nn.ModuleList()
        
        # Starting 'in_channels' = num_features
        in_channels = num_features
        
        # Use increasing dilation sizes, for example: 1, 2, 4, 8
        dilation_sizes = [2**i for i in range(4)]
        
        for i, out_channels in enumerate(self.channels):
            padding = (kernel_size - 1) * dilation_sizes[i] // 2
            self.temporal_blocks.append(
                TemporalConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation_sizes[i],
                    padding=padding,
                    dropout=dropout
                )
            )
            in_channels = out_channels

        # Final aggregation layer
        self.aggregator = nn.Conv1d(num_features, num_features, kernel_size=1)

        # Resize layer
        if seq_len >= different_len:
            self.resize_layer = nn.AdaptiveAvgPool1d(different_len)
        else:
            self.resize_layer = nn.Upsample(size=different_len, mode="linear", align_corners=False)
        
        # Forecasting layer
        self.forecast_fc = nn.Linear(seq_len, 1)  # Predicts the next timestep for each feature

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Convert to (batch_size, num_features, seq_len) for Conv1d
        # if torch.isnan(x).any():
        #     print('NaN values detected in input')
        x = x.permute(0, 2, 1)
        for i,block in enumerate(self.temporal_blocks):
            if torch.isnan(x).any():
                print(f'NaN values detected in input: {i}th TCN block')
            x = block(x)
        x = self.aggregator(x)
        # Forecast the next timestep directly from the aggregated features
        forecast = self.forecast_fc(x).squeeze(-1)  # Shape: (batch_size, num_features)

        # return resized_x.permute(0, 2, 1), forecast  # resized x to (batch_size, different_len, num_features) and forecast based on aggregated blocks
        return x.permute(0,2,1), forecast
###########################

#### External Memory With Attention Mechanism - Long Term (Global) Temporal Embeddings ###
class SharedMemoryAttention(nn.Module):
    def __init__(self, emb_len, mem_dim=64, num_memory_slots=64, dropout=0.1):
        super(SharedMemoryAttention, self).__init__()
        self.mem_dim = mem_dim
        self.num_memory_slots = num_memory_slots

        self.keys = nn.Parameter(torch.randn(num_memory_slots, mem_dim))
        self.values = nn.Parameter(torch.randn(num_memory_slots, mem_dim))

        self.query_proj = nn.Linear(emb_len, mem_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.keys)
        nn.init.xavier_uniform_(self.values)
        nn.init.xavier_uniform_(self.query_proj.weight)

    def forward(self, x):
        B, E, feat = x.shape
        x = x.transpose(1, 2)  # (B, F, E)
        
        # Query projection with dropout
        queries = self.query_proj(x)  # (B, F, mem_dim)
        queries = self.dropout(queries)

        keys_t = self.keys.transpose(0, 1)  # (mem_dim, num_memory_slots)
        attention_scores = torch.matmul(queries, keys_t)
        attention_weights = F.softmax(attention_scores, dim=-1)  
        
        # Optional: dropout on attention weights
        attention_weights = self.dropout(attention_weights)

        retrieved = torch.matmul(attention_weights, self.values)  # (B, F, mem_dim)
        retrieved = retrieved.transpose(1, 2)  # (B, mem_dim, F)
        return retrieved

######### Gated Fusion Model##########
class GatedFeedForwardFusion(nn.Module):
    def __init__(self, emb_len, mem_dim, out_dim):
        super(GatedFeedForwardFusion, self).__init__()
        self.emb_len = emb_len
        self.mem_dim = mem_dim
        combined_dim = emb_len + mem_dim

        # Gate and transform fully-connected layers
        self.gate_fc = nn.Linear(combined_dim, combined_dim)
        self.transform_fc = nn.Linear(combined_dim, out_dim)

    def forward(self, short_term, long_term):
        """
        short_term: (B, emb_len, num_features)
        long_term:  (B, mem_dim, num_features)

        Output: (B, out_dim, num_features)
        """
        B, E, feat = short_term.shape  # E = emb_len which could either be tcn_emb_dim or just the window_size
        _, M, _ = long_term.shape   # M = mem_dim

        # Concatenate along the embedding dimension: (B, E+M, feat)
        combined = torch.cat([short_term, long_term], dim=1)  # (B, E+M, feat)

        # We need to apply a linear transform along the embedding dimension. 
        # It's easiest to treat features as a batch dimension for this step.
        # (B, E+M, feat) -> (B, feat, E+M)
        combined = combined.permute(0, 2, 1)  # (B, feat, E+M)

        # Reshape to (B*F, E+M) so we can apply linear layers easily
        BF = B * feat
        combined_reshaped = combined.reshape(BF, E+M)

        # Gating: learn a mask that highlights important parts of combined_dim
        gate_scores = torch.sigmoid(self.gate_fc(combined_reshaped))  # (B*feat, E+M)

        # Apply the gate
        gated = gate_scores * combined_reshaped  # (B*feat, E+M)

        # Transform to out_dim
        transformed = self.transform_fc(gated)  # (B*feat, out_dim)

        # Reshape back to (B, F, out_dim)
        transformed = transformed.view(B, feat, -1)

        # Finally, transpose to (B, out_dim, F) if desired, or leave as (B, F, out_dim).
        # GAT often expects (batch, num_nodes, features), which would be (B, F, out_dim).
        # We'll keep it as (B, F, out_dim).
        
        # (B, F, out_dim) is suitable for a GAT where each feature is considered a node.
        # If you need (B, out_dim, F), you can transpose again.
        
        # Return in a node-friendly format: (B, F, out_dim)
        # return transformed 
        last_ts_transformed = transformed[:,:,-1]
        return transformed, last_ts_transformed


########## GAT Implementation ########


class GraphAttentionNetwork(nn.Module):
    def __init__(self, 
                 num_nodes, 
                 node_embedding_dim, 
                 dropout, 
                 alpha, 
                 use_gatv2=True, 
                 use_bias=True):
        """
        num_nodes: Number of nodes (features) in the graph
        node_embedding_dim: Dimension of the embedding for each node
        dropout: Dropout rate
        alpha: Negative slope for LeakyReLU
        use_gatv2: If True, use GATv2 style attention (Brody et al.)
        use_bias: If True, include a bias term in the attention mechanism
        """
        super(GraphAttentionNetwork, self).__init__()
        self.num_nodes = num_nodes
        self.node_embedding_dim = node_embedding_dim
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.use_bias = use_bias

        # In GATv2, linear transformation is applied after concatenation,
        # so the effective input dimension doubles.
        if self.use_gatv2:
            # For GATv2, we first concatenate pairs of node embeddings 
            # (each of dimension node_embedding_dim), so input to lin is 2 * node_embedding_dim
            lin_input_dim = 2 * node_embedding_dim
            a_input_dim = node_embedding_dim  # after lin transforms, we get node_embedding_dim back
        else:
            # Original GAT: a_input is constructed before linear transformation
            lin_input_dim = node_embedding_dim
            a_input_dim = 2 * node_embedding_dim

        self.lin = nn.Linear(lin_input_dim, node_embedding_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

        # Residual connection projection (if dimensions differ)
        self.residual_proj = nn.Linear(node_embedding_dim, node_embedding_dim) if not self.use_gatv2 else nn.Identity()

    def forward(self, x):
        """
        x: (B, num_nodes, node_embedding_dim)
           Represents a batch of graphs, each with num_nodes, and each node has a node_embedding_dim-dimensional embedding.
        """
        x =x.transpose(1,2)
        # We’ll assume x is already in the correct shape: (B, K, D) where K = num_nodes, D = node_embedding_dim
        residual = self.residual_proj(x)  # Project input for residual connection

        if self.use_gatv2:
            # GATv2: build pairwise concatenations in original embedding space
            a_input = self._make_attention_input(x)                 # (B, K, K, 2*D)
            a_input = self.leakyrelu(self.lin(a_input))             # (B, K, K, D)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (B, K, K, 1) -> (B, K, K)
        else:
            # Original GAT: transform x first, then form pairwise combinations
            Wx = self.lin(x)                                        # (B, K, D)
            a_input = self._make_attention_input(Wx)                # (B, K, K, 2*D)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)  # (B, K, K, 1) -> (B, K, K)

        if self.use_bias:
            e = e + self.bias

        # Compute attention weights
        attention = torch.softmax(e, dim=2)   # (B, K, K)
        attention = F.dropout(attention, self.dropout, training=self.training)  # (B, K, K)

        # Compute new node embeddings
        # (B, K, K) x (B, K, D) -> (B, K, D)
        h = self.sigmoid(torch.matmul(attention, x))

        #Add residual connection 
        h = h + residual
        h =h.transpose(1,2)
        return h  # (B, K, D)

    def _make_attention_input(self, v):
        """
        v: (B, K, D)
        Creates pairwise concatenations of node embeddings for all node pairs in the graph:
        For each pair of nodes (i, j), we form [v_i || v_j], resulting in a (B, K, K, 2*D) tensor.
        """
        B, K, D = v.size()

        # Repeat/reshape to form all pairs (i, j):
        # blocks_repeating selects node_i, expanded across all j
        blocks_repeating = v.unsqueeze(2).repeat(1, 1, K, 1)    # (B, K, K, D)
        # blocks_alternating selects node_j, repeated across all i
        blocks_alternating = v.unsqueeze(1).repeat(1, K, 1, 1)  # (B, K, K, D)

        # Concatenate node_i and node_j embeddings along the last dimension
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=-1)  # (B, K, K, 2*D)

        return combined
###########################

class GraphAttentionNetworkMH(nn.Module):
    def __init__(self, 
                 num_nodes, 
                 node_embedding_dim,    
                 dropout, 
                 alpha, 
                 num_heads=4,
                 use_gatv2=True, 
                 use_bias=True):
        """
        num_nodes: Number of nodes (features) in the graph
        node_embedding_dim: Dimension of the embedding for each node
        dropout: Dropout rate
        alpha: Negative slope for LeakyReLU
        use_gatv2: If True, use GATv2 style attention (Brody et al.)
        use_bias: If True, include a bias term in the attention mechanism
        """
        super(GraphAttentionNetworkMH, self).__init__()
        self.num_nodes = num_nodes
        self.node_embedding_dim = node_embedding_dim
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.use_bias = use_bias
        self.num_heads = num_heads


        # In GATv2, linear transformation is applied after concatenation,
        # so the effective input dimension doubles.
        if self.use_gatv2:
            # For GATv2, we first concatenate pairs of node embeddings 
            # (each of dimension node_embedding_dim), so input to lin is 2 * node_embedding_dim
            lin_input_dim = 2 * node_embedding_dim
            a_input_dim = node_embedding_dim  # after lin transforms, we get node_embedding_dim back
        else:
            # Original GAT: a_input is constructed before linear transformation
            lin_input_dim = node_embedding_dim
            a_input_dim = 2 * node_embedding_dim

         # For multi-head, we create separate parameters for each head
        self.lin = nn.ModuleList([nn.Linear(lin_input_dim, node_embedding_dim) for _ in range(num_heads)])
        self.a = nn.ParameterList([nn.Parameter(torch.empty(a_input_dim, 1)) for _ in range(num_heads)])

        for a_param in self.a:
            nn.init.xavier_uniform_(a_param, gain=1.414)

        if self.use_bias:
            # One bias per head or a single bias? Usually one bias per head is fine.
            self.bias = nn.ParameterList([nn.Parameter(torch.zeros(num_nodes, num_nodes)) for _ in range(num_heads)])

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, num_nodes, node_embedding_dim)
        """
        B, K, D = x.size()

        # For each head, compute attention
        head_outputs = []
        for i in range(self.num_heads):
            if self.use_gatv2:
                a_input = self._make_attention_input(x)              # (B, K, K, 2*D)
                a_input = self.leakyrelu(self.lin[i](a_input))       # (B, K, K, D)
                e = torch.matmul(a_input, self.a[i]).squeeze(3)      # (B, K, K)
            else:
                Wx = self.lin[i](x)                                  # (B, K, D)
                a_input = self._make_attention_input(Wx)             # (B, K, K, 2*D)
                e = self.leakyrelu(torch.matmul(a_input, self.a[i])).squeeze(3)  # (B, K, K)

            if self.use_bias:
                e = e + self.bias[i]

            attention = torch.softmax(e, dim=2)   # (B, K, K)
            attention = F.dropout(attention, self.dropout, training=self.training)

            h_head = torch.matmul(attention, x)  # (B, K, D)
            head_outputs.append(h_head)

        # Combine heads
        # Common approaches:
        # 1. Concatenate along the embedding dimension: (B, K, num_heads*D)
        # 2. Average the heads: (B, K, D)
        # averaging over heads  || Concatenation often gives the model more capacity:
        h = torch.stack(head_outputs, dim=3)   # (B, K, D, num_heads)
        h = h.mean(dim=3) # (B, K, D) 

        # Optionally add a final linear layer after concatenation if desired
        # h = self.sigmoid(h) # If you still want nonlinearity, apply it here.

        return h  # (B, K, num_heads * D)

    def _make_attention_input(self, v):
        """
        v: (B, K, D)
        Creates pairwise concatenations of node embeddings: (B, K, K, 2*D)
        """
        B, K, D = v.size()
        blocks_repeating = v.unsqueeze(2).repeat(1, 1, K, 1)
        blocks_alternating = v.unsqueeze(1).repeat(1, K, 1, 1)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=-1)
        return combined
###########################

######## Reconstruction Model ########
    
class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out

class ReconstructionModel(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)

        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out
####################

###### VAE UPDATED ##########
class VAE_UPDATED(nn.Module):
    """
    VAE for reconstructing the entire sequence of shape (seq_len, num_features)
    from a flattened GRU output of shape (seq_len * hidden_dim).
    """

    def __init__(self, seq_len, input_dim, hidden_dim, num_features, latent_dim, dropout=0.1):
        super(VAE_UPDATED, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.latent_dim = latent_dim

        input_dim = input_dim * hidden_dim          # Flattened (seq_len x hidden_dim)
        output_dim = seq_len * num_features       # Flattened (seq_len x num_features)

        # Encoder: from (seq_len * hidden_dim) to latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.mu = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)

        # Decoder: from latent to (seq_len * num_features)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.Sigmoid()  # optional
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, h_seq):
        """
        :param h_seq: shape (B, seq_len, hidden_dim)
        :return: reconstructed sequence (B, seq_len, num_features), mu, log_var
        """

        B = h_seq.size(0)

        # Flatten to (B, seq_len * hidden_dim)
        h_seq = h_seq.contiguous()
        h_flat = h_seq.view(B, -1)

        # Encode
        encoded = self.encoder(h_flat)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode to (B, seq_len * num_features)
        decoded = self.decoder(z)

        # Reshape to (B, seq_len, num_features)
        reconstructed = decoded.view(B, self.seq_len, self.num_features)

        return reconstructed, mu, log_var
####################

######## Forecasting Model ###########
class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)
##############################

#### Temporal Attention for GRU Ouputs ###
    
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1)

    def forward(self, gru_outputs):
        # gru_outputs shape: (batch_size, seq_len, hidden_dim)
        scores = torch.tanh(self.attn_weights(gru_outputs))  # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)              # Attention weights
        context = torch.sum(weights * gru_outputs, dim=1)   # Weighted sum
        return context, weights
################

### Simple RNN ###
class ShallowRNNEncoder(nn.Module):
    def __init__(self, 
                 input_dim,   # number of features in the time series
                 hidden_dim,  
                 output_dim,  # dimension of the embedding to feed into GAT
                 num_layers=1,
                 dropout=0.0):
        super(ShallowRNNEncoder, self).__init__()
        
        # A simple GRU (you can also use LSTM)
        self.rnn = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        
        # A linear layer to map from hidden_dim -> output_dim
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
           A batch of node time-series data.
           For example, if you have N nodes, you might process each node’s time series 
           separately or stack them appropriately.
        """
        # rnn_out: (batch_size, seq_len, hidden_dim)
        # h_n: (num_layers, batch_size, hidden_dim)
        rnn_out, h_n = self.rnn(x)
        
        # We’ll use the final hidden state from the last layer as the node embedding.
        # shape: (num_layers, batch_size, hidden_dim)
        # Select the top layer’s hidden state:
        h_last_layer = h_n[-1]  # (batch_size, hidden_dim)
        
        # Map to desired output dimension
        out = self.fc(h_last_layer)  # (batch_size, output_dim)
        return out

######### MAL_GATE Implementation ##########
class MAL_GATE(nn.Module):
    def __init__(self, 
    num_features,
    output_size, 
    window_size,
    memory_dim,
    num_memory_slots,
    node_embed_dim,
    num_heads,
    gru_num_layers,
    gru_hid_dim= 150, 
    tcn_emb_dim=32,
    tcn_kernel_size= 3, 
    forecast_n_layers=1,
    forecast_hid_dim=150,
    vae_latent_dim = 16,  
    recon_hid_dim = 150,
    recon_n_layers = 1,  
    dropout = 0.2, 
    alpha = 0.2, 
    ):
       
        
        super(MAL_GATE, self).__init__()
        self.num_features = num_features
        self.hyperparams = {
            "window_size": window_size,
            "memory_dim": memory_dim,
            "num_memory_slots":num_memory_slots,
            "dropout": dropout
        }
        self.epsilon = 0.5  # Distance threshold for ε-KNN graph


        self.shallow_rnn  = ShallowRNNEncoder(num_features, gru_hid_dim, gru_hid_dim )
        self.tcn =  TCNFeatureAggregator(num_features, window_size, tcn_emb_dim, tcn_kernel_size, dropout)
        self.memory_module = SharedMemoryAttention(window_size, memory_dim, num_memory_slots, dropout)
        self.gated_fusion = GatedFeedForwardFusion(window_size, memory_dim, node_embed_dim) #window_size can be replaced by tcn_emb_dim 
        self.gat = GraphAttentionNetwork(num_nodes=num_features, node_embedding_dim=node_embed_dim, dropout=dropout, alpha=alpha) #, num_heads = num_heads)
        self.gru = GRU(num_features, gru_hid_dim, gru_num_layers, dropout)
        
        # Add the Temporal Attention Layer
        self.temporal_attention = TemporalAttention(gru_hid_dim)

        self.forecasting_model = Forecasting_Model(num_features, forecast_hid_dim, output_size, forecast_n_layers, dropout)
        self.vae = VAE_UPDATED(window_size, node_embed_dim, num_features, output_size, vae_latent_dim, dropout)
  
    
        
        # Fusion layer
        # self.fc = nn.Linear(gru_hid_dim, num_features) 
    def epsilon_knn_graph(self, features):
        """
        Constructs a distance-based KNN (ε-KNN) adjacency matrix while ensuring num_features remains unchanged.
        """
        batch_size, num_nodes, feature_dim = features.shape
        adjacency_matrices = []
        
        for i in range(batch_size):  
            feature_slice = features[i].cpu().numpy()
            dist_matrix = np.linalg.norm(feature_slice[:, None, :] - feature_slice[None, :, :], axis=-1)
            
            # Ensure each node has at least ONE connection
            adj_matrix = (dist_matrix < self.epsilon).astype(float)

            # **Key Fix: Prevent Isolated Nodes**
            for j in range(num_nodes):
                if np.sum(adj_matrix[j]) == 0:  
                    nearest_idx = np.argmin(dist_matrix[j])  # Find closest node
                    adj_matrix[j, nearest_idx] = 1  # Force at least one connection

            np.fill_diagonal(adj_matrix, 0)  # Remove self-loops
            adjacency_matrices.append(torch.tensor(adj_matrix, dtype=torch.float, device=features.device))

        return torch.stack(adjacency_matrices)  # (batch_size, num_nodes, num_nodes)
    
    def hybrid_adjacency_matrix(x, mi_thresh=0.1, corr_thresh=0.3):
        """
        Constructs a hybrid MI + Pearson correlation adjacency matrix over features (nodes).
        Each feature is treated as a node, and its embedding is the time-series slice over the sliding window.
        
        Parameters:
            x: Tensor of shape (batch_size, window_size, num_features)
        
        Returns:
            adj: Tensor of shape (batch_size, num_features, num_features)
        """
        x_np = x.detach().cpu().numpy()  # shape: (B, W, F)
        batch_size, window_size, num_features = x_np.shape
        adj_matrices = []

        for b in range(batch_size):
            time_window = x_np[b]  # shape: (W, F)
            mi_matrix = np.zeros((num_features, num_features))
            corr_matrix = np.zeros((num_features, num_features))

            for i in range(num_features):
                for j in range(num_features):
                    if i != j:
                        try:
                            xi = time_window[:, i]
                            xj = time_window[:, j]

                            mi_matrix[i, j] = mutual_info_regression(xi.reshape(-1, 1), xj)[0]
                            corr_matrix[i, j], _ = pearsonr(xi, xj)
                        except:
                            mi_matrix[i, j] = 0
                            corr_matrix[i, j] = 0

            hybrid = ((mi_matrix > mi_thresh) | (np.abs(corr_matrix) > corr_thresh)).astype(float)
            np.fill_diagonal(hybrid, 0)
            adj_matrices.append(torch.tensor(hybrid, dtype=torch.float32))

        return torch.stack(adj_matrices).to(x.device)  # shape: (B, F, F)

    def forward(self, x):
        
        # adj_matrix = self.epsilon_knn_graph(x)
        # node_embedding = self.shallow_rnn(x)

        adj_matrix = self.hybrid_adjacency_matrix(x)
        gat_output = self.gat(x) 
        short_term_tcn, tcn_forecast = self.tcn(gat_output)
        long_term = self.memory_module(short_term_tcn)
        h_recon, h_end_pred = self.gated_fusion(short_term_tcn, long_term)
        
        forecast_pred = self.forecasting_model(h_end_pred) #(h_end_pred)
        reconstructed, _, _ = self.vae(h_recon)
        #reconstructed= self.recon_model(h_end_pred)
        return reconstructed, forecast_pred
        # x shape: (batch, seq_len, num_features)

        # # x = x.float()
        # short_term_tcn, tcn_forecast = self.tcn(x)
        # long_term = self.memory_module(short_term_tcn)
        # node_embedding = self.gated_fusion(short_term_tcn, long_term)
        # gat_output = self.gat(node_embedding)         # Pass node embedding to GAT
        
        # # Alternative #1
        # # reconstructed, _, _ = self.vae(gat_output)
        # # forecast_pred, _ =  self.gru(gat_output)
        # # return gat_output, reconstructed, forecast_pred

        # # Alternative #2
        # gru_input = gat_output.transpose(1, 2) 
        # out, h_end=  self.gru(gru_input)

        # # Apply Temporal Attention
        # context_vector, attn_weights = self.temporal_attention(out)


        # h_recon = out # because we want to reconstruct the entire sequence
        # h_end_pred = h_end.view(x.shape[0], -1)   # Hidden state for last timestamp
        # forecast_pred = self.forecasting_model(h_end_pred) #(h_end_pred)
        # reconstructed, _, _ = self.vae(h_recon)
        # #reconstructed= self.recon_model(h_end_pred)
        # return reconstructed, forecast_pred



    def get_hyperparam_summary(self):
        # Return a nicely formatted string or dict
        return ", ".join(f"{k}={v}" for k,v in self.hyperparams.items() if v is not None)

