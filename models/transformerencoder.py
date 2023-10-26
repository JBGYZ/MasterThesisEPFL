import torch.nn as nn
import torch
import math

from torch import Tensor
from .fcn import FCN

class OutputReducer(nn.Module):
    def __init__(self):
        super(OutputReducer, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        Apply global average pooling along the sequence dimension
        """
        reduced_output = torch.mean(x, dim=2)
        return reduced_output

class PositionalEncoding(nn.Module):
    """
        Absolute positional encoding for short sequences.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(3000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class LearnedPositionalEncoding(nn.Module):
    """
        learned positional encoding for short sequences.
    """
    def __init__(self, d_model, max_seq_len=512):
        super(LearnedPositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).expand(x.size(0), seq_len)
        position_embeddings = self.position_embeddings(positions)
        x = x + position_embeddings
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head self-attention layer 
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        
        # Position-wise feedforward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization for both attention and feedforward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # Multi-head self-attention
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        
        # Position-wise feedforward
        ff_output = self.feed_forward(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        
        return src

class TransformerEncoder(nn.Module):
    """
        Transformer encoder module for classification. Two permutations in forward method
    """
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, args, ch, input_dim, num_outputs, dropout=0.1, reducer_type="fc"):
        super(TransformerEncoder, self).__init__()
        if args.pos_encoder_type == "absolute":
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        elif args.pos_encoder_type == "learned":
            self.pos_encoder = LearnedPositionalEncoding(d_model)
        else:  
            raise NameError("Specify a valid positional encoder type in [absolute, learned]")
        # Stack multiple encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        if reducer_type == "fc":
            self.reducer = FCN(
            num_layers=args.reducer_layers,
            input_channels=ch * input_dim,
            h=args.reducer_size,
            out_dim=num_outputs,
            bias=args.bias,
            )
        elif reducer_type == "linear":
            self.reducer = nn.Linear(ch * input_dim, num_outputs)
        elif reducer_type == "none":
            self.reducer = OutputReducer()
        else:
            raise NameError("Specify a valid reducer type in [fc, linear, none]")
        
    def forward(self, src, src_mask=None):
        src = src.permute(2,0,1)
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        src = src.permute(1,2,0)
        src = src.flatten(1)
        src = self.reducer(src)
        return src