from torch import nn
import torch
import math

# Define a Transformer model class inheriting from nn.Module
class Transformer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()  # Initialize the parent class (nn.Module)

        # Store the model dimension and vocabulary size
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Define an embedding layer with vocabulary size and model dimension
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Forward pass of the transformer model
        # Multiply the embeddings by the square root of the model dimension to normalize
        return self.embedding(x) * math.sqrt(self.d_model)
    

# Define a Positional Encoding class inheriting from nn.Module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()  # Initialize the parent class (nn.Module)
        
        # Store model dimension, sequence length, and dropout rate
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)  # Define a dropout layer with the given dropout rate

        # Create a positional encoding matrix of shape (sequence length, model dimension)
        # Initialize it with zeros
        pe = torch.zeros(seq_len, d_model)

        # Create a tensor of positions (0 to seq_len-1), shaped as (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Calculate the divisor used for scaling the sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sinusoidal functions to even indices in the positional encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine functions to odd indices in the positional encoding matrix
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension (B, seq_len, d_model) with B=1
        pe = pe.unsqueeze(0)

        # Register 'pe' as a buffer that should not be considered a model parameter
        # Buffers, like parameters, are persistent states for the module but are not trained
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input tensor
        # The positional encoding is sliced to match the input size in case of shorter sequences
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)

        # Apply dropout to the resulting tensor for regularization
        return self.dropout(x)
    


# Define a class for Layer Normalization that extends nn.Module
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()  # Initialize the parent class (nn.Module)
        
        # Epsilon value to prevent division by zero during normalization
        self.eps = eps
        
        # Learnable scale parameter initialized to ones
        # This parameter is multiplied with the normalized input
        self.alpha = nn.Parameter(torch.ones(1))
        
        # Learnable bias parameter initialized to zeros
        # This parameter is added to the scaled normalized input
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Calculate the mean of the input tensor along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        
        # Calculate the standard deviation of the input tensor along the last dimension
        std = x.std(dim=-1, keepdim=True)
        
        # Normalize the input by subtracting the mean and dividing by the standard deviation
        # Then scale and shift the normalized output using the learnable parameters alpha and bias
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

# Define a class for Feed Forward Block that extends nn.Module
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super(FeedForwardBlock, self).__init__()  # Initialize the parent class (nn.Module)
        
        # First linear layer increases dimensionality from d_model to d_ff
        self.linear_1 = nn.Linear(d_model, d_ff)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        
        # Second linear layer decreases dimensionality back from d_ff to d_model
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Apply the first linear transformation
        x = self.linear_1(x)
        
        # Apply ReLU activation function to add non-linearity
        x = torch.relu(x)
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Apply the second linear transformation
        return self.linear_2(x)