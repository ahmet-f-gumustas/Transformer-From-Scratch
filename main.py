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
    

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
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
        super().__init__()  # Initialize the parent class (nn.Module)
        
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
    

# Define a class for Multi-Head Attention that extends nn.Module
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()  # Initialize the parent class (nn.Module)

        self.d_model = d_model  # Dimensionality of the model
        self.h = h  # Number of attention heads
        assert d_model % h == 0, f"{d_model} % {h} == 0 / d_model is not divisible by h"

        self.d_k = d_model // h  # Dimensionality of each head
        self.w_q = nn.Linear(d_model, d_model)  # Linear transformation for queries
        self.w_k = nn.Linear(d_model, d_model)  # Linear transformation for keys
        self.w_v = nn.Linear(d_model, d_model)  # Linear transformation for values

        self.w_o = nn.Linear(d_model, d_model)  # Final linear transformation for output
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]  # Get the dimension of keys/queries/values

        # Compute scaled dot-product attention scores
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Apply mask by setting masked positions to a large negative value
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # Apply softmax to get probabilities

        if dropout is not None:
            # Apply dropout to attention scores to prevent overfitting
            attention_scores = dropout(attention_scores)

        # Compute the output of the attention layer
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # Linear transformations of the inputs
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Reshape and transpose to get (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Apply attention function to get the output and attention scores
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Concatenate heads and apply the final linear layer
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        return self.w_o(x)



# Define a class for Residual Connection that extends nn.Module
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()  # Initialize the parent class (nn.Module)
        
        # Initialize a dropout layer with the specified dropout rate
        self.dropout = nn.Dropout(dropout)
        
        # Initialize the layer normalization component
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # Apply layer normalization to the input x
        normalized_input = self.norm(x)
        
        # Apply the sublayer function (like a feed-forward network) to the normalized input
        # and apply dropout to its output
        sublayer_output = self.dropout(sublayer(normalized_input))
        
        # Add the original input x to the output of the sublayer, creating a residual connection
        # This step helps in training deep networks by allowing gradients to flow directly through the residuals
        return x + sublayer_output



# Define a class for the Encoder Block of a Transformer that extends nn.Module
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()  # Initialize the parent class (nn.Module)
        
        # Store the self-attention block passed as a parameter
        self.self_attention_block = self_attention_block
        
        # Store the feed-forward block passed as a parameter
        self.feed_forward_block = feed_forward_block
        
        # Initialize two residual connections, one for each of the blocks above
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # First, process the input through the self-attention block within a residual connection
        # The lambda function allows passing the input x to self_attention_block with src_mask
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        
        # Then, process the result through the feed-forward block within another residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        # Return the output which now has been processed by both blocks with their respective residual connections
        return x
    

# Define the class for the Encoder of a Transformer that extends nn.Module
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()  # Initialize the parent class (nn.Module)
        
        # Store the list of encoder blocks passed as a parameter
        self.layers = layers
        
        # Initialize the layer normalization to be applied at the end of the encoder processing
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        # Process the input through each encoder block in sequence
        for layer in self.layers:
            x = layer(x, mask)  # Each layer processes the input and the mask
        
        # After processing through all the layers, apply normalization
        return self.norm(x)  # Return the normalized output


# Define a class for the Decoder Block of a Transformer that extends nn.Module
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block  # Self-attention mechanism for the decoder
        self.cross_attention_block = cross_attention_block  # Cross-attention mechanism that attends to the encoder's output
        self.feed_forward_block = feed_forward_block  # Feed-forward network used within the decoder
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])  # Three residual connections

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Process input through self-attention with residual connection
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # Process through cross-attention with residual connection
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # Feed-forward processing with residual connection
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

# Define the Decoder class that manages multiple layers of DecoderBlocks
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # List of decoder layers
        self.norm = LayerNormalization()  # Layer normalization to stabilize the output

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Sequentially process input through each decoder block
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        # Apply layer normalization after processing all layers
        return self.norm(x)

# Define the Projection Layer for mapping decoder output to vocabulary size
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)  # Linear transformation to vocabulary size

    def forward(self, x):
        # Apply softmax to convert to log probabilities for each vocabulary token
        return torch.log_softmax(self.proj(x), dim=-1)

# Define the Transformer model class that includes both encoder and decoder components
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder  # Encoder component
        self.decoder = decoder  # Decoder component
        self.src_embed = src_embed  # Source embeddings
        self.tgt_embed = tgt_embed  # Target embeddings
        self.src_pos = src_pos  # Source positional encoding
        self.tgt_pos = tgt_pos  # Target positional encoding
        self.projection_layer = projection_layer  # Projection layer to output vocabulary probabilities

    def encode(self, src, src_mask):
        # Embed and encode the source input
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # Embed and decode the target input using encoder's output
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # Project decoder output to vocabulary probabilities
        return self.projection_layer(x)



def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 5, h: int= 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Initialize source and target embedding layers with specified vocabulary sizes and model dimension.
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Initialize positional encoding layers for both source and target with respective sequence lengths.
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Construct encoder blocks. Each block consists of a multi-head attention and a feed-forward network.
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Construct decoder blocks. Each block includes self-attention, cross-attention, and a feed-forward network.
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Wrap the list of encoder and decoder blocks in ModuleList and create the encoder and decoder.
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create a projection layer that maps the decoder output to the target vocabulary size.
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Assemble the full Transformer model with encoder, decoder, embeddings, positional encodings, and projection layer.
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize weights of the model with Xavier uniform distribution, which is commonly used for initializing deep neural networks.
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer













