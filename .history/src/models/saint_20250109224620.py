"""
SAINT (Self-Attention and Intersample Attention Transformer, https://github.com/somepago/saint?tab=readme-ov-file) for NBA Player Performance Prediction

This implementation uses a transformer-based architecture to process both categorical
and continuous features for predicting NBA player performance metrics.
"""
class ResidualMLP(nn.Module):
    """Multi-layer perceptron with residual connections, batch normalization, and dropout."""
    def __init__(self, layer_dims, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_dims)-1):
            block = nn.Sequential(
                nn.Linear(layer_dims[i], layer_dims[i+1]),
                nn.BatchNorm1d(layer_dims[i+1]) if i < len(layer_dims)-2 else nn.Identity(),
                nn.ReLU() if i < len(layer_dims)-2 else nn.Identity(),
                nn.Dropout(dropout) if i < len(layer_dims)-2 else nn.Identity()
            )
            self.layers.append(block)
            
    def forward(self, x):
        residual = x
        for i, block in enumerate(self.layers[:-1]):
            x = block(x)
            # Add residual connection if dimensions match
            if x.shape == residual.shape:
                x = x + residual
                residual = x
        return self.layers[-1](x)


class LayerNormalization(nn.Module):
    """Layer normalization wrapper for transformer components."""
    def __init__(self, dim, transformer_layer):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.transformer_layer = transformer_layer
    
    def forward(self, x, **kwargs):
        return self.transformer_layer(self.norm(x), **kwargs)


class FeedForwardNetwork(nn.Module):
    """Feed-forward network used in transformer blocks."""
    def __init__(self, dim, expansion_factor=4, dropout=0.):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim)
        )

    def forward(self, x):
        return self.network(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, dim, num_heads=8, head_dim=16, dropout=0.):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_output = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.num_heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        attention = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.dropout(attention)
        
        out = einsum('b h i j, b h j d -> b h i d', attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_output(out)


class DualPathTransformer(nn.Module):
    """Transformer that processes both row-wise and column-wise attention."""
    def __init__(self, dim, num_features, depth, num_heads, head_dim=16, attention_dropout=0., ffn_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Row attention
                LayerNormalization(dim, nn.Sequential(
                    MultiHeadAttention(dim, num_heads=num_heads, head_dim=head_dim, dropout=attention_dropout),
                    nn.Dropout(attention_dropout)
                )),
                LayerNormalization(dim, nn.Sequential(
                    FeedForwardNetwork(dim, dropout=ffn_dropout),
                    nn.Dropout(ffn_dropout)
                )),
                # Column attention
                LayerNormalization(dim*num_features, nn.Sequential(
                    MultiHeadAttention(dim*num_features, num_heads=num_heads, head_dim=64, dropout=attention_dropout),
                    nn.Dropout(attention_dropout)
                )),
                LayerNormalization(dim*num_features, nn.Sequential(
                    FeedForwardNetwork(dim*num_features, dropout=ffn_dropout),
                    nn.Dropout(ffn_dropout)
                ))
            ]))

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        
        for row_attention, row_ffn, col_attention, col_ffn in self.layers:
            # Row attention (feature-wise)
            x = row_attention(x) + x
            x = row_ffn(x) + x
            
            # Column attention (between features)
            x_col = rearrange(x, 'b n d -> 1 b (n d)')
            x_col = col_attention(x_col) + x_col
            x_col = col_ffn(x_col) + x_col
            x = rearrange(x_col, '1 b (n d) -> b n d', n=seq_len)
            
        return x


class SAINT(nn.Module):
    """Segregated Attention-based Interpretation Network for Tabular Data.
    
    This model combines categorical and continuous features using embeddings and 
    processes them through a dual-path transformer architecture.
    """
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim=32,
        depth=6,
        num_heads=8,
        output_dim=7,
        num_players=12,
        head_dim=16,
        mlp_hidden_factors=(4, 2),
        continuous_embedding_type='MLP',
        attention_dropout=0.1,
        ffn_dropout=0.1,
        mlp_dropout=0.1
    ):
        super().__init__()
        
        # Architecture parameters
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.categories = categories
        self.num_continuous = num_continuous
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.num_players = num_players
        
        # Category embedding initialization
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0))
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('categories_offset', categories_offset)
        self.category_embeddings = nn.Embedding(self.num_unique_categories, dim)
        
        # Continuous feature processing
        if continuous_embedding_type == 'MLP':
            self.continuous_embeddings = nn.ModuleList([
                ResidualMLP([1, 100, dim], dropout=mlp_dropout) 
                for _ in range(self.num_continuous)
            ])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            num_features = self.num_categories + num_continuous
        else:
            input_size = (dim * self.num_categories) + num_continuous
            num_features = self.num_categories
        
        # Transformer layers
        self.transformer = DualPathTransformer(
            dim=dim,
            num_features=num_features,
            depth=depth,
            num_heads=num_heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout
        )
        
        # Output layers
        hidden_dim = input_size // 8
        hidden_dims = list(map(lambda t: hidden_dim * t, mlp_hidden_factors))
        all_dims = [input_size, *hidden_dims, output_dim]
        self.output_mlp = ResidualMLP(all_dims, dropout=mlp_dropout)
    
    def forward(self, categorical_features, continuous_features):
        """
        Forward pass of the SAINT model.
        
        Args:
            categorical_features: Tensor of shape [batch_size, num_players, num_categorical]
            continuous_features: Tensor of shape [batch_size, num_players, num_continuous]
        
        Returns:
            Tensor of shape [batch_size, num_players, output_dim]
        """
        batch_size = categorical_features.size(0)
        
        # Reshape features to process all players together
        cat_reshaped = categorical_features.view(batch_size * self.num_players, -1)
        cont_reshaped = continuous_features.view(batch_size * self.num_players, -1)
        
        # Process categorical features
        cat_reshaped = cat_reshaped + self.categories_offset
        cat_embedded = self.category_embeddings(cat_reshaped.long())
        
        # Process continuous features
        if hasattr(self, 'continuous_embeddings'):
            cont_embedded = torch.stack([
                self.continuous_embeddings[i](cont_reshaped[:, i:i+1])
                for i in range(self.num_continuous)
            ], dim=1)
        else:
            cont_embedded = cont_reshaped.unsqueeze(-1)
        
        # Combine embeddings
        combined = torch.cat((cat_embedded, cont_embedded), dim=1)
        
        # Transform features
        transformed = self.transformer(combined)
        
        # Generate output
        flattened = transformed.reshape(transformed.size(0), -1)
        output = self.output_mlp(flattened)
        
        return output.view(batch_size, self.num_players, -1)




