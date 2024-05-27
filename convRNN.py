import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2),
        )

    def forward(self, x):
        return self.block(x)

class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)

class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1()
        x = self.act(x)
        x = self.drop(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=786, d_ff=2048, dropout=0.1):
        super().__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int(
            (cube_size[0] * cube_size[1] * cube_size[2])
            / (patch_size * patch_size * patch_size)
        )
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(
            in_channels=input_dim,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.n_patches, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_dim = int(
            (cube_size[0] * cube_size[1] * cube_size[2])
            / (patch_size * patch_size * patch_size)
        )
        self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        self.attn = SelfAttention(num_heads, embed_dim, dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        cube_size,
        patch_size,
        num_heads,
        num_layers,
        dropout,
        extract_layers,
    ):
        super().__init__()
        self.embeddings = Embeddings(
            input_dim, embed_dim, cube_size, patch_size, dropout
        )
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(
                embed_dim, num_heads, dropout, cube_size, patch_size
            )
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            
            
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers

class ConvRNNCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int):
        """
        Initialize a ConvRNNCell.

        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Size of the convolutional kernel.
        """
        super(ConvRNNCell, self).__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2  # same padding, assuming kernel_size is odd
        self.conv = nn.Conv2d(in_channels=input_channels + hidden_channels,
                              out_channels=hidden_channels,
                              kernel_size=kernel_size,
                              padding=padding)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvRNNCell.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).
            hidden_state (torch.Tensor): Previous hidden state of shape (batch_size, hidden_channels, height, width).

        Returns:
            torch.Tensor: Updated hidden state.
        """
        combined = torch.cat([x, hidden_state], dim=1)  # concatenate along channel axis
        hidden_state = torch.tanh(self.conv(combined))
        return hidden_state


class ConvRNN(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int, num_layers: int):
        """
        Initialize a ConvRNN.

        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Size of the convolutional kernel.
            num_layers (int): Number of ConvRNN layers.
        """
        super(ConvRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.cells = nn.ModuleList([
            ConvRNNCell(input_channels if i == 0 else hidden_channels, hidden_channels, kernel_size)
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, h_0: list = None) -> (torch.Tensor, list):
        """
        Forward pass of the ConvRNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, channels, height, width).
            h_0 (list, optional): Initial hidden states for each layer. Defaults to None.

        Returns:
            torch.Tensor: Outputs for all time steps.
            list: Final hidden states for each layer.
        """
        batch_size, seq_len, channels, height, width = x.size()
        
        # Initialize hidden states if not provided
        if h_0 is None:
            h_0 = [torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device)
                   for _ in range(self.num_layers)]
        
        hidden_states = h_0
        outputs = []

        for t in range(seq_len):
            input_t = x[:, t]
            for i, cell in enumerate(self.cells):
                hidden_states[i] = cell(input_t, hidden_states[i])
                input_t = hidden_states[i]
            outputs.append(hidden_states[-1])
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden_states

class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.fc(x)
    
class TransformerConvRNN(nn.Module):
    def __init__(
        self,
        img_shape=(128, 128, 128),
        input_dim=4,
        output_dim=3,
        embed_dim=96, # reduce embedding
        patch_size=32, # incerese patch size
        num_heads=6,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 9
        self.ext_layers = [3, 6, 9]
        self.patch_dim = [int(x / patch_size) for x in img_shape]

        # RNN
        self.input_channel_size_rnn = 256
        self.hidden_channel_size_rnn = embed_dim
        
        # Transformer Encoder
        self.transformer = Transformer(
            input_dim,
            embed_dim,
            img_shape,
            patch_size,
            num_heads,
            self.num_layers,
            dropout,
            self.ext_layers,
        )

        self.rnn = ConvRNN(self.input_channel_size_rnn, 
                           self.hidden_channel_size_rnn, 
                           kernel_size=3,
                           num_layers=1)
        # Decoder
        self.decoder0 = nn.Sequential(
            Conv3DBlock(input_dim, 5, 3)
        )

        self.final_projection = FC(48, 128)
            
    
    
    
    def forward(self, x):
        z = self.transformer(x)
        z3, z6, z9 = z
        
        # reshape the residual streams to be of dim (B, embed_dim, H/patch_size, W/patch_size, D/patch_size)
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim) # shape: (B, 96, 4, 4, 4)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim) # shape: (B, 96, 4, 4, 4)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim) # shape: (B, 96, 4, 4, 4)
        
        feature_map = self.decoder0(x) # shape: (B, 5, 128, 128, 128)
        
        # we want to create a tensor of shape (B, seq_len, C, H, W) while x currently is
        # of shape (B, C, H, W, D), so we take a slice in the depth dimension and stack them in the sequence dimension
        
        x = x.permute(0, 4, 1, 2, 3) # shape: (B, 128, 3, 128, 128) <- (B, seq_len, C, H, W)
        # choose a height and width that works ( i dont use a formula to calculate this, just shapes have to match)
        
        h_dim, w_dim = 12, 16

        # reshape x to be of shape (B, seq_len, C, H, W)
        x = x.view(1, 128, -1, h_dim, w_dim) # shape: (B, 128, 256, 12, 16)
        print("x shape: ", x.shape)
        
        # stack the residual streams along the depth dimension
        h0 = torch.cat((z3, z6, z9), dim=-1) # shape: (B, 96, 4, 4, 12)
  
        # flatten the depth dimension over some arbitrary chosen H and W dimensions
        h0 = h0.view(1, -1, h_dim, w_dim) # shape: (B, 96, 12, 16)
        print("h0 shape: ", h0.shape)
        
        # hidden state should be a list, for each layer a hidden state
        h0_list = [h0]

        output, hidden_states = self.rnn(x, h0_list) # shape: (B, 128, 96, 12, 16)
        print("output shape: ", output.shape)
        # reshape back to [B, 3, 128, 128, 128]
        output = output.view(1, 3, 128, 128, -1) # shape: (B, 3, 128, 128, 48)
        print("output shape: ", output.shape)
        
        # project depth dimension to 128
        output = self.final_projection(output)
        print("output shape: ", output.shape)
        return output        