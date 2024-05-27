import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops

# NOTE: when running this model, its currently only possible to run with patch size 16. 
# When decreasing patch size to 8, we will need an additional downsampling layer on the input before inputting to the RNN. 
# This is possible, but not implemented yet.

class DownSampler(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.Conv3d(
            in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.block(x)
    

class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(
            in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0
        )

    def forward(self, x):
        return self.block(x)


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


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
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

# NOTE: added the conv3d rnn class
class Conv3DRNNCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int):
        """
        Initialize a Conv3DRNNCell.

        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Size of the convolutional kernel.
        """
        super(Conv3DRNNCell, self).__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2  # same padding, assuming kernel_size is odd
        self.conv = nn.Conv3d(in_channels=input_channels + hidden_channels,
                              out_channels=hidden_channels,
                              kernel_size=kernel_size,
                              padding=padding)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Conv3DRNNCell.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width, depth).
            hidden_state (torch.Tensor): Previous hidden state of shape (batch_size, hidden_channels, height, width, depth).

        Returns:
            torch.Tensor: Updated hidden state.
        """
        combined = torch.cat([x, hidden_state], dim=1)  # concatenate along channel axis
        hidden_state = torch.tanh(self.conv(combined))
        return hidden_state


class Conv3DRNN(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int, num_layers: int):
        """
        Initialize a Conv3DRNN.

        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Size of the convolutional kernel.
            num_layers (int): Number of Conv3DRNN layers.
        """
        super(Conv3DRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.cells = nn.ModuleList([
            Conv3DRNNCell(input_channels if i == 0 else hidden_channels, hidden_channels, kernel_size)
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, h_0: list = None) -> (torch.Tensor, list):
        """
        Forward pass of the Conv3DRNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, channels, depth, height, width).
            h_0 (list, optional): Initial hidden states for each layer. Defaults to None.

        Returns:
            torch.Tensor: Outputs for all time steps.
            list: Final hidden states for each layer.
        """
        batch_size, seq_len, channels, depth, height, width = x.size()
        
        if h_0 is None:
            h_0 = [torch.zeros(batch_size, self.hidden_channels, depth, height, width).to(x.device)
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


# This is needed to project residual stream to depth dimension to a desired depth of a slice
class ProjectionLayer(nn.Module):
    def __init__(self, d, d_new):
        super(ProjectionLayer, self).__init__()
        self.projection_matrix = nn.Parameter(torch.randn(d, d_new))

    def forward(self, x):
        x_projected = einops.einsum(x, self.projection_matrix, 'b c h w d, d d_new -> b c h w d_new')
        return x_projected
    

class UNET3Dconv(nn.Module):
    def __init__(
        self,
        img_shape=(128, 128, 128),
        input_dim=4,
        output_dim=3,
        embed_dim=96, # embedding dim is vastly reduced to adhere to memory constraints
        patch_size=16,
        num_heads=8,
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

        # Input channels for RNN
        self.input_channels = 12 # NOTE: we can greatly increase this (to like 64?) but i did not do this for testing purposes
        self.hidden_channels = embed_dim # this is a design choice, if changed we need to play around with convolutional layers though, but its possible
        
        start_dim = 16 # this is also a design choice, we can change this depending on memory constraints
        
        # U-Net Decoder
        self.decoder0 = nn.Sequential(
            Conv3DBlock(input_dim, 32, 3), Conv3DBlock(32, start_dim//8, 3)
        )

        self.decoder_z = nn.Sequential(
            Deconv3DBlock(embed_dim, self.hidden_channels),
        )
        
        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(self.hidden_channels, start_dim),
                Deconv3DBlock(start_dim, start_dim//2),
                Deconv3DBlock(start_dim//2, start_dim//4)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(self.hidden_channels, start_dim),
                Deconv3DBlock(start_dim, start_dim//2),
            )

        self.decoder9 = \
            Deconv3DBlock(self.hidden_channels, start_dim)
        
        
        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(start_dim, start_dim),
                SingleDeconv3DBlock(start_dim, start_dim//2)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(start_dim, start_dim//2),
                # Conv3DBlock(start_dim//2, start_dim//2), #NOTE: removed due to memory constraints
                SingleDeconv3DBlock(start_dim//2, start_dim//4)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(start_dim//2, start_dim//4),
                # Conv3DBlock(start_dim//4, start_dim//4), #NOTE: removed due to memory constraints
                SingleDeconv3DBlock(start_dim//4, start_dim//8)
            )
        self.decoder0_header = nn.Sequential(
            Conv3DBlock(start_dim//4, start_dim//8), 
            # Conv3DBlock(start_dim//8, start_dim//8), #NOTE: removed due to memory constraints
            SingleConv3DBlock(start_dim//8, output_dim, 1))

        projection_dim = (img_shape[0] // patch_size)  # we can calculate this dynamically
        
        self.projection_depth1 = ProjectionLayer(projection_dim, 1) # NOTE: chosen to be 1, we can change depending on size of slice
        self.projection_depth2 = ProjectionLayer(projection_dim, 1) # NOTE: chosen to be 1, we can change depending on size of slice
        self.projection_depth3 = ProjectionLayer(projection_dim, 1) # NOTE: chosen to be 1, we can change depending on size of slice
        
        # define rnns
        self.rnn1 = Conv3DRNN(input_channels=self.input_channels,
                        hidden_channels=self.hidden_channels,
                        kernel_size=3,
                        num_layers=1)

        self.rnn2 = Conv3DRNN(input_channels=self.input_channels,
                        hidden_channels=self.hidden_channels,
                        kernel_size=3,
                        num_layers=1)
        
        self.rnn3 = Conv3DRNN(input_channels=self.input_channels,
                        hidden_channels=self.hidden_channels,
                        kernel_size=3,
                        num_layers=1)
        
        # calculate the number of downsamples needed (NOTE: this is done for testing purposes, 
        # img shape will always be the same so this is not needed)
        times_to_downsample = int(math.log2(img_shape[0] / projection_dim))
        
        downsampler_layers = [
            DownSampler(input_dim if i == 0 else self.input_channels, self.input_channels)
            for i in range(times_to_downsample)
        ]
        
        # used to downsample inpput, so it matches that of hidden states when inputting in rnn
        self.downsampler = nn.Sequential(*downsampler_layers)
        
    def forward(self, x):
        z = self.transformer(x)
        
        feature_map = self.decoder0(x)
                
        # print("feature map shape: ", feature_map.shape)
        
        z3, z6, z9 = z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        
        # project z3, z6, z9 to depth 1
        z3 = self.projection_depth1(z3)
        z6 = self.projection_depth2(z6)
        z9 = self.projection_depth3(z9)
               
        print("z3 shape: ", z3.shape)
        print("z6 shape: ", z6.shape)
        print("z9 shape: ", z9.shape)
        
        # TODO: MAKE THIS CONV LAYERS INSTEAD TO INCREASE CHANNELS
        # z3 = self.decoder_z(z3)
        # z6 = self.decoder_z(z6)
        # z9 = self.decoder_z(z9)        
        
        print("z3 shape: ", z3.shape)
        print("z6 shape: ", z6.shape)
        print("z9 shape: ", z9.shape)
        
        #  downsample inpput, so it matches that of hidden states when inputting in rnn
        x = self.downsampler(x) # TODO: here we can do bigger channel dims, but this is smaller now for testing purposes
        
        x = x.unsqueeze(1) # add seq dimension
        x = x.permute(0, -1, 2, 3, 4, 1) # TODO: change this so we can have bigger slices, now we simply swap the dimensions and take slices of 1
        
        print("x shape: ", x.shape)

        # run through rnns
        output3, _ = self.rnn1(x, h_0=[z3])
        output6, _ = self.rnn2(x, h_0=[z6])
        output9, _ = self.rnn3(x, h_0=[z9])
        
        # permute back
        output3 = output3.permute(0, -1, 2, 3, 4, 1)
        output6 = output6.permute(0, -1, 2, 3, 4, 1)
        output9 = output9.permute(0, -1, 2, 3, 4, 1)

        # remove sequence dimension
        output3 = output3.squeeze(1)
        output6 = output6.squeeze(1)
        output9 = output9.squeeze(1)
        
        # TODO: set a start dim which can be bigger, smaller now for testing -> allows for increasing the start_dim, which increases everything 
        output9 = self.decoder9(output9)
        output6 = self.decoder6(output6)
        output3 = self.decoder3(output3)
        
        # from here on we cannot change anything, its all dynamic based on previous choices
        output9 = self.decoder9_upsampler(output9)
        
        output6 = self.decoder6_upsampler(torch.cat([output9, output6], dim=1))
        output3 = self.decoder3_upsampler(torch.cat([output6, output3], dim=1))
        output = self.decoder0_header(torch.cat([feature_map, output3], dim=1))
        return output