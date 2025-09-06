import typing as tp
import numpy as np
import torch
import torch.nn as nn


class StreamableConv1d(nn.Module):
    """Streamable 1D convolution with optional normalization."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True,
                 norm: str = 'none', norm_kwargs: dict = {}, causal: bool = False, 
                 pad_mode: str = 'reflect'):
        super().__init__()
        self.causal = causal
        self.pad_mode = pad_mode
        
        # Calculate padding
        if causal:
            self.padding = (kernel_size - 1) * dilation
        else:
            self.padding = ((kernel_size - 1) * dilation) // 2
            
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, 
                              padding=0, dilation=dilation, groups=groups, bias=bias)
        
        # Normalization
        if norm == 'weight_norm':
            self.conv = nn.utils.weight_norm(self.conv)
            self.norm = nn.Identity()  # Weight norm is applied to conv, not as separate layer
        elif norm == 'batch_norm':
            self.norm = nn.BatchNorm1d(out_channels, **norm_kwargs)
        elif norm == 'layer_norm':
            self.norm = nn.LayerNorm(out_channels, **norm_kwargs)
        else:
            self.norm = nn.Identity()
    
    def forward(self, x):
        if self.causal:
            x = torch.nn.functional.pad(x, (self.padding, 0), mode='constant')
        else:
            x = torch.nn.functional.pad(x, (self.padding, self.padding), mode=self.pad_mode)
        x = self.conv(x)
        return self.norm(x)


class StreamableConvTranspose1d(nn.Module):
    """Streamable 1D transposed convolution."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True,
                 norm: str = 'none', norm_kwargs: dict = {}, causal: bool = False,
                 trim_right_ratio: float = 1.0):
        super().__init__()
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        
        self.conv_tr = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, 
                                          stride, padding=0, dilation=dilation, 
                                          groups=groups, bias=bias)
        
        # Normalization
        if norm == 'weight_norm':
            self.conv_tr = nn.utils.weight_norm(self.conv_tr)
            self.norm = nn.Identity()  # Weight norm is applied to conv, not as separate layer
        elif norm == 'batch_norm':
            self.norm = nn.BatchNorm1d(out_channels, **norm_kwargs)
        elif norm == 'layer_norm':
            self.norm = nn.LayerNorm(out_channels, **norm_kwargs)
        else:
            self.norm = nn.Identity()
    
    def forward(self, x):
        x = self.conv_tr(x)
        x = self.norm(x)
        
        # Trim for causal setup
        if self.causal:
            trim_right = int(x.shape[-1] * self.trim_right_ratio)
            x = x[..., :trim_right]
            
        return x


class StreamableLSTM(nn.Module):
    """Streamable LSTM layer."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, 
                 bidirectional: bool = False, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           bidirectional=bidirectional, dropout=dropout, batch_first=False)
    
    def forward(self, x):
        # x: [B, C, T] -> [T, B, C]
        x = x.transpose(1, 2)
        output, _ = self.lstm(x)
        # [T, B, C] -> [B, C, T]
        return output.transpose(1, 2)


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model."""
    def __init__(self, dim: int, kernel_sizes: tp.List[int] = [3, 1], dilations: tp.List[int] = [1, 1],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False,
                 pad_mode: str = 'reflect', compress: int = 2, true_skip: bool = True):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), 'Number of kernel sizes should match number of dilations'
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                StreamableConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation,
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, pad_mode=pad_mode),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = StreamableConv1d(dim, dim, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                                             causal=causal, pad_mode=pad_mode)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Module):
    """SEANet encoder."""
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 3,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = True, compress: int = 2, lstm: int = 0,
                 disable_norm_outer_blocks: int = 0):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, \
            "Number of blocks for which to disable norm is invalid."

        act = getattr(nn, activation)
        mult = 1
        model: tp.List[nn.Module] = [
            StreamableConv1d(channels, mult * n_filters, kernel_size,
                             norm='none' if self.disable_norm_outer_blocks >= 1 else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = 'none' if self.disable_norm_outer_blocks >= i + 2 else norm
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      norm=block_norm, norm_params=norm_params,
                                      activation=activation, activation_params=activation_params,
                                      causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

            # Add downsampling layers
            model += [
                act(**activation_params),
                StreamableConv1d(mult * n_filters, mult * n_filters * 2,
                                 kernel_size=ratio * 2, stride=ratio,
                                 norm=block_norm, norm_kwargs=norm_params,
                                 causal=causal, pad_mode=pad_mode),
            ]
            mult *= 2

        if lstm:
            model += [StreamableLSTM(mult * n_filters, num_layers=lstm)]

        model += [
            act(**activation_params),
            StreamableConv1d(mult * n_filters, dimension, last_kernel_size,
                             norm='none' if self.disable_norm_outer_blocks == self.n_blocks else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class SEANetDecoder(nn.Module):
    """SEANet decoder."""
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 3,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 final_activation: tp.Optional[str] = None, final_activation_params: tp.Optional[dict] = None,
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = True, compress: int = 2, lstm: int = 0,
                 disable_norm_outer_blocks: int = 0, trim_right_ratio: float = 1.0):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, \
            "Number of blocks for which to disable norm is invalid."

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = [
            StreamableConv1d(dimension, mult * n_filters, kernel_size,
                             norm='none' if self.disable_norm_outer_blocks == self.n_blocks else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]

        if lstm:
            model += [StreamableLSTM(mult * n_filters, num_layers=lstm)]

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = 'none' if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1) else norm
            # Add upsampling layers
            model += [
                act(**activation_params),
                StreamableConvTranspose1d(mult * n_filters, mult * n_filters // 2,
                                          kernel_size=ratio * 2, stride=ratio,
                                          norm=block_norm, norm_kwargs=norm_params,
                                          causal=causal, trim_right_ratio=trim_right_ratio),
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      activation=activation, activation_params=activation_params,
                                      norm=block_norm, norm_params=norm_params, causal=causal,
                                      pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

            mult //= 2

        # Add final layers
        model += [
            act(**activation_params),
            StreamableConv1d(n_filters, channels, last_kernel_size,
                             norm='none' if self.disable_norm_outer_blocks >= 1 else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [
                final_act(**final_activation_params)
            ]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        y = self.model(z)
        return y
