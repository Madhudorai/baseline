from abc import ABC, abstractmethod
import logging
from pathlib import Path
import typing as tp

import torch
from torch import nn
import torch.nn.functional as F


logger = logging.getLogger()


class QuantizedResult:
    """Result of quantization containing quantized tensor, codes, bandwidth, and penalty."""
    def __init__(self, x: torch.Tensor, codes: torch.Tensor, bandwidth: torch.Tensor, penalty: torch.Tensor):
        self.x = x
        self.codes = codes
        self.bandwidth = bandwidth
        self.penalty = penalty


class TwoBranchQuantizedResult:
    """Result of 2-branch quantization containing quantized tensor, codes, bandwidth, penalty, diversity loss, and consistency loss."""
    def __init__(self, x: torch.Tensor, codes: torch.Tensor, bandwidth: torch.Tensor, penalty: torch.Tensor, diversity_loss: torch.Tensor, consistency_loss: torch.Tensor):
        self.x = x
        self.codes = codes
        self.bandwidth = bandwidth
        self.penalty = penalty
        self.diversity_loss = diversity_loss
        self.consistency_loss = consistency_loss


class EuclideanCodebook(nn.Module):
    """Euclidean codebook for vector quantization."""
    def __init__(self, dim: int, codebook_size: int, kmeans_init: bool = True, kmeans_iters: int = 10,
                 decay: float = 0.8, epsilon: float = 1e-5, threshold_ema_dead_code: float = 2.):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        
        # Initialize codebook
        self.register_buffer('embed', torch.randn(codebook_size, dim))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', torch.randn(codebook_size, dim))
        self.register_buffer('inited', torch.tensor(True))
        
        if kmeans_init:
            self.inited.data.fill_(False)
    
    def init_embed_(self, data):
        """Initialize codebook using k-means."""
        if self.inited:
            return
        
        data = data.reshape(-1, self.dim)
        embed, _ = self.kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.inited.data.fill_(True)
    
    def kmeans(self, data, k, iters):
        """Simple k-means implementation."""
        data = data.cpu()
        n_samples = data.size(0)
        
        # If we have fewer samples than k, repeat some samples
        if n_samples < k:
            # Repeat data to have at least k samples
            repeat_factor = (k // n_samples) + 1
            data = data.repeat(repeat_factor, 1)
            n_samples = data.size(0)
        
        # Select k random centroids
        centroids = data[torch.randperm(n_samples)[:k]]
        
        for _ in range(iters):
            distances = torch.cdist(data, centroids)
            labels = distances.argmin(dim=1)
            for i in range(k):
                mask = labels == i
                if mask.sum() > 0:
                    centroids[i] = data[mask].mean(dim=0)
        
        return centroids.to(data.device), labels.to(data.device)
    
    def expire_codes_(self, x):
        """Replace dead codes with random vectors from current batch."""
        if self.threshold_ema_dead_code == 0:
            return
        
        x = x.reshape(-1, self.dim)
        distances = torch.cdist(x, self.embed)
        labels = distances.argmin(dim=1)
        
        for i in range(self.codebook_size):
            if self.cluster_size[i] < self.threshold_ema_dead_code:
                # Replace with random vector from current batch
                mask = labels == i
                if mask.sum() > 0:
                    self.embed[i] = x[mask][torch.randint(0, mask.sum(), (1,))]
    
    def quantize(self, x):
        """Quantize input vectors to nearest codebook entries."""
        distances = torch.cdist(x, self.embed)
        return distances.argmin(dim=1)
    
    def dequantize(self, embed_ind):
        """Dequantize indices back to vectors."""
        return self.embed[embed_ind]
    
    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = x.reshape(-1, self.dim)
        
        self.init_embed_(x)
        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        quantize = self.dequantize(embed_ind)
        
        if self.training:
            self.expire_codes_(x)
            # Update cluster size and embed average
            self.cluster_size.mul_(self.decay).add_(embed_onehot.sum(0), alpha=1 - self.decay)
            embed_sum = x.t() @ embed_onehot
            self.embed_avg.mul_(self.decay).add_(embed_sum.t(), alpha=1 - self.decay)
            
            # Update codebook
            cluster_size = self.cluster_size + self.epsilon
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
        
        quantize = quantize.reshape(shape)
        return quantize, embed_ind.reshape(shape[:-1])


class VectorQuantization(nn.Module):
    """Vector quantization with commitment loss."""
    def __init__(self, dim: int, codebook_size: int, commitment_weight: float = 1.0, 
                 decay: float = 0.8, epsilon: float = 1e-5, kmeans_init: bool = True, 
                 kmeans_iters: int = 10, threshold_ema_dead_code: float = 2.):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        
        self._codebook = EuclideanCodebook(
            dim=dim, codebook_size=codebook_size, kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters, decay=decay, epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code
        )
    
    @property
    def codebook(self):
        return self._codebook.embed
    
    def forward(self, x):
        device = x.device
        x_flat = x.reshape(-1, self.dim)
        
        quantize, embed_ind = self._codebook(x_flat)
        quantize = quantize.reshape(x.shape)
        
        if self.training:
            # Straight-through estimator
            quantize = x + (quantize - x).detach()
        
        loss = torch.tensor(0.0, device=device, requires_grad=self.training)
        
        if self.training and self.commitment_weight > 0:
            commit_loss = F.mse_loss(quantize.detach(), x)
            loss = loss + commit_loss * self.commitment_weight
        
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization with multiple codebooks."""
    def __init__(self, dim: int, n_q: int, codebook_size: int, commitment_weight: float = 1.0,
                 decay: float = 0.8, epsilon: float = 1e-5, kmeans_init: bool = True,
                 kmeans_iters: int = 10, threshold_ema_dead_code: float = 2.):
        super().__init__()
        self.dim = dim
        self.n_q = n_q
        self.codebook_size = codebook_size
        
        self.layers = nn.ModuleList([
            VectorQuantization(
                dim=dim, codebook_size=codebook_size, commitment_weight=commitment_weight,
                decay=decay, epsilon=epsilon, kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters, threshold_ema_dead_code=threshold_ema_dead_code
            ) for _ in range(n_q)
        ])
    
    def forward(self, x, n_q: tp.Optional[int] = None):
        quantized_out = torch.zeros_like(x)
        residual = x
        
        all_losses = []
        all_indices = []
        
        n_q = n_q or self.n_q
        
        for i, layer in enumerate(self.layers[:n_q]):
            quantized, indices, loss = layer(residual)
            quantized = quantized.detach()
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)
        
        if self.training:
            # Straight-through estimator for RVQ
            quantized_out = x + (quantized_out - x).detach()
        
        out_losses = torch.stack(all_losses)
        out_indices = torch.stack(all_indices, dim=1)  # [B, n_q, T]
        
        return quantized_out, out_indices, out_losses
    
    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        """Encode input to discrete codes."""
        residual = x
        all_indices = []
        n_q = n_q or self.n_q
        
        for i, layer in enumerate(self.layers[:n_q]):
            _, indices, _ = layer(residual)
            quantized = layer._codebook.dequantize(indices.reshape(-1, 1)).reshape(x.shape)
            residual = residual - quantized
            all_indices.append(indices)
        
        return torch.stack(all_indices, dim=1)  # [B, n_q, T]
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode discrete codes back to continuous representation."""
        quantized_out = torch.zeros(codes.shape[0], self.dim, codes.shape[2], device=codes.device)
        
        for i, indices in enumerate(codes.unbind(1)):
            layer = self.layers[i]
            quantized = layer._codebook.dequantize(indices.reshape(-1, 1)).reshape(quantized_out.shape)
            quantized_out = quantized_out + quantized
        
        return quantized_out


class TwoBranchResidualVectorQuantization(nn.Module):
    """Two-branch residual vector quantization with shared first codebook and diverse second codebook.
    
    This quantizer is designed for 2-branch mode where:
    - First codebook tokens are shared across channels/segments (same audio content)
    - Second codebook tokens are maximally different to encourage diversity
    """
    def __init__(self, dim: int, n_q: int, codebook_size: int, commitment_weight: float = 1.0,
                 decay: float = 0.8, epsilon: float = 1e-5, kmeans_init: bool = True,
                 kmeans_iters: int = 10, threshold_ema_dead_code: float = 2.,
                 diversity_weight: float = 1.0):
        super().__init__()
        self.dim = dim
        self.n_q = n_q
        self.codebook_size = codebook_size
        self.diversity_weight = diversity_weight
        
        # Create quantizers for each codebook
        self.layers = nn.ModuleList([
            VectorQuantization(
                dim=dim, codebook_size=codebook_size, commitment_weight=commitment_weight,
                decay=decay, epsilon=epsilon, kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters, threshold_ema_dead_code=threshold_ema_dead_code
            ) for _ in range(n_q)
        ])
    
    def forward(self, x, n_q: tp.Optional[int] = None):
        """Forward pass for 2-branch quantization.
        
        Args:
            x: Input tensor of shape [B, dim, T] where B should be even (pairs)
            n_q: Number of codebooks to use
            
        Returns:
            quantized_out: Quantized output
            out_indices: Code indices [B, n_q, T]
            out_losses: Commitment losses
            diversity_loss: Diversity loss for second codebook
            consistency_loss: Consistency loss for first codebook
        """
        quantized_out = torch.zeros_like(x)
        residual = x
        
        all_losses = []
        all_indices = []
        
        n_q = n_q or self.n_q
        
        # Process all codebooks normally first
        for i, layer in enumerate(self.layers[:n_q]):
            quantized, indices, loss = layer(residual)
            quantized = quantized.detach()
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)
        
        if self.training:
            # Straight-through estimator for RVQ
            quantized_out = x + (quantized_out - x).detach()
        
        out_losses = torch.stack(all_losses)
        out_indices = torch.stack(all_indices, dim=1)  # [B, n_q, T]
        
        # Calculate consistency and diversity losses if we have pairs
        diversity_loss = torch.tensor(0.0, device=x.device, requires_grad=self.training)
        consistency_loss = torch.tensor(0.0, device=x.device, requires_grad=self.training)
        
        if self.training and n_q >= 2 and x.shape[0] % 2 == 0:
            # Reshape to pairs: [B//2, 2, n_q, T]
            codes_pairs = out_indices.view(x.shape[0] // 2, 2, n_q, -1)
            
            # CONSISTENCY LOSS: Ensure first codebook tokens are identical between pairs
            first_codebook_1 = codes_pairs[:, 0, 0, :]  # [B//2, T] - first codebook, branch 1
            first_codebook_2 = codes_pairs[:, 1, 0, :]  # [B//2, T] - first codebook, branch 2
            
            # Calculate consistency loss: encourage identical first codebook tokens
            # Use MSE loss between first codebook indices, normalized by sequence length
            mse_loss = F.mse_loss(first_codebook_1.float(), first_codebook_2.float())
            # Normalize by sequence length to get per-timestep loss
            consistency_loss = mse_loss / first_codebook_1.shape[1] * self.diversity_weight
            
            # DIVERSITY LOSS: Ensure second codebook tokens are different between pairs
            second_codebook_1 = codes_pairs[:, 0, 1, :]  # [B//2, T] - second codebook, branch 1
            second_codebook_2 = codes_pairs[:, 1, 1, :]  # [B//2, T] - second codebook, branch 2
            
            # Calculate diversity loss: encourage different second codebook tokens
            # Use normalized MSE loss between second codebook indices for stability
            mse_diversity = F.mse_loss(second_codebook_1.float(), second_codebook_2.float())
            # Normalize by sequence length and scale to reasonable range
            diversity_loss = (mse_diversity / second_codebook_1.shape[1]) * self.diversity_weight
        
        return quantized_out, out_indices, out_losses, diversity_loss, consistency_loss
    
    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        """Encode input to discrete codes."""
        residual = x
        all_indices = []
        n_q = n_q or self.n_q
        
        for i, layer in enumerate(self.layers[:n_q]):
            _, indices, _ = layer(residual)
            quantized = layer._codebook.dequantize(indices.reshape(-1, 1)).reshape(x.shape)
            residual = residual - quantized
            all_indices.append(indices)
        
        return torch.stack(all_indices, dim=1)  # [B, n_q, T]
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode discrete codes back to continuous representation."""
        quantized_out = torch.zeros(codes.shape[0], self.dim, codes.shape[2], device=codes.device)
        
        for i, indices in enumerate(codes.unbind(1)):
            layer = self.layers[i]
            quantized = layer._codebook.dequantize(indices.reshape(-1, 1)).reshape(quantized_out.shape)
            quantized_out = quantized_out + quantized
        
        return quantized_out


class CompressionModel(ABC, nn.Module):
    """Base API for all compression models that aim at being used as audio tokenizers
    with a language model.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """See `EncodecModel.encode`."""
        ...

    @abstractmethod
    def decode(self, codes: torch.Tensor):
        """See `EncodecModel.decode`."""
        ...

    @property
    @abstractmethod
    def channels(self) -> int:
        ...

    @property
    @abstractmethod
    def frame_rate(self) -> float:
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...


class EncodecModel(CompressionModel):
    """Encodec model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        frame_rate (int): Frame rate for the latent representation.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        renormalize (bool): Whether to renormalize the audio before running the model.
    """
    # we need assignment to override the property in the abstract class,
    # I couldn't find a better way...
    frame_rate: float = 0
    sample_rate: int = 0
    channels: int = 0

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 frame_rate: int,
                 sample_rate: int,
                 channels: int,
                 causal: bool = False,
                 renormalize: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.renormalize = renormalize
        self.causal = causal
        if self.causal:
            # we force disabling here to avoid handling linear overlap of segments
            # as supported in original EnCodec codebase.
            assert not self.renormalize, 'Causal model does not support renormalize'

    def preprocess(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        scale: tp.Optional[torch.Tensor]
        if self.renormalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        return x, scale

    def postprocess(self,
                    x: torch.Tensor,
                    scale: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        if scale is not None:
            assert self.renormalize
            x = x * scale.view(-1, 1, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3
        length = x.shape[-1]
        x, scale = self.preprocess(x)

        emb = self.encoder(x)
        out = self.decoder(emb)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        out = self.postprocess(out, scale)

        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the given input tensor to latent representation.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes (torch.Tensor): a float tensor of shape [B, K, T] with K the number of latent dimensions and T the timestep.
        """
        assert x.dim() == 3
        x, _ = self.preprocess(x)
        emb = self.encoder(x)
        return emb

    def decode(self, codes: torch.Tensor):
        """Decode the given codes to a reconstructed representation.

        Args:
            codes (torch.Tensor): Float tensor of shape [B, K, T]

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        out = self.decoder(codes)
        # out contains extra padding added by the encoder and decoder
        return out


class QuantizedEncodecModel(CompressionModel):
    """Encodec model with quantization support for tokenization.
    
    This model extends the base EncodecModel to include vector quantization,
    enabling discrete token representation of audio.
    """
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module, quantizer: ResidualVectorQuantization,
                 frame_rate: int, sample_rate: int, channels: int, causal: bool = False,
                 renormalize: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._channels = channels
        self.renormalize = renormalize
        self.causal = causal
        
        if self.causal:
            assert not self.renormalize, 'Causal model does not support renormalize'
    
    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return self._channels
    
    @property
    def frame_rate(self) -> float:
        """Frame rate of the latent representation."""
        return self._frame_rate
    
    @property
    def sample_rate(self) -> int:
        """Audio sample rate."""
        return self._sample_rate
    
    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.n_q
    
    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.n_q
    
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        assert 0 < n <= self.quantizer.n_q
        self.quantizer.n_q = n
    
    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.codebook_size
    
    def preprocess(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        scale: tp.Optional[torch.Tensor]
        if self.renormalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        return x, scale
    
    def postprocess(self, x: torch.Tensor, scale: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        if scale is not None:
            assert self.renormalize
            x = x * scale.view(-1, 1, 1)
        return x
    
    def forward(self, x: torch.Tensor) -> QuantizedResult:
        """Forward pass with quantization."""
        assert x.dim() == 3
        length = x.shape[-1]
        x, scale = self.preprocess(x)
        
        # Encode to continuous embeddings
        emb = self.encoder(x)
        
        # Quantize embeddings
        quantized_emb, codes, commit_losses = self.quantizer(emb)
        
        # Decode quantized embeddings
        out = self.decoder(quantized_emb)
        
        # Remove extra padding
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]
        
        out = self.postprocess(out, scale)
        
        # Calculate bandwidth (bits per second)
        import math
        bw_per_q = math.log2(self.quantizer.codebook_size) * self.frame_rate / 1000
        bandwidth = torch.tensor(self.quantizer.n_q * bw_per_q).to(x.device)
        
        # Calculate total commitment loss
        penalty = torch.mean(commit_losses)
        
        return QuantizedResult(out, codes, bandwidth, penalty)
    
    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Encode the given input tensor to quantized representation.
        
        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]
            
        Returns:
            codes, scale (tuple of torch.Tensor, torch.Tensor): Tuple composed of:
                codes: a float tensor of shape [B, K, T] with K the number of codebooks and T the timestep.
                scale: a float tensor containing the scale for audio renormalization.
        """
        assert x.dim() == 3
        x, scale = self.preprocess(x)
        emb = self.encoder(x)
        codes = self.quantizer.encode(emb)
        return codes, scale
    
    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """Decode the given codes to a reconstructed representation.
        
        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]
            scale (torch.Tensor, optional): Float tensor containing the scale value.
            
        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        emb = self.decode_latent(codes)
        out = self.decoder(emb)
        out = self.postprocess(out, scale)
        return out
    
    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.decode(codes)


class TwoBranchQuantizedEncodecModel(CompressionModel):
    """2-branch Encodec model with quantization support for tokenization.
    
    This model extends the base EncodecModel to include two-branch vector quantization,
    where the first codebook tokens are shared across channels/segments and the second
    codebook tokens are maximally different to encourage diversity.
    """
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module, quantizer: TwoBranchResidualVectorQuantization,
                 frame_rate: int, sample_rate: int, channels: int, causal: bool = False,
                 renormalize: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._channels = channels
        self.renormalize = renormalize
        self.causal = causal
        
        if self.causal:
            assert not self.renormalize, 'Causal model does not support renormalize'
    
    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return self._channels
    
    @property
    def frame_rate(self) -> float:
        """Frame rate of the latent representation."""
        return self._frame_rate
    
    @property
    def sample_rate(self) -> int:
        """Audio sample rate."""
        return self._sample_rate
    
    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.n_q
    
    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.n_q
    
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        assert 0 < n <= self.quantizer.n_q
        self.quantizer.n_q = n
    
    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.codebook_size
    
    def preprocess(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        scale: tp.Optional[torch.Tensor]
        if self.renormalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        return x, scale
    
    def postprocess(self, x: torch.Tensor, scale: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        if scale is not None:
            assert self.renormalize
            x = x * scale.view(-1, 1, 1)
        return x
    
    def forward(self, x: torch.Tensor) -> TwoBranchQuantizedResult:
        """Forward pass with 2-branch quantization.
        
        Args:
            x: Input audio tensor of shape [B, C, T] where B should be even (pairs)
            
        Returns:
            TwoBranchQuantizedResult: Result containing quantized audio, codes, bandwidth, penalty, and diversity loss
        """
        assert x.dim() == 3
        length = x.shape[-1]
        x, scale = self.preprocess(x)
        
        # Encode to continuous embeddings
        emb = self.encoder(x)
        
        # Quantize embeddings with 2-branch logic
        quantized_emb, codes, commit_losses, diversity_loss, consistency_loss = self.quantizer(emb)
        
        # Decode quantized embeddings
        out = self.decoder(quantized_emb)
        
        # Remove extra padding
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]
        
        out = self.postprocess(out, scale)
        
        # Calculate bandwidth (bits per second)
        import math
        bw_per_q = math.log2(self.quantizer.codebook_size) * self.frame_rate / 1000
        bandwidth = torch.tensor(self.quantizer.n_q * bw_per_q).to(x.device)
        
        # Calculate total commitment loss
        penalty = torch.mean(commit_losses)
        
        return TwoBranchQuantizedResult(out, codes, bandwidth, penalty, diversity_loss, consistency_loss)
    
    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Encode the given input tensor to quantized representation.
        
        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]
            
        Returns:
            codes, scale (tuple of torch.Tensor, torch.Tensor): Tuple composed of:
                codes: a float tensor of shape [B, K, T] with K the number of codebooks and T the timestep.
                scale: a float tensor containing the scale for audio renormalization.
        """
        assert x.dim() == 3
        x, scale = self.preprocess(x)
        emb = self.encoder(x)
        codes = self.quantizer.encode(emb)
        return codes, scale
    
    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """Decode the given codes to a reconstructed representation.
        
        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]
            scale (torch.Tensor, optional): Float tensor containing the scale value.
            
        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        emb = self.decode_latent(codes)
        out = self.decoder(emb)
        out = self.postprocess(out, scale)
        return out
    
    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.decode(codes)
