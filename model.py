from abc import ABC, abstractmethod
import logging
from pathlib import Path
import typing as tp

import torch
from torch import nn


logger = logging.getLogger()


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
