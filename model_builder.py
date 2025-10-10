import typing as tp
import logging
import numpy as np

from seanet import SEANetEncoder, SEANetDecoder
from model import EncodecModel

logger = logging.getLogger(__name__)


def build_encodec_model(sample_rate: int = 24000, channels: int = 32, 
                       n_filters: int = 4, n_residual_layers: int = 1, 
                       dimension: int = 32, causal: bool = False,
                       custom_ratios: tp.Optional[tp.List[int]] = None) -> EncodecModel:
    """Build EnCodec model with SEANet encoder and decoder."""
    
    # Use custom ratios if provided, otherwise use default
    if custom_ratios is not None:
        ratios = custom_ratios
        frame_rate = sample_rate / np.prod(ratios)
    else:
        # Model ratios based on sample rate (from AudioCraft)
        model_ratios = {
            16000: [10, 8, 8],    # 25 Hz at 16kHz
            24000: [10, 8, 12],   # 25 Hz at 24kHz  
            32000: [10, 8, 16],   # 25 Hz at 32kHz
            48000: [10, 8, 24],   # 25 Hz at 48kHz
        }
        
        if sample_rate not in model_ratios:
            raise ValueError(f"Unsupported sample rate: {sample_rate}. "
                           f"Supported: {list(model_ratios.keys())}")
        
        ratios = model_ratios[sample_rate]
        frame_rate = sample_rate / np.prod(ratios)  # Actual frame rate depends on ratios
    
    # SEANet configuration
    seanet_kwargs = {
        "n_filters": n_filters,
        "n_residual_layers": n_residual_layers,
        "dimension": dimension,
        "ratios": ratios,
        "channels": channels,
        "causal": causal,
        "activation": "ELU",
        "activation_params": {"alpha": 1.0},
        "norm": "weight_norm",
        "norm_params": {},
        "kernel_size": 7,
        "last_kernel_size": 7,
        "residual_kernel_size": 3,
        "dilation_base": 2,
        "pad_mode": "reflect",
        "true_skip": True,
        "compress": 2,
        "lstm": 2, 
        "disable_norm_outer_blocks": 0,
    }
    
    # Create encoder and decoder
    encoder = SEANetEncoder(**seanet_kwargs)
    decoder = SEANetDecoder(**seanet_kwargs)
    
    # Create EnCodec model
    model = EncodecModel(
        encoder=encoder,
        decoder=decoder,
        frame_rate=frame_rate,
        sample_rate=sample_rate,
        channels=channels,
        causal=causal,
        renormalize=False
    )
    
    logger.info(f"Built EnCodec model:")
    logger.info(f"  Sample rate: {sample_rate} Hz")
    logger.info(f"  Frame rate: {frame_rate:.1f} Hz")
    logger.info(f"  Channels: {channels}")
    logger.info(f"  Latent dimension: {dimension}")
    logger.info(f"  Ratios: {ratios}")
    logger.info(f"  Causal: {causal}")
    
    return model


def get_default_ratios(sample_rate: int) -> tp.List[int]:
    """Get default ratios for a given sample rate."""
    model_ratios = {
        16000: [10, 8, 8],    # 25 Hz at 16kHz
        24000: [10, 8, 12],   # 25 Hz at 24kHz  
    }
    
    if sample_rate not in model_ratios:
        raise ValueError(f"Unsupported sample rate: {sample_rate}. "
                       f"Supported: {list(model_ratios.keys())}")
    
    return model_ratios[sample_rate]


def calculate_frame_rate(sample_rate: int, ratios: tp.List[int]) -> float:
    """Calculate frame rate from sample rate and ratios."""
    return sample_rate / np.prod(ratios)
