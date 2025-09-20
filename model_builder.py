import typing as tp
import logging
import numpy as np

from seanet import SEANetEncoder, SEANetDecoder
from model import EncodecModel, QuantizedEncodecModel, ResidualVectorQuantization, TwoBranchQuantizedEncodecModel, TwoBranchResidualVectorQuantization

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
        frame_rate = sample_rate / np.prod(ratios)  # Should be ~25 Hz
    
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
        "lstm": 0,
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


def build_quantized_encodec_model(sample_rate: int = 24000, channels: int = 1, 
                                 n_filters: int = 4, n_residual_layers: int = 1, 
                                 dimension: int = 32, causal: bool = False,
                                 n_q: int = 8, codebook_size: int = 1024,
                                 commitment_weight: float = 1.0,
                                 custom_ratios: tp.Optional[tp.List[int]] = None) -> QuantizedEncodecModel:
    """Build quantized EnCodec model with SEANet encoder/decoder and residual vector quantization.
    
    Args:
        sample_rate: Audio sample rate
        channels: Number of audio channels
        n_filters: Number of filters in SEANet
        n_residual_layers: Number of residual layers
        dimension: Latent dimension
        causal: Whether to use causal convolutions
        n_q: Number of quantizer codebooks
        codebook_size: Size of each codebook
        commitment_weight: Weight for commitment loss
        custom_ratios: Custom downsampling ratios
        
    Returns:
        QuantizedEncodecModel: Model with quantization support
    """
    
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
        frame_rate = sample_rate / np.prod(ratios)  # Should be ~25 Hz
    
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
        "lstm": 0,
        "disable_norm_outer_blocks": 0,
    }
    
    # Create encoder and decoder
    encoder = SEANetEncoder(**seanet_kwargs)
    decoder = SEANetDecoder(**seanet_kwargs)
    
    # Create residual vector quantizer
    quantizer = ResidualVectorQuantization(
        dim=dimension,
        n_q=n_q,
        codebook_size=codebook_size,
        commitment_weight=commitment_weight,
        decay=0.99,
        epsilon=1e-5,
        kmeans_init=True,
        kmeans_iters=50,
        threshold_ema_dead_code=2.0
    )
    
    # Create quantized EnCodec model
    model = QuantizedEncodecModel(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        frame_rate=frame_rate,
        sample_rate=sample_rate,
        channels=channels,
        causal=causal,
        renormalize=False
    )
    
    logger.info(f"Built Quantized EnCodec model:")
    logger.info(f"  Sample rate: {sample_rate} Hz")
    logger.info(f"  Frame rate: {frame_rate:.1f} Hz")
    logger.info(f"  Channels: {channels}")
    logger.info(f"  Latent dimension: {dimension}")
    logger.info(f"  Ratios: {ratios}")
    logger.info(f"  Causal: {causal}")
    logger.info(f"  Number of codebooks: {n_q}")
    logger.info(f"  Codebook size: {codebook_size}")
    logger.info(f"  Commitment weight: {commitment_weight}")
    
    return model


def build_2branch_quantized_encodec_model(sample_rate: int = 24000, channels: int = 1, 
                                         n_filters: int = 4, n_residual_layers: int = 1, 
                                         dimension: int = 32, causal: bool = False,
                                         n_q: int = 2, codebook_size: int = 1024,
                                         commitment_weight: float = 1.0,
                                         diversity_weight: float = 1.0,
                                         custom_ratios: tp.Optional[tp.List[int]] = None) -> TwoBranchQuantizedEncodecModel:
    """Build 2-branch quantized EnCodec model with SEANet encoder/decoder and two-branch residual vector quantization.
    
    This model is designed for 2-branch mode where:
    - First codebook tokens are shared across channels/segments (same audio content)
    - Second codebook tokens are maximally different to encourage diversity
    
    Args:
        sample_rate: Audio sample rate
        channels: Number of audio channels
        n_filters: Number of filters in SEANet
        n_residual_layers: Number of residual layers
        dimension: Latent dimension
        causal: Whether to use causal convolutions
        n_q: Number of quantizer codebooks
        codebook_size: Size of each codebook
        commitment_weight: Weight for commitment loss
        diversity_weight: Weight for diversity loss
        custom_ratios: Custom downsampling ratios
        
    Returns:
        TwoBranchQuantizedEncodecModel: Model with 2-branch quantization support
    """
    
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
        frame_rate = sample_rate / np.prod(ratios)  # Should be ~25 Hz
    
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
        "lstm": 0,
        "disable_norm_outer_blocks": 0,
    }
    
    # Create encoder and decoder
    encoder = SEANetEncoder(**seanet_kwargs)
    decoder = SEANetDecoder(**seanet_kwargs)
    
    # Create two-branch residual vector quantizer
    quantizer = TwoBranchResidualVectorQuantization(
        dim=dimension,
        n_q=n_q,
        codebook_size=codebook_size,
        commitment_weight=commitment_weight,
        diversity_weight=diversity_weight,
        decay=0.99,
        epsilon=1e-5,
        kmeans_init=True,
        kmeans_iters=50,
        threshold_ema_dead_code=2.0
    )
    
    # Create 2-branch quantized EnCodec model
    model = TwoBranchQuantizedEncodecModel(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        frame_rate=frame_rate,
        sample_rate=sample_rate,
        channels=channels,
        causal=causal,
        renormalize=False
    )
    
    logger.info(f"Built 2-Branch Quantized EnCodec model:")
    logger.info(f"  Sample rate: {sample_rate} Hz")
    logger.info(f"  Frame rate: {frame_rate:.1f} Hz")
    logger.info(f"  Channels: {channels}")
    logger.info(f"  Latent dimension: {dimension}")
    logger.info(f"  Ratios: {ratios}")
    logger.info(f"  Causal: {causal}")
    logger.info(f"  Number of codebooks: {n_q}")
    logger.info(f"  Codebook size: {codebook_size}")
    logger.info(f"  Commitment weight: {commitment_weight}")
    logger.info(f"  Diversity weight: {diversity_weight}")
    
    return model
