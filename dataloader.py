"""
Simple dataloader for multichannel audio files using soundfile.
"""

import random
import typing as tp
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np

class MultiChannelAudioDataset(Dataset):
    """Dataset for multichannel audio with random sampling.
    
    Instead of one item per file, you can set a virtual dataset size
    and each __getitem__ will pick a random file + random segment.
    """
    
    def __init__(self, 
                 audio_dir: str,
                 sample_rate: int = 24000,
                 segment_duration: float = 10.0,
                 channels: int = 32,
                 dataset_size: int = None,
                 pad: bool = True,
                 random_crop: bool = True,
                 file_extensions: tp.List[str] = None,
                 min_file_duration: float = None,
                 max_segments_per_file: int = None):
        """
        Args:
            audio_dir: Directory containing audio files
            sample_rate: Target sample rate
            segment_duration: Duration of segments in seconds
            channels: Number of channels to load
            dataset_size: Virtual dataset size (if None = number of files)
            pad: Whether to pad shorter segments
            random_crop: Whether to randomly crop longer segments
            file_extensions: List of file extensions to include (default: ['.wav'])
            min_file_duration: Minimum file duration to include (in seconds)
            max_segments_per_file: Maximum segments to extract from each file
        """
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.channels = channels
        self.pad = pad
        self.random_crop = random_crop
        self.dataset_size = dataset_size
        self.min_file_duration = min_file_duration
        self.max_segments_per_file = max_segments_per_file
        
        if file_extensions is None:
            file_extensions = ['.wav']
        self.file_extensions = [ext.lower() for ext in file_extensions]
        
        # Find all audio files and get their durations
        self.audio_files = self._find_audio_files()
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")
        
        # Filter files by duration if specified
        if min_file_duration is not None:
            self.audio_files = self._filter_by_duration(self.audio_files, min_file_duration)
            if len(self.audio_files) == 0:
                raise ValueError(f"No audio files longer than {min_file_duration}s found")
    
    def _find_audio_files(self) -> tp.List[Path]:
        """Find all audio files in the directory."""
        audio_files = []
        for ext in self.file_extensions:
            audio_files.extend(self.audio_dir.glob(f"**/*{ext}"))
            audio_files.extend(self.audio_dir.glob(f"**/*{ext.upper()}"))
        return sorted(audio_files)
    
    def _filter_by_duration(self, files: tp.List[Path], min_duration: float) -> tp.List[Path]:
        """Filter files by minimum duration."""
        valid_files = []
        min_samples = int(min_duration * self.sample_rate)
        
        for file_path in files:
            try:
                info = sf.info(str(file_path))
                if info.duration >= min_duration:
                    valid_files.append(file_path)
            except Exception as e:
                print(f"Warning: Could not read duration of {file_path}: {e}")
                continue
        
        return valid_files
    
    def _get_file_duration(self, file_path: Path) -> float:
        """Get duration of a file in seconds."""
        try:
            info = sf.info(str(file_path))
            return info.duration
        except Exception:
            return 0.0
    
    def __len__(self) -> int:
        # Virtual size if defined
        if self.dataset_size is not None:
            return self.dataset_size
        return len(self.audio_files)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        """Get a random segment from a random file.
        
        If dataset_size is defined, this will always pick random files and segments.
        If dataset_size is None, this will pick the file at the given index.
        """
        
        # Always pick a random file for better variety
        audio_file = random.choice(self.audio_files)
        
        try:
            audio, sr = sf.read(str(audio_file), dtype=np.float32)
            
            # (channels, samples)
            if len(audio.shape) == 1:
                audio = np.tile(audio, (self.channels, 1))
            else:
                audio = audio.T
            
            # Fix channel count
            if audio.shape[0] < self.channels:
                padding = np.zeros((self.channels - audio.shape[0], audio.shape[1]), dtype=np.float32)
                audio = np.vstack([audio, padding])
            elif audio.shape[0] > self.channels:
                audio = audio[:self.channels, :]
            
            audio = torch.from_numpy(audio).float()
            
            # Resample
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)
            
            # Segmenting
            target_samples = int(self.segment_duration * self.sample_rate)
            current_samples = audio.shape[1]
            
            if current_samples < target_samples:
                if self.pad:
                    padding = torch.zeros(self.channels, target_samples - current_samples)
                    audio = torch.cat([audio, padding], dim=1)
                else:
                    repeats = (target_samples + current_samples - 1) // current_samples
                    audio = audio.repeat(1, repeats)[:, :target_samples]
            elif current_samples > target_samples:
                if self.random_crop:
                    start = random.randint(0, current_samples - target_samples)
                    audio = audio[:, start:start + target_samples]
                else:
                    start = (current_samples - target_samples) // 2
                    audio = audio[:, start:start + target_samples]
            
            return audio
        
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            return torch.zeros(self.channels, int(self.segment_duration * self.sample_rate))
    
    def _resample(self, audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        if orig_sr == target_sr:
            return audio
        orig_length = audio.shape[1]
        target_length = int(orig_length * target_sr / orig_sr)
        audio = audio.unsqueeze(0)
        audio = F.interpolate(audio, size=target_length, mode='linear', align_corners=False)
        return audio.squeeze(0)


def create_dataloader(audio_dir: str,
                     batch_size: int = 4,
                     sample_rate: int = 24000,
                     segment_duration: float = 10.0,
                     channels: int = 32,
                     num_workers: int = 4,
                     shuffle: bool = True,
                     dataset_size: int = None,
                     min_file_duration: float = None,
                     **kwargs) -> DataLoader:
    """Create a DataLoader for multichannel audio.
    
    Args:
        audio_dir: Directory containing audio files
        batch_size: Batch size for training
        sample_rate: Target sample rate
        segment_duration: Duration of segments in seconds
        channels: Number of channels to load
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the dataset
        dataset_size: Virtual dataset size (if None = number of files)
        min_file_duration: Minimum file duration to include (in seconds)
        **kwargs: Additional arguments passed to MultiChannelAudioDataset
    """
    
    dataset = MultiChannelAudioDataset(
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        segment_duration=segment_duration,
        channels=channels,
        dataset_size=dataset_size,
        min_file_duration=min_file_duration,
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


class FixedValidationDataset(Dataset):
    """Fixed validation dataset that uses the same segments every epoch."""
    
    def __init__(self, 
                 audio_files: tp.List[Path],
                 sample_rate: int = 24000,
                 segment_duration: float = 10.0,
                 channels: int = 32,
                 min_file_duration: float = None,
                 random_crop: bool = True):
        """
        Args:
            audio_files: List of audio files to use for validation
            sample_rate: Target sample rate
            segment_duration: Duration of segments in seconds
            channels: Number of channels to load
            min_file_duration: Minimum file duration to include (in seconds)
            random_crop: Whether to randomly crop longer segments
        """
        self.audio_files = audio_files
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.channels = channels
        self.min_file_duration = min_file_duration
        self.random_crop = random_crop
        
        # Pre-generate fixed segments
        self.fixed_segments = self._generate_fixed_segments()
    
    def _generate_fixed_segments(self) -> tp.List[tp.Tuple[Path, int]]:
        """Pre-generate fixed segments (file_path, start_sample) for consistent validation."""
        segments = []
        
        for file_path in self.audio_files:
            try:
                info = sf.info(str(file_path))
                if self.min_file_duration and info.duration < self.min_file_duration:
                    continue
                    
                # Calculate how many segments we can get from this file
                file_samples = int(info.duration * self.sample_rate)
                segment_samples = int(self.segment_duration * self.sample_rate)
                
                if file_samples < segment_samples:
                    continue
                
                # Generate multiple segments from this file
                max_segments = min(10, file_samples // segment_samples)  # Max 10 segments per file
                for _ in range(max_segments):
                    if self.random_crop:
                        start_sample = random.randint(0, file_samples - segment_samples)
                    else:
                        start_sample = 0
                    segments.append((file_path, start_sample))
                    
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
                continue
        
        return segments
    
    def __len__(self) -> int:
        return len(self.fixed_segments)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        """Get a fixed segment."""
        file_path, start_sample = self.fixed_segments[index]
        
        try:
            audio, sr = sf.read(str(file_path), dtype=np.float32)
            
            # (channels, samples)
            if len(audio.shape) == 1:
                audio = np.tile(audio, (self.channels, 1))
            else:
                audio = audio.T
            
            # Fix channel count
            if audio.shape[0] < self.channels:
                padding = np.zeros((self.channels - audio.shape[0], audio.shape[1]), dtype=np.float32)
                audio = np.vstack([audio, padding])
            elif audio.shape[0] > self.channels:
                audio = audio[:self.channels, :]
            
            audio = torch.from_numpy(audio).float()
            
            # Resample
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)
            
            # Extract the fixed segment
            target_samples = int(self.segment_duration * self.sample_rate)
            current_samples = audio.shape[1]
            
            if current_samples < target_samples:
                # Pad if too short
                padding = torch.zeros(self.channels, target_samples - current_samples)
                audio = torch.cat([audio, padding], dim=1)
            elif current_samples > target_samples:
                # Crop the fixed segment
                audio = audio[:, start_sample:start_sample + target_samples]
            
            return audio
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros(self.channels, int(self.segment_duration * self.sample_rate))
    
    def _resample(self, audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        if orig_sr == target_sr:
            return audio
        orig_length = audio.shape[1]
        target_length = int(orig_length * target_sr / orig_sr)
        audio = audio.unsqueeze(0)
        audio = F.interpolate(audio, size=target_length, mode='linear', align_corners=False)
        return audio.squeeze(0)


def create_folder_based_dataloaders(audio_dir: str,
                                   train_folders: tp.List[str],
                                   val_folders: tp.List[str],
                                   batch_size: int = 4,
                                   sample_rate: int = 24000,
                                   segment_duration: float = 10.0,
                                   channels: int = 32,
                                   num_workers: int = 4,
                                   train_dataset_size: int = 20000,
                                   val_dataset_size: int = 5000,
                                   min_file_duration: float = None,
                                   **kwargs) -> tp.Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders based on specific folders.
    
    Args:
        audio_dir: Base directory containing audio files
        train_folders: List of folder names to use for training
        val_folders: List of folder names to use for validation
        batch_size: Batch size for training
        sample_rate: Target sample rate
        segment_duration: Duration of segments in seconds
        channels: Number of channels to load
        num_workers: Number of worker processes
        train_dataset_size: Number of training samples per epoch (default: 20000)
        val_dataset_size: Number of validation samples per epoch (default: 5000)
        min_file_duration: Minimum file duration to include (in seconds)
        **kwargs: Additional arguments passed to MultiChannelAudioDataset
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    audio_dir_path = Path(audio_dir)
    file_extensions = kwargs.get('file_extensions', ['.wav'])
    file_extensions = [ext.lower() for ext in file_extensions]
    
    # Find training files from specified folders
    train_files = []
    for folder in train_folders:
        folder_path = audio_dir_path / folder
        if folder_path.exists():
            for ext in file_extensions:
                train_files.extend(folder_path.glob(f"**/*{ext}"))
                train_files.extend(folder_path.glob(f"**/*{ext.upper()}"))
        else:
            print(f"Warning: Training folder '{folder}' not found in {audio_dir}")
    
    # Find validation files from specified folders
    val_files = []
    for folder in val_folders:
        folder_path = audio_dir_path / folder
        if folder_path.exists():
            for ext in file_extensions:
                val_files.extend(folder_path.glob(f"**/*{ext}"))
                val_files.extend(folder_path.glob(f"**/*{ext.upper()}"))
        else:
            print(f"Warning: Validation folder '{folder}' not found in {audio_dir}")
    
    train_files = sorted(train_files)
    val_files = sorted(val_files)
    
    if len(train_files) == 0:
        raise ValueError(f"No training files found in folders: {train_folders}")
    if len(val_files) == 0:
        raise ValueError(f"No validation files found in folders: {val_folders}")
    
    # Filter files by duration if specified
    if min_file_duration is not None:
        train_files = _filter_files_by_duration(train_files, min_file_duration, sample_rate)
        val_files = _filter_files_by_duration(val_files, min_file_duration, sample_rate)
        
        if len(train_files) == 0:
            raise ValueError(f"No training files longer than {min_file_duration}s found")
        if len(val_files) == 0:
            raise ValueError(f"No validation files longer than {min_file_duration}s found")
    
    print(f"Found {len(train_files)} training files from folders: {train_folders}")
    print(f"Found {len(val_files)} validation files from folders: {val_folders}")
    
    # Create training dataset
    train_dataset = MultiChannelAudioDataset(
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        segment_duration=segment_duration,
        channels=channels,
        dataset_size=train_dataset_size,
        min_file_duration=min_file_duration,
        **kwargs
    )
    # Override the audio_files with our training files
    train_dataset.audio_files = train_files
    
    # Create fixed validation dataset with same segments every epoch
    val_dataset = FixedValidationDataset(
        audio_files=val_files,
        sample_rate=sample_rate,
        segment_duration=segment_duration,
        channels=channels,
        min_file_duration=min_file_duration,
        random_crop=kwargs.get('random_crop', True)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data for consistent evaluation
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True
    )
    
    return train_loader, val_loader


def _filter_files_by_duration(files: tp.List[Path], min_duration: float, sample_rate: int) -> tp.List[Path]:
    """Filter files by minimum duration."""
    valid_files = []
    
    for file_path in files:
        try:
            info = sf.info(str(file_path))
            if info.duration >= min_duration:
                valid_files.append(file_path)
        except Exception as e:
            print(f"Warning: Could not read duration of {file_path}: {e}")
            continue
    
    return valid_files


def create_train_test_dataloaders(audio_dir: str,
                                train_ratio: float = 0.8,
                                batch_size: int = 4,
                                sample_rate: int = 24000,
                                segment_duration: float = 10.0,
                                channels: int = 32,
                                num_workers: int = 4,
                                train_dataset_size: int = None,
                                test_dataset_size: int = None,
                                min_file_duration: float = None,
                                **kwargs) -> tp.Tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders from a single directory with random split.
    
    Args:
        audio_dir: Directory containing audio files
        train_ratio: Ratio of files to use for training (default: 0.8 for 80/20 split)
        batch_size: Batch size for training
        sample_rate: Target sample rate
        segment_duration: Duration of segments in seconds
        channels: Number of channels to load
        num_workers: Number of worker processes
        train_dataset_size: Virtual dataset size for training (if None = number of files)
        test_dataset_size: Virtual dataset size for testing (if None = number of files)
        min_file_duration: Minimum file duration to include (in seconds)
        **kwargs: Additional arguments passed to MultiChannelAudioDataset
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    
    # Find all audio files first
    audio_dir_path = Path(audio_dir)
    file_extensions = kwargs.get('file_extensions', ['.wav'])
    file_extensions = [ext.lower() for ext in file_extensions]
    
    # Find all audio files
    audio_files = []
    for ext in file_extensions:
        audio_files.extend(audio_dir_path.glob(f"**/*{ext}"))
        audio_files.extend(audio_dir_path.glob(f"**/*{ext.upper()}"))
    audio_files = sorted(audio_files)
    
    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {audio_dir}")
    
    # Filter files by duration if specified
    if min_file_duration is not None:
        valid_files = []
        min_samples = int(min_file_duration * sample_rate)
        
        for file_path in audio_files:
            try:
                info = sf.info(str(file_path))
                if info.duration >= min_file_duration:
                    valid_files.append(file_path)
            except Exception as e:
                print(f"Warning: Could not read duration of {file_path}: {e}")
                continue
        
        audio_files = valid_files
        if len(audio_files) == 0:
            raise ValueError(f"No audio files longer than {min_file_duration}s found")
    
    print(f"Found {len(audio_files)} audio files")
    
    # Randomly split files into train/test
    random.seed(42)  # For reproducible splits
    random.shuffle(audio_files)
    
    split_idx = int(len(audio_files) * train_ratio)
    train_files = audio_files[:split_idx]
    test_files = audio_files[split_idx:]
    
    print(f"Train files: {len(train_files)} ({len(train_files)/len(audio_files)*100:.1f}%)")
    print(f"Test files: {len(test_files)} ({len(test_files)/len(audio_files)*100:.1f}%)")
    
    # Create train dataset with train files
    train_dataset = MultiChannelAudioDataset(
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        segment_duration=segment_duration,
        channels=channels,
        dataset_size=train_dataset_size,
        min_file_duration=min_file_duration,
        **kwargs
    )
    # Override the audio_files with our split
    train_dataset.audio_files = train_files
    
    # Create test dataset with test files
    test_dataset = MultiChannelAudioDataset(
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        segment_duration=segment_duration,
        channels=channels,
        dataset_size=test_dataset_size,
        min_file_duration=min_file_duration,
        **kwargs
    )
    # Override the audio_files with our split
    test_dataset.audio_files = test_files
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Always shuffle training data
        num_workers=num_workers,
        pin_memory=False,  # Disable pin_memory to avoid issues
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=num_workers,
        pin_memory=False,  # Disable pin_memory to avoid issues
        drop_last=True
    )
    
    return train_loader, test_loader


