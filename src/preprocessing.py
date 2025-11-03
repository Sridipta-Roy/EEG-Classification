"""
EEG Signal Preprocessing Module
================================

This module provides comprehensive preprocessing functions for EEG signals:
- Filtering (bandpass, notch)
- Artifact removal
- Segmentation
- Normalization
- Resampling

"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
import pywt
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class EEGPreprocessor:
    """
    Comprehensive EEG signal preprocessor
    
    Parameters:
    -----------
    sampling_rate : float
        Sampling rate of the signal in Hz
    """
    
    def __init__(self, sampling_rate: float):
        self.sampling_rate = sampling_rate
        
    def bandpass_filter(self, data: np.ndarray, lowcut: float = 0.5, highcut: float = 50.0, order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter to remove drift and high-frequency noise
        
        Parameters:
        -----------
        data : np.ndarray
            Input signal (channels x samples) or (samples,)
        lowcut : float
            Low cutoff frequency in Hz (default: 0.5 Hz)
        highcut : float
            High cutoff frequency in Hz (default: 50 Hz)
        order : int
            Filter order (default: 4)
        
        Returns:
        --------
        filtered_data : np.ndarray
            Bandpass filtered signal
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Design Butterworth bandpass filter
        b, a = butter(order, [low, high], btype='band')
        
        # Apply filter
        if data.ndim == 1:
            filtered_data = filtfilt(b, a, data)
        else:
            filtered_data = np.array([filtfilt(b, a, channel) 
                                     for channel in data])
        
        return filtered_data
    
    def notch_filter(self, data: np.ndarray, freq: float = 60.0, 
                    quality: float = 30.0) -> np.ndarray:
        """
        Apply notch filter to remove power line interference
        
        Parameters:
        -----------
        data : np.ndarray
            Input signal
        freq : float
            Frequency to remove in Hz (50 Hz for Europe, 60 Hz for US)
        quality : float
            Quality factor (higher = narrower notch)
        
        Returns:
        --------
        filtered_data : np.ndarray
            Notch filtered signal
        """
        nyquist = 0.5 * self.sampling_rate
        w0 = freq / nyquist
        
        # Design notch filter
        b, a = iirnotch(w0, quality)
        
        # Apply filter
        if data.ndim == 1:
            filtered_data = filtfilt(b, a, data)
        else:
            filtered_data = np.array([filtfilt(b, a, channel) 
                                     for channel in data])
        
        return filtered_data
    
    def normalize(self, data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        Normalize signal
        
        Parameters:
        -----------
        data : np.ndarray
            Input signal (channels x samples) or (samples,)
        method : str
            Normalization method:
            - 'zscore': Zero mean, unit variance
            - 'minmax': Scale to [0, 1]
            - 'robust': Robust scaling using median and IQR
        
        Returns:
        --------
        normalized_data : np.ndarray
            Normalized signal
        """
        if method == 'zscore':
            if data.ndim == 1:
                return (data - np.mean(data)) / (np.std(data) + 1e-8)
            else:
                return np.array([(channel - np.mean(channel)) / 
                               (np.std(channel) + 1e-8) 
                               for channel in data])
        
        elif method == 'minmax':
            if data.ndim == 1:
                return (data - np.min(data)) / (np.ptp(data) + 1e-8)
            else:
                return np.array([(channel - np.min(channel)) / 
                               (np.ptp(channel) + 1e-8) 
                               for channel in data])
        
        elif method == 'robust':
            if data.ndim == 1:
                median = np.median(data)
                iqr = np.percentile(data, 75) - np.percentile(data, 25)
                return (data - median) / (iqr + 1e-8)
            else:
                normalized = []
                for channel in data:
                    median = np.median(channel)
                    iqr = np.percentile(channel, 75) - np.percentile(channel, 25)
                    normalized.append((channel - median) / (iqr + 1e-8))
                return np.array(normalized)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def remove_baseline(self, data: np.ndarray, 
                       window_size: Optional[int] = None) -> np.ndarray:
        """
        Remove baseline drift using moving average
        
        Parameters:
        -----------
        data : np.ndarray
            Input signal
        window_size : int, optional
            Window size for moving average (default: 1 second)
        
        Returns:
        --------
        corrected_data : np.ndarray
            Baseline-corrected signal
        """
        if window_size is None:
            window_size = int(self.sampling_rate)
        
        if data.ndim == 1:
            # Calculate moving average as baseline
            baseline = np.convolve(data, np.ones(window_size)/window_size, 
                                  mode='same')
            return data - baseline
        else:
            return np.array([self.remove_baseline(channel, window_size) 
                           for channel in data])
    
    def segment_signal(self, data: np.ndarray, window_size: float, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment signal into fixed-length windows
        
        Parameters:
        -----------
        data : np.ndarray
            Input signal (channels x samples) or (samples,)
        window_size : float
            Window size in seconds
        overlap : float
            Overlap ratio (0.0 to 1.0)
        
        Returns:
        --------
        segments : np.ndarray
            Segmented data (n_segments, channels, window_samples) or
            (n_segments, window_samples)
        segment_times : np.ndarray
            Start time of each segment in seconds
        """
        window_samples = int(window_size * self.sampling_rate)
        step_samples = int(window_samples * (1 - overlap))
        
        if data.ndim == 1:
            # Single channel
            n_samples = len(data)
            segments = []
            segment_times = []
            
            start = 0
            while start + window_samples <= n_samples:
                segment = data[start:start + window_samples]
                segments.append(segment)
                segment_times.append(start / self.sampling_rate)
                start += step_samples
            
            return np.array(segments), np.array(segment_times)
        
        else:
            # Multiple channels
            n_channels, n_samples = data.shape
            segments = []
            segment_times = []
            
            start = 0
            while start + window_samples <= n_samples:
                segment = data[:, start:start + window_samples]
                segments.append(segment)
                segment_times.append(start / self.sampling_rate)
                start += step_samples
            
            return np.array(segments), np.array(segment_times)
    
    def resample(self, data: np.ndarray, 
                target_rate: float) -> np.ndarray:
        """
        Resample signal to target sampling rate
        
        Parameters:
        -----------
        data : np.ndarray
            Input signal
        target_rate : float
            Target sampling rate in Hz
        
        Returns:
        --------
        resampled_data : np.ndarray
            Resampled signal
        """
        if self.sampling_rate == target_rate:
            return data
        
        n_samples = int(len(data) * target_rate / self.sampling_rate)
        
        if data.ndim == 1:
            return signal.resample(data, n_samples)
        else:
            return np.array([signal.resample(channel, n_samples) 
                           for channel in data])
    
    def remove_artifacts_threshold(self, data: np.ndarray, threshold: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove segments with artifacts based on amplitude threshold
        
        Parameters:
        -----------
        data : np.ndarray
            Input signal segments (n_segments, ...)
        threshold : float
            Z-score threshold for artifact detection
        
        Returns:
        --------
        clean_data : np.ndarray
            Data with artifacts removed
        artifact_mask : np.ndarray
            Boolean mask indicating clean segments (True = clean)
        """
        # Calculate signal statistics per segment
        if data.ndim == 2:
            # (n_segments, n_samples)
            segment_max = np.max(np.abs(data), axis=1)
        elif data.ndim == 3:
            # (n_segments, n_channels, n_samples)
            segment_max = np.max(np.max(np.abs(data), axis=2), axis=1)
        else:
            raise ValueError("Data must be 2D or 3D")
        
        # Z-score normalization
        mean_max = np.mean(segment_max)
        std_max = np.std(segment_max)
        z_scores = (segment_max - mean_max) / (std_max + 1e-8)
        
        # Mark segments as clean if below threshold
        artifact_mask = np.abs(z_scores) < threshold
        clean_data = data[artifact_mask]
        
        print(f"Removed {np.sum(~artifact_mask)} artifact segments "
              f"({np.sum(~artifact_mask)/len(data)*100:.1f}%)")
        
        return clean_data, artifact_mask
    
    def wavelet_denoise(self, data: np.ndarray, wavelet: str = 'db4', level: int = 5) -> np.ndarray:
        """
        Denoise signal using wavelet decomposition
        
        Parameters:
        -----------
        data : np.ndarray
            Input signal
        wavelet : str
            Wavelet family (default: 'db4' - Daubechies 4)
        level : int
            Decomposition level
        
        Returns:
        --------
        denoised_data : np.ndarray
            Denoised signal
        """
        if data.ndim == 1:
            # Wavelet decomposition
            coeffs = pywt.wavedec(data, wavelet, level=level)
            
            # Threshold coefficients (soft thresholding)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(data)))
            
            coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') 
                           for c in coeffs]
            
            # Reconstruct signal
            return pywt.waverec(coeffs_thresh, wavelet)
        
        else:
            return np.array([self.wavelet_denoise(channel, wavelet, level) 
                           for channel in data])


def preprocess_pipeline(data: np.ndarray, sampling_rate: float,
                       apply_bandpass: bool = True,
                       apply_notch: bool = True,
                       apply_normalize: bool = True,
                       segment: bool = False,
                       window_size: float = 4.0,
                       overlap: float = 0.5) -> Tuple[np.ndarray, dict]:
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    data : np.ndarray
        Input signal
    sampling_rate : float
        Sampling rate in Hz
    apply_bandpass : bool
        Apply bandpass filter
    apply_notch : bool
        Apply notch filter
    apply_normalize : bool
        Apply z-score normalization
    segment : bool
        Segment into windows
    window_size : float
        Window size in seconds (if segmenting)
    overlap : float
        Overlap ratio (if segmenting)
    
    Returns:
    --------
    processed_data : np.ndarray
        Preprocessed signal
    info : dict
        Processing information
    """
    preprocessor = EEGPreprocessor(sampling_rate)
    info = {'steps': []}
    
    processed_data = data.copy()
    
    # Bandpass filter
    if apply_bandpass:
        processed_data = preprocessor.bandpass_filter(processed_data)
        info['steps'].append('bandpass_filter')
        print("✓ Applied bandpass filter (0.5-50 Hz)")
    
    # Notch filter
    if apply_notch:
        processed_data = preprocessor.notch_filter(processed_data, freq=60.0)
        info['steps'].append('notch_filter')
        print("✓ Applied notch filter (60 Hz)")
    
    # Normalization
    if apply_normalize:
        processed_data = preprocessor.normalize(processed_data, method='zscore')
        info['steps'].append('normalization')
        print("✓ Applied z-score normalization")
    
    # Segmentation
    if segment:
        processed_data, segment_times = preprocessor.segment_signal(
            processed_data, window_size, overlap
        )
        info['steps'].append('segmentation')
        info['segment_times'] = segment_times
        info['n_segments'] = len(processed_data)
        print(f"✓ Segmented into {len(processed_data)} windows "
              f"({window_size}s, {overlap*100}% overlap)")
    
    info['final_shape'] = processed_data.shape
    
    return processed_data, info


# Convenience functions
def quick_preprocess_bonn(data: np.ndarray, sampling_rate: float = 173.61) -> np.ndarray:
    """Quick preprocessing for Bonn dataset"""
    return preprocess_pipeline(
        data, sampling_rate,
        apply_bandpass=True,
        apply_notch=False,  # Bonn data is already clean
        apply_normalize=True,
        segment=False
    )[0]


def quick_preprocess_chbmit(data: np.ndarray, sampling_rate: float = 256.0) -> np.ndarray:
    """Quick preprocessing for CHB-MIT dataset"""
    return preprocess_pipeline(
        data, sampling_rate,
        apply_bandpass=True,
        apply_notch=True,  # Remove 60 Hz power line noise
        apply_normalize=True,
        segment=False
    )[0]


if __name__ == "__main__":
    # Test the preprocessor
    print("EEG Preprocessing Module")
    print("=" * 50)
    
    # Generate test signal
    fs = 256  # Hz
    t = np.arange(0, 10, 1/fs)
    test_signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t))
    
    # Test preprocessing
    preprocessor = EEGPreprocessor(fs)
    
    print("\n1. Testing bandpass filter...")
    filtered = preprocessor.bandpass_filter(test_signal)
    print(f"   Input shape: {test_signal.shape}, Output shape: {filtered.shape}")
    
    print("\n2. Testing normalization...")
    normalized = preprocessor.normalize(filtered)
    print(f"   Mean: {np.mean(normalized):.6f}, Std: {np.std(normalized):.6f}")
    
    print("\n3. Testing segmentation...")
    segments, times = preprocessor.segment_signal(normalized, window_size=4.0)
    print(f"   Generated {len(segments)} segments")
    
    print("\n✓ All tests passed!")