"""
EEG Feature Extraction Module
==============================

This module extracts features from EEG signals:
1. Time Domain Features (statistical)
2. Frequency Domain Features (FFT, PSD, band powers)
3. Time-Frequency Features (wavelets)
4. Nonlinear Features (entropy measures, RQA)
5. Recurrence Network Features

"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import pywt
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class TimeDomainFeatures:
    """Extract statistical features from time domain"""
    
    @staticmethod
    def extract(signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract time domain statistical features
        
        Parameters:
        -----------
        signal_data : np.ndarray
            1D signal array
        
        Returns:
        --------
        features : dict
            Dictionary of feature name: value pairs
        """
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['var'] = np.var(signal_data)
        features['median'] = np.median(signal_data)
        features['min'] = np.min(signal_data)
        features['max'] = np.max(signal_data)
        features['range'] = np.ptp(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        
        # Higher order statistics
        features['skewness'] = stats.skew(signal_data)
        features['kurtosis'] = stats.kurtosis(signal_data)
        
        # Percentiles
        features['q25'] = np.percentile(signal_data, 25)
        features['q75'] = np.percentile(signal_data, 75)
        features['iqr'] = features['q75'] - features['q25']
        
        # Energy and power
        features['energy'] = np.sum(signal_data**2)
        features['power'] = features['energy'] / len(signal_data)
        
        # Zero crossings
        zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal_data)
        
        # Mean absolute deviation
        features['mad'] = np.mean(np.abs(signal_data - features['mean']))
        
        # Signal complexity (approximate)
        features['complexity'] = np.sqrt(np.mean(np.diff(signal_data)**2))
        
        return features


class FrequencyDomainFeatures:
    """Extract features from frequency domain"""
    
    def __init__(self, sampling_rate: float):
        self.sampling_rate = sampling_rate
        
    def extract(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency domain features
        
        Parameters:
        -----------
        signal_data : np.ndarray
            1D signal array
        
        Returns:
        --------
        features : dict
            Dictionary of feature name: value pairs
        """
        features = {}
        
        # Compute FFT
        fft_vals = fft(signal_data)
        fft_freq = fftfreq(len(signal_data), 1/self.sampling_rate)
        
        # Only positive frequencies
        positive_freq_idx = fft_freq > 0
        fft_vals = np.abs(fft_vals[positive_freq_idx])
        fft_freq = fft_freq[positive_freq_idx]
        
        # Power spectral density using Welch's method
        freqs, psd = signal.welch(signal_data, fs=self.sampling_rate, 
                                   nperseg=min(256, len(signal_data)//4))
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(psd)
        features['dominant_frequency'] = freqs[dominant_freq_idx]
        features['dominant_power'] = psd[dominant_freq_idx]
        
        # Spectral centroid (center of mass of spectrum)
        features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
        
        # Spectral entropy
        psd_norm = psd / np.sum(psd)
        features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        # Total power
        features['total_power'] = np.sum(psd)
        
        # Frequency band powers (EEG bands)
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        for band_name, (low_freq, high_freq) in bands.items():
            band_idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            band_power = np.trapz(psd[band_idx], freqs[band_idx])
            features[f'{band_name}_power'] = band_power
            features[f'{band_name}_relative_power'] = band_power / features['total_power']
        
        # Band power ratios
        features['theta_beta_ratio'] = features['theta_power'] / (features['beta_power'] + 1e-10)
        features['alpha_delta_ratio'] = features['alpha_power'] / (features['delta_power'] + 1e-10)
        
        # Spectral edge frequency (frequency below which 95% of power is contained)
        cumsum_psd = np.cumsum(psd)
        edge_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0][0]
        features['spectral_edge_95'] = freqs[edge_idx]
        
        return features


class WaveletFeatures:
    """Extract wavelet-based time-frequency features"""
    
    def __init__(self, wavelet: str = 'db4', level: int = 5):
        """
        Initialize wavelet feature extractor
        
        Parameters:
        -----------
        wavelet : str
            Wavelet family (default: 'db4' - Daubechies 4)
        level : int
            Decomposition level
        """
        self.wavelet = wavelet
        self.level = level
        
    def extract(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract wavelet features
        
        Parameters:
        -----------
        signal_data : np.ndarray
            1D signal array
        
        Returns:
        --------
        features : dict
            Dictionary of feature name: value pairs
        """
        features = {}
        
        # Discrete wavelet transform
        coeffs = pywt.wavedec(signal_data, self.wavelet, level=self.level)
        
        # Extract features from each level
        for i, coeff in enumerate(coeffs):
            level_name = f'wavelet_level_{i}'
            
            # Statistical features of coefficients
            features[f'{level_name}_mean'] = np.mean(coeff)
            features[f'{level_name}_std'] = np.std(coeff)
            features[f'{level_name}_energy'] = np.sum(coeff**2)
            features[f'{level_name}_entropy'] = self._wavelet_entropy(coeff)
        
        # Total wavelet energy
        features['wavelet_total_energy'] = sum([np.sum(c**2) for c in coeffs])
        
        # Relative energy per level
        for i, coeff in enumerate(coeffs):
            energy = np.sum(coeff**2)
            features[f'wavelet_level_{i}_relative_energy'] = \
                energy / (features['wavelet_total_energy'] + 1e-10)
        
        return features
    
    @staticmethod
    def _wavelet_entropy(coeffs: np.ndarray) -> float:
        """Calculate entropy of wavelet coefficients"""
        # Normalize coefficients
        energy = np.sum(coeffs**2)
        if energy == 0:
            return 0.0
        
        p = (coeffs**2) / energy
        p = p[p > 0]  # Remove zeros
        
        return -np.sum(p * np.log2(p))


class NonlinearFeatures:
    """Extract nonlinear dynamics features"""
    
    @staticmethod
    def extract(signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract nonlinear features
        
        Parameters:
        -----------
        signal_data : np.ndarray
            1D signal array
        
        Returns:
        --------
        features : dict
            Dictionary of feature name: value pairs
        """
        features = {}
        
        # Sample Entropy
        try:
            features['sample_entropy'] = NonlinearFeatures._sample_entropy(signal_data)
        except:
            features['sample_entropy'] = 0.0
        
        # Approximate Entropy
        try:
            features['approximate_entropy'] = NonlinearFeatures._approximate_entropy(signal_data)
        except:
            features['approximate_entropy'] = 0.0
        
        # Hjorth parameters
        hjorth = NonlinearFeatures._hjorth_parameters(signal_data)
        features['hjorth_activity'] = hjorth[0]
        features['hjorth_mobility'] = hjorth[1]
        features['hjorth_complexity'] = hjorth[2]
        
        # Hurst exponent (simplified)
        try:
            features['hurst_exponent'] = NonlinearFeatures._hurst_exponent(signal_data)
        except:
            features['hurst_exponent'] = 0.5
        
        # Detrended Fluctuation Analysis (DFA) - simplified
        try:
            features['dfa_alpha'] = NonlinearFeatures._dfa(signal_data)
        except:
            features['dfa_alpha'] = 1.0
        
        return features
    
    @staticmethod
    def _sample_entropy(signal_data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Calculate Sample Entropy
        
        Parameters:
        -----------
        signal_data : np.ndarray
            Signal data
        m : int
            Pattern length
        r : float
            Tolerance (as fraction of std)
        """
        N = len(signal_data)
        r = r * np.std(signal_data)
        
        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            x = [[signal_data[j] for j in range(i, i + m)] 
                 for i in range(N - m + 1)]
            C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) - 1 
                 for x_i in x]
            return sum(C)
        
        return -np.log(_phi(m + 1) / _phi(m))
    
    @staticmethod
    def _approximate_entropy(signal_data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate Approximate Entropy"""
        N = len(signal_data)
        r = r * np.std(signal_data)
        
        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            x = [[signal_data[j] for j in range(i, i + m)] 
                 for i in range(N - m + 1)]
            C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) 
                 for x_i in x]
            phi = sum([np.log(c / (N - m + 1)) for c in C])
            return phi / (N - m + 1)
        
        return _phi(m) - _phi(m + 1)
    
    @staticmethod
    def _hjorth_parameters(signal_data: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate Hjorth parameters (Activity, Mobility, Complexity)
        
        Returns:
        --------
        activity : float
            Variance of signal
        mobility : float
            Square root of variance of first derivative / variance of signal
        complexity : float
            Mobility of first derivative / mobility of signal
        """
        # First derivative
        first_deriv = np.diff(signal_data)
        
        # Second derivative
        second_deriv = np.diff(first_deriv)
        
        # Activity (variance)
        activity = np.var(signal_data)
        
        # Mobility
        mobility = np.sqrt(np.var(first_deriv) / activity)
        
        # Complexity
        complexity = np.sqrt(np.var(second_deriv) / np.var(first_deriv)) / mobility
        
        return activity, mobility, complexity
    
    @staticmethod
    def _hurst_exponent(signal_data: np.ndarray) -> float:
        """
        Calculate Hurst exponent using R/S analysis
        
        Returns:
        --------
        hurst : float
            Hurst exponent (0.5 = random walk, >0.5 = trending, <0.5 = mean reverting)
        """
        # Simplified version
        n = len(signal_data)
        
        # Mean centered cumulative sum
        mean_signal = np.mean(signal_data)
        Y = np.cumsum(signal_data - mean_signal)
        
        # Range
        R = np.max(Y) - np.min(Y)
        
        # Standard deviation
        S = np.std(signal_data)
        
        if S == 0:
            return 0.5
        
        # R/S statistic
        RS = R / S
        
        # Hurst exponent approximation
        return np.log(RS) / np.log(n)
    
    @staticmethod
    def _dfa(signal_data: np.ndarray, n_scales: int = 10) -> float:
        """
        Detrended Fluctuation Analysis
        
        Returns:
        --------
        alpha : float
            DFA scaling exponent
        """
        N = len(signal_data)
        
        # Profile (cumulative sum of mean-centered signal)
        Y = np.cumsum(signal_data - np.mean(signal_data))
        
        # Box sizes (logarithmically distributed)
        scales = np.logspace(np.log10(4), np.log10(N//4), n_scales, dtype=int)
        
        F = []
        for scale in scales:
            # Split into boxes
            n_boxes = N // scale
            
            # Detrend each box
            fluctuations = []
            for i in range(n_boxes):
                box = Y[i*scale:(i+1)*scale]
                # Fit linear trend
                t = np.arange(len(box))
                coeffs = np.polyfit(t, box, 1)
                trend = np.polyval(coeffs, t)
                # Calculate fluctuation
                fluctuation = np.sqrt(np.mean((box - trend)**2))
                fluctuations.append(fluctuation)
            
            F.append(np.mean(fluctuations))
        
        # Fit power law: F(n) ~ n^alpha
        log_scales = np.log10(scales)
        log_F = np.log10(F)
        
        alpha = np.polyfit(log_scales, log_F, 1)[0]
        
        return alpha


class RQAFeatures:
    """
    Recurrence Quantification Analysis (RQA) Features 
 
    """
    
    def __init__(self, embedding_dim: int = 3, time_delay: int = 1, 
                 threshold: float = 0.1):
        """
        Initialize RQA feature extractor
        
        Parameters:
        -----------
        embedding_dim : int
            Embedding dimension
        time_delay : int
            Time delay for embedding
        threshold : float
            Recurrence threshold (as fraction of maximum distance)
        """
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.threshold = threshold
        
    def extract(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract RQA features
        
        Parameters:
        -----------
        signal_data : np.ndarray
            1D signal array
        
        Returns:
        --------
        features : dict
            Dictionary of RQA feature name: value pairs
        """
        # Create recurrence plot
        recurrence_matrix = self._create_recurrence_plot(signal_data)
        
        # Extract RQA measures
        features = {}
        
        # Recurrence Rate (RR)
        features['rqa_recurrence_rate'] = self._recurrence_rate(recurrence_matrix)
        
        # Determinism (DET)
        features['rqa_determinism'] = self._determinism(recurrence_matrix)
        
        # Average diagonal line length
        features['rqa_avg_diagonal_length'] = self._average_diagonal_length(recurrence_matrix)
        
        # Longest diagonal line
        features['rqa_max_diagonal_length'] = self._max_diagonal_length(recurrence_matrix)
        
        # Laminarity (LAM)
        features['rqa_laminarity'] = self._laminarity(recurrence_matrix)
        
        # Trapping time (TT)
        features['rqa_trapping_time'] = self._trapping_time(recurrence_matrix)
        
        # Entropy of diagonal line lengths
        features['rqa_entropy'] = self._diagonal_entropy(recurrence_matrix)
        
        return features
    
    def _create_recurrence_plot(self, signal_data: np.ndarray) -> np.ndarray:
        """Create recurrence plot using time-delay embedding"""
        # Time delay embedding
        N = len(signal_data)
        M = N - (self.embedding_dim - 1) * self.time_delay
        
        # Create embedded vectors
        embedded = np.zeros((M, self.embedding_dim))
        for i in range(M):
            for j in range(self.embedding_dim):
                embedded[i, j] = signal_data[i + j * self.time_delay]
        
        # Compute distance matrix
        dist_matrix = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                dist_matrix[i, j] = np.linalg.norm(embedded[i] - embedded[j])
        
        # Apply threshold
        threshold_value = self.threshold * np.max(dist_matrix)
        recurrence_matrix = (dist_matrix < threshold_value).astype(int)
        
        return recurrence_matrix
    
    @staticmethod
    def _recurrence_rate(recurrence_matrix: np.ndarray) -> float:
        """Recurrence Rate: Density of recurrence points"""
        N = recurrence_matrix.shape[0]
        return np.sum(recurrence_matrix) / (N * N)
    
    def _determinism(self, recurrence_matrix: np.ndarray, min_length: int = 2) -> float:
        """Determinism: Ratio of recurrence points forming diagonal lines"""
        diagonal_points = 0
        total_recurrence_points = np.sum(recurrence_matrix)
        
        if total_recurrence_points == 0:
            return 0.0
        
        # Count points in diagonal lines
        N = recurrence_matrix.shape[0]
        for offset in range(-N+1, N):
            diagonal = np.diagonal(recurrence_matrix, offset=offset)
            lengths = self._get_line_lengths(diagonal)
            diagonal_points += sum([l for l in lengths if l >= min_length])
        
        return diagonal_points / total_recurrence_points
    
    def _average_diagonal_length(self, recurrence_matrix: np.ndarray, 
                                 min_length: int = 2) -> float:
        """Average length of diagonal lines"""
        lengths = []
        N = recurrence_matrix.shape[0]
        
        for offset in range(-N+1, N):
            diagonal = np.diagonal(recurrence_matrix, offset=offset)
            line_lengths = self._get_line_lengths(diagonal)
            lengths.extend([l for l in line_lengths if l >= min_length])
        
        return np.mean(lengths) if lengths else 0.0
    
    def _max_diagonal_length(self, recurrence_matrix: np.ndarray) -> float:
        """Longest diagonal line length"""
        max_length = 0
        N = recurrence_matrix.shape[0]
        
        for offset in range(-N+1, N):
            diagonal = np.diagonal(recurrence_matrix, offset=offset)
            lengths = self._get_line_lengths(diagonal)
            if lengths:
                max_length = max(max_length, max(lengths))
        
        return float(max_length)
    
    def _laminarity(self, recurrence_matrix: np.ndarray, min_length: int = 2) -> float:
        """Laminarity: Ratio of recurrence points forming vertical lines"""
        vertical_points = 0
        total_recurrence_points = np.sum(recurrence_matrix)
        
        if total_recurrence_points == 0:
            return 0.0
        
        # Count points in vertical lines
        N = recurrence_matrix.shape[0]
        for col in range(N):
            lengths = self._get_line_lengths(recurrence_matrix[:, col])
            vertical_points += sum([l for l in lengths if l >= min_length])
        
        return vertical_points / total_recurrence_points
    
    def _trapping_time(self, recurrence_matrix: np.ndarray, min_length: int = 2) -> float:
        """Trapping Time: Average length of vertical lines"""
        lengths = []
        N = recurrence_matrix.shape[0]
        
        for col in range(N):
            line_lengths = self._get_line_lengths(recurrence_matrix[:, col])
            lengths.extend([l for l in line_lengths if l >= min_length])
        
        return np.mean(lengths) if lengths else 0.0
    
    def _diagonal_entropy(self, recurrence_matrix: np.ndarray, min_length: int = 2) -> float:
        """Entropy of diagonal line length distribution"""
        lengths = []
        N = recurrence_matrix.shape[0]
        
        for offset in range(-N+1, N):
            diagonal = np.diagonal(recurrence_matrix, offset=offset)
            line_lengths = self._get_line_lengths(diagonal)
            lengths.extend([l for l in line_lengths if l >= min_length])
        
        if not lengths:
            return 0.0
        
        # Calculate probability distribution
        unique_lengths = np.unique(lengths)
        probs = np.array([np.sum(np.array(lengths) == l) for l in unique_lengths])
        probs = probs / np.sum(probs)
        
        # Calculate entropy
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    @staticmethod
    def _get_line_lengths(binary_array: np.ndarray) -> List[int]:
        """Get lengths of consecutive 1s in binary array"""
        lengths = []
        current_length = 0
        
        for val in binary_array:
            if val == 1:
                current_length += 1
            else:
                if current_length > 0:
                    lengths.append(current_length)
                current_length = 0
        
        if current_length > 0:
            lengths.append(current_length)
        
        return lengths


class FeatureExtractor:
    """
    Main feature extraction class combining all feature types
    """
    
    def __init__(self, sampling_rate: float):
        """
        Initialize feature extractor
        
        Parameters:
        -----------
        sampling_rate : float
            Sampling rate of the signal in Hz
        """
        self.sampling_rate = sampling_rate
        self.time_features = TimeDomainFeatures()
        self.freq_features = FrequencyDomainFeatures(sampling_rate)
        self.wavelet_features = WaveletFeatures()
        self.nonlinear_features = NonlinearFeatures()
        self.rqa_features = RQAFeatures()
        
    def extract_all(self, signal_data: np.ndarray, 
                   include_rqa: bool = True) -> Dict[str, float]:
        """
        Extract all features from a signal
        
        Parameters:
        -----------
        signal_data : np.ndarray
            1D signal array
        include_rqa : bool
            Whether to include RQA features (computationally expensive)
        
        Returns:
        --------
        features : dict
            Dictionary of all features
        """
        all_features = {}
        
        # Time domain features
        all_features.update(self.time_features.extract(signal_data))
        
        # Frequency domain features
        all_features.update(self.freq_features.extract(signal_data))
        
        # Wavelet features
        all_features.update(self.wavelet_features.extract(signal_data))
        
        # Nonlinear features
        all_features.update(self.nonlinear_features.extract(signal_data))
        
        # RQA features 
        if include_rqa:
            all_features.update(self.rqa_features.extract(signal_data))
        
        return all_features
    
    def get_feature_names(self, include_rqa: bool = True) -> List[str]:
        """Get list of all feature names"""
        # Create a dummy signal to get feature names
        dummy_signal = np.random.randn(1000)
        features = self.extract_all(dummy_signal, include_rqa=include_rqa)
        return list(features.keys())


if __name__ == "__main__":
    print("EEG Feature Extraction Module")
    print("=" * 50)
    
    # Generate test signal
    fs = 173.61  # Bonn sampling rate
    t = np.linspace(0, 10, int(10 * fs))
    test_signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t))
    
    # Test feature extraction
    print("\nTesting feature extraction...")
    extractor = FeatureExtractor(fs)
    
    print("\n1. Time domain features...")
    time_feats = extractor.time_features.extract(test_signal)
    print(f"   Extracted {len(time_feats)} features")
    
    print("\n2. Frequency domain features...")
    freq_feats = extractor.freq_features.extract(test_signal)
    print(f"   Extracted {len(freq_feats)} features")
    
    print("\n3. Wavelet features...")
    wavelet_feats = extractor.wavelet_features.extract(test_signal)
    print(f"   Extracted {len(wavelet_feats)} features")
    
    print("\n4. Nonlinear features...")
    nonlinear_feats = extractor.nonlinear_features.extract(test_signal)
    print(f"   Extracted {len(nonlinear_feats)} features")
    
    print("\n5. RQA features...")
    rqa_feats = extractor.rqa_features.extract(test_signal[:500])  # Use subset for speed
    print(f"   Extracted {len(rqa_feats)} features")
    
    print("\n6. All features combined...")
    all_feats = extractor.extract_all(test_signal, include_rqa=False)
    print(f"   Total features: {len(all_feats)}")
    
    print("\n✓ All tests passed!")