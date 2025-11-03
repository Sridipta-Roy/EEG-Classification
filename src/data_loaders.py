"""
Data Loading Utilities for EEG Classification Project
======================================================

This module provides functions to load and parse EEG data from:
1. Bonn EEG Dataset
2. CHB-MIT Scalp EEG Database

"""

import numpy as np
import pandas as pd
import mne
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


class BonnDataLoader:
    """
    Loader for Bonn EEG Dataset
    
    Dataset Structure:
    - Set Z: Healthy subjects (eyes open)
    - Set O: Healthy subjects (eyes closed)
    - Set N: Interictal (seizure-free) from hippocampus
    - Set F: Interictal (seizure-free) from cortex
    - Set S: Ictal (seizure activity)
    """
    
    SAMPLING_RATE = 173.61  # Hz
    DURATION = 23.6  # seconds
    SAMPLES_PER_SEGMENT = 4097
    
    SET_LABELS = {
        'Z': 0,  # Healthy (eyes open)
        'O': 0,  # Healthy (eyes closed)
        'N': 1,  # Interictal (hippocampus)
        'F': 1,  # Interictal (cortex)
        'S': 2   # Ictal (seizure)
    }
    
    SET_DESCRIPTIONS = {
        'Z': 'Healthy (Eyes Open)',
        'O': 'Healthy (Eyes Closed)',
        'N': 'Interictal - Hippocampus',
        'F': 'Interictal - Cortex',
        'S': 'Ictal (Seizure)'
    }
    
    def __init__(self, data_path: str):
        """
        Initialize Bonn dataset loader
        
        Parameters:
        -----------
        data_path : str
            Path to Bonn dataset directory
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
    
    def load_set(self, set_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a specific set from Bonn dataset
        
        Parameters:
        -----------
        set_name : str
            Set identifier ('Z', 'O', 'N', 'F', or 'S')
        
        Returns:
        --------
        data : np.ndarray
            EEG data array of shape (n_segments, n_samples)
        labels : np.ndarray
            Label array of shape (n_segments,)
        """
        if set_name not in self.SET_LABELS:
            raise ValueError(f"Invalid set name. Choose from: {list(self.SET_LABELS.keys())}")
        
        set_path = self.data_path / set_name
        
        if not set_path.exists():
            raise ValueError(f"Set path does not exist: {set_path}")
        
        data_list = []
        labels_list = []
        
        # Load all .txt files in the set directory
        files = sorted(set_path.glob('*.txt'))
        
        if not files:
            # Try alternative naming pattern
            files = sorted(set_path.glob(f'{set_name}*.txt'))
        
        for file in files:
            try:
                segment = np.loadtxt(file)
                
                # Validate segment length
                if len(segment) == self.SAMPLES_PER_SEGMENT:
                    data_list.append(segment)
                    labels_list.append(self.SET_LABELS[set_name])
                else:
                    print(f"Warning: Skipping {file.name} - unexpected length {len(segment)}")
            
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
        
        data = np.array(data_list)
        labels = np.array(labels_list)
        
        print(f"Loaded Set {set_name} ({self.SET_DESCRIPTIONS[set_name]})")
        print(f"  - Segments: {len(data)}")
        print(f"  - Shape: {data.shape}")
        print(f"  - Label: {self.SET_LABELS[set_name]}")
        
        return data, labels
    
    def load_all_sets(self, sets: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load multiple sets and combine them
        
        Parameters:
        -----------
        sets : List[str], optional
            List of set names to load. If None, loads all sets.
        
        Returns:
        --------
        data : np.ndarray
            Combined EEG data
        labels : np.ndarray
            Combined labels
        set_ids : np.ndarray
            Array indicating which set each segment came from
        """
        if sets is None:
            sets = list(self.SET_LABELS.keys())
        
        all_data = []
        all_labels = []
        all_set_ids = []
        
        for set_name in sets:
            data, labels = self.load_set(set_name)
            all_data.append(data)
            all_labels.append(labels)
            all_set_ids.append([set_name] * len(data))
        
        combined_data = np.vstack(all_data)
        combined_labels = np.concatenate(all_labels)
        combined_set_ids = np.concatenate(all_set_ids)
        
        print(f"\nCombined Dataset:")
        print(f"  - Total segments: {len(combined_data)}")
        print(f"  - Shape: {combined_data.shape}")
        print(f"  - Label distribution: {np.bincount(combined_labels)}")
        
        return combined_data, combined_labels, combined_set_ids
    
    def get_binary_classification_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data for binary classification (Seizure vs Non-Seizure)
        
        Returns:
        --------
        data : np.ndarray
            EEG data
        labels : np.ndarray
            Binary labels (0: Non-Seizure, 1: Seizure)
        """
        # Load non-seizure sets (Z, O, N, F)
        non_seizure_sets = ['Z', 'O', 'N', 'F']
        non_seizure_data = []
        
        for set_name in non_seizure_sets:
            data, _ = self.load_set(set_name)
            non_seizure_data.append(data)
        
        non_seizure_data = np.vstack(non_seizure_data)
        non_seizure_labels = np.zeros(len(non_seizure_data))
        
        # Load seizure set (S)
        seizure_data, _ = self.load_set('S')
        seizure_labels = np.ones(len(seizure_data))
        
        # Combine
        data = np.vstack([non_seizure_data, seizure_data])
        labels = np.concatenate([non_seizure_labels, seizure_labels])
        
        print(f"\nBinary Classification Dataset:")
        print(f"  - Non-Seizure samples: {len(non_seizure_labels)}")
        print(f"  - Seizure samples: {len(seizure_labels)}")
        print(f"  - Class balance: {len(seizure_labels) / len(labels):.2%} seizure")
        
        return data, labels


class CHBMITDataLoader:
    """
    Loader for CHB-MIT Scalp EEG Database
 
    """
    
    SAMPLING_RATE = 256  # Hz
    
    def __init__(self, data_path: str):
        """
        Initialize CHB-MIT dataset loader
        
        Parameters:
        -----------
        data_path : str
            Path to CHB-MIT dataset directory
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
    
    def get_available_patients(self) -> List[str]:
        """
        Get list of available patient directories
        
        Returns:
        --------
        patients : List[str]
            List of patient IDs
        """
        patients = [d.name for d in self.data_path.iterdir() 
                   if d.is_dir() and d.name.startswith('chb')]
        return sorted(patients)
    
    def load_patient_file(self, patient: str, file_idx: int = 0, 
                         preload: bool = True) -> Optional[mne.io.Raw]:
        """
        Load a specific EDF file for a patient
        
        Parameters:
        -----------
        patient : str
            Patient ID (e.g., 'chb01')
        file_idx : int
            Index of file to load
        preload : bool
            Whether to load data into memory
        
        Returns:
        --------
        raw : mne.io.Raw or None
            MNE Raw object containing EEG data
        """
        patient_path = self.data_path / patient
        
        if not patient_path.exists():
            print(f"Patient directory not found: {patient}")
            return None
        
        # Get all EDF files
        edf_files = sorted(patient_path.glob('*.edf'))
        
        if not edf_files:
            print(f"No EDF files found for patient {patient}")
            return None
        
        if file_idx >= len(edf_files):
            print(f"File index {file_idx} out of range. Available: {len(edf_files)} files")
            return None
        
        file_path = edf_files[file_idx]
        
        try:
            raw = mne.io.read_raw_edf(file_path, preload=preload, verbose=False)
            print(f"Loaded: {file_path.name}")
            print(f"  - Channels: {len(raw.ch_names)}")
            print(f"  - Duration: {raw.times[-1]:.2f} seconds")
            print(f"  - Sampling Rate: {raw.info['sfreq']} Hz")
            
            return raw
        
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def load_patient_summary(self, patient: str) -> Dict:
        """
        Load summary information for a patient
        
        Parameters:
        -----------
        patient : str
            Patient ID
        
        Returns:
        --------
        summary : dict
            Dictionary containing patient summary
        """
        patient_path = self.data_path / patient
        summary_file = patient_path / f"{patient}-summary.txt"
        
        summary = {
            'patient_id': patient,
            'edf_files': [],
            'seizure_files': [],
            'total_seizures': 0
        }
        
        # Get all EDF files
        edf_files = sorted(patient_path.glob('*.edf'))
        summary['edf_files'] = [f.name for f in edf_files]
        summary['total_files'] = len(edf_files)
        
        # Parse summary file if it exists
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    content = f.read()
                    
                    # Look for seizure information
                    if 'Seizures' in content or 'seizure' in content.lower():
                        # Parse seizure information (simplified)
                        lines = content.split('\n')
                        for line in lines:
                            if 'File Name' in line or 'seizure' in line.lower():
                                # Extract file names with seizures
                                for edf_file in summary['edf_files']:
                                    if edf_file in line:
                                        summary['seizure_files'].append(edf_file)
                                        summary['total_seizures'] += 1
            
            except Exception as e:
                print(f"Error parsing summary file: {e}")
        
        return summary
    
    def extract_segments(self, raw: mne.io.Raw, window_size: float = 4.0,
                        overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract segments from raw data using sliding window
        
        Parameters:
        -----------
        raw : mne.io.Raw
            MNE Raw object
        window_size : float
            Window size in seconds
        overlap : float
            Overlap ratio (0.0 to 1.0)
        
        Returns:
        --------
        segments : np.ndarray
            Array of segments (n_segments, n_channels, n_samples)
        times : np.ndarray
            Start time of each segment
        """
        sfreq = raw.info['sfreq']
        window_samples = int(window_size * sfreq)
        step_samples = int(window_samples * (1 - overlap))
        
        data = raw.get_data()
        n_channels, n_samples = data.shape
        
        segments = []
        segment_times = []
        
        start = 0
        while start + window_samples <= n_samples:
            segment = data[:, start:start + window_samples]
            segments.append(segment)
            segment_times.append(start / sfreq)
            start += step_samples
        
        segments = np.array(segments)
        segment_times = np.array(segment_times)
        
        print(f"Extracted {len(segments)} segments")
        print(f"  - Segment shape: {segments.shape}")
        print(f"  - Window size: {window_size}s")
        print(f"  - Overlap: {overlap * 100}%")
        
        return segments, segment_times


def load_bonn_dataset(data_path: str, sets: Optional[List[str]] = None,
                     binary: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to load Bonn dataset
    
    Parameters:
    -----------
    data_path : str
        Path to Bonn dataset
    sets : List[str], optional
        Specific sets to load
    binary : bool
        If True, return binary labels (Seizure vs Non-Seizure)
    
    Returns:
    --------
    data : np.ndarray
        EEG data
    labels : np.ndarray
        Labels
    """
    loader = BonnDataLoader(data_path)
    
    if binary:
        return loader.get_binary_classification_data()
    else:
        data, labels, _ = loader.load_all_sets(sets)
        return data, labels


def load_chb_mit_patient(data_path: str, patient: str, 
                         file_idx: int = 0) -> Optional[mne.io.Raw]:
    """
    Convenience function to load CHB-MIT patient data
    
    Parameters:
    -----------
    data_path : str
        Path to CHB-MIT dataset
    patient : str
        Patient ID
    file_idx : int
        File index
    
    Returns:
    --------
    raw : mne.io.Raw or None
        MNE Raw object
    """
    loader = CHBMITDataLoader(data_path)
    return loader.load_patient_file(patient, file_idx)

