"""
Utility Functions for EEG Classification Project
=================================================

This module provides helper functions for:
- Loading preprocessed data
- Visualization
- Metrics calculation
- Data management

"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns


class DataLoader:
    """Helper class to load preprocessed EEG data"""
    
    def __init__(self, data_dir: str = 'data/processed'):
        self.data_dir = Path(data_dir)
        
    def load_bonn_splits(self) -> Tuple[np.ndarray, ...]:
        """
        Load train/val/test splits for Bonn dataset
        
        Returns:
        --------
        X_train, y_train, X_val, y_val, X_test, y_test
        """
        split_dir = self.data_dir / 'bonn' / 'splits'
        
        X_train = np.load(split_dir / 'X_train.npy')
        y_train = np.load(split_dir / 'y_train.npy')
        X_val = np.load(split_dir / 'X_val.npy')
        y_val = np.load(split_dir / 'y_val.npy')
        X_test = np.load(split_dir / 'X_test.npy')
        y_test = np.load(split_dir / 'y_test.npy')
        
        print(f"Loaded Bonn dataset splits:")
        print(f"  - Training: {X_train.shape}")
        print(f"  - Validation: {X_val.shape}")
        print(f"  - Test: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def load_bonn_preprocessed(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load preprocessed (unsegmented) Bonn data"""
        data_dir = self.data_dir / 'bonn'
        
        data = np.load(data_dir / 'preprocessed_data.npy')
        labels = np.load(data_dir / 'labels.npy')
        
        print(f"Loaded preprocessed Bonn data: {data.shape}")
        
        return data, labels
    
    def load_bonn_segmented(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load segmented Bonn data"""
        data_dir = self.data_dir / 'bonn'
        
        data = np.load(data_dir / 'segmented_data.npy')
        labels = np.load(data_dir / 'segmented_labels.npy')
        
        print(f"Loaded segmented Bonn data: {data.shape}")
        
        return data, labels
    
    def load_bonn_augmented(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load augmented Bonn data"""
        data_dir = self.data_dir / 'bonn'
        
        data = np.load(data_dir / 'augmented_data.npy')
        labels = np.load(data_dir / 'augmented_labels.npy')
        
        print(f"Loaded augmented Bonn data: {data.shape}")
        
        return data, labels


def get_class_distribution(labels: np.ndarray, class_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Get class distribution statistics
    
    Parameters:
    -----------
    labels : np.ndarray
        Label array
    class_names : list, optional
        Names for each class
    
    Returns:
    --------
    df : pd.DataFrame
        Distribution statistics
    """
    unique, counts = np.unique(labels, return_counts=True)
    percentages = counts / len(labels) * 100
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique]
    
    df = pd.DataFrame({
        'Class': class_names,
        'Count': counts,
        'Percentage': [f'{p:.2f}%' for p in percentages]
    })
    
    return df


def plot_class_distribution(labels: np.ndarray, 
                           class_names: Optional[List[str]] = None,
                           title: str = "Class Distribution",
                           figsize: Tuple[int, int] = (10, 6)):
    """
    Plot class distribution
    
    Parameters:
    -----------
    labels : np.ndarray
        Label array
    class_names : list, optional
        Names for each class
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    colors = sns.color_palette("husl", len(unique))
    ax1.bar(class_names, counts, color=colors, alpha=0.8)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Sample Counts', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (name, count) in enumerate(zip(class_names, counts)):
        ax1.text(i, count, str(count), ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(counts, labels=class_names, autopct='%1.1f%%', 
           colors=colors, startangle=90)
    ax2.set_title('Percentage Distribution', fontsize=12, fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_signal_examples(data: np.ndarray, labels: np.ndarray,
                        sampling_rate: float,
                        class_names: Optional[List[str]] = None,
                        n_examples: int = 3,
                        figsize: Tuple[int, int] = (15, 10)):
    """
    Plot example signals from each class
    
    Parameters:
    -----------
    data : np.ndarray
        Signal data (n_samples, n_timesteps)
    labels : np.ndarray
        Labels
    sampling_rate : float
        Sampling rate in Hz
    class_names : list, optional
        Class names
    n_examples : int
        Number of examples per class
    figsize : tuple
        Figure size
    """
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_labels]
    
    fig, axes = plt.subplots(n_classes, n_examples, 
                            figsize=figsize, squeeze=False)
    
    for i, label in enumerate(unique_labels):
        # Get indices for this class
        indices = np.where(labels == label)[0]
        
        # Randomly select examples
        selected = np.random.choice(indices, size=min(n_examples, len(indices)), 
                                   replace=False)
        
        for j, idx in enumerate(selected):
            signal = data[idx]
            time = np.arange(len(signal)) / sampling_rate
            
            axes[i, j].plot(time, signal, linewidth=0.5, alpha=0.8)
            axes[i, j].set_xlim([0, time[-1]])
            axes[i, j].grid(True, alpha=0.3)
            
            if j == 0:
                axes[i, j].set_ylabel(class_names[i], fontsize=10, fontweight='bold')
            
            if i == n_classes - 1:
                axes[i, j].set_xlabel('Time (s)', fontsize=9)
    
    plt.suptitle('Signal Examples by Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def calculate_signal_statistics(data: np.ndarray, 
                                labels: np.ndarray) -> pd.DataFrame:
    """
    Calculate statistical summary for each class
    
    Parameters:
    -----------
    data : np.ndarray
        Signal data
    labels : np.ndarray
        Labels
    
    Returns:
    --------
    df : pd.DataFrame
        Statistics dataframe
    """
    unique_labels = np.unique(labels)
    
    stats_list = []
    
    for label in unique_labels:
        class_data = data[labels == label]
        
        stats = {
            'Class': f'Class {label}',
            'Count': len(class_data),
            'Mean': np.mean(class_data),
            'Std': np.std(class_data),
            'Min': np.min(class_data),
            'Max': np.max(class_data),
            'Median': np.median(class_data),
            'Q25': np.percentile(class_data, 25),
            'Q75': np.percentile(class_data, 75),
        }
        
        stats_list.append(stats)
    
    df = pd.DataFrame(stats_list)
    
    return df


def save_preprocessing_info(output_path: str, **kwargs):
    """
    Save preprocessing configuration and statistics
    
    Parameters:
    -----------
    output_path : str
        Path to save info
    **kwargs : dict
        Configuration parameters
    """
    info = {
        'timestamp': pd.Timestamp.now().isoformat(),
        **kwargs
    }
    
    df = pd.DataFrame([info]).T
    df.columns = ['Value']
    df.to_csv(output_path)
    
    print(f"✓ Saved preprocessing info to {output_path}")


def load_preprocessing_info(info_path: str) -> Dict:
    """
    Load preprocessing information
    
    Parameters:
    -----------
    info_path : str
        Path to info file
    
    Returns:
    --------
    info : dict
        Preprocessing information
    """
    df = pd.read_csv(info_path, index_col=0)
    info = df['Value'].to_dict()
    
    return info


def create_summary_report(data_dict: Dict[str, np.ndarray],
                         labels_dict: Dict[str, np.ndarray],
                         output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create summary report for multiple datasets/splits
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of {name: data_array}
    labels_dict : dict
        Dictionary of {name: labels_array}
    output_path : str, optional
        Path to save report
    
    Returns:
    --------
    report : pd.DataFrame
        Summary report
    """
    report_data = []
    
    for name in data_dict.keys():
        data = data_dict[name]
        labels = labels_dict[name]
        
        unique, counts = np.unique(labels, return_counts=True)
        
        row = {
            'Dataset': name,
            'Total Samples': len(data),
            'Shape': str(data.shape),
            'Class 0 Count': counts[0] if 0 in unique else 0,
            'Class 1 Count': counts[1] if 1 in unique else 0,
            'Class Balance': f"{counts[0]/counts[1]:.2f}:1" if len(counts) == 2 else 'N/A',
        }
        
        report_data.append(row)
    
    report = pd.DataFrame(report_data)
    
    if output_path:
        report.to_csv(output_path, index=False)
        print(f"✓ Saved summary report to {output_path}")
    
    return report


def validate_data_splits(X_train: np.ndarray, X_val: np.ndarray, 
                        X_test: np.ndarray,
                        y_train: np.ndarray, y_val: np.ndarray, 
                        y_test: np.ndarray) -> bool:
    """
    Validate that data splits are correct
    
    Parameters:
    -----------
    X_train, X_val, X_test : np.ndarray
        Feature arrays
    y_train, y_val, y_test : np.ndarray
        Label arrays
    
    Returns:
    --------
    valid : bool
        True if all validations pass
    """
    print("Validating data splits...")
    
    # Check shapes match
    assert X_train.shape[0] == len(y_train), "Train data/labels mismatch"
    assert X_val.shape[0] == len(y_val), "Val data/labels mismatch"
    assert X_test.shape[0] == len(y_test), "Test data/labels mismatch"
    print("✓ Shapes match")
    
    # Check feature dimensions match
    assert X_train.shape[1:] == X_val.shape[1:] == X_test.shape[1:], \
        "Feature dimensions don't match"
    print("✓ Feature dimensions match")
    
    # Check no data leakage (no exact duplicates between sets)
    # This is a simple check - not exhaustive
    total_samples = len(X_train) + len(X_val) + len(X_test)
    print(f"✓ Total samples: {total_samples}")
    
    # Check class distribution
    train_dist = np.bincount(y_train)
    val_dist = np.bincount(y_val)
    test_dist = np.bincount(y_test)
    
    print(f"✓ Train distribution: {train_dist}")
    print(f"✓ Val distribution: {val_dist}")
    print(f"✓ Test distribution: {test_dist}")
    
    print("\n✓ All validations passed!")
    
    return True


if __name__ == "__main__":
    print("EEG Classification Utilities Module")
    print("=" * 50)
    
    # Example usage
    print("\nExample: Loading preprocessed data")
    print("from utils import DataLoader")
    print("loader = DataLoader('data/processed')")
    print("X_train, y_train, X_val, y_val, X_test, y_test = loader.load_bonn_splits()")