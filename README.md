# EEG Time Series Classification for Epileptic Seizure Detection

A comprehensive machine learning project for automated detection of epileptic seizures from EEG signals using both traditional machine learning and deep learning approaches.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)

---

## 📋 Project Overview

This project implements an end-to-end pipeline for classifying EEG signals to detect epileptic seizures. The work demonstrates advanced time series analysis, comprehensive feature engineering, and comparison of multiple machine learning and deep learning models.

### Key Highlights

- **Dataset**: Bonn EEG Dataset (500 single-channel recordings)
- **Classification**: Binary (Normal vs Seizure)
- **Features**: 70+ engineered features across 5 categories
- **Models**: 8 different models (5 ML + 3 DL)
- **Best Performance**: ~96% F1-Score, ~98% ROC-AUC

---

## 🎯 Objectives

1. **Develop** a comprehensive preprocessing pipeline for EEG signals
2. **Extract** meaningful features from time domain, frequency domain, and nonlinear dynamics
3. **Implement** and compare multiple classification models (ML and DL)
4. **Evaluate** models rigorously using appropriate metrics for medical applications
5. **Demonstrate** best practices in machine learning project development

---

## 🗂️ Project Structure

```
eeg-classification/
│
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── bonn/                     # Bonn EEG Dataset
│   │   └── chb-mit/                  # CHB-MIT Dataset (not used for model development)
│   └── processed/                    # Preprocessed data and features
│       └── bonn/
│           ├── features/             # Extracted features
│           └── splits/               # Train/val/test splits
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb           # Data exploration and visualization
│   ├── 02_preprocessing.ipynb                  # Signal preprocessing pipeline
│   ├── 03_feature_engineering.ipynb            # Feature extraction
│   ├── 04_model_development.ipynb              # Model training and comparison
│   └── 05_evaluation.ipynb                     # Final evaluation and reporting
│
├── src/
│   ├── data_loader.py                # Data loading utilities
│   ├── preprocessing.py              # Preprocessing functions
│   ├── features.py                   # Feature extraction methods
│   ├── models.py                     # Model architectures
│   ├── train.py                      # Training utilities
│   ├── evaluate.py                   # Evaluation functions
│   ├── feature_selection.py          # Feature selection methods
│   └── utils.py                      # Helper functions
│
├── results/
│   ├── models/                       # Trained models and scalers
│   └── figures/                      # Generated visualizations
│
├── README.md
└── requirements.txt
```

---

## 🔬 Methodology

### Phase 1: Exploratory Data Analysis
- **Dataset analysis**: Bonn EEG Dataset with 5 sets (Z, O, N, F, S)
- **Signal visualization**: Multi-channel EEG patterns
- **Statistical analysis**: Time domain and frequency domain characteristics
- **Class distribution**: Balanced binary classification (Seizure vs Normal)

### Phase 2: Data Preprocessing
- **Bandpass filtering**: 0.5-50 Hz (remove drift and high-frequency noise)
- **Notch filtering**: 60 Hz (power line interference removal for CHB-MIT)
- **Segmentation**: 4-second windows with 50% overlap
- **Normalization**: Z-score normalization per channel
- **Data augmentation**: Balance class distribution
- **Train/Val/Test split**: 70% / 15% / 15%

### Phase 3: Feature Engineering
Extracted **70+ features** across five categories:

1. **Time Domain Features**
   - Statistical: mean, std, variance, skewness, kurtosis
   - Dynamics: zero-crossing rate, energy, RMS

2. **Frequency Domain Features**
   - FFT-based: dominant frequency, spectral centroid, spectral entropy
   - EEG bands: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-50 Hz)
   - Band power ratios

3. **Wavelet Features**
   - Discrete Wavelet Transform (Daubechies db4)
   - 5 decomposition levels
   - Energy and entropy per level

4. **Nonlinear Features**
   - Sample entropy, Approximate entropy
   - Hjorth parameters (Activity, Mobility, Complexity)
   - Hurst exponent, DFA alpha

5. **RQA Features**
   - Recurrence rate, Determinism
   - Laminarity, Trapping time
   - Diagonal line analysis

### Phase 4: Model Development
Trained and compared **8 models**:

**Traditional Machine Learning (Feature-Based)**
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost
4. LightGBM
5. Support Vector Machine (SVM)

**Deep Learning Models**

6. 1D CNN (Convolutional Neural Network)
7. Bidirectional LSTM
8. Attention-LSTM

All models trained with:
- Proper regularization (dropout, early stopping)
- Class balancing
- Hyperparameter optimization
- Validation monitoring

### Phase 5: Evaluation & Analysis
Comprehensive evaluation including:
- **Performance metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Clinical metrics**: Sensitivity, Specificity, PPV, NPV
- **Error analysis**: False positive/negative investigation
- **Cross-validation**: 5-fold CV for traditional ML models
- **Statistical testing**: McNemar's test for model comparison
- **Feature importance**: Analysis for interpretability

---

## 📊 Results Summary

### Best Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 96.5% |
| Precision | 95.8% |
| Recall (Sensitivity) | 97.3% |
| F1-Score | 96.5% |
| Specificity | 95.7% |
| ROC-AUC | 98.3% |

### Confusion Matrix (Test Set)
```
                Predicted
              Normal  Seizure
Actual Normal   359      16
       Seizure   10     365
```

### Key Findings
- **High sensitivity (97.3%)** ensures most seizures are detected (critical for patient safety)
- **High specificity (95.7%)** minimizes false alarms
- **Balanced performance** suitable for clinical applications
- **Feature importance** reveals spectral power and entropy measures as most discriminative
- **Model comparison** shows both traditional ML and deep learning achieve excellent results

---

## 🛠️ Technologies Used

### Core Libraries
- **Python 3.8+**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning algorithms
- **TensorFlow/Keras** - Deep learning models

### Signal Processing
- **SciPy** - Signal processing and filtering
- **PyWavelets** - Wavelet transforms
- **MNE** - EEG data handling

### Visualization
- **Matplotlib** - Plotting
- **Seaborn** - Statistical visualizations
- **Plotly** - Interactive plots

### Model Training
- **XGBoost** - Gradient boosting
- **LightGBM** - Efficient gradient boosting

---

## 🔍 Project Highlights

### Time Series Analysis
- Emphasizes temporal dependencies in EEG signals
- Multi-scale analysis using wavelets
- Captures both local patterns (CNN) and long-term dependencies (LSTM)

### Feature Engineering
- Domain-specific features based on neuroscience literature
- Comprehensive coverage of signal characteristics
- Feature selection strategies to optimize performance

### Model Comparison
- Fair comparison using same train/test splits
- Both feature-based and end-to-end learning approaches
- Demonstrates trade-offs between interpretability and performance

### Clinical Relevance
- Metrics focused on medical applications (sensitivity/specificity)
- Error analysis with clinical implications
- Considerations for real-world deployment

---

## 🔮 Future Work

- Validate on CHB-MIT dataset for generalization assessment
- Implement real-time seizure prediction (not just detection)
- Explore patient-specific model fine-tuning
- Develop web-based demo application
- Investigate additional feature selection techniques
- Extend to multi-class classification (seizure types)

---

**Note**: This is an academic project demonstrating machine learning techniques for EEG analysis. It is not intended for clinical use without proper validation and regulatory approval.
