"""
EEG Classification Models Module
=================================

This module implements various models for EEG time series classification:
1. Traditional ML Models (Random Forest, XGBoost, SVM)
2. Deep Learning Models (1D CNN, LSTM, CNN-LSTM Hybrid)
3. Attention-based Models

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class TraditionalMLModels:
    """
    Traditional machine learning models for feature-based classification
    """
    
    @staticmethod
    def get_random_forest(n_estimators: int = 100, 
                         max_depth: Optional[int] = None,
                         random_state: int = 42) -> RandomForestClassifier:
        """
        Random Forest classifier
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees
        max_depth : int, optional
            Maximum depth of trees
        random_state : int
            Random seed
        """
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
    
    @staticmethod
    def get_xgboost(n_estimators: int = 100,
                   max_depth: int = 6,
                   learning_rate: float = 0.1,
                   random_state: int = 42) -> XGBClassifier:
        """
        XGBoost classifier
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Learning rate
        random_state : int
            Random seed
        """
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=1  
        )
    
    @staticmethod
    def get_lightgbm(n_estimators: int = 100,
                    max_depth: int = -1,
                    learning_rate: float = 0.1,
                    random_state: int = 42) -> LGBMClassifier:
        """
        LightGBM classifier
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth (-1 = no limit)
        learning_rate : float
            Learning rate
        random_state : int
            Random seed
        """
        return LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            class_weight='balanced',
            verbose=-1
        )
    
    @staticmethod
    def get_svm(kernel: str = 'rbf',
               C: float = 1.0,
               gamma: str = 'scale',
               random_state: int = 42) -> SVC:
        """
        Support Vector Machine classifier
        
        Parameters:
        -----------
        kernel : str
            Kernel type ('linear', 'rbf', 'poly')
        C : float
            Regularization parameter
        gamma : str or float
            Kernel coefficient
        random_state : int
            Random seed
        """
        return SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            class_weight='balanced',
            probability=True
        )
    
    @staticmethod
    def get_logistic_regression(C: float = 1.0,
                               max_iter: int = 1000,
                               random_state: int = 42) -> LogisticRegression:
        """
        Logistic Regression classifier (baseline)
        
        Parameters:
        -----------
        C : float
            Inverse regularization strength
        max_iter : int
            Maximum iterations
        random_state : int
            Random seed
        """
        return LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            class_weight='balanced'
        )


class DeepLearningModels:
    """
    Deep learning models for end-to-end time series classification
    """
    
    @staticmethod
    def build_1d_cnn(input_shape: Tuple[int, int],
                    num_classes: int = 2,
                    filters: list = [64, 128, 256],
                    kernel_sizes: list = [7, 5, 3],
                    dropout_rate: float = 0.5) -> keras.Model:
        """
        1D CNN for time series classification        
        
        Parameters:
        -----------
        input_shape : tuple
            (timesteps, channels) - e.g., (694, 1)
        num_classes : int
            Number of output classes
        filters : list
            Number of filters for each conv layer
        kernel_sizes : list
            Kernel size for each conv layer
        dropout_rate : float
            Dropout rate
        
        Returns:
        --------
        model : keras.Model
            Compiled CNN model
        """
        inputs = layers.Input(shape=input_shape)
        
        x = inputs
        
        # Convolutional blocks
        for i, (n_filters, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            x = layers.Conv1D(
                filters=n_filters,
                kernel_size=kernel_size,
                padding='same',
                name=f'conv1d_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            x = layers.Activation('relu', name=f'relu_{i+1}')(x)
            x = layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}')(x)
            x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu', name='dense_1')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense')(x)
        x = layers.Dense(64, activation='relu', name='dense_2')(x)
        
        # Output layer
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', keras.metrics.AUC(name='auc')]
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model = models.Model(inputs=inputs, outputs=outputs, name='1D_CNN')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    @staticmethod
    def build_lstm(input_shape: Tuple[int, int],
                  num_classes: int = 2,
                  lstm_units: list = [128, 64],
                  dropout_rate: float = 0.5,
                  bidirectional: bool = True) -> keras.Model:
        """
        LSTM for time series classification
        
        Parameters:
        -----------
        input_shape : tuple
            (timesteps, features)
        num_classes : int
            Number of output classes
        lstm_units : list
            Number of units for each LSTM layer
        dropout_rate : float
            Dropout rate
        bidirectional : bool
            Whether to use bidirectional LSTM
        
        Returns:
        --------
        model : keras.Model
            Compiled LSTM model
        """
        inputs = layers.Input(shape=input_shape)
        
        x = inputs
        
        # LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1)
            
            lstm_layer = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate * 0.5,
                name=f'lstm_{i+1}'
            )
            
            if bidirectional:
                x = layers.Bidirectional(lstm_layer, name=f'bilstm_{i+1}')(x)
            else:
                x = lstm_layer(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense')(x)
        
        # Output layer
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', keras.metrics.AUC(name='auc')]
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model = models.Model(inputs=inputs, outputs=outputs, name='LSTM')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    @staticmethod
    def build_cnn_lstm_hybrid(input_shape: Tuple[int, int],
                             num_classes: int = 2,
                             cnn_filters: list = [64, 128],
                             lstm_units: int = 64,
                             dropout_rate: float = 0.5) -> keras.Model:
        """
        CNN-LSTM Hybrid model
        
        Architecture:
        - CNN layers for feature extraction
        - LSTM layers for temporal modeling
        - Dense layers for classification
        
        Parameters:
        -----------
        input_shape : tuple
            (timesteps, channels)
        num_classes : int
            Number of output classes
        cnn_filters : list
            Number of filters for each CNN layer
        lstm_units : int
            Number of LSTM units
        dropout_rate : float
            Dropout rate
        
        Returns:
        --------
        model : keras.Model
            Compiled CNN-LSTM model
        """
        inputs = layers.Input(shape=input_shape)
        
        x = inputs
        
        # CNN feature extraction
        for i, n_filters in enumerate(cnn_filters):
            x = layers.Conv1D(
                filters=n_filters,
                kernel_size=5,
                padding='same',
                name=f'conv_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            x = layers.Activation('relu', name=f'relu_{i+1}')(x)
            x = layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}')(x)
            x = layers.Dropout(dropout_rate * 0.5, name=f'dropout_cnn_{i+1}')(x)
        
        # LSTM temporal modeling
        x = layers.Bidirectional(
            layers.LSTM(
                units=lstm_units,
                return_sequences=False,
                dropout=dropout_rate,
                name='lstm'
            ),
            name='bilstm'
        )(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense')(x)
        
        # Output layer
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', keras.metrics.AUC(name='auc')]
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model = models.Model(inputs=inputs, outputs=outputs, name='CNN_LSTM')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    @staticmethod
    def build_attention_lstm(input_shape: Tuple[int, int],
                            num_classes: int = 2,
                            lstm_units: int = 128,
                            attention_units: int = 64,
                            dropout_rate: float = 0.5) -> keras.Model:
        """
        LSTM with self-attention mechanism
        
        Parameters:
        -----------
        input_shape : tuple
            (timesteps, features)
        num_classes : int
            Number of output classes
        lstm_units : int
            Number of LSTM units
        attention_units : int
            Number of attention units
        dropout_rate : float
            Dropout rate
        
        Returns:
        --------
        model : keras.Model
            Compiled Attention-LSTM model
        """
        inputs = layers.Input(shape=input_shape)
        
        # Bidirectional LSTM
        lstm_out = layers.Bidirectional(
            layers.LSTM(
                units=lstm_units,
                return_sequences=True,
                dropout=dropout_rate,
                name='lstm'
            ),
            name='bilstm'
        )(inputs)
        
        # Attention mechanism
        attention = layers.Dense(attention_units, activation='tanh', name='attention_dense')(lstm_out)
        attention = layers.Dense(1, name='attention_weights')(attention)
        attention = layers.Flatten(name='attention_flatten')(attention)
        attention = layers.Activation('softmax', name='attention_softmax')(attention)
        attention = layers.RepeatVector(lstm_units * 2, name='attention_repeat')(attention)
        attention = layers.Permute([2, 1], name='attention_permute')(attention)
        
        # Apply attention
        merged = layers.Multiply(name='attention_multiply')([lstm_out, attention])
        merged = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name='attention_sum')(merged)
        
        # Dense layers
        x = layers.Dense(64, activation='relu', name='dense_1')(merged)
        x = layers.Dropout(dropout_rate, name='dropout_dense')(x)
        
        # Output layer
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', keras.metrics.AUC(name='auc')]
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model = models.Model(inputs=inputs, outputs=outputs, name='Attention_LSTM')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        return model


def get_callbacks(model_name: str,
                 patience: int = 10,
                 min_delta: float = 0.001) -> list:
    """
    Get training callbacks
    
    Parameters:
    -----------
    model_name : str
        Name for saved model
    patience : int
        Early stopping patience
    min_delta : float
        Minimum change to qualify as improvement
    
    Returns:
    --------
    callbacks : list
        List of Keras callbacks
    """
    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=f'../results/models/{model_name}_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callback_list


if __name__ == "__main__":
    print("EEG Classification Models Module")
    print("=" * 50)
    
    # Test model creation
    print("\n1. Testing Traditional ML Models...")
    rf = TraditionalMLModels.get_random_forest()
    print(f" Random Forest: {rf}")
    
    xgb = TraditionalMLModels.get_xgboost()
    print(f" XGBoost: {xgb}")
    
    print("\n2. Testing Deep Learning Models...")
    
    # Test 1D CNN
    input_shape = (694, 1) 
    cnn_model = DeepLearningModels.build_1d_cnn(input_shape)
    print(f" 1D CNN: {cnn_model.count_params():,} parameters")
    
    # Test LSTM
    lstm_model = DeepLearningModels.build_lstm(input_shape)
    print(f" LSTM: {lstm_model.count_params():,} parameters")
    
    # Test CNN-LSTM
    hybrid_model = DeepLearningModels.build_cnn_lstm_hybrid(input_shape)
    print(f" CNN-LSTM: {hybrid_model.count_params():,} parameters")
  