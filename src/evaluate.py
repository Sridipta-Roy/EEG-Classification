"""
Model Evaluation and Visualization Module
==========================================

This module provides comprehensive evaluation for the models developed

Features:
- Detailed performance metrics
- Error analysis
- Cross-validation
- Statistical significance testing

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, cross_validate
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Compute all evaluation metrics
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_proba : np.ndarray, optional
            Prediction probabilities
        
        Returns:
        --------
        metrics : dict
            Dictionary of all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Derived metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        
        # Probabilistic metrics
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        return metrics
    
    @staticmethod
    def cross_validate_model(model, X: np.ndarray, y: np.ndarray,
                           cv: int = 5) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation
        
        Parameters:
        -----------
        model : sklearn model
            Model to evaluate
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        cv : int
            Number of folds
        
        Returns:
        --------
        cv_results : dict
            Cross-validation results
        """
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        return cv_results
    
    @staticmethod
    def print_classification_report(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   target_names: List[str] = None) -> None:
        """
        Print detailed classification report
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        target_names : list, optional
            Class names
        """
        if target_names is None:
            target_names = ['Normal', 'Seizure']
        
        print("\nDetailed Classification Report:")
        print("=" * 60)
        print(classification_report(y_true, y_pred, target_names=target_names))


class ErrorAnalyzer:
    """Analyze model errors in detail"""
    
    @staticmethod
    def analyze_errors(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      y_proba: np.ndarray,
                      X: np.ndarray = None) -> pd.DataFrame:
        """
        Analyze prediction errors
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predictions
        y_proba : np.ndarray
            Prediction probabilities
        X : np.ndarray, optional
            Feature matrix
        
        Returns:
        --------
        error_df : pd.DataFrame
            DataFrame with error analysis
        """
        # Create error mask
        errors = y_true != y_pred
        
        # Analyze by error type
        false_positives = (y_true == 0) & (y_pred == 1)
        false_negatives = (y_true == 1) & (y_pred == 0)
        
        error_analysis = {
            'Total Errors': int(np.sum(errors)),
            'Error Rate': float(np.mean(errors)),
            'False Positives': int(np.sum(false_positives)),
            'False Negatives': int(np.sum(false_negatives)),
            'FP Rate': float(np.mean(false_positives)),
            'FN Rate': float(np.mean(false_negatives))
        }
        
        # Confidence analysis
        error_proba = y_proba[errors]
        correct_proba = y_proba[~errors]
        
        error_analysis['Avg Error Confidence'] = float(np.mean(np.abs(error_proba - 0.5)))
        error_analysis['Avg Correct Confidence'] = float(np.mean(np.abs(correct_proba - 0.5)))
        
        return pd.DataFrame([error_analysis]).T
    
    @staticmethod
    def get_misclassified_indices(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  error_type: str = 'all') -> np.ndarray:
        """
        Get indices of misclassified samples
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predictions
        error_type : str
            'all', 'fp' (false positive), or 'fn' (false negative)
        
        Returns:
        --------
        indices : np.ndarray
            Indices of misclassified samples
        """
        if error_type == 'all':
            return np.where(y_true != y_pred)[0]
        elif error_type == 'fp':
            return np.where((y_true == 0) & (y_pred == 1))[0]
        elif error_type == 'fn':
            return np.where((y_true == 1) & (y_pred == 0))[0]
        else:
            raise ValueError("error_type must be 'all', 'fp', or 'fn'")


class StatisticalTests:
    """Statistical significance testing"""
    
    @staticmethod
    def mcnemar_test(y_true: np.ndarray,
                    y_pred1: np.ndarray,
                    y_pred2: np.ndarray) -> Tuple[float, float]:
        """
        McNemar's test for comparing two models
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred1 : np.ndarray
            Predictions from model 1
        y_pred2 : np.ndarray
            Predictions from model 2
        
        Returns:
        --------
        statistic : float
            Test statistic
        p_value : float
            P-value
        """
        # Create contingency table
        correct1 = (y_true == y_pred1)
        correct2 = (y_true == y_pred2)
        
        # Both correct, both wrong
        both_correct = np.sum(correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        
        # One correct, one wrong
        only1_correct = np.sum(correct1 & ~correct2)
        only2_correct = np.sum(~correct1 & correct2)
        
        # McNemar's test
        statistic = ((only1_correct - only2_correct) ** 2) / (only1_correct + only2_correct)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        return statistic, p_value
    
    @staticmethod
    def paired_ttest(scores1: np.ndarray,
                    scores2: np.ndarray) -> Tuple[float, float]:
        """
        Paired t-test for comparing cross-validation scores
        
        Parameters:
        -----------
        scores1 : np.ndarray
            Scores from model 1
        scores2 : np.ndarray
            Scores from model 2
        
        Returns:
        --------
        statistic : float
            T-statistic
        p_value : float
            P-value
        """
        statistic, p_value = stats.ttest_rel(scores1, scores2)
        return statistic, p_value


class Visualizer:
    """Visualizations"""
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             class_names: List[str] = None,
                             normalize: bool = False,
                             title: str = 'Confusion Matrix',
                             figsize: Tuple = (8, 6),
                             cmap: str = 'Blues') -> plt.Figure:
        """
        Plot confusion matrix        
        """
        if class_names is None:
            class_names = ['Normal', 'Seizure']
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap=cmap, ax=ax, cbar_kws={'label': 'Count'},
                   xticklabels=class_names, yticklabels=class_names,
                   square=True, linewidths=2)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add metrics text
        tn, fp, fn, tp = cm.ravel() if not normalize else (cm * len(y_true)).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
        ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray,
                      y_proba: np.ndarray,
                      title: str = 'ROC Curve',
                      figsize: Tuple = (8, 6)) -> plt.Figure:
        """
        Plot ROC curve
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray,
                                    y_proba: np.ndarray,
                                    title: str = 'Precision-Recall Curve',
                                    figsize: Tuple = (8, 6)) -> plt.Figure:
        """
        Plot precision-recall curve
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(recall, precision, linewidth=2, label='PR Curve')
        ax.axhline(y=np.sum(y_true) / len(y_true), color='r', 
                  linestyle='--', label='Baseline (Random)')
        
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_confidence_distribution(y_true: np.ndarray,
                                    y_proba: np.ndarray,
                                    title: str = 'Prediction Confidence Distribution',
                                    figsize: Tuple = (10, 6)) -> plt.Figure:
        """
        Plot distribution of prediction confidence
        """
        # Separate correct and incorrect predictions
        y_pred = (y_proba > 0.5).astype(int)
        correct = y_true == y_pred
        
        correct_proba = y_proba[correct]
        incorrect_proba = y_proba[~correct]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Correct predictions
        ax1.hist(correct_proba, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Probability', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'Correct Predictions (n={len(correct_proba)})',
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Incorrect predictions
        ax2.hist(incorrect_proba, bins=30, alpha=0.7, color='red', edgecolor='black')
        ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Probability', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title(f'Incorrect Predictions (n={len(incorrect_proba)})',
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

