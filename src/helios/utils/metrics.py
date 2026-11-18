"""
Evaluation metrics for model training and validation
"""

from typing import Dict, List, Optional
import numpy as np
from scipy import stats
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate regression metrics
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        prefix: Prefix for metric names (e.g., "train_", "val_")
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {f"{prefix}rmse": float('nan')}
    
    # RMSE
    mse = mean_squared_error(y_true, y_pred)
    metrics[f"{prefix}rmse"] = np.sqrt(mse)
    
    # MAE
    metrics[f"{prefix}mae"] = mean_absolute_error(y_true, y_pred)
    
    # R²
    metrics[f"{prefix}r2"] = r2_score(y_true, y_pred)
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    metrics[f"{prefix}pearson_r"] = pearson_r
    metrics[f"{prefix}pearson_p"] = pearson_p
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    metrics[f"{prefix}spearman_r"] = spearman_r
    metrics[f"{prefix}spearman_p"] = spearman_p
    
    return metrics


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC)
        prefix: Prefix for metric names
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Accuracy
    metrics[f"{prefix}accuracy"] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    metrics[f"{prefix}precision"] = precision
    metrics[f"{prefix}recall"] = recall
    metrics[f"{prefix}f1"] = f1
    
    # AUC metrics if probabilities provided
    if y_prob is not None:
        try:
            metrics[f"{prefix}roc_auc"] = roc_auc_score(y_true, y_prob)
            metrics[f"{prefix}pr_auc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            pass  # Not enough classes
    
    return metrics


def calculate_enrichment_factor(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fraction: float = 0.01
) -> float:
    """
    Calculate enrichment factor at given fraction
    
    Args:
        y_true: Ground truth binary labels
        y_score: Predicted scores
        fraction: Top fraction to consider (e.g., 0.01 for 1%)
    
    Returns:
        Enrichment factor
    """
    # Sort by score (descending)
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Calculate enrichment
    n_top = max(1, int(len(y_true) * fraction))
    n_actives_top = np.sum(y_true_sorted[:n_top])
    n_actives_total = np.sum(y_true)
    
    if n_actives_total == 0:
        return 0.0
    
    enrichment = (n_actives_top / n_top) / (n_actives_total / len(y_true))
    return enrichment


def calculate_bedroc_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    alpha: float = 20.0
) -> float:
    """
    Calculate BEDROC (Boltzmann-Enhanced Discrimination of ROC)
    
    Args:
        y_true: Ground truth binary labels
        y_score: Predicted scores
        alpha: Early recognition parameter
    
    Returns:
        BEDROC score
    """
    # Sort by score (descending)
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    n = len(y_true)
    n_actives = np.sum(y_true)
    
    if n_actives == 0 or n_actives == n:
        return 0.0
    
    # Calculate BEDROC
    sum_term = 0.0
    for i, label in enumerate(y_true_sorted):
        if label == 1:
            sum_term += np.exp(-alpha * (i + 1) / n)
    
    ra = n_actives / n
    random_sum = ra * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)
    
    bedroc = (sum_term / n_actives - random_sum) / (1 - random_sum)
    
    return bedroc


def calculate_molecular_metrics(
    smiles_true: List[str],
    smiles_pred: List[str]
) -> Dict[str, float]:
    """
    Calculate molecular generation metrics
    
    Args:
        smiles_true: Ground truth SMILES
        smiles_pred: Generated SMILES
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Exact match
    n_exact = sum(s1 == s2 for s1, s2 in zip(smiles_true, smiles_pred))
    metrics['exact_match'] = n_exact / len(smiles_true)
    
    return metrics


class MetricTracker:
    """Track metrics across training"""
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {}
    
    def update(self, metrics: Dict[str, float], step: int):
        """Update metrics"""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((step, value))
    
    def get_best(self, metric_name: str, mode: str = "min") -> tuple:
        """Get best value for a metric"""
        if metric_name not in self.history:
            return None, None
        
        values = self.history[metric_name]
        if mode == "min":
            best_idx = np.argmin([v[1] for v in values])
        else:
            best_idx = np.argmax([v[1] for v in values])
        
        return values[best_idx]
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric"""
        if metric_name not in self.history or len(self.history[metric_name]) == 0:
            return None
        return self.history[metric_name][-1][1]
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}
        for key, values in self.history.items():
            vals = [v[1] for v in values]
            summary[key] = {
                'mean': np.mean(vals),
                'std': np.std(vals),
                'min': np.min(vals),
                'max': np.max(vals),
                'latest': vals[-1],
            }
        return summary
