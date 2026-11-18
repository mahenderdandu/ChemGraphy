"""
Visualization utilities for molecules, attention maps, and results
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64


def plot_training_curves(
    history: Dict[str, List[Tuple[int, float]]],
    metrics: List[str],
    title: str = "Training Curves",
    save_path: Optional[str] = None
) -> None:
    """
    Plot training curves
    
    Args:
        history: Dictionary of metric histories
        metrics: List of metric names to plot
        title: Plot title
        save_path: Path to save figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric in history:
            steps, values = zip(*history[metric])
            ax.plot(steps, values, marker='o', markersize=3)
            ax.set_xlabel('Step')
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Ground Truth",
    xlabel: str = "Ground Truth",
    ylabel: str = "Predicted",
    save_path: Optional[str] = None
) -> None:
    """
    Plot prediction scatter plot
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Identity line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    # Calculate metrics
    from scipy import stats
    r, p = stats.pearsonr(y_true, y_pred)
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Add text box with metrics
    textstr = f'Pearson r: {r:.3f}\nRMSE: {rmse:.3f}\nn: {len(y_true)}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        save_path: Path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curve
    
    Args:
        y_true: Ground truth binary labels
        y_score: Predicted scores
        title: Plot title
        save_path: Path to save figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Attention Heatmap",
    save_path: Optional[str] = None
) -> None:
    """
    Plot attention weight heatmap
    
    Args:
        attention_weights: Attention weight matrix (2D)
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(attention_weights, cmap='viridis', ax=ax,
                xticklabels=x_labels, yticklabels=y_labels,
                cbar_kws={'label': 'Attention Weight'})
    
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_molecule_2d(smiles: str, size: Tuple[int, int] = (300, 300)) -> Optional[str]:
    """
    Generate 2D molecule image as base64 string
    
    Args:
        smiles: SMILES string
        size: Image size
    
    Returns:
        Base64 encoded image string or None
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        img = Draw.MolToImage(mol, size=size)
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        print(f"Error generating molecule image: {e}")
        return None


def plot_molecule_3d_html(smiles: str) -> Optional[str]:
    """
    Generate 3D molecule visualization as HTML
    
    Args:
        smiles: SMILES string
    
    Returns:
        HTML string for 3D visualization or None
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import py3Dmol
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Convert to mol block
        mol_block = Chem.MolToMolBlock(mol)
        
        # Create 3D view
        view = py3Dmol.view(width=400, height=400)
        view.addModel(mol_block, 'mol')
        view.setStyle({'stick': {}, 'sphere': {'scale': 0.3}})
        view.zoomTo()
        
        return view._make_html()
        
    except Exception as e:
        print(f"Error generating 3D molecule: {e}")
        return None


def create_summary_report(
    metrics: Dict[str, float],
    plots: Dict[str, str],
    save_path: str
) -> None:
    """
    Create HTML summary report
    
    Args:
        metrics: Dictionary of metrics
        plots: Dictionary of plot paths
        save_path: Path to save HTML report
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Helios Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .metric {{ margin: 10px 0; }}
            .plot {{ margin: 20px 0; }}
            img {{ max-width: 800px; }}
        </style>
    </head>
    <body>
        <h1>Helios Training Report</h1>
        
        <h2>Metrics</h2>
        <div class="metrics">
    """
    
    for name, value in metrics.items():
        val = value.item() if hasattr(value, 'item') else value
        html += f'<div class="metric"><strong>{name}:</strong> {float(val):.4f}</div>\n'
    
    html += """
        </div>
        
        <h2>Plots</h2>
    """
    
    for name, path in plots.items():
        html += f"""
        <div class="plot">
            <h3>{name}</h3>
            <img src="{path}" alt="{name}">
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    with open(save_path, 'w') as f:
        f.write(html)
