"""
Training script for Graphormer on MoleculeNet datasets
No preprocessing required - uses public datasets
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import logging
from pathlib import Path
import os
import yaml
from tqdm import tqdm
import wandb
from typing import Dict

from helios.models.graphormer.model import load_graphormer_from_pretrained
from helios.data.moleculenet_loader import get_moleculenet_loaders
from helios.utils.metrics import calculate_regression_metrics
from helios.utils.visualization import plot_training_curves

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphormerTrainer:
    """Trainer for Graphormer models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Loss function
        self.criterion = nn.MSELoss() if config['task_type'] == 'regression' else nn.BCEWithLogitsLoss()
        
        # Tracking
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Checkpoints
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized on {device}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch['node_features'] = batch['node_features'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            _, predictions = self.model(batch)
            predictions = predictions.squeeze(-1)  # Remove last dim
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, dataloader, split='val') -> Dict:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
            # Move to device
            batch['node_features'] = batch['node_features'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            _, predictions = self.model(batch)
            predictions = predictions.squeeze(-1)  # Remove last dim
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            total_loss += loss.item()
            
            # Store predictions
            all_preds.append(predictions.cpu())
            all_targets.append(targets.cpu())
        
        # Combine all predictions
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        # Compute metrics
        avg_loss = total_loss / len(dataloader)
        metrics = calculate_regression_metrics(all_targets, all_preds, prefix=f"{split}_")
        metrics[f'{split}_loss'] = avg_loss
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def train(self, num_epochs: int):
        """Full training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.evaluate(self.val_loader, 'val')
            val_loss = val_metrics['val_loss']
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Log metrics
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                       f"val_rmse={val_metrics['val_rmse']:.4f}, val_r2={val_metrics['val_r2']:.4f}")
            
            # Wandb logging
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    **val_metrics
                })
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % self.config['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        # Final evaluation on test set
        logger.info("\n" + "="*60)
        logger.info("Evaluating on test set...")
        test_metrics = self.evaluate(self.test_loader, 'test')
        logger.info(f"Test results: rmse={test_metrics['test_rmse']:.4f}, r2={test_metrics['test_r2']:.4f}")
        
        # Plot training curves
        if self.config.get('save_plots', True):
            plot_path = self.checkpoint_dir / 'training_curves.png'
            plot_training_curves(
                self.train_losses,
                self.val_losses,
                title='Graphormer Training',
                save_path=str(plot_path)
            )
            logger.info(f"Saved training curves to {plot_path}")
        
        return test_metrics


def prune_non_best_checkpoints(checkpoint_dir: str):
    """Delete non-best epoch checkpoints and keep only best_model.pt"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return

    removed = 0
    for file_path in checkpoint_path.glob('checkpoint_epoch_*.pt'):
        try:
            os.remove(file_path)
            removed += 1
        except OSError as e:
            logger.warning(f"Could not remove {file_path}: {e}")

    logger.info(f"Pruned {removed} non-best checkpoints in {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Graphormer on MoleculeNet')
    parser.add_argument('--dataset', type=str, default='ESOL',
                       choices=['ESOL', 'FreeSolv', 'Lipophilicity', 'BACE'],
                       help='Dataset name')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints/graphormer',
                       help='Checkpoint directory')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--keep-best-only', action='store_true',
                       help='Delete checkpoint_epoch_*.pt files after training')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)['graphormer']
    else:
        config = {
            'learning_rate': args.lr,
            'weight_decay': 1e-5,
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'task_type': 'regression',
            'checkpoint_dir': args.checkpoint_dir,
            'save_every': 10,
            'use_wandb': args.use_wandb,
            'save_plots': True
        }
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project='helios-graphormer',
            config=config,
            name=f'{args.dataset}_graphormer'
        )
    
    logger.info(f"Training Graphormer on {args.dataset} dataset")
    logger.info(f"Device: {args.device}")
    logger.info(f"Config: {config}")
    
    # Load data
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = get_moleculenet_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=0,  # Set to 0 for Mac compatibility
        root_dir='data/moleculenet'
    )
    
    # Load model
    logger.info("Loading Graphormer model...")
    model = load_graphormer_from_pretrained(
        output_dim=1,
        task_type='regression',
        device=args.device
    )
    
    # Create trainer
    trainer = GraphormerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=args.device
    )
    
    # Train
    test_metrics = trainer.train(config['num_epochs'])
    
    logger.info("\n" + "="*60)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Test RMSE: {test_metrics['test_rmse']:.4f}")
    logger.info(f"Test R²: {test_metrics['test_r2']:.4f}")

    if args.keep_best_only:
        prune_non_best_checkpoints(config['checkpoint_dir'])
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
