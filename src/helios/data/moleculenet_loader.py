"""
MoleculeNet dataset loader
Loads public molecular datasets without preprocessing
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from rdkit import Chem

logger = logging.getLogger(__name__)


def smiles_to_graph(smiles: str) -> Dict[str, torch.Tensor]:
    """
    Convert SMILES to graph representation for Graphormer
    
    Args:
        smiles: SMILES string
    
    Returns:
        Dictionary with graph tensors
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic())
        ]
        atom_features.append(features)
    
    # Edge features
    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])  # Undirected
        
        bond_features = [
            int(bond.GetBondType()),
            int(bond.GetIsConjugated())
        ]
        edge_features.append(bond_features)
        edge_features.append(bond_features)  # Duplicate for undirected
    
    # Convert to tensors
    num_atoms = len(atom_features)
    
    return {
        'node_features': torch.tensor(atom_features, dtype=torch.float32),
        'edge_index': torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros((2, 0), dtype=torch.long),
        'edge_features': torch.tensor(edge_features, dtype=torch.float32) if edge_features else torch.zeros((0, 2), dtype=torch.float32),
        'num_nodes': num_atoms,
        'smiles': smiles
    }


class MoleculeNetDataset(Dataset):
    """
    Dataset for MoleculeNet benchmarks
    Supports ESOL, FreeSolv, Lipophilicity, etc.
    """
    
    # Dataset information
    DATASETS = {
        'ESOL': {
            'task': 'regression',
            'metric': 'rmse',
            'target': 'measured log solubility in mols per litre',
            'size': 1128,
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
        },
        'FREESOLV': {
            'task': 'regression',
            'metric': 'rmse',
            'target': 'expt',
            'size': 642,
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv'
        },
        'LIPOPHILICITY': {
            'task': 'regression',
            'metric': 'rmse',
            'target': 'exp',
            'size': 4200,
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv'
        },
        'BACE': {
            'task': 'classification',
            'metric': 'roc_auc',
            'target': 'Class',
            'size': 1513,
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv'
        },
    }
    
    def __init__(
        self,
        dataset_name: str = 'ESOL',
        split: str = 'train',
        root_dir: str = 'data/moleculenet',
        transform=None
    ):
        """
        Args:
            dataset_name: Name of dataset (ESOL, FreeSolv, etc.)
            split: train/val/test
            root_dir: Root directory for datasets
            transform: Optional transform function
        """
        self.dataset_name = dataset_name.upper()
        self.split = split
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        if self.dataset_name not in self.DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {list(self.DATASETS.keys())}")
        
        self.dataset_info = self.DATASETS[self.dataset_name]
        self.task_type = self.dataset_info['task']
        
        # Load data
        self.data = self._load_data()
        
        logger.info(f"Loaded {dataset_name} {split} split: {len(self.data)} samples")
    
    def _load_data(self) -> pd.DataFrame:
        """Load and split dataset"""
        cache_path = self.root_dir / f"{self.dataset_name.lower()}.csv"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        # Download if not cached
        if not cache_path.exists():
            logger.info(f"Downloading {self.dataset_name} dataset...")
            import requests
            response = requests.get(self.dataset_info['url'])
            with open(cache_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Saved to {cache_path}")
        
        # Load data
        df = pd.read_csv(cache_path)
        
        # Get SMILES and target columns
        smiles_col = self._find_smiles_column(df)
        target_col = self.dataset_info['target']
        
        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column not found in dataset")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Clean data
        df = df[[smiles_col, target_col]].dropna()
        df.columns = ['smiles', 'target']
        
        # Split data (80/10/10)
        n = len(df)
        train_idx = int(0.8 * n)
        val_idx = int(0.9 * n)
        
        if self.split == 'train':
            df = df.iloc[:train_idx]
        elif self.split == 'val':
            df = df.iloc[train_idx:val_idx]
        else:  # test
            df = df.iloc[val_idx:]
        
        return df.reset_index(drop=True)
    
    def _find_smiles_column(self, df: pd.DataFrame) -> str:
        """Find SMILES column name"""
        possible_names = ['smiles', 'SMILES', 'smile', 'mol', 'molecule']
        for name in possible_names:
            if name in df.columns:
                return name
        # Default to first column
        return df.columns[0]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get item by index
        
        Returns:
            Dictionary with smiles, target, and optionally graph data
        """
        row = self.data.iloc[idx]
        
        item = {
            'smiles': row['smiles'],
            'target': torch.tensor(row['target'], dtype=torch.float32),
            'idx': idx
        }
        
        if self.transform:
            item = self.transform(item)
        
        return item


def collate_molecules(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching molecules
    
    Args:
        batch: List of data items
    
    Returns:
        Batched data dictionary
    """
    from helios.models.graphormer.model import smiles_to_graph
    
    graphs = []
    targets = []
    smiles_list = []
    
    for item in batch:
        try:
            graph = smiles_to_graph(item['smiles'])
            graphs.append(graph)
            targets.append(item['target'].item())  # Extract scalar value
            smiles_list.append(item['smiles'])
        except Exception as e:
            logger.warning(f"Failed to process SMILES {item['smiles']}: {e}")
            continue
    
    if not graphs:
        raise ValueError("No valid molecules in batch")
    
    # Batch graphs (simple version - pad to max size)
    max_nodes = max(g['num_nodes'] for g in graphs)
    
    batched_node_features = []
    for g in graphs:
        node_feat = g['node_features']
        # Pad to max_nodes
        if node_feat.size(0) < max_nodes:
            padding = torch.zeros(max_nodes - node_feat.size(0), node_feat.size(1))
            node_feat = torch.cat([node_feat, padding], dim=0)
        batched_node_features.append(node_feat)
    
    return {
        'node_features': torch.stack(batched_node_features),
        'targets': torch.tensor(targets, dtype=torch.float32),
        'smiles': smiles_list
    }


def get_moleculenet_loaders(
    dataset_name: str = 'ESOL',
    batch_size: int = 32,
    num_workers: int = 4,
    root_dir: str = 'data/moleculenet'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train/val/test dataloaders for a MoleculeNet dataset
    
    Args:
        dataset_name: Dataset name
        batch_size: Batch size
        num_workers: Number of workers for data loading
        root_dir: Root directory for datasets
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = MoleculeNetDataset(dataset_name, 'train', root_dir)
    val_dataset = MoleculeNetDataset(dataset_name, 'val', root_dir)
    test_dataset = MoleculeNetDataset(dataset_name, 'test', root_dir)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_molecules
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_molecules
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_molecules
    )
    
    logger.info(f"Created dataloaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    logging.basicConfig(level=logging.INFO)
    
    print("Testing MoleculeNet loader...")
    
    # Load ESOL dataset
    train_loader, val_loader, test_loader = get_moleculenet_loaders('ESOL', batch_size=8)
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Node features shape: {batch['node_features'].shape}")
    print(f"Targets shape: {batch['targets'].shape}")
    print(f"SMILES samples: {batch['smiles'][:3]}")
