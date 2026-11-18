"""
Molecule cleaner and standardizer
Handles SMILES cleaning, standardization, and filtering
"""

import logging
from typing import Optional, Dict, Tuple
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from helios.utils.chemistry import (
    get_molecule_processor,
    convert_to_pactivity,
    classify_activity
)

logger = logging.getLogger(__name__)


class MoleculeCleaner:
    """Clean and standardize molecules"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.processor = get_molecule_processor()
        
        # Default configuration
        self.config = {
            'canonicalize_smiles': True,
            'remove_salts': True,
            'neutralize_charges': True,
            'remove_mixtures': True,
            'deduplicate': True,
            'min_mw': 0,
            'max_mw': 800,
            'min_heavy_atoms': 3,
            'max_heavy_atoms': 50,
            'allow_metals': False,
        }
        
        if config:
            self.config.update(config)
    
    def clean_smiles(self, smiles: str) -> Optional[str]:
        """
        Clean and standardize a SMILES string
        
        Args:
            smiles: Input SMILES
        
        Returns:
            Cleaned SMILES or None if invalid
        """
        if pd.isna(smiles) or not smiles:
            return None
        
        # Standardize
        clean_smiles = self.processor.standardize_molecule(smiles)
        
        if clean_smiles is None:
            return None
        
        # Validate
        is_valid = self.processor.is_valid_molecule(
            clean_smiles,
            min_mw=self.config['min_mw'],
            max_mw=self.config['max_mw'],
            min_heavy_atoms=self.config['min_heavy_atoms'],
            max_heavy_atoms=self.config['max_heavy_atoms'],
            allow_metals=self.config['allow_metals']
        )
        
        if not is_valid:
            return None
        
        return clean_smiles
    
    def clean_dataframe(
        self,
        df: pd.DataFrame,
        smiles_column: str = 'canonical_smiles',
        add_descriptors: bool = True,
        add_inchi: bool = True
    ) -> pd.DataFrame:
        """
        Clean molecules in a DataFrame
        
        Args:
            df: Input DataFrame
            smiles_column: Name of SMILES column
            add_descriptors: Add molecular descriptors
            add_inchi: Add InChI and InChI Key
        
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning {len(df)} molecules...")
        
        # Clean SMILES
        tqdm.pandas(desc="Cleaning SMILES")
        df['clean_smiles'] = df[smiles_column].progress_apply(self.clean_smiles)
        
        # Remove invalid molecules
        n_invalid = df['clean_smiles'].isna().sum()
        df = df[df['clean_smiles'].notna()].copy()
        logger.info(f"Removed {n_invalid} invalid molecules")
        
        # Deduplicate
        if self.config['deduplicate']:
            n_before = len(df)
            df = df.drop_duplicates(subset=['clean_smiles'])
            n_removed = n_before - len(df)
            logger.info(f"Removed {n_removed} duplicate molecules")
        
        # Add descriptors
        if add_descriptors:
            logger.info("Calculating molecular descriptors...")
            tqdm.pandas(desc="Descriptors")
            descriptors = df['clean_smiles'].progress_apply(
                self.processor.calculate_descriptors
            )
            
            # Convert to DataFrame and join
            desc_df = pd.DataFrame(descriptors.tolist())
            df = pd.concat([df.reset_index(drop=True), desc_df], axis=1)
        
        # Add InChI
        if add_inchi:
            logger.info("Generating InChI keys...")
            tqdm.pandas(desc="InChI")
            df['inchi_key'] = df['clean_smiles'].progress_apply(
                self.processor.smiles_to_inchi_key
            )
        
        logger.info(f"Cleaned dataset: {len(df)} molecules")
        
        return df


class ActivityDataCleaner:
    """Clean and process activity data"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            'activity_types': ['IC50', 'Ki', 'Kd', 'EC50', 'AC50'],
            'convert_to_log': True,
            'remove_missing': True,
            'remove_inconsistent': True,
            'active_threshold': 1000,  # nM
            'inactive_threshold': 10000,  # nM
        }
        
        if config:
            self.config.update(config)
    
    def clean_activities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean activity data
        
        Args:
            df: Input DataFrame with activity data
        
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning {len(df)} activity records...")
        
        # Filter by activity type
        if 'standard_type' in df.columns:
            df = df[df['standard_type'].isin(self.config['activity_types'])].copy()
            logger.info(f"After activity type filter: {len(df)} records")
        
        # Remove missing values
        if self.config['remove_missing']:
            required_cols = ['standard_value', 'standard_units']
            for col in required_cols:
                if col in df.columns:
                    n_before = len(df)
                    df = df[df[col].notna()]
                    logger.info(f"Removed {n_before - len(df)} records with missing {col}")
        
        # Convert to pActivity
        if self.config['convert_to_log'] and 'standard_value' in df.columns:
            logger.info("Converting to pActivity...")
            df['pactivity'] = df.apply(
                lambda row: convert_to_pactivity(
                    row['standard_value'],
                    row.get('standard_units', 'nM')
                ),
                axis=1
            )
            
            # Remove invalid conversions
            n_before = len(df)
            df = df[df['pactivity'].notna()]
            logger.info(f"Removed {n_before - len(df)} records with invalid pActivity")
        
        # Add activity classification
        if 'standard_value' in df.columns:
            logger.info("Classifying activity...")
            df['activity_class'] = df.apply(
                lambda row: classify_activity(
                    row['standard_value'],
                    row.get('standard_units', 'nM'),
                    self.config['active_threshold'],
                    self.config['inactive_threshold']
                ),
                axis=1
            )
        
        # Remove inconsistent data (optional)
        if self.config['remove_inconsistent']:
            # Group by compound-target pairs and check for inconsistencies
            if all(col in df.columns for col in ['clean_smiles', 'target_id', 'activity_class']):
                logger.info("Checking for inconsistent activity annotations...")
                
                # Group and check variance
                grouped = df.groupby(['clean_smiles', 'target_id'])['activity_class']
                inconsistent = grouped.apply(lambda x: x.nunique() > 1 if x.notna().any() else False)
                inconsistent_pairs = inconsistent[inconsistent].index
                
                n_before = len(df)
                df = df[~df.set_index(['clean_smiles', 'target_id']).index.isin(inconsistent_pairs)]
                logger.info(f"Removed {n_before - len(df)} inconsistent records")
        
        logger.info(f"Cleaned dataset: {len(df)} activity records")
        
        return df


def clean_dataset(
    input_path: str,
    output_path: str,
    smiles_column: str = 'canonical_smiles',
    molecule_config: Optional[Dict] = None,
    activity_config: Optional[Dict] = None,
    has_activity_data: bool = True
) -> pd.DataFrame:
    """
    Complete dataset cleaning pipeline
    
    Args:
        input_path: Input CSV path
        output_path: Output CSV path
        smiles_column: Name of SMILES column
        molecule_config: Molecule cleaning configuration
        activity_config: Activity cleaning configuration
        has_activity_data: Whether dataset has activity data
    
    Returns:
        Cleaned DataFrame
    """
    # Load data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records")
    
    # Clean molecules
    mol_cleaner = MoleculeCleaner(molecule_config)
    df = mol_cleaner.clean_dataframe(df, smiles_column=smiles_column)
    
    # Clean activities if present
    if has_activity_data:
        act_cleaner = ActivityDataCleaner(activity_config)
        df = act_cleaner.clean_activities(df)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean molecular data')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file')
    parser.add_argument('--smiles-column', type=str, default='canonical_smiles',
                       help='Name of SMILES column')
    parser.add_argument('--no-activity', action='store_true',
                       help='Dataset does not have activity data')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    df = clean_dataset(
        args.input,
        args.output,
        smiles_column=args.smiles_column,
        has_activity_data=not args.no_activity
    )
    
    print(f"\n=== Cleaning Summary ===")
    print(f"Total records: {len(df)}")
    print(f"\nColumns: {', '.join(df.columns)}")
    print(f"\nSample data:")
    print(df.head())
