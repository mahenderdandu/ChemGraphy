"""
Main data pipeline script
Orchestrates downloading, cleaning, and preparing all data for Helios
"""

import logging
import argparse
from pathlib import Path
import yaml
import pandas as pd
from typing import Dict, Optional

from helios.data.downloaders import download_chembl_data, download_pubchem_compounds
from helios.data.processors.chembl_processor import process_chembl_data
from helios.data.processors.molecule_cleaner import clean_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Main data pipeline for Helios"""
    
    def __init__(self, config_path: str, limit: Optional[int] = None, chunk_size: int = 10000):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.limit = limit
        self.chunk_size = chunk_size
        self.output_dir = Path(self.data_config['output']['final_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.limit:
            logger.info(f"Running in TEST MODE with limit={self.limit} records")
        logger.info(f"Processing in chunks of {self.chunk_size:,} records")
    
    def step1_download_data(self):
        """Step 1: Download data from sources"""
        logger.info("=" * 60)
        logger.info("STEP 1: Downloading Data")
        logger.info("=" * 60)
        
        raw_dir = Path(self.data_config['output']['raw_dir'])
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Download ChEMBL
        if self.data_config['sources']['chembl']['download']:
            logger.info("\n--- Downloading ChEMBL ---")
            chembl_dir = raw_dir / 'chembl'
            chembl_files = download_chembl_data(
                str(chembl_dir),
                version=self.data_config['sources']['chembl']['version'],
                download_sqlite=True,
                download_sdf=False
            )
            logger.info(f"ChEMBL SQLite: {chembl_files['sqlite']}")
        
        # Download PubChem (optional)
        if self.data_config['sources']['pubchem'].get('download', False):
            logger.info("\n--- Downloading PubChem ---")
            pubchem_dir = raw_dir / 'pubchem'
            pubchem_file = download_pubchem_compounds(
                str(pubchem_dir),
                query="drug",
                max_compounds=10000
            )
            logger.info(f"PubChem compounds: {pubchem_file}")
    
    def step2_process_chembl(self):
        """Step 2: Process ChEMBL database"""
        logger.info("=" * 60)
        logger.info("STEP 2: Processing ChEMBL")
        logger.info("=" * 60)
        
        raw_dir = Path(self.data_config['output']['raw_dir'])
        processed_dir = Path(self.data_config['output']['processed_dir'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Find ChEMBL database (search recursively)
        chembl_dir = raw_dir / 'chembl'
        db_files = list(chembl_dir.glob('**/*.db'))
        if not db_files:
            raise FileNotFoundError(f"No .db file found in {chembl_dir}")
        db_file = db_files[0]
        
        # Process activities
        output_file = processed_dir / 'chembl_raw.csv'
        df = process_chembl_data(
            str(db_file),
            str(output_file),
            activity_types=self.data_config['labels']['activity_types'],
            min_confidence=self.data_config['labels']['filters']['min_confidence'],
            limit=self.limit,
            chunk_size=self.chunk_size
        )
        
        logger.info(f"Processed {len(df)} ChEMBL records")
    
    def step3_clean_data(self):
        """Step 3: Clean and standardize molecules"""
        logger.info("=" * 60)
        logger.info("STEP 3: Cleaning Data")
        logger.info("=" * 60)
        
        processed_dir = Path(self.data_config['output']['processed_dir'])
        
        # Clean ChEMBL
        logger.info("\n--- Cleaning ChEMBL ---")
        chembl_input = processed_dir / 'chembl_raw.csv'
        chembl_output = self.output_dir / self.data_config['output']['files']['chembl_clean']
        
        df_chembl = clean_dataset(
            str(chembl_input),
            str(chembl_output),
            smiles_column='canonical_smiles',
            molecule_config=self.data_config['cleaning'],
            activity_config=self.data_config['labels'],
            has_activity_data=True
        )
        
        logger.info(f"Cleaned ChEMBL: {len(df_chembl)} records")
    
    def step4_prepare_training_data(self):
        """Step 4: Prepare final training datasets"""
        logger.info("=" * 60)
        logger.info("STEP 4: Preparing Training Data")
        logger.info("=" * 60)
        
        # Load cleaned data
        chembl_file = self.output_dir / self.data_config['output']['files']['chembl_clean']
        df = pd.read_csv(chembl_file)
        
        # Create master training file
        master_file = self.output_dir / self.data_config['output']['files']['master_training']
        df.to_csv(master_file, index=False)
        logger.info(f"Saved master training data: {len(df)} records")
        
        # Split into train/val/test
        from sklearn.model_selection import train_test_split
        
        split_config = self.data_config['split']
        
        # First split: train + val, test
        train_val, test = train_test_split(
            df,
            test_size=split_config['test'],
            random_state=split_config['random_seed'],
            stratify=df['target_id'] if 'target_id' in df.columns else None
        )
        
        # Second split: train, val
        val_ratio = split_config['val'] / (split_config['train'] + split_config['val'])
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=split_config['random_seed'],
            stratify=train_val['target_id'] if 'target_id' in train_val.columns else None
        )
        
        # Save splits
        train.to_csv(self.output_dir / self.data_config['output']['files']['train_split'], index=False)
        val.to_csv(self.output_dir / self.data_config['output']['files']['val_split'], index=False)
        test.to_csv(self.output_dir / self.data_config['output']['files']['test_split'], index=False)
        
        logger.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    
    def run_full_pipeline(self):
        """Run complete data pipeline"""
        logger.info("\n" + "=" * 60)
        logger.info("HELIOS DATA PIPELINE")
        logger.info("=" * 60 + "\n")
        
        try:
            self.step1_download_data()
            self.step2_process_chembl()
            self.step3_clean_data()
            self.step4_prepare_training_data()
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ DATA PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    parser = argparse.ArgumentParser(description='Helios Data Pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--step',
        type=str,
        choices=['download', 'process', 'clean', 'prepare', 'all'],
        default='all',
        help='Pipeline step to run'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of records for testing (e.g., --limit 10000)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000,
        help='Process data in chunks of this size (default: 10000, smaller=less RAM)'
    )
    
    args = parser.parse_args()
    
    pipeline = DataPipeline(args.config, limit=args.limit, chunk_size=args.chunk_size)
    
    if args.step == 'download':
        pipeline.step1_download_data()
    elif args.step == 'process':
        pipeline.step2_process_chembl()
    elif args.step == 'clean':
        pipeline.step3_clean_data()
    elif args.step == 'prepare':
        pipeline.step4_prepare_training_data()
    else:
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
