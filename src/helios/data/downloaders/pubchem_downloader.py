"""
PubChem data downloader
Downloads compound and bioassay data from PubChem
"""

import os
import logging
from typing import List, Optional, Dict
import requests
import time
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import pubchempy as pcp

logger = logging.getLogger(__name__)


class PubChemDownloader:
    """Download data from PubChem"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.delay = 0.2  # Rate limiting
    
    def search_compounds(
        self,
        query: str,
        max_results: int = 1000
    ) -> List[int]:
        """
        Search for compounds by name/query
        
        Args:
            query: Search query
            max_results: Maximum number of results
        
        Returns:
            List of compound IDs (CIDs)
        """
        try:
            time.sleep(self.delay)
            compounds = pcp.get_compounds(query, 'name', listkey_count=max_results)
            cids = [c.cid for c in compounds]
            logger.info(f"Found {len(cids)} compounds for query: {query}")
            return cids
        except Exception as e:
            logger.error(f"Error searching compounds: {e}")
            return []
    
    def get_compound_properties(
        self,
        cids: List[int],
        properties: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get properties for compound IDs
        
        Args:
            cids: List of compound IDs
            properties: List of properties to retrieve
        
        Returns:
            DataFrame with compound properties
        """
        if properties is None:
            properties = [
                'MolecularFormula',
                'MolecularWeight',
                'CanonicalSMILES',
                'IsomericSMILES',
                'InChI',
                'InChIKey',
                'IUPACName',
                'XLogP',
                'TPSA',
                'Complexity',
                'HBondDonorCount',
                'HBondAcceptorCount',
                'RotatableBondCount',
            ]
        
        results = []
        
        # Process in batches
        batch_size = 100
        for i in tqdm(range(0, len(cids), batch_size), desc="Fetching properties"):
            batch_cids = cids[i:i+batch_size]
            
            try:
                time.sleep(self.delay)
                
                # Build URL
                cid_str = ','.join(map(str, batch_cids))
                prop_str = ','.join(properties)
                url = f"{self.base_url}/compound/cid/{cid_str}/property/{prop_str}/JSON"
                
                response = requests.get(url)
                response.raise_for_status()
                
                data = response.json()
                if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                    results.extend(data['PropertyTable']['Properties'])
                
            except Exception as e:
                logger.warning(f"Error fetching batch {i}: {e}")
                continue
        
        df = pd.DataFrame(results)
        return df
    
    def get_compound_descriptions(self, cids: List[int]) -> Dict[int, str]:
        """
        Get compound descriptions/summaries
        
        Args:
            cids: List of compound IDs
        
        Returns:
            Dictionary mapping CID to description
        """
        descriptions = {}
        
        for cid in tqdm(cids, desc="Fetching descriptions"):
            try:
                time.sleep(self.delay)
                
                url = f"{self.base_url}/compound/cid/{cid}/description/JSON"
                response = requests.get(url)
                response.raise_for_status()
                
                data = response.json()
                if 'InformationList' in data and 'Information' in data['InformationList']:
                    info = data['InformationList']['Information'][0]
                    if 'Description' in info:
                        descriptions[cid] = info['Description']
                
            except Exception as e:
                logger.debug(f"No description for CID {cid}: {e}")
                continue
        
        return descriptions
    
    def search_bioassays(
        self,
        min_compounds: int = 100,
        max_compounds: int = 10000,
        target_type: Optional[str] = None,
        max_results: int = 1000
    ) -> List[int]:
        """
        Search for bioassays
        
        Args:
            min_compounds: Minimum number of tested compounds
            max_compounds: Maximum number of tested compounds
            target_type: Target type (e.g., 'protein')
            max_results: Maximum number of results
        
        Returns:
            List of assay IDs (AIDs)
        """
        aids = []
        
        try:
            # This is a simplified search - PubChem's REST API is limited
            # For production, use PUG-REST advanced search or download bulk data
            logger.info("Searching bioassays...")
            
            # Example: search for confirmed assays
            url = f"{self.base_url}/assay/type/confirmatory/aids/JSON"
            
            time.sleep(self.delay)
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            if 'IdentifierList' in data and 'AID' in data['IdentifierList']:
                aids = data['IdentifierList']['AID'][:max_results]
            
            logger.info(f"Found {len(aids)} bioassays")
            
        except Exception as e:
            logger.error(f"Error searching bioassays: {e}")
        
        return aids
    
    def get_bioassay_data(self, aid: int) -> Optional[pd.DataFrame]:
        """
        Get bioassay data for an assay ID
        
        Args:
            aid: Assay ID
        
        Returns:
            DataFrame with bioassay data
        """
        try:
            time.sleep(self.delay)
            
            # Get concise bioactivity data
            url = f"{self.base_url}/assay/aid/{aid}/concise/JSON"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse data structure
            if 'Table' in data:
                table = data['Table']
                rows = table.get('Row', [])
                
                results = []
                for row in rows:
                    result = {}
                    for key, value in row.items():
                        result[key] = value
                    results.append(result)
                
                df = pd.DataFrame(results)
                df['aid'] = aid
                return df
            
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching bioassay {aid}: {e}")
            return None
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """Save data to file"""
        output_path = self.output_dir / filename
        data.to_csv(output_path, index=False)
        logger.info(f"Saved data to {output_path}")
        return output_path


def download_pubchem_compounds(
    output_dir: str,
    query: str = "drug",
    max_compounds: int = 10000
) -> Path:
    """
    Download PubChem compound data
    
    Args:
        output_dir: Output directory
        query: Search query
        max_compounds: Maximum number of compounds
    
    Returns:
        Path to output file
    """
    downloader = PubChemDownloader(output_dir)
    
    # Search compounds
    cids = downloader.search_compounds(query, max_compounds)
    
    if not cids:
        logger.warning("No compounds found")
        return None
    
    # Get properties
    df = downloader.get_compound_properties(cids)
    
    # Save
    output_path = downloader.save_data(df, "pubchem_compounds.csv")
    
    return output_path


def download_pubchem_bioassays(
    output_dir: str,
    aids: Optional[List[int]] = None,
    max_assays: int = 100
) -> Path:
    """
    Download PubChem bioassay data
    
    Args:
        output_dir: Output directory
        aids: List of assay IDs (if None, will search)
        max_assays: Maximum number of assays
    
    Returns:
        Path to output file
    """
    downloader = PubChemDownloader(output_dir)
    
    if aids is None:
        aids = downloader.search_bioassays(max_results=max_assays)
    
    all_data = []
    
    for aid in tqdm(aids[:max_assays], desc="Downloading bioassays"):
        df = downloader.get_bioassay_data(aid)
        if df is not None:
            all_data.append(df)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        output_path = downloader.save_data(combined, "pubchem_bioassays.csv")
        return output_path
    
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download PubChem data')
    parser.add_argument('--output-dir', type=str, default='data/raw/pubchem',
                       help='Output directory')
    parser.add_argument('--type', type=str, choices=['compounds', 'bioassays'],
                       default='compounds', help='Data type to download')
    parser.add_argument('--query', type=str, default='drug',
                       help='Search query (for compounds)')
    parser.add_argument('--max-results', type=int, default=10000,
                       help='Maximum number of results')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.type == 'compounds':
        path = download_pubchem_compounds(
            args.output_dir,
            args.query,
            args.max_results
        )
    else:
        path = download_pubchem_bioassays(
            args.output_dir,
            max_assays=args.max_results
        )
    
    if path:
        print(f"\nDownloaded data to: {path}")
