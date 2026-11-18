"""
ChEMBL data processor
Extracts and processes activity data from ChEMBL database
"""

import logging
from typing import Optional, List, Dict
import pandas as pd
import sqlite3
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ChEMBLProcessor:
    """Process ChEMBL database"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"ChEMBL database not found: {db_path}")
        
        self.conn = sqlite3.connect(str(self.db_path))
    
    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def extract_activities(
        self,
        activity_types: List[str] = None,
        min_confidence: int = 5,
        target_types: Optional[List[str]] = None,
        limit: Optional[int] = None,
        chunk_size: int = 50000
    ) -> pd.DataFrame:
        """
        Extract bioactivity data from ChEMBL in chunks (memory-efficient)
        
        Args:
            activity_types: List of activity types (IC50, Ki, etc.)
            min_confidence: Minimum confidence score
            target_types: Target types to include
            limit: Maximum number of records
            chunk_size: Process this many records at a time
        
        Returns:
            DataFrame with activity data
        """
        if activity_types is None:
            activity_types = ['IC50', 'Ki', 'Kd', 'EC50', 'AC50']
        
        # First, count total records
        count_query = """
        SELECT COUNT(*) as total
        FROM activities act
        JOIN molecule_dictionary md ON act.molregno = md.molregno
        JOIN compound_structures cs ON md.molregno = cs.molregno
        JOIN assays ass ON act.assay_id = ass.assay_id
        JOIN target_dictionary td ON ass.tid = td.tid
        WHERE 
            act.standard_type IN ({})
            AND act.standard_value IS NOT NULL
            AND act.standard_units IS NOT NULL
            AND cs.canonical_smiles IS NOT NULL
            AND ass.confidence_score >= ?
        """.format(','.join(['?'] * len(activity_types)))
        
        params = activity_types + [min_confidence]
        
        if target_types:
            count_query += " AND td.target_type IN ({})".format(','.join(['?'] * len(target_types)))
            params.extend(target_types)
        
        cursor = self.conn.cursor()
        cursor.execute(count_query, params)
        total_records = cursor.fetchone()[0]
        
        if limit and limit < total_records:
            total_records = limit
        
        logger.info(f"Total records to process: {total_records:,}")
        logger.info(f"Processing in chunks of {chunk_size:,} records")
        
        # Build main query
        query = """
        SELECT 
            cs.canonical_smiles,
            cs.standard_inchi_key,
            md.chembl_id as compound_id,
            act.standard_type,
            act.standard_value,
            act.standard_units,
            act.pchembl_value,
            act.activity_comment,
            act.data_validity_comment,
            td.chembl_id as target_id,
            td.pref_name as target_name,
            td.target_type,
            td.organism,
            ass.chembl_id as assay_id,
            ass.assay_type,
            ass.confidence_score,
            doc.chembl_id as document_id,
            doc.journal,
            doc.year
        FROM activities act
        JOIN molecule_dictionary md ON act.molregno = md.molregno
        JOIN compound_structures cs ON md.molregno = cs.molregno
        JOIN assays ass ON act.assay_id = ass.assay_id
        JOIN target_dictionary td ON ass.tid = td.tid
        LEFT JOIN docs doc ON act.doc_id = doc.doc_id
        WHERE 
            act.standard_type IN ({})
            AND act.standard_value IS NOT NULL
            AND act.standard_units IS NOT NULL
            AND cs.canonical_smiles IS NOT NULL
            AND ass.confidence_score >= ?
        """.format(','.join(['?'] * len(activity_types)))
        
        if target_types:
            query += " AND td.target_type IN ({})".format(','.join(['?'] * len(target_types)))
        
        # Process in chunks
        chunks = []
        num_chunks = (total_records + chunk_size - 1) // chunk_size
        
        logger.info(f"Extracting data in {num_chunks} chunks...")
        
        for i in range(num_chunks):
            offset = i * chunk_size
            chunk_query = query + f" LIMIT {chunk_size} OFFSET {offset}"
            
            logger.info(f"Processing chunk {i+1}/{num_chunks} (records {offset:,} to {min(offset+chunk_size, total_records):,})")
            
            chunk_df = pd.read_sql_query(chunk_query, self.conn, params=params)
            
            if len(chunk_df) == 0:
                break
                
            chunks.append(chunk_df)
            
            # Stop if we hit the limit
            if limit and sum(len(c) for c in chunks) >= limit:
                break
        
        # Combine all chunks
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            if limit:
                df = df.head(limit)
        else:
            df = pd.DataFrame()
        
        logger.info(f"Extracted {len(df):,} activity records")
        
        return df
    
    def extract_compound_properties(
        self,
        compound_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract compound properties
        
        Args:
            compound_ids: List of ChEMBL IDs (if None, extract all)
        
        Returns:
            DataFrame with compound properties
        """
        query = """
        SELECT 
            md.chembl_id,
            cs.canonical_smiles,
            cs.standard_inchi,
            cs.standard_inchi_key,
            cp.mw_freebase as molecular_weight,
            cp.alogp,
            cp.hba,
            cp.hbd,
            cp.psa,
            cp.rtb as rotatable_bonds,
            cp.aromatic_rings,
            cp.heavy_atoms,
            cp.num_ro5_violations
        FROM molecule_dictionary md
        JOIN compound_structures cs ON md.molregno = cs.molregno
        LEFT JOIN compound_properties cp ON md.molregno = cp.molregno
        WHERE cs.canonical_smiles IS NOT NULL
        """
        
        params = []
        if compound_ids:
            query += " AND md.chembl_id IN ({})".format(','.join(['?'] * len(compound_ids)))
            params = compound_ids
        
        df = pd.read_sql_query(query, self.conn, params=params)
        
        logger.info(f"Extracted {len(df)} compound records")
        
        return df
    
    def extract_target_info(self, target_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract target information
        
        Args:
            target_ids: List of target ChEMBL IDs
        
        Returns:
            DataFrame with target info
        """
        query = """
        SELECT 
            td.chembl_id,
            td.pref_name,
            td.target_type,
            td.organism,
            pc.accession as uniprot_id,
            pc.description as protein_description
        FROM target_dictionary td
        LEFT JOIN target_components tc ON td.tid = tc.tid
        LEFT JOIN component_sequences cs ON tc.component_id = cs.component_id
        LEFT JOIN protein_classification pc ON cs.component_id = pc.protein_class_id
        """
        
        params = []
        if target_ids:
            query += " WHERE td.chembl_id IN ({})".format(','.join(['?'] * len(target_ids)))
            params = target_ids
        
        df = pd.read_sql_query(query, self.conn, params=params)
        
        return df
    
    def get_compound_annotations(self, compound_ids: List[str]) -> Dict[str, str]:
        """
        Get compound annotations/descriptions for RAG corpus
        
        Args:
            compound_ids: List of ChEMBL IDs
        
        Returns:
            Dictionary mapping compound ID to annotation
        """
        query = """
        SELECT 
            md.chembl_id,
            md.pref_name,
            GROUP_CONCAT(DISTINCT ms.synonyms, '; ') as synonyms,
            cp.full_mwt,
            cp.alogp
        FROM molecule_dictionary md
        LEFT JOIN molecule_synonyms ms ON md.molregno = ms.molregno
        LEFT JOIN compound_properties cp ON md.molregno = cp.molregno
        WHERE md.chembl_id IN ({})
        GROUP BY md.chembl_id
        """.format(','.join(['?'] * len(compound_ids)))
        
        df = pd.read_sql_query(query, self.conn, params=compound_ids)
        
        annotations = {}
        for _, row in df.iterrows():
            parts = [f"Compound: {row['chembl_id']}"]
            if pd.notna(row['pref_name']):
                parts.append(f"Name: {row['pref_name']}")
            if pd.notna(row['synonyms']):
                parts.append(f"Synonyms: {row['synonyms']}")
            if pd.notna(row['full_mwt']):
                parts.append(f"MW: {row['full_mwt']:.2f}")
            if pd.notna(row['alogp']):
                parts.append(f"LogP: {row['alogp']:.2f}")
            
            annotations[row['chembl_id']] = '. '.join(parts)
        
        return annotations


def process_chembl_data(
    db_path: str,
    output_path: str,
    activity_types: List[str] = None,
    min_confidence: int = 5,
    limit: Optional[int] = None,
    chunk_size: int = 10000
) -> pd.DataFrame:
    """
    Process ChEMBL database and save to file
    
    Args:
        db_path: Path to ChEMBL SQLite database
        output_path: Output CSV path
        activity_types: Activity types to extract
        min_confidence: Minimum confidence score
        limit: Maximum records
        chunk_size: Process in chunks of this size
    
    Returns:
        DataFrame with processed data
    """
    processor = ChEMBLProcessor(db_path)
    
    # Extract activities
    df = processor.extract_activities(
        activity_types=activity_types,
        min_confidence=min_confidence,
        limit=limit,
        chunk_size=chunk_size
    )
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process ChEMBL data')
    parser.add_argument('--db-path', type=str, required=True,
                       help='Path to ChEMBL SQLite database')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV path')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of records')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    df = process_chembl_data(
        args.db_path,
        args.output,
        limit=args.limit
    )
    
    print(f"\nProcessed {len(df)} records")
    print("\nColumn summary:")
    print(df.info())
