"""
ChEMBL RAG corpus builder
Extracts text data from ChEMBL for retrieval-augmented generation
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ChEMBLRAGProcessor:
    """Extract text corpus from ChEMBL database for RAG"""
    
    def __init__(self, db_path: str):
        """
        Args:
            db_path: Path to ChEMBL SQLite database
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"ChEMBL database not found: {db_path}")
        
        self.conn = sqlite3.connect(str(self.db_path))
        logger.info(f"Connected to ChEMBL database: {db_path}")
    
    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def extract_compound_documents(self, limit: int = None) -> List[Dict]:
        """
        Extract compound descriptions
        
        Args:
            limit: Maximum number of documents
        
        Returns:
            List of document dictionaries
        """
        logger.info("Extracting compound documents...")
        
        query = """
        SELECT DISTINCT
            md.chembl_id,
            md.pref_name,
            cs.canonical_smiles,
            cp.full_mwt,
            cp.alogp,
            cp.hba,
            cp.hbd,
            cp.num_ro5_violations
        FROM molecule_dictionary md
        LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
        LEFT JOIN compound_properties cp ON md.molregno = cp.molregno
        WHERE md.pref_name IS NOT NULL
        AND cs.canonical_smiles IS NOT NULL
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor = self.conn.cursor()
        cursor.execute(query)
        
        documents = []
        for row in tqdm(cursor.fetchall(), desc="Processing compounds"):
            chembl_id, name, smiles, mw, logp, hba, hbd, ro5 = row
            
            # Build descriptive text
            text_parts = [f"Compound: {name} ({chembl_id})"]
            
            if mw:
                text_parts.append(f"Molecular weight: {mw:.2f} g/mol")
            if logp:
                text_parts.append(f"LogP: {logp:.2f}")
            if hba:
                text_parts.append(f"Hydrogen bond acceptors: {hba}")
            if hbd:
                text_parts.append(f"Hydrogen bond donors: {hbd}")
            if ro5 is not None:
                text_parts.append(f"Lipinski Rule of 5 violations: {ro5}")
            
            text_parts.append(f"SMILES: {smiles}")
            
            documents.append({
                'id': f"compound_{chembl_id}",
                'type': 'compound',
                'title': f"{name} ({chembl_id})",
                'text': '. '.join(text_parts),
                'metadata': {
                    'chembl_id': chembl_id,
                    'smiles': smiles,
                    'name': name
                }
            })
        
        logger.info(f"Extracted {len(documents)} compound documents")
        return documents
    
    def extract_target_documents(self, limit: int = None) -> List[Dict]:
        """
        Extract target/protein descriptions
        
        Args:
            limit: Maximum number of documents
        
        Returns:
            List of document dictionaries
        """
        logger.info("Extracting target documents...")
        
        query = """
        SELECT DISTINCT
            td.chembl_id,
            td.pref_name,
            td.target_type,
            td.organism,
            cs.accession,
            cs.description
        FROM target_dictionary td
        LEFT JOIN target_components tc ON td.tid = tc.tid
        LEFT JOIN component_sequences cs ON tc.component_id = cs.component_id
        WHERE td.pref_name IS NOT NULL
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor = self.conn.cursor()
        cursor.execute(query)
        
        documents = []
        for row in tqdm(cursor.fetchall(), desc="Processing targets"):
            chembl_id, name, target_type, organism, accession, description = row
            
            # Build descriptive text
            text_parts = [f"Target: {name} ({chembl_id})"]
            
            if target_type:
                text_parts.append(f"Type: {target_type}")
            if organism:
                text_parts.append(f"Organism: {organism}")
            if accession:
                text_parts.append(f"Accession: {accession}")
            if description:
                text_parts.append(f"Description: {description}")
            
            documents.append({
                'id': f"target_{chembl_id}",
                'type': 'target',
                'title': f"{name} ({chembl_id})",
                'text': '. '.join(text_parts),
                'metadata': {
                    'chembl_id': chembl_id,
                    'name': name,
                    'organism': organism
                }
            })
        
        logger.info(f"Extracted {len(documents)} target documents")
        return documents
    
    def extract_mechanism_documents(self, limit: int = None) -> List[Dict]:
        """
        Extract drug mechanism of action descriptions
        
        Args:
            limit: Maximum number of documents
        
        Returns:
            List of document dictionaries
        """
        logger.info("Extracting mechanism documents...")
        
        query = """
        SELECT DISTINCT
            md.chembl_id,
            md.pref_name,
            dm.mechanism_of_action,
            dm.action_type,
            td.chembl_id as target_chembl_id,
            td.pref_name as target_name
        FROM drug_mechanism dm
        JOIN molecule_dictionary md ON dm.molregno = md.molregno
        LEFT JOIN target_dictionary td ON dm.tid = td.tid
        WHERE dm.mechanism_of_action IS NOT NULL
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor = self.conn.cursor()
        cursor.execute(query)
        
        documents = []
        for row in tqdm(cursor.fetchall(), desc="Processing mechanisms"):
            drug_id, drug_name, mechanism, action, target_id, target_name = row
            
            # Build descriptive text
            text_parts = [f"Drug mechanism: {drug_name} ({drug_id})"]
            text_parts.append(f"Mechanism of action: {mechanism}")
            
            if action:
                text_parts.append(f"Action type: {action}")
            if target_name:
                text_parts.append(f"Target: {target_name} ({target_id})")
            
            documents.append({
                'id': f"mechanism_{drug_id}_{target_id or 'unknown'}",
                'type': 'mechanism',
                'title': f"{drug_name} mechanism",
                'text': '. '.join(text_parts),
                'metadata': {
                    'drug_chembl_id': drug_id,
                    'target_chembl_id': target_id,
                    'mechanism': mechanism
                }
            })
        
        logger.info(f"Extracted {len(documents)} mechanism documents")
        return documents
    
    def build_corpus(
        self,
        output_path: str,
        max_compounds: int = 10000,
        max_targets: int = 5000,
        max_mechanisms: int = 5000
    ) -> int:
        """
        Build complete RAG corpus
        
        Args:
            output_path: Path to save corpus JSON
            max_compounds: Max compound documents
            max_targets: Max target documents
            max_mechanisms: Max mechanism documents
        
        Returns:
            Total number of documents
        """
        logger.info("Building RAG corpus from ChEMBL...")
        
        all_documents = []
        
        # Extract compounds
        try:
            compound_docs = self.extract_compound_documents(limit=max_compounds)
            all_documents.extend(compound_docs)
        except Exception as e:
            logger.error(f"Failed to extract compounds: {e}")
        
        # Extract targets
        try:
            target_docs = self.extract_target_documents(limit=max_targets)
            all_documents.extend(target_docs)
        except Exception as e:
            logger.error(f"Failed to extract targets: {e}")
        
        # Extract mechanisms
        try:
            mechanism_docs = self.extract_mechanism_documents(limit=max_mechanisms)
            all_documents.extend(mechanism_docs)
        except Exception as e:
            logger.error(f"Failed to extract mechanisms: {e}")
        
        # Save to JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(all_documents, f, indent=2)
        
        logger.info(f"Saved {len(all_documents)} documents to {output_path}")
        logger.info(f"Corpus breakdown:")
        logger.info(f"  - Compounds: {len([d for d in all_documents if d['type'] == 'compound'])}")
        logger.info(f"  - Targets: {len([d for d in all_documents if d['type'] == 'target'])}")
        logger.info(f"  - Mechanisms: {len([d for d in all_documents if d['type'] == 'mechanism'])}")
        
        return len(all_documents)


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Build RAG corpus from ChEMBL')
    parser.add_argument('--db-path', type=str, required=True,
                       help='Path to ChEMBL SQLite database')
    parser.add_argument('--output', type=str, default='data/rag/chembl_corpus.json',
                       help='Output JSON path')
    parser.add_argument('--max-compounds', type=int, default=10000,
                       help='Maximum compound documents')
    parser.add_argument('--max-targets', type=int, default=5000,
                       help='Maximum target documents')
    parser.add_argument('--max-mechanisms', type=int, default=5000,
                       help='Maximum mechanism documents')
    
    args = parser.parse_args()
    
    processor = ChEMBLRAGProcessor(args.db_path)
    num_docs = processor.build_corpus(
        args.output,
        max_compounds=args.max_compounds,
        max_targets=args.max_targets,
        max_mechanisms=args.max_mechanisms
    )
    
    print(f"\n✅ Created RAG corpus with {num_docs} documents")
    print(f"📁 Saved to: {args.output}")
