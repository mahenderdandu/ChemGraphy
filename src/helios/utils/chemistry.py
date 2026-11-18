"""
Chemistry utility functions using RDKit
Handles SMILES processing, molecule standardization, and basic calculations
"""

from typing import Optional, List, Dict, Tuple
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, SaltRemover, MolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np

logger = logging.getLogger(__name__)


class MoleculeProcessor:
    """Process and standardize molecules"""
    
    def __init__(self):
        self.salt_remover = SaltRemover.SaltRemover()
        self.normalizer = rdMolStandardize.Normalizer()
        self.uncharger = rdMolStandardize.Uncharger()
        
    def canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """Convert SMILES to canonical form"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            logger.warning(f"Failed to canonicalize SMILES {smiles}: {e}")
            return None
    
    def standardize_molecule(self, smiles: str) -> Optional[str]:
        """
        Full standardization pipeline:
        1. Parse SMILES
        2. Remove salts
        3. Normalize functional groups
        4. Neutralize charges
        5. Canonicalize
        """
        try:
            # Parse
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Remove salts
            mol = self.salt_remover.StripMol(mol)
            
            # Normalize
            mol = self.normalizer.normalize(mol)
            
            # Neutralize
            mol = self.uncharger.uncharge(mol)
            
            # Canonicalize
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            
            return canonical_smiles
            
        except Exception as e:
            logger.warning(f"Failed to standardize SMILES {smiles}: {e}")
            return None
    
    def is_valid_molecule(
        self, 
        smiles: str,
        min_mw: float = 0,
        max_mw: float = 800,
        min_heavy_atoms: int = 3,
        max_heavy_atoms: int = 50,
        allow_metals: bool = False
    ) -> bool:
        """Check if molecule passes quality filters"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # Check molecular weight
            mw = Descriptors.MolWt(mol)
            if not (min_mw <= mw <= max_mw):
                return False
            
            # Check heavy atom count
            heavy_atoms = mol.GetNumHeavyAtoms()
            if not (min_heavy_atoms <= heavy_atoms <= max_heavy_atoms):
                return False
            
            # Check for metals if not allowed
            if not allow_metals:
                metals = [
                    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V',
                    'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr',
                    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                    'In', 'Sn', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os',
                    'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi'
                ]
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() in metals:
                        return False
            
            # Check for mixtures (multiple fragments)
            if len(Chem.GetMolFrags(mol)) > 1:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating molecule {smiles}: {e}")
            return False
    
    def calculate_descriptors(self, smiles: str) -> Optional[Dict[str, float]]:
        """Calculate common molecular descriptors"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            descriptors = {
                'mw': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'heavy_atoms': mol.GetNumHeavyAtoms(),
                'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
            }
            
            return descriptors
            
        except Exception as e:
            logger.warning(f"Error calculating descriptors for {smiles}: {e}")
            return None
    
    def smiles_to_inchi(self, smiles: str) -> Optional[str]:
        """Convert SMILES to InChI"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToInchi(mol)
        except Exception as e:
            logger.warning(f"Error converting SMILES to InChI: {e}")
            return None
    
    def smiles_to_inchi_key(self, smiles: str) -> Optional[str]:
        """Convert SMILES to InChI Key"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToInchiKey(mol)
        except Exception as e:
            logger.warning(f"Error converting SMILES to InChI Key: {e}")
            return None
    
    def generate_fingerprint(
        self, 
        smiles: str, 
        fp_type: str = "morgan",
        radius: int = 2,
        n_bits: int = 2048
    ) -> Optional[np.ndarray]:
        """Generate molecular fingerprint"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            if fp_type == "morgan":
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            elif fp_type == "maccs":
                fp = AllChem.GetMACCSKeysFingerprint(mol)
            elif fp_type == "topological":
                fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
            else:
                raise ValueError(f"Unknown fingerprint type: {fp_type}")
            
            return np.array(fp)
            
        except Exception as e:
            logger.warning(f"Error generating fingerprint for {smiles}: {e}")
            return None


def convert_to_pactivity(value: float, unit: str = "nM") -> Optional[float]:
    """
    Convert activity value to pActivity = -log10(activity_in_molar)
    
    Args:
        value: Activity value
        unit: Unit (nM, uM, mM, M)
    
    Returns:
        pActivity value or None if invalid
    """
    try:
        if value <= 0:
            return None
        
        # Convert to molar
        if unit == "nM":
            molar = value * 1e-9
        elif unit == "uM":
            molar = value * 1e-6
        elif unit == "mM":
            molar = value * 1e-3
        elif unit == "M":
            molar = value
        else:
            logger.warning(f"Unknown unit: {unit}")
            return None
        
        # Calculate pActivity
        pactivity = -np.log10(molar)
        
        return pactivity
        
    except Exception as e:
        logger.warning(f"Error converting to pActivity: {e}")
        return None


def classify_activity(value: float, unit: str = "nM", 
                     active_threshold: float = 1000,
                     inactive_threshold: float = 10000) -> Optional[int]:
    """
    Classify molecule as active (1) or inactive (0)
    
    Args:
        value: Activity value
        unit: Unit
        active_threshold: Threshold for active (in nM)
        inactive_threshold: Threshold for inactive (in nM)
    
    Returns:
        0 (inactive), 1 (active), or None (intermediate)
    """
    try:
        # Convert to nM
        if unit == "nM":
            value_nm = value
        elif unit == "uM":
            value_nm = value * 1000
        elif unit == "mM":
            value_nm = value * 1000000
        elif unit == "M":
            value_nm = value * 1000000000
        else:
            return None
        
        if value_nm <= active_threshold:
            return 1  # Active
        elif value_nm >= inactive_threshold:
            return 0  # Inactive
        else:
            return None  # Intermediate (usually excluded)
            
    except Exception as e:
        logger.warning(f"Error classifying activity: {e}")
        return None


# Singleton instance
_processor = None

def get_molecule_processor() -> MoleculeProcessor:
    """Get singleton MoleculeProcessor instance"""
    global _processor
    if _processor is None:
        _processor = MoleculeProcessor()
    return _processor
