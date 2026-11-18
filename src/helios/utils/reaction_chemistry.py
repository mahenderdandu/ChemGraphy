"""
Reaction Chemistry Utilities for Helios
Provides reaction visualization, prediction, and analysis using RDKit

This module enables:
- Reaction SMILES/SMIRKS parsing and validation
- Reaction visualization
- Product prediction from reactants + reaction rules
- Reaction mechanism analysis
- Common reaction template library
"""

from typing import Optional, List, Dict, Tuple, Any, Union
import logging
from dataclasses import dataclass
from io import BytesIO
import base64

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdChemReactions
from rdkit.Chem import Descriptors
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReactionResult:
    """Container for reaction prediction results"""
    success: bool
    products: List[str]  # SMILES of products
    product_sets: List[List[str]]  # Multiple possible product sets
    reactants: List[str]
    reaction_smirks: str
    message: str
    product_images: Optional[List[Image.Image]] = None
    product_properties: Optional[List[Dict[str, float]]] = None


@dataclass
class ReactionComponents:
    """Parsed reaction components"""
    reactants: List[str]
    products: List[str]
    agents: Optional[List[str]] = None  # Optional reagents/catalysts


# Common reaction templates (SMIRKS patterns)
COMMON_REACTIONS = {
    "amide_coupling": {
        "name": "Amide Coupling (Condensation)",
        "smirks": "[C:1](=[O:2])[OH1:3].[N!H0:4]>>[C:1](=[O:2])[N:4].[OH2:3]",
        "description": "Carboxylic acid + Amine → Amide + Water"
    },
    "esterification": {
        "name": "Fischer Esterification",
        "smirks": "[C:1](=[O:2])[OH:3].[OH:4][C:5]>>[C:1](=[O:2])[O:4][C:5].[OH2:3]",
        "description": "Carboxylic acid + Alcohol → Ester + Water"
    },
    "hydrolysis_ester": {
        "name": "Ester Hydrolysis",
        "smirks": "[C:1](=[O:2])[O:3][C:4].[OH2:5]>>[C:1](=[O:2])[OH:5].[OH:3][C:4]",
        "description": "Ester + Water → Carboxylic acid + Alcohol"
    },
    "suzuki_coupling": {
        "name": "Suzuki Coupling",
        "smirks": "[c:1][Br:2].[c:3][B]>>[c:1][c:3]",
        "description": "Aryl halide + Aryl boronic acid → Biaryl"
    },
    "buchwald_hartwig": {
        "name": "Buchwald-Hartwig Amination", 
        "smirks": "[c:1][Br:2].[NH2:3][C:4]>>[c:1][N:3][C:4]",
        "description": "Aryl halide + Amine → Arylamine"
    },
    "snar": {
        "name": "SNAr (Nucleophilic Aromatic Substitution)",
        "smirks": "[c:1][F,Cl,Br,I:2].[NH2:3]>>[c:1][N:3]",
        "description": "Activated aryl halide + Nucleophile → Substituted arene"
    },
    "reductive_amination": {
        "name": "Reductive Amination",
        "smirks": "[C:1]=[O:2].[NH2:3][C:4]>>[C:1][N:3][C:4]",
        "description": "Aldehyde/Ketone + Amine → Amine (via imine reduction)"
    },
    "alkylation": {
        "name": "N-Alkylation",
        "smirks": "[N:1]([H]).[C:2][Cl,Br,I:3]>>[N:1][C:2]",
        "description": "Amine + Alkyl halide → N-alkylated amine"
    },
    "oxidation_alcohol": {
        "name": "Alcohol Oxidation",
        "smirks": "[C:1][OH:2]>>[C:1]=[O:2]",
        "description": "Primary/Secondary alcohol → Aldehyde/Ketone"
    },
    "reduction_carbonyl": {
        "name": "Carbonyl Reduction",
        "smirks": "[C:1]=[O:2]>>[C:1][OH:2]",
        "description": "Aldehyde/Ketone → Alcohol"
    },
    "hydrogenation": {
        "name": "Alkene Hydrogenation",
        "smirks": "[C:1]=[C:2]>>[C:1][C:2]",
        "description": "Alkene → Alkane"
    },
    "epoxidation": {
        "name": "Alkene Epoxidation",
        "smirks": "[C:1]=[C:2]>>[C:1]1[O][C:2]1",
        "description": "Alkene → Epoxide"
    }
}


class ReactionProcessor:
    """Process and analyze chemical reactions"""
    
    def __init__(self):
        """Initialize reaction processor with common templates"""
        self.reaction_templates = COMMON_REACTIONS.copy()
        
    def parse_reaction_smiles(self, reaction_smiles: str) -> Optional[ReactionComponents]:
        """
        Parse a reaction SMILES string into components
        
        Args:
            reaction_smiles: Reaction string (e.g., "A.B>>C.D" or "A.B>reagent>C.D")
            
        Returns:
            ReactionComponents or None if parsing fails
        """
        try:
            # Handle different reaction formats
            if ">>" in reaction_smiles:
                parts = reaction_smiles.split(">>")
                if len(parts) == 2:
                    reactants = parts[0].split(".")
                    products = parts[1].split(".")
                    return ReactionComponents(
                        reactants=[r.strip() for r in reactants if r.strip()],
                        products=[p.strip() for p in products if p.strip()],
                        agents=None
                    )
            elif ">" in reaction_smiles:
                # Format: reactants>agents>products
                parts = reaction_smiles.split(">")
                if len(parts) == 3:
                    reactants = parts[0].split(".")
                    agents = parts[1].split(".") if parts[1] else None
                    products = parts[2].split(".")
                    return ReactionComponents(
                        reactants=[r.strip() for r in reactants if r.strip()],
                        products=[p.strip() for p in products if p.strip()],
                        agents=[a.strip() for a in agents if a.strip()] if agents else None
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing reaction SMILES: {e}")
            return None
    
    def validate_reaction(self, reaction_smiles: str, use_smiles: bool = True) -> Tuple[bool, str]:
        """
        Validate a reaction SMILES/SMARTS string
        
        Args:
            reaction_smiles: Reaction string
            use_smiles: Whether to interpret as SMILES (True) or SMARTS (False)
            
        Returns:
            (is_valid, message)
        """
        try:
            rxn = rdChemReactions.ReactionFromSmarts(reaction_smiles, useSmiles=use_smiles)
            if rxn is None:
                return False, "Could not parse reaction"
            
            # Basic validation
            n_reactants = rxn.GetNumReactantTemplates()
            n_products = rxn.GetNumProductTemplates()
            
            if n_reactants == 0:
                return False, "No reactants found in reaction"
            if n_products == 0:
                return False, "No products found in reaction"
            
            return True, f"Valid reaction: {n_reactants} reactant(s) → {n_products} product(s)"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def visualize_reaction(
        self, 
        reaction_smiles: str, 
        use_smiles: bool = True,
        img_size: Tuple[int, int] = (800, 300)
    ) -> Optional[Image.Image]:
        """
        Create an image of the reaction scheme
        
        Args:
            reaction_smiles: Reaction SMILES/SMARTS string
            use_smiles: Whether to interpret as SMILES
            img_size: Image dimensions
            
        Returns:
            PIL Image or None
        """
        try:
            rxn = rdChemReactions.ReactionFromSmarts(reaction_smiles, useSmiles=use_smiles)
            if rxn is None:
                return None
            
            img = Draw.ReactionToImage(rxn, subImgSize=(img_size[0]//4, img_size[1]))
            return img
            
        except Exception as e:
            logger.error(f"Error visualizing reaction: {e}")
            return None
    
    def run_reaction(
        self,
        reactant_smiles: List[str],
        reaction_smirks: str,
        max_products: int = 10
    ) -> ReactionResult:
        """
        Run a reaction with given reactants and reaction rule
        
        Args:
            reactant_smiles: List of reactant SMILES
            reaction_smirks: SMIRKS pattern for the reaction
            max_products: Maximum number of product sets to return
            
        Returns:
            ReactionResult with products and analysis
        """
        try:
            # Parse reactants
            mols = []
            for smiles in reactant_smiles:
                mol = Chem.MolFromSmiles(smiles.strip())
                if mol is None:
                    return ReactionResult(
                        success=False,
                        products=[],
                        product_sets=[],
                        reactants=reactant_smiles,
                        reaction_smirks=reaction_smirks,
                        message=f"Invalid reactant SMILES: {smiles}"
                    )
                mols.append(mol)
            
            # Parse reaction
            rxn = rdChemReactions.ReactionFromSmarts(reaction_smirks)
            if rxn is None:
                return ReactionResult(
                    success=False,
                    products=[],
                    product_sets=[],
                    reactants=reactant_smiles,
                    reaction_smirks=reaction_smirks,
                    message="Invalid reaction SMIRKS pattern"
                )
            
            # Run reaction
            results = rxn.RunReactants(tuple(mols))
            
            if not results:
                return ReactionResult(
                    success=False,
                    products=[],
                    product_sets=[],
                    reactants=reactant_smiles,
                    reaction_smirks=reaction_smirks,
                    message="No reaction occurred. Reactants may not match the reaction pattern."
                )
            
            # Process results
            all_products = []
            product_sets = []
            product_images = []
            product_properties = []
            
            for i, product_set in enumerate(results[:max_products]):
                set_smiles = []
                set_images = []
                set_props = []
                
                for mol in product_set:
                    try:
                        Chem.SanitizeMol(mol)
                        smiles = Chem.MolToSmiles(mol)
                        set_smiles.append(smiles)
                        
                        if smiles not in all_products:
                            all_products.append(smiles)
                        
                        # Generate image
                        img = Draw.MolToImage(mol, size=(200, 200))
                        set_images.append(img)
                        
                        # Calculate properties
                        props = self._calculate_properties(mol)
                        set_props.append(props)
                        
                    except Exception as e:
                        logger.warning(f"Error processing product: {e}")
                        continue
                
                if set_smiles:
                    product_sets.append(set_smiles)
                    product_images.extend(set_images)
                    product_properties.extend(set_props)
            
            return ReactionResult(
                success=True,
                products=all_products,
                product_sets=product_sets,
                reactants=reactant_smiles,
                reaction_smirks=reaction_smirks,
                message=f"Generated {len(product_sets)} possible product set(s)",
                product_images=product_images,
                product_properties=product_properties
            )
            
        except Exception as e:
            logger.error(f"Error running reaction: {e}")
            return ReactionResult(
                success=False,
                products=[],
                product_sets=[],
                reactants=reactant_smiles,
                reaction_smirks=reaction_smirks,
                message=f"Error: {str(e)}"
            )
    
    def _calculate_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate molecular properties"""
        try:
            return {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol)
            }
        except:
            return {}
    
    def get_available_reactions(self) -> Dict[str, Dict[str, str]]:
        """Get list of available reaction templates"""
        return self.reaction_templates
    
    def add_reaction_template(
        self,
        key: str,
        name: str,
        smirks: str,
        description: str
    ) -> bool:
        """Add a custom reaction template"""
        try:
            # Validate SMIRKS
            rxn = rdChemReactions.ReactionFromSmarts(smirks)
            if rxn is None:
                return False
            
            self.reaction_templates[key] = {
                "name": name,
                "smirks": smirks,
                "description": description
            }
            return True
            
        except Exception as e:
            logger.error(f"Error adding reaction template: {e}")
            return False
    
    def predict_products_from_template(
        self,
        reactant_smiles: List[str],
        template_key: str
    ) -> ReactionResult:
        """
        Run reaction using a named template
        
        Args:
            reactant_smiles: List of reactant SMILES
            template_key: Key from reaction_templates
            
        Returns:
            ReactionResult
        """
        if template_key not in self.reaction_templates:
            return ReactionResult(
                success=False,
                products=[],
                product_sets=[],
                reactants=reactant_smiles,
                reaction_smirks="",
                message=f"Unknown reaction template: {template_key}"
            )
        
        template = self.reaction_templates[template_key]
        return self.run_reaction(reactant_smiles, template["smirks"])
    
    def analyze_reaction_components(
        self,
        reaction_smiles: str
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze all components of a reaction
        
        Args:
            reaction_smiles: Full reaction SMILES string
            
        Returns:
            Dictionary with analysis of reactants and products
        """
        components = self.parse_reaction_smiles(reaction_smiles)
        if components is None:
            return None
        
        analysis = {
            'reactants': [],
            'products': [],
            'agents': [],
            'delta_mw': 0,
            'delta_logp': 0
        }
        
        total_reactant_mw = 0
        total_product_mw = 0
        total_reactant_logp = 0
        total_product_logp = 0
        
        # Analyze reactants
        for smiles in components.reactants:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                props = self._calculate_properties(mol)
                analysis['reactants'].append({
                    'smiles': smiles,
                    'properties': props
                })
                total_reactant_mw += props.get('molecular_weight', 0)
                total_reactant_logp += props.get('logp', 0)
        
        # Analyze products
        for smiles in components.products:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                props = self._calculate_properties(mol)
                analysis['products'].append({
                    'smiles': smiles,
                    'properties': props
                })
                total_product_mw += props.get('molecular_weight', 0)
                total_product_logp += props.get('logp', 0)
        
        # Analyze agents if present
        if components.agents:
            for smiles in components.agents:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    props = self._calculate_properties(mol)
                    analysis['agents'].append({
                        'smiles': smiles,
                        'properties': props
                    })
        
        # Calculate deltas
        analysis['delta_mw'] = total_product_mw - total_reactant_mw
        analysis['delta_logp'] = total_product_logp - total_reactant_logp
        
        return analysis
    
    def suggest_reaction(
        self,
        reactant_smiles: List[str]
    ) -> List[Tuple[str, str, float]]:
        """
        Suggest possible reactions based on functional groups in reactants
        
        Args:
            reactant_smiles: List of reactant SMILES
            
        Returns:
            List of (template_key, description, confidence) tuples
        """
        suggestions = []
        
        # Parse reactants and identify functional groups
        functional_groups = set()
        for smiles in reactant_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fg = self._identify_functional_groups(mol)
                functional_groups.update(fg)
        
        # Match functional groups to reactions
        if 'carboxylic_acid' in functional_groups and 'amine' in functional_groups:
            suggestions.append(("amide_coupling", "Amide formation is possible", 0.9))
        
        if 'carboxylic_acid' in functional_groups and 'alcohol' in functional_groups:
            suggestions.append(("esterification", "Ester formation is possible", 0.85))
        
        if 'aryl_halide' in functional_groups and 'boronic_acid' in functional_groups:
            suggestions.append(("suzuki_coupling", "Suzuki coupling is possible", 0.9))
        
        if 'aryl_halide' in functional_groups and 'amine' in functional_groups:
            suggestions.append(("buchwald_hartwig", "Buchwald-Hartwig amination is possible", 0.8))
        
        if 'carbonyl' in functional_groups and 'amine' in functional_groups:
            suggestions.append(("reductive_amination", "Reductive amination is possible", 0.75))
        
        if 'alkene' in functional_groups:
            suggestions.append(("hydrogenation", "Hydrogenation is possible", 0.7))
            suggestions.append(("epoxidation", "Epoxidation is possible", 0.6))
        
        if 'alcohol' in functional_groups:
            suggestions.append(("oxidation_alcohol", "Alcohol oxidation is possible", 0.65))
        
        return sorted(suggestions, key=lambda x: x[2], reverse=True)
    
    def _identify_functional_groups(self, mol: Chem.Mol) -> List[str]:
        """Identify functional groups in a molecule using SMARTS patterns"""
        patterns = {
            'carboxylic_acid': '[CX3](=O)[OX2H1]',
            'amine': '[NX3;H2,H1;!$(NC=O)]',
            'alcohol': '[OX2H][CX4]',
            'carbonyl': '[CX3]=[OX1]',
            'aryl_halide': '[c][F,Cl,Br,I]',
            'boronic_acid': '[c][B]([OH])([OH])',
            'alkene': '[CX3]=[CX3]',
            'ester': '[CX3](=O)[OX2][CX4]',
            'amide': '[NX3][CX3](=[OX1])',
            'nitrile': '[CX2]#[NX1]',
            'alkyl_halide': '[CX4][F,Cl,Br,I]',
            'aldehyde': '[CX3H1](=O)',
            'ketone': '[CX3](=O)[CX4]',
            'ether': '[OX2]([CX4])[CX4]',
            'sulfonamide': '[SX4](=[OX1])(=[OX1])([NX3])',
            'nitro': '[NX3+](=O)[O-]'
        }
        
        found = []
        for name, smarts in patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                found.append(name)
        
        return found


# Singleton instance
_reaction_processor = None


def get_reaction_processor() -> ReactionProcessor:
    """Get singleton ReactionProcessor instance"""
    global _reaction_processor
    if _reaction_processor is None:
        _reaction_processor = ReactionProcessor()
    return _reaction_processor


def visualize_molecule_with_properties(
    smiles: str,
    img_size: Tuple[int, int] = (400, 400)
) -> Tuple[Optional[Image.Image], Optional[Dict[str, float]]]:
    """
    Visualize a molecule and return its properties
    
    Args:
        smiles: SMILES string
        img_size: Image dimensions
        
    Returns:
        (PIL Image, properties dict) or (None, None)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        
        # Generate image
        img = Draw.MolToImage(mol, size=img_size)
        
        # Calculate properties
        props = {
            'molecular_weight': round(Descriptors.MolWt(mol), 3),
            'logp': round(Descriptors.MolLogP(mol), 3),
            'h_donors': Descriptors.NumHDonors(mol),
            'h_acceptors': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'tpsa': round(Descriptors.TPSA(mol), 3),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'heavy_atoms': mol.GetNumHeavyAtoms(),
            'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
            'num_rings': Descriptors.RingCount(mol),
            'fraction_sp3': round(Descriptors.FractionCSP3(mol), 3)
        }
        
        return img, props
        
    except Exception as e:
        logger.error(f"Error visualizing molecule: {e}")
        return None, None


def check_lipinski_rules(smiles: str) -> Dict[str, Any]:
    """
    Check Lipinski's Rule of Five for drug-likeness
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary with rule checks and overall pass/fail
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'valid': False, 'error': 'Invalid SMILES'}
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        rules = {
            'molecular_weight': {'value': mw, 'limit': 500, 'pass': mw <= 500},
            'logp': {'value': logp, 'limit': 5, 'pass': logp <= 5},
            'h_donors': {'value': hbd, 'limit': 5, 'pass': hbd <= 5},
            'h_acceptors': {'value': hba, 'limit': 10, 'pass': hba <= 10}
        }
        
        violations = sum(1 for r in rules.values() if not r['pass'])
        
        return {
            'valid': True,
            'rules': rules,
            'violations': violations,
            'drug_like': violations <= 1,
            'message': f"{'Drug-like' if violations <= 1 else 'Not drug-like'} ({violations} violation(s))"
        }
        
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def check_veber_rules(smiles: str) -> Dict[str, Any]:
    """
    Check Veber rules for oral bioavailability
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary with rule checks
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'valid': False, 'error': 'Invalid SMILES'}
        
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        
        rules = {
            'rotatable_bonds': {'value': rotatable_bonds, 'limit': 10, 'pass': rotatable_bonds <= 10},
            'tpsa': {'value': tpsa, 'limit': 140, 'pass': tpsa <= 140}
        }
        
        all_pass = all(r['pass'] for r in rules.values())
        
        return {
            'valid': True,
            'rules': rules,
            'bioavailable': all_pass,
            'message': f"{'Good oral bioavailability' if all_pass else 'Poor oral bioavailability expected'}"
        }
        
    except Exception as e:
        return {'valid': False, 'error': str(e)}
