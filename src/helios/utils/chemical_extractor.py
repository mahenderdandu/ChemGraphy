"""
Chemical Name Extractor & SMILES Resolver

Extracts chemical/drug names from text and resolves them to SMILES strings.
Uses:
1. Built-in dictionary of ~200 common chemicals/drugs
2. PubChem PUG-REST API as fallback
3. Regex patterns for SMILES-like strings already in text

This enables the GNN (Graphormer) pipeline to work automatically
without the user needing to provide explicit SMILES input.
"""

import re
import logging
import json
from typing import List, Dict, Optional, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)


# ============================================================
# Common Chemical Name → SMILES Dictionary
# ============================================================

CHEMICAL_SMILES_DB: Dict[str, str] = {
    # --- Analgesics / NSAIDs ---
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "acetylsalicylic acid": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "naproxen": "COc1ccc2cc(CC(C)C(=O)O)ccc2c1",
    "diclofenac": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "celecoxib": "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "indomethacin": "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1",
    "piroxicam": "CN1C(=C(O)c2ccccc2S1(=O)=O)C(=O)Nc1ccccn1",
    "morphine": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2C(=C[C@@H]1[C@@H]3O)C5",
    "codeine": "COc1ccc2C[C@H]3N(C)CC[C@@]45c2c1O[C@H]4C(=C[C@@H]35)O",

    # --- Antibiotics ---
    "penicillin": "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@@H]1C(=O)O",
    "penicillin g": "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@@H]1C(=O)O",
    "amoxicillin": "CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccc(O)cc3)C(=O)N2[C@@H]1C(=O)O",
    "ciprofloxacin": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "metronidazole": "Cc1ncc([N+](=O)[O-])n1CCO",
    "erythromycin": "CC[C@@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@@H]([C@H]2O)N(C)C)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O",
    "tetracycline": "CN(C)[C@@H]1C(=O)C(C(N)=O)=C(O)[C@@]2(O)C(=O)c3c(O)cccc3[C@@H](O)[C@H]12",
    "doxycycline": "O=C(N)[C@@]1(O)C(=O)c2c(O)cccc2[C@@H](O)[C@H]2C[C@H]3[C@@H](N(C)C)C(=O)C(=C([C@@]3(O)C2=C1O)O)O",
    "vancomycin": "CC1OC(OC2C(O)C(O)C(OC2CO)Oc2cc3cc(Oc4cc(cc(Cl)c4O)C(NC(=O)C(CC(N)=O)NC(=O)C4NC(=O)C(NC3=O)c3cc(O)c(c(Cl)c3)Oc3ccc(cc3)C(O)C4NC=O)C(=O)NC(C(O)c3ccc(O)c(c3)C)C(=O)O)c(O)c2)C(O)C(O)C1NC(C)=O",

    # --- Cardiovascular ---
    "atorvastatin": "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1",
    "metoprolol": "COCCc1ccc(OCC(O)CNC(C)C)cc1",
    "amlodipine": "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl",
    "lisinopril": "NCCCC[C@@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O",
    "losartan": "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1",
    "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "digoxin": "C[C@H]1O[C@@H](O[C@@H]2C[C@H](O)[C@@H](O[C@@H]3C[C@H](O)[C@H](O[C@@H]4C[C@H](O)[C@@H](O[C@@H]5CC[C@]6(C)[C@@H](CC[C@@H]7[C@@H]6C[C@H](O)[C@]6(C)[C@@H](CC=C8COC(=O)C8)CC[C@H]76)C5)O4)O3)O2)[C@@H](O)[C@H]1O",

    # --- CNS / Psychiatric ---
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "diazepam": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "fluoxetine": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
    "prozac": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
    "sertraline": "CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc21",
    "risperidone": "Cc1nc2n(CC3CCN(CCc4noc5cc(F)ccc45)CC3)c(=O)cc2cc1",
    "haloperidol": "OC1(c2ccc(Cl)cc2)CCN(CCCC(=O)c2ccc(F)cc2)CC1",
    "donepezil": "COc1cc2CC(CC3CCN(Cc4ccccc4)CC3)C(=O)c2cc1OC",
    "melatonin": "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",

    # --- Antidiabetics ---
    "metformin": "CN(C)C(=N)NC(=N)N",
    "glipizide": "Cc1cnc(NS(=O)(=O)c2ccc(CCNC(=O)c3ccccn3)cc2)cn1",
    "pioglitazone": "CCc1ccc(CCOc2ccc(CC3SC(=O)NC3=O)cc2)nc1",
    "sitagliptin": "N[C@@H](CC(=O)N1CCn2c(nnc2C(F)(F)F)C1)c1cc(F)c(F)cc1F",

    # --- Anticancer ---
    "tamoxifen": "CCC(=C(c1ccccc1)c1ccccc1)c1ccc(OCCN(C)C)cc1",
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "erlotinib": "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "doxorubicin": "COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)C[C@@](O)(C(=O)CO)C[C@@H]3O[C@H]1C[C@H](N)[C@@H](O)[C@H](C)O1",
    "paclitaxel": "CC(=O)O[C@@H]1C(=O)[C@@]2(C)C(C[C@H]3OC[C@@]23OC(C)=O)C2C(OC(=O)c3ccccc3)(C1(C)C)C(O)C(OC(=O)C(O)C(NC(=O)c1ccccc1)c1ccccc1)C2=O",
    "methotrexate": "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1",
    "5-fluorouracil": "O=c1[nH]cc(F)c(=O)[nH]1",
    "fluorouracil": "O=c1[nH]cc(F)c(=O)[nH]1",
    "cisplatin": "[NH3][Pt]([NH3])(Cl)Cl",
    "cyclophosphamide": "ClCCN(CCCl)P1(=O)NCCCO1",

    # --- Antiviral ---
    "acyclovir": "Nc1nc2c(ncn2COCCO)c(=O)[nH]1",
    "oseltamivir": "CCC(CC)O[C@@H]1C=C(C(=O)OCC)[C@@H](N)[C@H](NC(C)=O)[C@@H]1O",
    "remdesivir": "CCC(CC)COC(=O)[C@H](C)NP(=O)(OC[C@H]1OC(c2ccc3c(N)ncnn23)[C@H](O)[C@@H]1O)Oc1ccccc1",
    "ritonavir": "CC(C)[C@H](NC(=O)OCC)C(=O)N[C@@H](C[C@H](O)[C@H](Cc1ccccc1)NC(=O)OCc1cncs1)Cc1ccccc1",
    "lopinavir": "Cc1cccc(C)c1OCC(=O)N[C@@H](C[C@H](O)[C@@H](Cc1ccccc1)NC(=O)[C@@H](CC(C)C)NC(=O)c1ccc2ccccc2n1)Cc1ccccc1",

    # --- Steroids / Hormones ---
    "testosterone": "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]43C)[C@@H]1CC[C@@H]2O",
    "estradiol": "C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc43)[C@@H]1CC[C@@H]2O",
    "cortisol": "O[C@@]1(C(=O)CO)CC[C@@H]2[C@@]1(C)C[C@H](O)[C@H]1[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@@H]3C(=O)C[C@@]21C",
    "hydrocortisone": "O[C@@]1(C(=O)CO)CC[C@@H]2[C@@]1(C)C[C@H](O)[C@H]1[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@@H]3C(=O)C[C@@]21C",
    "prednisone": "O=C1C=C[C@@]2(C)[C@@H]3C(=O)C[C@]4(C)[C@@H](C(=O)CO)CC[C@]4(O)[C@@H]3CC[C@@H]2[C@@H]1O",
    "dexamethasone": "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO",
    "cholesterol": "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(C)CC[C@H]1[C@H]2CC=C2C[C@@H](O)CC[C@@]21C",

    # --- Simple / Common Chemicals ---
    "ethanol": "CCO",
    "methanol": "CO",
    "acetone": "CC(=O)C",
    "acetic acid": "CC(=O)O",
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "phenol": "Oc1ccccc1",
    "aniline": "Nc1ccccc1",
    "glycine": "NCC(=O)O",
    "alanine": "CC(N)C(=O)O",
    "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "fructose": "OC[C@@H](O)[C@@H](O)[C@H](O)C(=O)CO",
    "sucrose": "OC[C@H]1OC(O[C@@]2(CO)O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O",
    "urea": "NC(=O)N",
    "glycerol": "OCC(O)CO",
    "formic acid": "OC=O",
    "citric acid": "OC(=O)CC(O)(CC(=O)O)C(=O)O",
    "nicotine": "CN1CCC[C@H]1c1cccnc1",
    "capsaicin": "COc1cc(CNC(=O)CCCC/C=C/C(C)C)ccc1O",
    "dopamine": "NCCc1ccc(O)c(O)c1",
    "serotonin": "NCCc1c[nH]c2ccc(O)cc12",
    "adrenaline": "CNC[C@H](O)c1ccc(O)c(O)c1",
    "epinephrine": "CNC[C@H](O)c1ccc(O)c(O)c1",
    "norepinephrine": "NC[C@H](O)c1ccc(O)c(O)c1",
    "histamine": "NCCc1c[nH]cn1",
    "acetylcholine": "CC(=O)OCC[N+](C)(C)C",
    "gaba": "NCCCC(=O)O",
    "gamma-aminobutyric acid": "NCCCC(=O)O",

    # --- Vitamins ---
    "vitamin c": "OC[C@@H](O)[C@H]1OC(=O)C(O)=C1O",
    "ascorbic acid": "OC[C@@H](O)[C@H]1OC(=O)C(O)=C1O",
    "vitamin e": "CC1=C(C)C(O)=C2OC(C)(CCCC(C)CCCC(C)CCCC(C)C)CCC2=C1C",
    "retinol": "CC1=C(C=CC(C)=CC=CC(C)=CCO)C(C)(C)CCC1",
    "vitamin a": "CC1=C(C=CC(C)=CC=CC(C)=CCO)C(C)(C)CCC1",
    "niacin": "OC(=O)c1cccnc1",
    "vitamin b3": "OC(=O)c1cccnc1",

    # --- Other important drugs ---
    "penicillamine": "CC(S)(C(N)C(=O)O)C",
    "omeprazole": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
    "ranitidine": "CNC(/C=C/[N+](=O)[O-])=C\\NCCSCc1ccc(CN(C)C)o1",
    "sildenafil": "CCCc1nn(C)c2c1nc(nc2OCC)-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1",
    "viagra": "CCCc1nn(C)c2c1nc(nc2OCC)-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1",
    "loratadine": "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1",
    "cetirizine": "OC(=O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1",
    "diphenhydramine": "CN(C)CCOC(c1ccccc1)c1ccccc1",
    "benadryl": "CN(C)CCOC(c1ccccc1)c1ccccc1",
    "clopidogrel": "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1",
    "plavix": "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1",
    "thalidomide": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
    "penicillin v": "CC1(C)S[C@@H]2[C@H](NC(=O)COc3ccccc3)C(=O)N2[C@@H]1C(=O)O",
    "ampicillin": "CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccccc3)C(=O)N2[C@@H]1C(=O)O",

    # --- Amino acids ---
    "leucine": "CC(C)CC(N)C(=O)O",
    "isoleucine": "CCC(C)C(N)C(=O)O",
    "valine": "CC(C)C(N)C(=O)O",
    "phenylalanine": "NC(Cc1ccccc1)C(=O)O",
    "tryptophan": "NC(Cc1c[nH]c2ccccc12)C(=O)O",
    "tyrosine": "NC(Cc1ccc(O)cc1)C(=O)O",
    "serine": "NC(CO)C(=O)O",
    "threonine": "CC(O)C(N)C(=O)O",
    "cysteine": "NC(CS)C(=O)O",
    "methionine": "CSCCC(N)C(=O)O",
    "proline": "OC(=O)C1CCCN1",
    "histidine": "NC(Cc1c[nH]cn1)C(=O)O",
    "arginine": "NC(CCCNC(=N)N)C(=O)O",
    "lysine": "NCCCCC(N)C(=O)O",
    "aspartic acid": "NC(CC(=O)O)C(=O)O",
    "glutamic acid": "NC(CCC(=O)O)C(=O)O",
    "asparagine": "NC(=O)CC(N)C(=O)O",
    "glutamine": "NC(=O)CCC(N)C(=O)O",

    # --- Solvents / Industrial ---
    "dmso": "CS(=O)C",
    "dimethyl sulfoxide": "CS(=O)C",
    "chloroform": "ClC(Cl)Cl",
    "dichloromethane": "ClCCl",
    "thf": "C1CCOC1",
    "tetrahydrofuran": "C1CCOC1",
    "diethyl ether": "CCOCC",
    "acetonitrile": "CC#N",
    "dmf": "CN(C)C=O",
    "pyridine": "c1ccncc1",
    "triethylamine": "CCN(CC)CC",
    "hexane": "CCCCCC",
    "pentane": "CCCCC",
    "butanol": "CCCCO",
    "propanol": "CCCO",
    "isopropanol": "CC(O)C",
    "ethylene glycol": "OCCO",
    "formaldehyde": "C=O",
    "acetaldehyde": "CC=O",
    "benzoic acid": "OC(=O)c1ccccc1",
    "salicylic acid": "OC(=O)c1ccccc1O",
    "oxalic acid": "OC(=O)C(=O)O",
    "tartaric acid": "OC(C(O)C(=O)O)C(=O)O",
    "lactic acid": "CC(O)C(=O)O",
    "succinic acid": "OC(=O)CCC(=O)O",
    "fumaric acid": "OC(=O)/C=C/C(=O)O",
    "maleic acid": "OC(=O)/C=C\\C(=O)O",
    "naphthalene": "c1ccc2ccccc2c1",
    "anthracene": "c1ccc2cc3ccccc3cc2c1",
    "pyrene": "c1cc2ccc3cccc4ccc(c1)c2c34",
}


# ============================================================
# SMILES pattern (detect explicit SMILES in user text)
# ============================================================

# Regex that matches typical SMILES strings (contains ring digits, stereochem, etc.)
# Requires at least one non-alphabetic SMILES character (ring-closure digit, bracket,
# bond =/#, stereo @, slash \/\\) to avoid false-positives on plain English words.
SMILES_PATTERN = re.compile(
    r'(?<![a-zA-Z])'           # not preceded by letter
    r'(?=[A-Za-z@+\-\[\]\(\)\\\/=#$:.%0-9]*[0-9\[\]\(\)\\\/@=#])'  # lookahead: must contain ring-closure digit, bracket, or bond symbol
    r'('
    r'[A-Z][a-z]?'             # starts with element symbol
    r'(?:[A-Za-z0-9@+\-\[\]\(\)\\\/=#$:.%]){4,}'  # continues with SMILES chars, min 5 total
    r')'
    r'(?![a-zA-Z])',           # not followed by letter
    re.MULTILINE
)


# ============================================================
# PubChem Resolver (fallback)
# ============================================================

def _resolve_via_pubchem(name: str) -> Optional[str]:
    """
    Resolve a chemical name to SMILES using PubChem PUG-REST API.
    Returns canonical SMILES or None.
    """
    try:
        import urllib.request
        import urllib.parse

        encoded_name = urllib.parse.quote(name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/CanonicalSMILES/JSON"

        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        props = data.get("PropertyTable", {}).get("Properties", [])
        if props:
            smiles = props[0].get("CanonicalSMILES")
            if smiles:
                logger.info(f"PubChem resolved '{name}' → {smiles}")
                return smiles
    except Exception as e:
        logger.debug(f"PubChem lookup failed for '{name}': {e}")

    return None


@lru_cache(maxsize=512)
def resolve_name_to_smiles(name: str) -> Optional[str]:
    """
    Resolve a chemical name to SMILES.
    1. Check built-in dictionary
    2. Fall back to PubChem API
    """
    key = name.lower().strip()

    # Check dictionary
    if key in CHEMICAL_SMILES_DB:
        return CHEMICAL_SMILES_DB[key]

    # Try PubChem
    smiles = _resolve_via_pubchem(name)
    if smiles:
        # Cache for future
        CHEMICAL_SMILES_DB[key] = smiles
    return smiles


# ============================================================
# Chemical Name Extraction from Text
# ============================================================

def _build_name_patterns() -> re.Pattern:
    """Build a compiled regex of all dictionary chemical names, longest first."""
    names = sorted(CHEMICAL_SMILES_DB.keys(), key=len, reverse=True)
    escaped = [re.escape(n) for n in names]
    pattern = r'\b(' + '|'.join(escaped) + r')\b'
    return re.compile(pattern, re.IGNORECASE)

_NAME_PATTERN: Optional[re.Pattern] = None

def _get_name_pattern() -> re.Pattern:
    global _NAME_PATTERN
    if _NAME_PATTERN is None:
        _NAME_PATTERN = _build_name_patterns()
    return _NAME_PATTERN


def extract_chemicals_from_text(text: str, use_pubchem: bool = True) -> List[Dict[str, str]]:
    """
    Extract all chemical names from text and resolve to SMILES.

    Returns list of dicts: [{"name": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}, ...]
    Deduplicates by SMILES.
    """
    results: Dict[str, Dict[str, str]] = {}  # smiles → {name, smiles}

    # 1. Look for known names from dictionary
    pattern = _get_name_pattern()
    for match in pattern.finditer(text):
        name = match.group(0).lower().strip()
        smiles = CHEMICAL_SMILES_DB.get(name)
        if smiles and smiles not in results:
            results[smiles] = {"name": match.group(0), "smiles": smiles}

    # 2. Look for explicit SMILES-like strings in text
    for match in SMILES_PATTERN.finditer(text):
        candidate = match.group(1)
        # Validate with RDKit if available
        try:
            from rdkit import Chem
            from rdkit import RDLogger as _RDLogger
            _RDLogger.DisableLog('rdApp.*')  # suppress parse error spam
            mol = Chem.MolFromSmiles(candidate)
            _RDLogger.EnableLog('rdApp.error')
            if mol and mol.GetNumAtoms() >= 2:
                canonical = Chem.MolToSmiles(mol)
                if canonical not in results:
                    results[canonical] = {"name": candidate, "smiles": canonical}
        except ImportError:
            # Without RDKit, just skip SMILES detection
            pass

    # 3. If nothing found, try harder with common chemical word patterns and PubChem
    if not results and use_pubchem:
        # Look for capitalized multi-word chemical-ish names
        chem_candidates = re.findall(
            r'\b([A-Z][a-z]+(?:[ -][a-z]+)*(?:ine|ide|ene|ane|one|ate|ol|al|acid|amine|amide|azole|oxin|mycin|cillin|phen|fen|mab|nib|zole|pril|artan|olol|statin|gliptin|tide|dronate|prazole|vir|navir))\b',
            text
        )
        for candidate in chem_candidates:
            smiles = resolve_name_to_smiles(candidate)
            if smiles and smiles not in results:
                results[smiles] = {"name": candidate, "smiles": smiles}

    return list(results.values())


def extract_first_smiles(text: str, use_pubchem: bool = True) -> Optional[str]:
    """
    Extract the first/primary chemical from text and return its SMILES.
    Used when the pipeline needs a single molecule for GNN processing.
    """
    chemicals = extract_chemicals_from_text(text, use_pubchem=use_pubchem)
    if chemicals:
        return chemicals[0]["smiles"]
    return None


def extract_all_smiles(text: str, use_pubchem: bool = True) -> List[Tuple[str, str]]:
    """
    Extract all chemicals from text.
    Returns list of (name, smiles) tuples.
    """
    chemicals = extract_chemicals_from_text(text, use_pubchem=use_pubchem)
    return [(c["name"], c["smiles"]) for c in chemicals]


# ============================================================
# Generate molecule images (base64-encoded) for inline display
# ============================================================

def generate_molecule_image_base64(
    smiles: str,
    size: Tuple[int, int] = (300, 200),
    fmt: str = "png"
) -> Optional[str]:
    """
    Generate a base64-encoded molecule image from SMILES.
    Returns data URI string: "data:image/png;base64,..."
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from io import BytesIO
        import base64

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        img = Draw.MolToImage(mol, size=size)
        buf = BytesIO()
        img.save(buf, format=fmt.upper())
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/{fmt};base64,{b64}"
    except Exception as e:
        logger.error(f"Error generating molecule image for {smiles}: {e}")
        return None


def generate_molecule_svg(smiles: str, size: Tuple[int, int] = (350, 250)) -> Optional[str]:
    """Generate SVG of a molecule from SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw, AllChem
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as e:
        logger.error(f"Error generating SVG for {smiles}: {e}")
        return None
