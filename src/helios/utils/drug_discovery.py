"""
Drug Discovery Toolkit for Helios
Comprehensive ADMET, drug-likeness, lead optimization, and toxicity analysis

This module provides pharmaceutical-grade molecular analysis including:
- ADMET property prediction (Absorption, Distribution, Metabolism, Excretion, Toxicity)
- Multi-rule drug-likeness scoring (Lipinski, Veber, Ghose, Egan, Muegge)
- Lead-likeness and fragment-likeness assessment
- Structural alerts (PAINS, Brenk, toxicophores)
- Pharmacophore feature identification
- Lead optimization suggestions with bioisostere recommendations
- Synthetic accessibility scoring
- Bioavailability scoring
"""

import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import (
    Descriptors, AllChem, Fragments, rdMolDescriptors,
    FilterCatalog, rdchem, Lipinski, QED
)
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from rdkit import DataStructs
import numpy as np

logger = logging.getLogger(__name__)


# ==========================================================================
# ADMET Prediction
# ==========================================================================

def predict_admet(smiles: str) -> Optional[Dict[str, Any]]:
    """
    Predict ADMET properties using RDKit descriptors and empirical rules.
    Returns a rich dictionary of absorption, distribution, metabolism,
    excretion, and toxicity indicators.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    mr = Descriptors.MolMR(mol)
    num_rings = Descriptors.RingCount(mol)

    # --- Absorption ---
    # Lipinski absorption rules
    good_absorption = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
    # Estimated human intestinal absorption (HIA) heuristic
    hia_score = max(0, min(100, 100 - 0.1 * max(0, tpsa - 140) - 0.5 * max(0, mw - 500)))
    # Caco-2 permeability estimate (log cm/s)
    caco2_est = -5.5 + 0.006 * (140 - min(tpsa, 140)) + 0.01 * min(logp, 5) - 0.008 * max(0, rot_bonds - 5)
    # Oral bioavailability heuristic
    oral_bioavail = hia_score > 70 and tpsa <= 140 and rot_bonds <= 10

    # --- Distribution ---
    # Blood-Brain Barrier penetration (Egan model)
    bbb_penetrant = (tpsa <= 90 and logp >= -0.5 and logp <= 4.0 and mw <= 450)
    # Plasma protein binding estimate
    ppb_est = min(99.5, max(10, 50 + 10 * logp - 0.1 * tpsa))
    # Volume of distribution estimate (rough empirical)
    vd_est = 0.04 + 0.2 * logp - 0.01 * tpsa + 0.001 * mw

    # --- Metabolism ---
    # CYP substrate likelihood heuristics
    cyp_flags = {}
    cyp_flags["CYP3A4_substrate_likely"] = logp > 1 and mw > 300
    cyp_flags["CYP2D6_substrate_likely"] = (
        mol.HasSubstructMatch(Chem.MolFromSmarts("[NH]")) and logp > 0.5
    )
    cyp_flags["CYP2C9_substrate_likely"] = logp > 2 and tpsa < 80
    # Metabolic stability heuristic
    metabolic_stability = "High" if (logp < 3 and mw < 400 and rot_bonds < 7) else (
        "Moderate" if (logp < 5 and mw < 500) else "Low"
    )

    # --- Excretion ---
    # Renal clearance heuristic
    renal_clearance = "High" if (mw < 350 and logp < 1.5 and tpsa > 60) else (
        "Moderate" if mw < 500 else "Low"
    )
    # Half-life estimate bucket
    half_life_class = (
        "Short (<4h)" if (logp < 1 and mw < 300) else
        "Medium (4-12h)" if (logp < 3 and mw < 500) else
        "Long (>12h)"
    )

    # --- Toxicity ---
    alerts = get_structural_alerts(smiles)
    pains_count = alerts.get("pains_count", 0)
    brenk_count = alerts.get("brenk_count", 0)
    tox_risk = "Low" if (pains_count == 0 and brenk_count == 0) else (
        "Medium" if (pains_count + brenk_count <= 2) else "High"
    )
    # hERG liability heuristic (cardiotoxicity)
    herg_risk = logp > 3.7 and mw > 350 and any(
        mol.HasSubstructMatch(Chem.MolFromSmarts(p))
        for p in ["[NH]", "c1ccncc1", "N1CCN(CC1)"]
        if Chem.MolFromSmarts(p) is not None
    )
    # Ames mutagenicity rough proxy
    ames_risk = any(
        mol.HasSubstructMatch(Chem.MolFromSmarts(s))
        for s in ["[N+](=O)[O-]", "[NH]N=O", "c1cc[nH+]cc1"]
        if Chem.MolFromSmarts(s) is not None
    )

    return {
        "absorption": {
            "human_intestinal_absorption": round(hia_score, 1),
            "hia_class": "High" if hia_score > 70 else ("Moderate" if hia_score > 40 else "Low"),
            "caco2_permeability_est": round(caco2_est, 2),
            "oral_bioavailability": oral_bioavail,
            "lipinski_compliant": good_absorption,
            "pgp_substrate_likely": tpsa > 120 or mw > 600,
        },
        "distribution": {
            "bbb_penetrant": bbb_penetrant,
            "plasma_protein_binding_pct": round(ppb_est, 1),
            "vd_est_l_kg": round(max(0.04, vd_est), 2),
        },
        "metabolism": {
            **cyp_flags,
            "metabolic_stability": metabolic_stability,
        },
        "excretion": {
            "renal_clearance": renal_clearance,
            "half_life_class": half_life_class,
        },
        "toxicity": {
            "overall_risk": tox_risk,
            "pains_alerts": pains_count,
            "brenk_alerts": brenk_count,
            "herg_liability": herg_risk,
            "ames_mutagenicity_flag": ames_risk,
            "structural_alert_details": alerts.get("details", []),
        },
        "descriptors_used": {
            "mw": round(mw, 2),
            "logp": round(logp, 2),
            "tpsa": round(tpsa, 2),
            "hbd": hbd,
            "hba": hba,
            "rotatable_bonds": rot_bonds,
            "aromatic_rings": aromatic_rings,
            "heavy_atoms": heavy_atoms,
            "fsp3": round(fsp3, 3),
            "molar_refractivity": round(mr, 2),
            "num_rings": num_rings,
        },
    }


# ==========================================================================
# Multi-Rule Drug-Likeness
# ==========================================================================

def assess_drug_likeness(smiles: str) -> Optional[Dict[str, Any]]:
    """
    Comprehensive drug-likeness assessment using multiple rule systems.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    mr = Descriptors.MolMR(mol)
    num_rings = Descriptors.RingCount(mol)

    # QED (Quantitative Estimate of Drug-likeness)
    qed_score = QED.qed(mol)

    # Lipinski Rule of 5
    lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    lipinski = {
        "pass": lipinski_violations <= 1,
        "violations": lipinski_violations,
        "rules": {
            "MW <= 500": {"value": round(mw, 1), "pass": mw <= 500},
            "LogP <= 5": {"value": round(logp, 2), "pass": logp <= 5},
            "HBD <= 5": {"value": hbd, "pass": hbd <= 5},
            "HBA <= 10": {"value": hba, "pass": hba <= 10},
        },
    }

    # Veber rules (oral bioavailability)
    veber = {
        "pass": rot_bonds <= 10 and tpsa <= 140,
        "rules": {
            "RotBonds <= 10": {"value": rot_bonds, "pass": rot_bonds <= 10},
            "TPSA <= 140": {"value": round(tpsa, 1), "pass": tpsa <= 140},
        },
    }

    # Ghose filter
    ghose_pass = (-0.4 <= logp <= 5.6 and 160 <= mw <= 480 and
                  40 <= mr <= 130 and 20 <= heavy_atoms <= 70)
    ghose = {
        "pass": ghose_pass,
        "rules": {
            "-0.4 <= LogP <= 5.6": {"value": round(logp, 2), "pass": -0.4 <= logp <= 5.6},
            "160 <= MW <= 480": {"value": round(mw, 1), "pass": 160 <= mw <= 480},
            "40 <= MR <= 130": {"value": round(mr, 1), "pass": 40 <= mr <= 130},
            "20 <= Atoms <= 70": {"value": heavy_atoms, "pass": 20 <= heavy_atoms <= 70},
        },
    }

    # Egan filter (bioavailability)
    egan = {
        "pass": tpsa <= 132 and logp <= 5.88,
        "rules": {
            "TPSA <= 132": {"value": round(tpsa, 1), "pass": tpsa <= 132},
            "LogP <= 5.88": {"value": round(logp, 2), "pass": logp <= 5.88},
        },
    }

    # Muegge filter
    muegge_pass = (200 <= mw <= 600 and -2 <= logp <= 5 and tpsa <= 150 and
                   num_rings <= 7 and hba <= 10 and hbd <= 5 and rot_bonds <= 15 and heavy_atoms >= 5)
    muegge = {
        "pass": muegge_pass,
        "rules": {
            "200 <= MW <= 600": {"value": round(mw, 1), "pass": 200 <= mw <= 600},
            "-2 <= LogP <= 5": {"value": round(logp, 2), "pass": -2 <= logp <= 5},
            "TPSA <= 150": {"value": round(tpsa, 1), "pass": tpsa <= 150},
            "Rings <= 7": {"value": num_rings, "pass": num_rings <= 7},
        },
    }

    # Lead-likeness (optimized for lead optimization)
    lead_like = {
        "pass": mw <= 350 and logp <= 3.5 and rot_bonds <= 7 and hbd <= 3 and hba <= 7,
        "rules": {
            "MW <= 350": {"value": round(mw, 1), "pass": mw <= 350},
            "LogP <= 3.5": {"value": round(logp, 2), "pass": logp <= 3.5},
            "RotBonds <= 7": {"value": rot_bonds, "pass": rot_bonds <= 7},
        },
    }

    # Fragment-likeness (Rule of 3 for fragment-based drug discovery)
    frag_like = {
        "pass": mw <= 300 and logp <= 3 and hbd <= 3 and hba <= 3 and rot_bonds <= 3,
        "rules": {
            "MW <= 300": {"value": round(mw, 1), "pass": mw <= 300},
            "LogP <= 3": {"value": round(logp, 2), "pass": logp <= 3},
            "HBD <= 3": {"value": hbd, "pass": hbd <= 3},
            "HBA <= 3": {"value": hba, "pass": hba <= 3},
        },
    }

    # Overall score
    filters_passed = sum([
        lipinski["pass"], veber["pass"], ghose["pass"],
        egan["pass"], muegge["pass"],
    ])
    overall_score = round((filters_passed / 5) * 100, 1)

    return {
        "qed_score": round(qed_score, 3),
        "overall_drug_likeness_pct": overall_score,
        "filters_passed": f"{filters_passed}/5",
        "lipinski": lipinski,
        "veber": veber,
        "ghose": ghose,
        "egan": egan,
        "muegge": muegge,
        "lead_likeness": lead_like,
        "fragment_likeness": frag_like,
    }


# ==========================================================================
# Structural Alerts & Toxicophores
# ==========================================================================

# PAINS patterns (Pan-Assay Interference Compounds) - key patterns
PAINS_SMARTS = [
    ("[#6]1:[#6]:[#6](:[#6]:[#6]:[#6]:1)-[#6]=[#6]-[#6]=[#8]", "Michael acceptor (acrylate)"),
    ("c1cc(/C=C/C=O)ccc1", "Cinnamaldehyde-type"),
    ("c1ccc2c(c1)[nH]c1ccccc12", "Carbazole"),
    ("[#16]-[#16]", "Disulfide"),
    ("[#6](=[#8])([#6])[#16]", "Thioester"),
    ("[N;R1]=[N;R1]", "Azo compound"),
    ("c1ccnc(SSc2ncccc2)c1", "2,2'-Bipyridyl disulfide"),
]

# Brenk structural alert patterns
BRENK_SMARTS = [
    ("[S-]", "Thiol anion"),
    ("[SH]", "Free thiol"),
    ("C#N", "Nitrile (reactive)"),
    ("[N+](=O)[O-]", "Nitro group"),
    ("C(=O)Cl", "Acyl chloride"),
    ("[P](=O)([OH])([OH])", "Phosphoric acid"),
    ("OS(=O)(=O)O", "Sulfate ester"),
    ("C=CC(=O)", "Michael acceptor"),
    ("c1ccc([N+](=O)[O-])cc1", "Aromatic nitro"),
    ("[NH]N=O", "Nitrosamine"),
    ("[N;X2]=[N;X2]", "Diazo compound"),
    ("ClC(Cl)Cl", "Chloroform-like"),
]


def get_structural_alerts(smiles: str) -> Dict[str, Any]:
    """
    Screen molecule for PAINS filters, Brenk alerts, and toxicophores.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"pains_count": 0, "brenk_count": 0, "details": []}

    details = []
    pains_count = 0
    brenk_count = 0

    for smarts, name in PAINS_SMARTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            pains_count += 1
            details.append({"type": "PAINS", "alert": name, "pattern": smarts})

    for smarts, name in BRENK_SMARTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            brenk_count += 1
            details.append({"type": "Brenk", "alert": name, "pattern": smarts})

    # Use RDKit FilterCatalog if available
    try:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog.FilterCatalog(params)
        entry = catalog.GetFirstMatch(mol)
        if entry is not None:
            desc = entry.GetDescription()
            if desc not in [d["alert"] for d in details]:
                pains_count += 1
                details.append({"type": "PAINS_catalog", "alert": desc, "pattern": ""})
    except Exception:
        pass

    return {
        "pains_count": pains_count,
        "brenk_count": brenk_count,
        "total_alerts": pains_count + brenk_count,
        "clean": pains_count + brenk_count == 0,
        "details": details,
    }


# ==========================================================================
# Pharmacophore Analysis
# ==========================================================================

PHARMACOPHORE_SMARTS = {
    "H-bond donor": "[!$([#1])][NH,$([NH2]),$([OH]),$([SH])]",
    "H-bond acceptor": "[$([O,S;H1;v2;!$(*-*=*)]),$([O,S;H0;v2]),$([#7;v3;!$([nH])]),$([nH])]",
    "Positive ionizable": "[$([NH2]),$([NH]([#6])[#6]),$([N]([#6])([#6])[#6]),n]",
    "Negative ionizable": "[$([OH])&$([CX3](=O)[OH]),$([SH](=O)(=O)),$(P([OH])(=O))]",
    "Aromatic ring": "a1aaaa1",
    "Hydrophobic": "[CH2,CH3,$([CH;!$([CH](-[OH])-[OH])])]",
    "Halogen bond donor": "[F,Cl,Br,I]",
}


def identify_pharmacophores(smiles: str) -> Optional[Dict[str, Any]]:
    """
    Identify pharmacophoric features in a molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = {}
    for name, smarts in PHARMACOPHORE_SMARTS.items():
        pat = Chem.MolFromSmarts(smarts)
        if pat:
            matches = mol.GetSubstructMatches(pat)
            features[name] = {
                "count": len(matches),
                "atom_indices": [list(m) for m in matches[:10]],  # limit
            }

    # Summarize
    total_features = sum(f["count"] for f in features.values())
    return {
        "features": features,
        "total_pharmacophore_features": total_features,
        "has_hb_donor": features.get("H-bond donor", {}).get("count", 0) > 0,
        "has_hb_acceptor": features.get("H-bond acceptor", {}).get("count", 0) > 0,
        "has_aromatic": features.get("Aromatic ring", {}).get("count", 0) > 0,
        "has_hydrophobic": features.get("Hydrophobic", {}).get("count", 0) > 0,
        "has_ionizable": (
            features.get("Positive ionizable", {}).get("count", 0) > 0 or
            features.get("Negative ionizable", {}).get("count", 0) > 0
        ),
    }


# ==========================================================================
# Lead Optimization Suggestions
# ==========================================================================

# Bioisostere replacement ideas
BIOISOSTERE_MAP = {
    "carboxylic_acid": {
        "smarts": "[CX3](=O)[OX2H1]",
        "suggestions": [
            "Tetrazole (improves metabolic stability)",
            "Acylsulfonamide (maintains acidity, better permeability)",
            "Hydroxamic acid (adds metalloenzyme binding)",
            "Phosphonic acid (increased polarity, target-specific)",
        ],
    },
    "amide": {
        "smarts": "[NX3][CX3](=[OX1])[#6]",
        "suggestions": [
            "N-methylamide (blocks metabolism at NH)",
            "Reversed amide (different H-bonding pattern)",
            "Urea (-NH-C(=O)-NH-, increased H-bonding)",
            "1,2,4-Oxadiazole (metabolically stable, rigid)",
        ],
    },
    "ester": {
        "smarts": "[CX3](=O)[OX2][CX4]",
        "suggestions": [
            "Amide (improved metabolic stability)",
            "Oxadiazole (bioisosteric replacement)",
            "Carbamate (retains H-bond acceptor capability)",
        ],
    },
    "phenol": {
        "smarts": "c1ccc(O)cc1",
        "suggestions": [
            "Indazole (reduce Phase II metabolism)",
            "Benzimidazole (maintain H-bonding, add basicity)",
            "Replace -OH with -F (reduce glucuronidation)",
        ],
    },
    "aniline": {
        "smarts": "c1ccc(N)cc1",
        "suggestions": [
            "Aminopyridine (reduce CYP metabolism, maintain basicity)",
            "Amide NH (remove aniline toxicity risk)",
            "Sulfonamide (replace NH2 with SO2NH2)",
        ],
    },
    "nitro_group": {
        "smarts": "[N+](=O)[O-]",
        "suggestions": [
            "Cyano group (-CN, reduced toxicity)",
            "Sulfonamide (less toxic, maintains polarity)",
            "Amide (eliminated mutagenicity risk)",
        ],
    },
}


def suggest_lead_optimization(smiles: str) -> Optional[Dict[str, Any]]:
    """
    Analyze a molecule and suggest lead optimization strategies.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    drug_likeness = assess_drug_likeness(smiles)
    admet = predict_admet(smiles)
    alerts = get_structural_alerts(smiles)

    suggestions = []
    priority_actions = []

    # Check bioisostere opportunities
    bioisostere_opportunities = []
    for group_name, info in BIOISOSTERE_MAP.items():
        pat = Chem.MolFromSmarts(info["smarts"])
        if pat and mol.HasSubstructMatch(pat):
            bioisostere_opportunities.append({
                "functional_group": group_name,
                "replacements": info["suggestions"],
            })

    # MW-based suggestions
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)

    if mw > 500:
        priority_actions.append({
            "issue": "High molecular weight",
            "value": f"{mw:.0f} Da",
            "action": "Remove non-essential substituents or use ring contraction strategies",
            "severity": "high",
        })
    if logp > 5:
        priority_actions.append({
            "issue": "High lipophilicity",
            "value": f"LogP = {logp:.2f}",
            "action": "Introduce polar groups (-OH, -NH2, =O) or replace C-H with C-F",
            "severity": "high",
        })
    if logp < 0:
        priority_actions.append({
            "issue": "Low lipophilicity",
            "value": f"LogP = {logp:.2f}",
            "action": "Add hydrophobic moieties (aromatic rings, alkyl chains) to improve permeability",
            "severity": "medium",
        })
    if tpsa > 140:
        priority_actions.append({
            "issue": "High polar surface area",
            "value": f"TPSA = {tpsa:.0f} A²",
            "action": "Reduce H-bond donors/acceptors or mask polar groups via prodrug strategy",
            "severity": "high",
        })

    if admet:
        if admet["toxicity"]["herg_liability"]:
            priority_actions.append({
                "issue": "hERG liability (cardiotoxicity risk)",
                "value": "Predicted positive",
                "action": "Reduce basicity, lower LogP, add polarity near basic nitrogen",
                "severity": "critical",
            })
        if admet["metabolism"]["metabolic_stability"] == "Low":
            priority_actions.append({
                "issue": "Low metabolic stability",
                "value": "Predicted unstable",
                "action": "Block metabolic soft spots with -F, -CF3, or cyclopropyl groups",
                "severity": "high",
            })
        if not admet["distribution"]["bbb_penetrant"]:
            suggestions.append("If CNS target: Reduce TPSA < 90, increase LogP, reduce MW < 450")

    if alerts["total_alerts"] > 0:
        priority_actions.append({
            "issue": "Structural alerts detected",
            "value": f"{alerts['total_alerts']} alerts",
            "action": "Remove PAINS/Brenk substructures to reduce assay interference and toxicity risk",
            "severity": "critical" if alerts["total_alerts"] > 2 else "high",
        })

    return {
        "smiles": smiles,
        "priority_actions": priority_actions,
        "bioisostere_opportunities": bioisostere_opportunities,
        "general_suggestions": suggestions,
        "current_profile": {
            "qed": drug_likeness["qed_score"] if drug_likeness else None,
            "drug_likeness_pct": drug_likeness["overall_drug_likeness_pct"] if drug_likeness else None,
            "admet_summary": {
                "absorption": admet["absorption"]["hia_class"] if admet else None,
                "bbb": admet["distribution"]["bbb_penetrant"] if admet else None,
                "metabolism": admet["metabolism"]["metabolic_stability"] if admet else None,
                "toxicity_risk": admet["toxicity"]["overall_risk"] if admet else None,
            } if admet else None,
        },
    }


# ==========================================================================
# Molecular Similarity
# ==========================================================================

def compute_similarity_profile(
    query_smiles: str,
    reference_smiles_list: List[str],
    fp_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[Dict[str, Any]]:
    """Compute Tanimoto similarity of query vs a list of references."""
    mol_q = Chem.MolFromSmiles(query_smiles)
    if mol_q is None:
        return None

    if fp_type == "morgan":
        fp_q = AllChem.GetMorganFingerprintAsBitVect(mol_q, radius, nBits=n_bits)
    else:
        fp_q = Chem.RDKFingerprint(mol_q, fpSize=n_bits)

    results = []
    for ref_smi in reference_smiles_list:
        mol_r = Chem.MolFromSmiles(ref_smi)
        if mol_r is None:
            continue
        if fp_type == "morgan":
            fp_r = AllChem.GetMorganFingerprintAsBitVect(mol_r, radius, nBits=n_bits)
        else:
            fp_r = Chem.RDKFingerprint(mol_r, fpSize=n_bits)
        sim = DataStructs.TanimotoSimilarity(fp_q, fp_r)
        results.append({"smiles": ref_smi, "tanimoto": round(sim, 4)})

    results.sort(key=lambda x: x["tanimoto"], reverse=True)
    return {"query": query_smiles, "similarities": results}


# ==========================================================================
# Synthetic Accessibility Score
# ==========================================================================

def estimate_synthetic_accessibility(smiles: str) -> Optional[Dict[str, Any]]:
    """
    Estimate synthetic accessibility using RDKit's SA Score.
    Score ranges from 1 (easy) to 10 (hard).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        from rdkit.Chem import RDConfig
        import os, sys
        sa_module_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
        if sa_module_path not in sys.path:
            sys.path.insert(0, sa_module_path)
        import sascorer
        sa_score = sascorer.calculateScore(mol)
    except Exception:
        # Fallback heuristic based on complexity
        ring_count = Descriptors.RingCount(mol)
        stereo_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        heavy = mol.GetNumHeavyAtoms()
        sa_score = 1 + 0.3 * ring_count + 0.5 * stereo_centers + 0.05 * heavy
        sa_score = min(10, max(1, sa_score))

    if sa_score <= 3:
        category = "Easy to synthesize"
    elif sa_score <= 5:
        category = "Moderately accessible"
    elif sa_score <= 7:
        category = "Challenging synthesis"
    else:
        category = "Very difficult to synthesize"

    return {
        "sa_score": round(sa_score, 2),
        "category": category,
        "scale": "1 (easy) to 10 (hard)",
    }


# ==========================================================================
# Full Drug Discovery Report
# ==========================================================================

def generate_full_report(smiles: str) -> Optional[Dict[str, Any]]:
    """
    Generate a comprehensive drug discovery report for a molecule.
    Combines all analyses into a single result.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    report = {
        "smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "admet": predict_admet(smiles),
        "drug_likeness": assess_drug_likeness(smiles),
        "structural_alerts": get_structural_alerts(smiles),
        "pharmacophores": identify_pharmacophores(smiles),
        "lead_optimization": suggest_lead_optimization(smiles),
        "synthetic_accessibility": estimate_synthetic_accessibility(smiles),
    }

    return report
