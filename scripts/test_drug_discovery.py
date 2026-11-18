#!/usr/bin/env python3
"""Quick integration test for drug discovery modules"""
import sys
sys.path.insert(0, 'src')

from helios.utils.drug_discovery import (
    predict_admet, assess_drug_likeness, get_structural_alerts,
    identify_pharmacophores, suggest_lead_optimization,
    estimate_synthetic_accessibility, generate_full_report
)
from helios.utils.figure_generator import generate_all_figures

smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'  # Aspirin
print(f"Testing with Aspirin: {smiles}\n")

admet = predict_admet(smiles)
print(f"✓ ADMET prediction: HIA class = {admet['absorption']['hia_class']}")

dl = assess_drug_likeness(smiles)
print(f"✓ Drug-likeness: QED = {dl['qed_score']:.3f}, Lipinski pass = {dl['lipinski']['pass']}")

alerts = get_structural_alerts(smiles)
print(f"✓ Structural alerts: {alerts['total_alerts']} alerts")

pharm = identify_pharmacophores(smiles)
print(f"✓ Pharmacophores: {pharm['total_pharmacophore_features']} features")

opt = suggest_lead_optimization(smiles)
print(f"✓ Lead optimization: {len(opt['priority_actions'])} actions")

sa = estimate_synthetic_accessibility(smiles)
print(f"✓ SA: score={sa['sa_score']:.1f}, category={sa['category']}")

report = generate_full_report(smiles)
print(f"✓ Full report: {len(report)} keys: {list(report.keys())}")

# Test figure generation
figures = generate_all_figures(
    smiles=smiles,
    molecule_name='Aspirin',
    admet_data=admet,
    drug_likeness_data=dl,
    alerts_data=alerts,
    pharm_data=pharm,
    opt_data=opt,
)
print(f"\n✓ Generated {len(figures)} figures:")
for name, url in figures.items():
    print(f"  - {name}: {len(url)} chars (starts with {url[:30]}...)")

print("\n=== ALL TESTS PASSED ===")
