#!/usr/bin/env python3
"""
Helios Chemistry Lab - Standalone Streamlit App
Molecule analysis, reaction visualization, and reaction prediction tools

Usage:
    streamlit run scripts/chemistry_lab.py
"""

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdChemReactions, AllChem
from typing import Dict, Any, List, Optional, Tuple

# Import Helios utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from helios.utils.reaction_chemistry import (
    get_reaction_processor,
    visualize_molecule_with_properties,
    check_lipinski_rules,
    check_veber_rules,
    COMMON_REACTIONS
)


def get_molecule_properties(mol) -> Dict[str, float]:
    """Calculate comprehensive molecule properties."""
    return {
        "Molecular Weight": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "H-Bond Donors": Descriptors.NumHDonors(mol),
        "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
        "TPSA": Descriptors.TPSA(mol),
        "Aromatic Rings": Descriptors.NumAromaticRings(mol),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Ring Count": Descriptors.RingCount(mol),
        "Fraction SP3": Descriptors.FractionCSP3(mol),
        "Formal Charge": Chem.rdmolops.GetFormalCharge(mol)
    }


def visualize_molecule(smiles: str, key_prefix: str):
    """Helper to visualize and show properties of a single molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Structure")
                img = Draw.MolToImage(mol, size=(300, 300))
                st.image(img, use_column_width=True)
            with col2:
                st.subheader("Properties")
                props = get_molecule_properties(mol)
                st.write(f"**SMILES:** `{smiles}`")
                for k, v in props.items():
                    if isinstance(v, float):
                        st.write(f"**{k}:** {v:.3f}")
                    else:
                        st.write(f"**{k}:** {v}")
        else:
            st.error("Invalid SMILES input")
    except Exception as e:
        st.error(f"Error: {e}")


def show_drug_likeness(smiles: str):
    """Display drug-likeness assessment."""
    lipinski = check_lipinski_rules(smiles)
    veber = check_veber_rules(smiles)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Lipinski's Rule of Five")
        if lipinski.get('valid'):
            status = "✅ Drug-like" if lipinski['drug_like'] else "❌ Not drug-like"
            st.write(f"**Status:** {status} ({lipinski['violations']} violation(s))")
            
            for rule, data in lipinski['rules'].items():
                icon = "✅" if data['pass'] else "❌"
                st.write(f"{icon} {rule.replace('_', ' ').title()}: {data['value']:.2f} (limit: ≤{data['limit']})")
        else:
            st.error(f"Error: {lipinski.get('error')}")
    
    with col2:
        st.subheader("Veber Rules (Oral Bioavailability)")
        if veber.get('valid'):
            status = "✅ Good" if veber['bioavailable'] else "❌ Poor"
            st.write(f"**Status:** {status}")
            
            for rule, data in veber['rules'].items():
                icon = "✅" if data['pass'] else "❌"
                st.write(f"{icon} {rule.replace('_', ' ').title()}: {data['value']:.2f} (limit: ≤{data['limit']})")
        else:
            st.error(f"Error: {veber.get('error')}")


def main():
    st.set_page_config(
        page_title="Helios Chemistry Lab",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Helios Chemistry Lab")
    st.markdown("*Advanced molecule analysis and reaction prediction powered by RDKit*")
    
    # Initialize reaction processor
    rxn_processor = get_reaction_processor()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧬 Molecule Analyzer",
        "⚗️ Reaction Visualizer", 
        "🔮 Reaction Predictor",
        "📚 Reaction Templates"
    ])

    # ============ TAB 1: Molecule Analyzer ============
    with tab1:
        st.header("Single Molecule Analyzer")
        st.markdown("Enter a SMILES string to analyze molecular properties and drug-likeness.")
        
        smiles_input = st.text_input(
            "Enter SMILES String",
            value="CC(=O)OC1=CC=CC=C1C(=O)O",
            key="mol_input",
            help="e.g., Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_btn = st.button("🔬 Analyze Molecule", key="btn_mol", type="primary")
        
        if analyze_btn and smiles_input:
            st.divider()
            visualize_molecule(smiles_input, "tab1")
            st.divider()
            show_drug_likeness(smiles_input)
        
        # Example molecules
        st.divider()
        st.subheader("Example Molecules")
        examples = {
            "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "Acetaminophen": "CC(=O)Nc1ccc(O)cc1",
            "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "Metformin": "CN(C)C(=N)NC(=N)N"
        }
        
        cols = st.columns(len(examples))
        for i, (name, smi) in enumerate(examples.items()):
            with cols[i]:
                if st.button(name, key=f"ex_{name}"):
                    st.session_state['mol_input'] = smi
                    st.rerun()

    # ============ TAB 2: Reaction Visualizer ============
    with tab2:
        st.header("Reaction System Visualizer")
        st.markdown("Visualize a complete reaction and analyze all components.")
        st.info("**Format:** Reactants >> Products (e.g., `CC(=O)O.OCC>>CC(=O)OCC.O`)")
        
        rxn_string = st.text_input(
            "Enter Reaction SMILES",
            value="CC(=O)O.OCC>>CC(=O)OCC.O",
            key="rxn_input"
        )
        
        if st.button("🔬 Visualize Reaction", key="btn_viz_rxn", type="primary"):
            if rxn_string:
                try:
                    # Visualize reaction
                    rxn_img = rxn_processor.visualize_reaction(rxn_string)
                    if rxn_img:
                        st.subheader("Reaction Scheme")
                        st.image(rxn_img, use_column_width=True)
                    
                    # Analyze components
                    analysis = rxn_processor.analyze_reaction_components(rxn_string)
                    
                    if analysis:
                        st.divider()
                        st.subheader("Component Analysis")
                        
                        r_col, p_col = st.columns(2)
                        
                        with r_col:
                            st.markdown("#### Reactants")
                            for i, r in enumerate(analysis['reactants']):
                                with st.expander(f"Reactant {i+1}: {r['smiles']}"):
                                    visualize_molecule(r['smiles'], key_prefix=f"r_{i}")
                        
                        with p_col:
                            st.markdown("#### Products")
                            for i, p in enumerate(analysis['products']):
                                with st.expander(f"Product {i+1}: {p['smiles']}"):
                                    visualize_molecule(p['smiles'], key_prefix=f"p_{i}")
                        
                        # Show deltas
                        st.divider()
                        st.subheader("Transformation Summary")
                        delta_col1, delta_col2 = st.columns(2)
                        with delta_col1:
                            st.metric("ΔMolecular Weight", f"{analysis['delta_mw']:+.2f}")
                        with delta_col2:
                            st.metric("ΔLogP", f"{analysis['delta_logp']:+.2f}")
                            
                except Exception as e:
                    st.error(f"Could not parse reaction: {e}")

    # ============ TAB 3: Reaction Predictor ============
    with tab3:
        st.header("Reaction Predictor")
        st.markdown("Provide reactants and a reaction rule to predict products.")
        
        c1, c2 = st.columns(2)
        with c1:
            reactants_input = st.text_input(
                "Reactants (dot-separated SMILES)",
                value="CC(=O)O.CN",
                help="Separate multiple reactants with dots"
            )
        
        with c2:
            reaction_type = st.selectbox(
                "Reaction Type",
                options=["custom"] + list(COMMON_REACTIONS.keys()),
                format_func=lambda x: COMMON_REACTIONS[x]['name'] if x in COMMON_REACTIONS else "Custom SMIRKS"
            )
        
        if reaction_type == "custom":
            smirks_input = st.text_input(
                "Custom SMIRKS Pattern",
                value="[C:1](=[O:2])[OH1:3].[N!H0:4]>>[C:1](=[O:2])[N:4].[OH2:3]"
            )
        else:
            smirks_input = COMMON_REACTIONS[reaction_type]['smirks']
            st.info(f"**{COMMON_REACTIONS[reaction_type]['name']}:** {COMMON_REACTIONS[reaction_type]['description']}")
            st.code(smirks_input, language=None)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            run_btn = st.button("🚀 Run Reaction", key="btn_run_rxn", type="primary")
        with col2:
            suggest_btn = st.button("💡 Suggest Reactions", key="btn_suggest")
        
        if suggest_btn and reactants_input:
            st.divider()
            st.subheader("Suggested Reactions")
            reactant_list = [r.strip() for r in reactants_input.split(".") if r.strip()]
            suggestions = rxn_processor.suggest_reaction(reactant_list)
            
            if suggestions:
                for key, desc, confidence in suggestions:
                    rxn_info = COMMON_REACTIONS.get(key, {})
                    st.markdown(f"**{rxn_info.get('name', key)}** (confidence: {confidence:.0%})")
                    st.markdown(f"- {desc}")
                    st.code(rxn_info.get('smirks', 'N/A'), language=None)
            else:
                st.warning("No common reactions suggested for these reactants.")

        if run_btn and reactants_input:
            try:
                reactant_list = [r.strip() for r in reactants_input.split(".") if r.strip()]
                
                # Validate reactants
                mols = [Chem.MolFromSmiles(s) for s in reactant_list]
                if None in mols:
                    st.error("Invalid reactant SMILES provided.")
                else:
                    result = rxn_processor.run_reaction(reactant_list, smirks_input)
                    
                    if not result.success:
                        st.warning(f"⚠️ {result.message}")
                    else:
                        st.success(f"✅ {result.message}")
                        
                        for i, product_set in enumerate(result.product_sets):
                            st.divider()
                            st.markdown(f"**Outcome Set {i+1}**")
                            cols = st.columns(min(len(product_set), 4))
                            
                            for idx, p_smiles in enumerate(product_set):
                                mol = Chem.MolFromSmiles(p_smiles)
                                if mol:
                                    with cols[idx % len(cols)]:
                                        st.image(
                                            Draw.MolToImage(mol, size=(200, 200)),
                                            caption=p_smiles,
                                            use_column_width=True
                                        )
                                        props = get_molecule_properties(mol)
                                        st.caption(f"MW: {props['Molecular Weight']:.2f}")
                                        st.caption(f"LogP: {props['LogP']:.2f}")

            except Exception as e:
                st.error(f"Error running reaction: {e}")

    # ============ TAB 4: Reaction Templates ============
    with tab4:
        st.header("Available Reaction Templates")
        st.markdown("Reference library of common organic reactions with SMIRKS patterns.")
        
        for key, info in COMMON_REACTIONS.items():
            with st.expander(f"**{info['name']}** (`{key}`)"):
                st.markdown(f"**Description:** {info['description']}")
                st.code(info['smirks'], language=None)
                
                # Test button
                if st.button(f"Test {key}", key=f"test_{key}"):
                    st.session_state['test_reaction'] = key
        
        st.divider()
        st.subheader("Understanding SMIRKS")
        st.markdown("""
        **SMIRKS** is a reaction transform language based on SMILES/SMARTS:
        
        - `[C:1]` - Atom mapping (`:1` is the mapping number)
        - `>>` - Separates reactants from products
        - `[OH1]` - SMARTS for hydroxyl with exactly 1 hydrogen
        - `[N!H0]` - Nitrogen with at least one hydrogen
        
        **Example:** Amide coupling
        ```
        [C:1](=[O:2])[OH1:3].[N!H0:4]>>[C:1](=[O:2])[N:4].[OH2:3]
        ```
        - Carboxylic acid + Amine → Amide + Water
        """)


if __name__ == "__main__":
    main()
