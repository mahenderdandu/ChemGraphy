import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import rdChemReactions
from rdkit.Chem import AllChem

def get_molecule_properties(mol):
    """Calculate basic molecule properties."""
    return {
        "Molecular Weight": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "Num H Donors": Descriptors.NumHDonors(mol),
        "Num H Acceptors": Descriptors.NumHAcceptors(mol),
        "Num Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
        "TPSA": Descriptors.TPSA(mol)
    }

def visualize_molecule(smiles, key_prefix):
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
                    st.write(f"**{k}:** {v:.3f}")
        else:
            st.error("Invalid SMILES input")
    except Exception as e:
        st.error(f"Error: {e}")

def main():
    st.set_page_config(page_title="Chemoinformatics App", layout="wide")
    
    st.title("🧪 chemical.ai")
    
    tab1, tab2, tab3 = st.tabs(["Single Molecule", "Reaction Visualizer", "Run Reaction"])

    # --- TAB 1: Single Molecule ---
    with tab1:
        st.header("Single Molecule Viewer")
        smiles_input = st.text_input("Enter SMILES String", value="CC(=O)OC1=CC=CC=C1C(=O)O", key="mol_input")
        if st.button("Analyze Molecule", key="btn_mol"):
            visualize_molecule(smiles_input, "tab1")

    # --- TAB 2: Reaction Visualizer (Existing Reaction SMILES) ---
    with tab2:
        st.header("Reaction System Visualizer")
        st.info("Visualize a full reaction string (Reactants >> Products)")
        rxn_string = st.text_input("Enter Reaction SMILES", value="CC(=O)O.OCC>>CC(=O)OCC.O", key="rxn_input")
        
        if st.button("Visualize Reaction", key="btn_viz_rxn"):
            if rxn_string:
                try:
                    rxn = rdChemReactions.ReactionFromSmarts(rxn_string, useSmiles=True)
                    if rxn:
                        st.subheader("Reaction Scheme")
                        img = Draw.ReactionToImage(rxn, subImgSize=(300, 200))
                        st.image(img, use_column_width=True)
                        
                        st.divider()
                        st.subheader("Component Analysis")
                        
                        # Split reactants and products
                        parts = rxn_string.split(">>")
                        if len(parts) == 2:
                            reactants = parts[0].split(".")
                            products = parts[1].split(".")
                            
                            r_col, p_col = st.columns(2)
                            with r_col:
                                st.markdown("#### Reactants")
                                for i, r in enumerate(reactants):
                                    with st.expander(f"Reactant {i+1}: {r}"):
                                        visualize_molecule(r, key_prefix=f"r_{i}")
                            
                            with p_col:
                                st.markdown("#### Products")
                                for i, p in enumerate(products):
                                    with st.expander(f"Product {i+1}: {p}"):
                                        visualize_molecule(p, key_prefix=f"p_{i}")
                except Exception as e:
                    st.error(f"Could not parse reaction: {e}")

    # --- TAB 3: Run Unknown Reaction ---
    with tab3:
        st.header("Run/Predict Reaction")
        st.write("Provide reactants and a reaction rule (SMIRKS) to generate products.")
        
        c1, c2 = st.columns(2)
        with c1:
            reactants_input = st.text_input("Reactants (dot separated SMILES)", value="CC(=O)O.CN")
        with c2:
            # Default is Amide coupling
            smirks_input = st.text_input("Reaction SMIRKS (Pattern)", value="[C:1](=[O:2])[OH1:3].[N!H0:4]>>[C:1](=[O:2])[N:4].[OH2:3]")

        if st.button("Run Reaction", key="btn_run_rxn"):
            try:
                # 1. Parse Reactants
                reactant_smiles_list = reactants_input.split(".")
                mols = [Chem.MolFromSmiles(s) for s in reactant_smiles_list if s.strip()]
                
                if None in mols:
                    st.error("Invalid reactant SMILES provided.")
                else:
                    # 2. Parse Reaction Rule
                    rxn = rdChemReactions.ReactionFromSmarts(smirks_input)
                    
                    # 3. Run
                    # RunReactants expects a tuple of reactants
                    results = rxn.RunReactants(tuple(mols))
                    
                    if not results:
                        st.warning("No reaction occurred! Check if reactants match the SMIRKS logic.")
                    else:
                        st.success(f"Generated {len(results)} possible product sets.")
                        
                        for i, product_set in enumerate(results):
                            st.divider()
                            st.markdown(f"**Outcome Set {i+1}**")
                            cols = st.columns(len(product_set))
                            for idx, mol in enumerate(product_set):
                                Chem.SanitizeMol(mol)
                                p_smiles = Chem.MolToSmiles(mol)
                                with cols[idx]:
                                    st.image(Draw.MolToImage(mol, size=(200,200)), caption=p_smiles, use_column_width=True)
                                    props = get_molecule_properties(mol)
                                    st.caption(f"MW: {props['Molecular Weight']:.2f}")

            except Exception as e:
                st.error(f"Error running reaction: {e}")

if __name__ == "__main__":
    main()
