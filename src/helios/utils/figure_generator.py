"""
Textbook-Quality Figure Generator for Helios

Generates publication-ready, inline figures for LLM responses.
Figures are returned as base64-encoded data URIs for embedding in chat messages.

Figure types:
- Molecular structure diagrams with highlighted pharmacophores
- ADMET radar/spider charts
- Drug-likeness scorecard visuals
- Property comparison bar charts
- Bioavailability radar plots
- SAR (Structure-Activity Relationship) heatmaps
- Mechanism-of-action pathway diagrams
- Lipinski space plots
- Molecular property distribution histograms
"""

import logging
import base64
import io
import math
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Publication-quality style
HELIOS_STYLE = {
    "figure.facecolor": "#0f172a",  # slate-900
    "axes.facecolor": "#1e293b",    # slate-800
    "axes.edgecolor": "#475569",    # slate-600
    "axes.labelcolor": "#e2e8f0",   # slate-200
    "text.color": "#e2e8f0",
    "xtick.color": "#94a3b8",       # slate-400
    "ytick.color": "#94a3b8",
    "grid.color": "#334155",        # slate-700
    "grid.alpha": 0.5,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
}

# Color palette
COLORS = {
    "emerald": "#34d399",
    "blue": "#60a5fa",
    "purple": "#a78bfa",
    "pink": "#f472b6",
    "amber": "#fbbf24",
    "red": "#f87171",
    "teal": "#2dd4bf",
    "cyan": "#22d3ee",
    "slate": "#94a3b8",
    "green_pass": "#22c55e",
    "red_fail": "#ef4444",
}

RADAR_COLORS = [COLORS["emerald"], COLORS["blue"], COLORS["purple"], COLORS["pink"], COLORS["amber"]]


def _fig_to_base64(fig, dpi=150, fmt="png") -> str:
    """Convert matplotlib figure to base64 data URI string."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor='none',
                transparent=False)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt};base64,{b64}"


def _fig_to_svg_string(fig) -> str:
    """Convert matplotlib figure to SVG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor='none',
                transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue().decode("utf-8")


# ==========================================================================
# 1. ADMET Radar Chart
# ==========================================================================

def generate_admet_radar(admet_data: Dict[str, Any], molecule_name: str = "") -> str:
    """
    Generate a radar/spider chart showing ADMET profile.
    Returns base64 PNG data URI.
    """
    with plt.rc_context(HELIOS_STYLE):
        # Normalize ADMET properties to 0-1 scale for radar
        categories = [
            "Absorption",
            "Distribution",
            "Metabolism",
            "Excretion",
            "Safety"
        ]

        # Score each category 0-1
        scores = []

        # Absorption: HIA score
        hia = admet_data.get("absorption", {}).get("human_intestinal_absorption", 50)
        scores.append(hia / 100)

        # Distribution: inverse PPB (less binding = more free drug)
        ppb = admet_data.get("distribution", {}).get("plasma_protein_binding_pct", 80)
        scores.append(1 - ppb / 100)

        # Metabolism: stability score
        met_map = {"High": 0.9, "Moderate": 0.5, "Low": 0.2}
        met = admet_data.get("metabolism", {}).get("metabolic_stability", "Moderate")
        scores.append(met_map.get(met, 0.5))

        # Excretion: clearance balance
        cl_map = {"High": 0.4, "Moderate": 0.7, "Low": 0.9}
        cl = admet_data.get("excretion", {}).get("renal_clearance", "Moderate")
        scores.append(cl_map.get(cl, 0.5))

        # Safety: inverse tox risk
        tox_map = {"Low": 0.9, "Medium": 0.5, "High": 0.15}
        tox = admet_data.get("toxicity", {}).get("overall_risk", "Medium")
        scores.append(tox_map.get(tox, 0.5))

        # Create radar chart
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        scores_plot = scores + scores[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")

        # Draw the polygon
        ax.fill(angles, scores_plot, color=COLORS["emerald"], alpha=0.25)
        ax.plot(angles, scores_plot, color=COLORS["emerald"], linewidth=2.5, marker='o',
                markersize=8, markerfacecolor=COLORS["emerald"], markeredgecolor="white",
                markeredgewidth=1.5)

        # Draw reference circles
        for val in [0.25, 0.5, 0.75, 1.0]:
            circle_vals = [val] * (N + 1)
            ax.plot(angles, circle_vals, color="#475569", linewidth=0.5, linestyle='--', alpha=0.5)

        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12, fontweight='bold', color="#e2e8f0")
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="#94a3b8")
        ax.spines['polar'].set_color('#475569')

        # Score labels at each point
        for angle, score, cat in zip(angles[:-1], scores, categories):
            color = COLORS["green_pass"] if score >= 0.6 else (COLORS["amber"] if score >= 0.4 else COLORS["red_fail"])
            ax.annotate(f"{score*100:.0f}%", xy=(angle, score), fontsize=10,
                        fontweight='bold', ha='center', va='bottom', color=color,
                        path_effects=[path_effects.withStroke(linewidth=3, foreground="#0f172a")])

        title = f"ADMET Profile"
        if molecule_name:
            title += f" — {molecule_name}"
        ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)

        return _fig_to_base64(fig)


# ==========================================================================
# 2. Drug-Likeness Scorecard
# ==========================================================================

def generate_drug_likeness_scorecard(drug_likeness_data: Dict[str, Any], molecule_name: str = "") -> str:
    """
    Generate a visual scorecard showing pass/fail for each drug-likeness filter.
    """
    with plt.rc_context(HELIOS_STYLE):
        filters = {
            "Lipinski\n(Rule of 5)": drug_likeness_data.get("lipinski", {}).get("pass", False),
            "Veber\n(Oral Bioavail.)": drug_likeness_data.get("veber", {}).get("pass", False),
            "Ghose\n(Drug Range)": drug_likeness_data.get("ghose", {}).get("pass", False),
            "Egan\n(Bioavailability)": drug_likeness_data.get("egan", {}).get("pass", False),
            "Muegge\n(Pharmacology)": drug_likeness_data.get("muegge", {}).get("pass", False),
        }

        qed = drug_likeness_data.get("qed_score", 0)
        overall = drug_likeness_data.get("overall_drug_likeness_pct", 0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                                 gridspec_kw={"width_ratios": [2, 1]})
        fig.patch.set_facecolor("#0f172a")

        # Left: Filter pass/fail bars
        ax = axes[0]
        ax.set_facecolor("#1e293b")
        names = list(filters.keys())
        passes = [1 if v else 0 for v in filters.values()]
        colors = [COLORS["green_pass"] if v else COLORS["red_fail"] for v in filters.values()]
        icons = ["PASS" if v else "FAIL" for v in filters.values()]

        bars = ax.barh(range(len(names)), passes, color=colors, height=0.6,
                       edgecolor='none', alpha=0.85)

        for i, (bar, icon, passed) in enumerate(zip(bars, icons, filters.values())):
            color = "white"
            ax.text(0.5, i, icon, ha='center', va='center', fontsize=13,
                    fontweight='bold', color=color,
                    path_effects=[path_effects.withStroke(linewidth=2, foreground="#0f172a")])

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=11, fontweight='bold', color="#e2e8f0")
        ax.set_xlim(0, 1.2)
        ax.set_xticks([])
        ax.set_title("Drug-Likeness Filters", fontsize=14, fontweight='bold', color='white')
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Right: QED gauge
        ax2 = axes[1]
        ax2.set_facecolor("#1e293b")
        ax2.set_aspect('equal')

        # Draw QED as a donut
        theta = np.linspace(0, 2 * np.pi, 100)
        r_outer, r_inner = 1.0, 0.65
        # Background ring
        ax2.fill_between(theta, r_inner, r_outer, alpha=0.2, color="#475569")
        # QED fill
        qed_angle = 2 * np.pi * qed
        theta_fill = np.linspace(np.pi / 2, np.pi / 2 - qed_angle, 100)
        x_outer = r_outer * np.cos(theta_fill)
        y_outer = r_outer * np.sin(theta_fill)
        x_inner = r_inner * np.cos(theta_fill[::-1])
        y_inner = r_inner * np.sin(theta_fill[::-1])
        x = np.concatenate([x_outer, x_inner])
        y = np.concatenate([y_outer, y_inner])

        qed_color = COLORS["green_pass"] if qed >= 0.5 else (COLORS["amber"] if qed >= 0.3 else COLORS["red_fail"])
        ax2.fill(x, y, color=qed_color, alpha=0.8)

        # Center text
        ax2.text(0, 0.05, f"{qed:.2f}", ha='center', va='center', fontsize=28,
                 fontweight='bold', color='white')
        ax2.text(0, -0.2, "QED Score", ha='center', va='center', fontsize=11,
                 color="#94a3b8")
        ax2.text(0, -0.38, f"Overall: {overall:.0f}%", ha='center', va='center',
                 fontsize=10, color=qed_color, fontweight='bold')

        ax2.set_xlim(-1.3, 1.3)
        ax2.set_ylim(-1.3, 1.3)
        ax2.axis('off')
        ax2.set_title("Drug Quality", fontsize=14, fontweight='bold', color='white')

        title = "Drug-Likeness Assessment"
        if molecule_name:
            title += f" — {molecule_name}"
        fig.suptitle(title, fontsize=16, fontweight='bold', color='white', y=1.02)
        fig.tight_layout()

        return _fig_to_base64(fig)


# ==========================================================================
# 3. Property Comparison Chart
# ==========================================================================

def generate_property_comparison(
    molecules: List[Dict[str, Any]],
    title: str = "Molecular Property Comparison"
) -> str:
    """
    Generate a grouped bar chart comparing properties of multiple molecules.
    Each dict: {"name": str, "smiles": str}
    """
    if not RDKIT_AVAILABLE or not molecules:
        return ""

    with plt.rc_context(HELIOS_STYLE):
        # Compute properties
        property_names = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds"]
        mol_data = []
        mol_names = []

        for m in molecules[:6]:  # Max 6 molecules
            mol = Chem.MolFromSmiles(m["smiles"])
            if mol is None:
                continue
            mol_names.append(m.get("name", m["smiles"][:20]))
            mol_data.append([
                Descriptors.MolWt(mol) / 100,  # Normalize MW
                Descriptors.MolLogP(mol),
                rdMolDescriptors.CalcTPSA(mol) / 30,  # Normalize TPSA
                rdMolDescriptors.CalcNumHBD(mol),
                rdMolDescriptors.CalcNumHBA(mol),
                rdMolDescriptors.CalcNumRotatableBonds(mol),
            ])

        if not mol_data:
            return ""

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")

        n_mols = len(mol_names)
        n_props = len(property_names)
        x = np.arange(n_props)
        width = 0.8 / n_mols

        for i, (name, data) in enumerate(zip(mol_names, mol_data)):
            offset = (i - n_mols / 2 + 0.5) * width
            color = RADAR_COLORS[i % len(RADAR_COLORS)]
            bars = ax.bar(x + offset, data, width, label=name, color=color, alpha=0.85,
                         edgecolor='none')
            for bar, val in zip(bars, data):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8,
                        color=color, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}\n{'(×100)' if p=='MW' else '(×30)' if p=='TPSA' else ''}"
                           for p in property_names], fontsize=10)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', color='white')
        ax.legend(loc='upper right', fontsize=9, facecolor="#1e293b",
                  edgecolor="#475569", labelcolor="#e2e8f0")
        ax.grid(axis='y', alpha=0.3)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()

        return _fig_to_base64(fig)


# ==========================================================================
# 4. Bioavailability Radar
# ==========================================================================

def generate_bioavailability_radar(smiles: str, molecule_name: str = "") -> str:
    """
    Generate a bioavailability radar (like SwissADME's BOILED-Egg style radar).
    Shows 6 key properties vs ideal ranges.
    """
    if not RDKIT_AVAILABLE:
        return ""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""

    with plt.rc_context(HELIOS_STYLE):
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        rot = rdMolDescriptors.CalcNumRotatableBonds(mol)

        categories = ["MW\n(150-500)", "LogP\n(-0.7-5)", "TPSA\n(20-130)",
                      "HBD\n(0-5)", "HBA\n(0-10)", "RotBonds\n(0-10)"]

        # Normalize to 0-1 based on ideal ranges
        def norm(val, low, high, max_val):
            """Normalize: 1.0 in the ideal range, drops off outside."""
            if low <= val <= high:
                return 1.0
            elif val < low:
                return max(0, 1.0 - abs(val - low) / max_val)
            else:
                return max(0, 1.0 - abs(val - high) / max_val)

        scores = [
            norm(mw, 150, 500, 300),
            norm(logp, -0.7, 5.0, 4),
            norm(tpsa, 20, 130, 80),
            norm(hbd, 0, 5, 5),
            norm(hba, 0, 10, 8),
            norm(rot, 0, 10, 8),
        ]

        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        scores_plot = scores + scores[:1]
        angles += angles[:1]

        # Ideal zone (all 1.0)
        ideal = [1.0] * (N + 1)

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")

        # Ideal zone fill
        ax.fill(angles, ideal, color=COLORS["green_pass"], alpha=0.1)
        ax.plot(angles, ideal, color=COLORS["green_pass"], linewidth=1, linestyle='--', alpha=0.4)

        # Actual values
        ax.fill(angles, scores_plot, color=COLORS["blue"], alpha=0.3)
        ax.plot(angles, scores_plot, color=COLORS["blue"], linewidth=2.5, marker='o',
                markersize=8, markerfacecolor=COLORS["blue"], markeredgecolor="white",
                markeredgewidth=1.5)

        # Labels with actual values
        actual_vals = [f"{mw:.0f}", f"{logp:.1f}", f"{tpsa:.0f}", str(hbd), str(hba), str(rot)]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, fontweight='bold', color="#e2e8f0")
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "", "", ""], fontsize=8)
        ax.spines['polar'].set_color('#475569')

        # Value annotations
        for angle, score, val in zip(angles[:-1], scores, actual_vals):
            color = COLORS["green_pass"] if score >= 0.8 else (COLORS["amber"] if score >= 0.5 else COLORS["red_fail"])
            ax.annotate(val, xy=(angle, score + 0.08), fontsize=11,
                        fontweight='bold', ha='center', va='bottom', color=color,
                        path_effects=[path_effects.withStroke(linewidth=3, foreground="#0f172a")])

        title = "Bioavailability Radar"
        if molecule_name:
            title += f" — {molecule_name}"
        ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)

        return _fig_to_base64(fig)


# ==========================================================================
# 5. Structural Alerts Visual
# ==========================================================================

def generate_alerts_visual(alerts_data: Dict[str, Any], molecule_name: str = "") -> str:
    """
    Generate a visual summary of structural alerts.
    """
    with plt.rc_context(HELIOS_STYLE):
        details = alerts_data.get("details", [])
        pains = alerts_data.get("pains_count", 0)
        brenk = alerts_data.get("brenk_count", 0)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 2]})
        fig.patch.set_facecolor("#0f172a")

        # Left: Summary gauge
        ax = axes[0]
        ax.set_facecolor("#1e293b")
        total = pains + brenk
        categories = ["PAINS", "Brenk"]
        counts = [pains, brenk]
        colors_bar = [COLORS["amber"], COLORS["red"]]

        bars = ax.barh(categories, counts, color=colors_bar, height=0.5, edgecolor='none', alpha=0.85)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                        str(count), va='center', fontsize=13, fontweight='bold', color='white')

        status_color = COLORS["green_pass"] if total == 0 else (COLORS["amber"] if total <= 2 else COLORS["red_fail"])
        status_text = "CLEAN" if total == 0 else f"{total} ALERT{'S' if total > 1 else ''}"
        ax.text(0.5, -0.15, status_text, transform=ax.transAxes, ha='center',
                fontsize=16, fontweight='bold', color=status_color)

        ax.set_xlim(0, max(3, max(counts) + 1) if counts else 3)
        ax.set_title("Alert Summary", fontsize=12, fontweight='bold', color='white')
        for spine in ['top', 'right', 'bottom']:
            ax.spines[spine].set_visible(False)

        # Right: Alert details table
        ax2 = axes[1]
        ax2.set_facecolor("#1e293b")
        ax2.axis('off')

        if details:
            cell_text = [[d["type"], d["alert"]] for d in details[:8]]
            table = ax2.table(cellText=cell_text, colLabels=["Type", "Alert"],
                             loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.auto_set_column_width([0, 1])
            for key, cell in table.get_celld().items():
                cell.set_facecolor("#1e293b")
                cell.set_edgecolor("#475569")
                cell.set_text_props(color="#e2e8f0")
                if key[0] == 0:  # header
                    cell.set_facecolor("#334155")
                    cell.set_text_props(color="white", fontweight='bold')
        else:
            ax2.text(0.5, 0.5, "No structural alerts detected\nMolecule appears clean",
                     ha='center', va='center', fontsize=14, color=COLORS["green_pass"],
                     fontweight='bold', transform=ax2.transAxes)

        title = "Structural Alert Analysis"
        if molecule_name:
            title += f" — {molecule_name}"
        fig.suptitle(title, fontsize=14, fontweight='bold', color='white', y=1.02)
        fig.tight_layout()

        return _fig_to_base64(fig)


# ==========================================================================
# 6. Pharmacophore Feature Map
# ==========================================================================

def generate_pharmacophore_chart(pharm_data: Dict[str, Any], molecule_name: str = "") -> str:
    """
    Generate a horizontal bar chart of pharmacophoric features.
    """
    if not pharm_data:
        return ""

    with plt.rc_context(HELIOS_STYLE):
        features = pharm_data.get("features", {})
        names = []
        counts = []
        for name, info in features.items():
            c = info.get("count", 0)
            if c > 0:
                names.append(name)
                counts.append(c)

        if not names:
            return ""

        fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.6 + 1)))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")

        colors_list = [RADAR_COLORS[i % len(RADAR_COLORS)] for i in range(len(names))]
        bars = ax.barh(range(len(names)), counts, color=colors_list, height=0.6,
                       edgecolor='none', alpha=0.85)

        for bar, count, color in zip(bars, counts, colors_list):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    str(count), va='center', fontsize=12, fontweight='bold', color=color)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=11, fontweight='bold', color="#e2e8f0")
        ax.set_xlabel("Count", fontsize=12)
        ax.invert_yaxis()

        title = "Pharmacophore Features"
        if molecule_name:
            title += f" — {molecule_name}"
        ax.set_title(title, fontsize=14, fontweight='bold', color='white')

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()

        return _fig_to_base64(fig)


# ==========================================================================
# 7. Lead Optimization Priority Chart
# ==========================================================================

def generate_optimization_chart(opt_data: Dict[str, Any], molecule_name: str = "") -> str:
    """
    Generate a visual of priority optimization actions.
    """
    if not opt_data:
        return ""

    with plt.rc_context(HELIOS_STYLE):
        actions = opt_data.get("priority_actions", [])
        if not actions:
            return ""

        fig, ax = plt.subplots(figsize=(10, max(3, len(actions) * 1.2 + 1)))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")

        severity_colors = {
            "critical": COLORS["red"],
            "high": COLORS["amber"],
            "medium": COLORS["blue"],
            "low": COLORS["green_pass"],
        }
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        actions_sorted = sorted(actions, key=lambda a: severity_order.get(a.get("severity", "low"), 0), reverse=True)

        for i, action in enumerate(actions_sorted[:8]):
            sev = action.get("severity", "medium")
            color = severity_colors.get(sev, COLORS["slate"])

            # Severity bar
            ax.barh(i, severity_order.get(sev, 1), color=color, height=0.7,
                    edgecolor='none', alpha=0.3)

            # Issue text
            ax.text(0.1, i, f"{action['issue']}", fontsize=11, fontweight='bold',
                    color=color, va='center')
            # Value text
            ax.text(0.1, i - 0.25, f"  {action.get('value', '')}", fontsize=9,
                    color="#94a3b8", va='center', style='italic')

        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(0, 5)

        title = "Lead Optimization Priorities"
        if molecule_name:
            title += f" — {molecule_name}"
        ax.set_title(title, fontsize=14, fontweight='bold', color='white')
        ax.invert_yaxis()

        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()

        return _fig_to_base64(fig)


# ==========================================================================
# 8. Molecule Structure with Annotations
# ==========================================================================

def generate_annotated_molecule(smiles: str, molecule_name: str = "",
                                highlight_pharmacophores: bool = True,
                                size: Tuple[int, int] = (500, 400)) -> str:
    """
    Generate a high-quality annotated molecule structure image.
    """
    if not RDKIT_AVAILABLE:
        return ""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""

    AllChem.Compute2DCoords(mol)

    # Find atoms to highlight (pharmacophores)
    highlight_atoms = []
    highlight_colors = {}
    highlight_bonds = []
    bond_colors = {}

    if highlight_pharmacophores:
        # H-bond donors (blue)
        donor_smarts = Chem.MolFromSmarts("[NH,$([NH2]),$([OH])]")
        if donor_smarts:
            for match in mol.GetSubstructMatches(donor_smarts):
                for idx in match:
                    highlight_atoms.append(idx)
                    highlight_colors[idx] = (0.2, 0.5, 1.0)  # blue

        # H-bond acceptors (green)
        acceptor_smarts = Chem.MolFromSmarts("[$([O;H0;v2]),$([#7;v3;!$([nH])])]")
        if acceptor_smarts:
            for match in mol.GetSubstructMatches(acceptor_smarts):
                for idx in match:
                    if idx not in highlight_colors:
                        highlight_atoms.append(idx)
                        highlight_colors[idx] = (0.1, 0.8, 0.4)  # green

        # Aromatic (purple tint)
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic() and atom.GetIdx() not in highlight_colors:
                highlight_atoms.append(atom.GetIdx())
                highlight_colors[atom.GetIdx()] = (0.6, 0.4, 0.9)  # purple

    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().addAtomIndices = False
    drawer.drawOptions().bondLineWidth = 2.5

    if highlight_atoms:
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms,
                           highlightAtomColors=highlight_colors)
    else:
        drawer.DrawMolecule(mol)

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return svg


# ==========================================================================
# 9. Comprehensive Figure Set Generator
# ==========================================================================

def generate_all_figures(
    smiles: str,
    molecule_name: str = "",
    admet_data: Optional[Dict] = None,
    drug_likeness_data: Optional[Dict] = None,
    alerts_data: Optional[Dict] = None,
    pharm_data: Optional[Dict] = None,
    opt_data: Optional[Dict] = None,
) -> Dict[str, str]:
    """
    Generate all relevant figures for a molecule.
    Returns dict of {figure_name: base64_data_uri}.
    """
    figures = {}

    try:
        # Annotated structure
        svg = generate_annotated_molecule(smiles, molecule_name)
        if svg:
            figures["molecule_structure_svg"] = svg
    except Exception as e:
        logger.warning(f"Failed to generate molecule structure: {e}")

    try:
        if admet_data:
            fig = generate_admet_radar(admet_data, molecule_name)
            if fig:
                figures["admet_radar"] = fig
    except Exception as e:
        logger.warning(f"Failed to generate ADMET radar: {e}")

    try:
        if drug_likeness_data:
            fig = generate_drug_likeness_scorecard(drug_likeness_data, molecule_name)
            if fig:
                figures["drug_likeness_scorecard"] = fig
    except Exception as e:
        logger.warning(f"Failed to generate drug-likeness scorecard: {e}")

    try:
        fig = generate_bioavailability_radar(smiles, molecule_name)
        if fig:
            figures["bioavailability_radar"] = fig
    except Exception as e:
        logger.warning(f"Failed to generate bioavailability radar: {e}")

    try:
        if alerts_data:
            fig = generate_alerts_visual(alerts_data, molecule_name)
            if fig:
                figures["structural_alerts"] = fig
    except Exception as e:
        logger.warning(f"Failed to generate alerts visual: {e}")

    try:
        if pharm_data:
            fig = generate_pharmacophore_chart(pharm_data, molecule_name)
            if fig:
                figures["pharmacophore_features"] = fig
    except Exception as e:
        logger.warning(f"Failed to generate pharmacophore chart: {e}")

    try:
        if opt_data:
            fig = generate_optimization_chart(opt_data, molecule_name)
            if fig:
                figures["optimization_priorities"] = fig
    except Exception as e:
        logger.warning(f"Failed to generate optimization chart: {e}")

    return figures
