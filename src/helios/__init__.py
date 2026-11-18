"""
Helios Chemistry AI Assistant
Main package initialization
"""

__version__ = "0.2.0"
__author__ = "Your Name"

from .utils import chemistry, metrics, visualization

# Drug discovery modules (optional - require rdkit)
try:
    from .utils import drug_discovery, figure_generator
    __all__ = ["chemistry", "metrics", "visualization", "drug_discovery", "figure_generator"]
except ImportError:
    __all__ = ["chemistry", "metrics", "visualization"]
