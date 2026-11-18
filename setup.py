from setuptools import setup, find_packages

setup(
    name="helios",
    version="0.1.0",
    description="Chemistry AI Assistant with RAG, GNN, and LLM",
    author="Helios Contributors",
    author_email="maintainers@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "rdkit>=2023.9.1",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.2",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "helios-chat=helios.ui.app:main",
        ],
    },
)
