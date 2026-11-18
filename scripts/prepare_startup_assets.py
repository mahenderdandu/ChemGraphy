#!/usr/bin/env python3
"""Prepare all runtime assets required by Helios startup.

This script is intended to be invoked by the startup scripts before launching
the API/frontend. It performs idempotent preparation:
1. Download required datasets
2. Download/prepare raw ChEMBL data
3. Build RAG corpus + embeddings
4. Warm-download embedding + LLM models
5. Train Graphormer and keep only best checkpoint
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from helios.data.downloaders import download_chembl_data
from helios.data.moleculenet_loader import MoleculeNetDataset
from helios.data.processors.chembl_rag_processor import ChEMBLRAGProcessor
from helios.models.graphormer.model import load_graphormer_from_pretrained
from build_numpy_index import build_numpy_index


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("helios.prepare_startup_assets")


def _load_data_config(root_dir: Path) -> dict:
    config_path = root_dir / "configs" / "data_config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _run_command(command: list[str], cwd: Path) -> None:
    logger.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def _find_chembl_db(raw_chembl_dir: Path) -> Path | None:
    candidates = sorted(raw_chembl_dir.glob("**/*.db"))
    return candidates[0] if candidates else None


def ensure_moleculenet_datasets(root_dir: Path) -> None:
    logger.info("Ensuring MoleculeNet datasets are downloaded...")
    data_root = root_dir / "data" / "moleculenet"
    for dataset_name in ("ESOL", "FreeSolv", "Lipophilicity", "BACE"):
        MoleculeNetDataset(dataset_name=dataset_name, split="train", root_dir=str(data_root))
        logger.info("Dataset ready: %s", dataset_name)


def ensure_chembl_db(root_dir: Path, config: dict, force: bool) -> Path:
    raw_chembl_dir = root_dir / "data" / "raw" / "chembl"
    raw_chembl_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        existing = _find_chembl_db(raw_chembl_dir)
        if existing:
            logger.info("Using existing ChEMBL database: %s", existing)
            return existing

    version = str(config["data"]["sources"]["chembl"]["version"])
    logger.info("Downloading ChEMBL v%s database...", version)
    downloads = download_chembl_data(
        output_dir=str(raw_chembl_dir),
        version=version,
        download_sqlite=True,
        download_sdf=False,
        force=force,
    )
    db_path = Path(downloads["sqlite"])
    logger.info("ChEMBL database ready: %s", db_path)
    return db_path


def ensure_rag_assets(
    root_dir: Path,
    chembl_db_path: Path,
    force: bool,
    max_compounds: int,
    max_targets: int,
    max_mechanisms: int,
    embedding_model: str,
) -> None:
    rag_dir = root_dir / "data" / "rag"
    rag_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = rag_dir / "chembl_corpus.json"
    embeddings_path = rag_dir / "embeddings.npy"
    metadata_path = rag_dir / "metadata.json"

    need_corpus = force or not corpus_path.exists()
    need_embeddings = force or not embeddings_path.exists() or not metadata_path.exists()

    if need_corpus:
        logger.info("Building RAG corpus from ChEMBL...")
        processor = ChEMBLRAGProcessor(str(chembl_db_path))
        processor.build_corpus(
            output_path=str(corpus_path),
            max_compounds=max_compounds,
            max_targets=max_targets,
            max_mechanisms=max_mechanisms,
        )
    else:
        logger.info("RAG corpus already present: %s", corpus_path)

    if need_embeddings:
        logger.info("Building RAG embeddings index...")
        build_numpy_index(
            corpus_path=str(corpus_path),
            output_dir=str(rag_dir),
            encoder_name=embedding_model,
            batch_size=16,
        )
    else:
        logger.info("RAG embeddings already present: %s", embeddings_path)


def ensure_model_downloads(embedding_model: str, llm_model: str) -> None:
    logger.info("Downloading/validating embedding model cache: %s", embedding_model)
    SentenceTransformer(embedding_model, device="cpu")

    logger.info("Downloading/validating LLM tokenizer cache: %s", llm_model)
    AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)

    logger.info("Downloading/validating LLM model cache: %s", llm_model)
    AutoModelForCausalLM.from_pretrained(
        llm_model,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    logger.info("Loading Graphormer model implementation...")
    load_graphormer_from_pretrained(device="cpu")


def prune_non_best_checkpoints(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.exists():
        return

    for file_path in checkpoint_dir.glob("checkpoint_epoch_*.pt"):
        file_path.unlink(missing_ok=True)
    logger.info("Removed non-best Graphormer checkpoints in %s", checkpoint_dir)


def ensure_graphormer_checkpoint(
    root_dir: Path,
    force: bool,
    dataset: str,
    epochs: int,
    batch_size: int,
    device: str,
) -> None:
    checkpoint_dir = root_dir / "models" / "checkpoints" / "graphormer"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pt"

    if not force and best_path.exists():
        logger.info("Best Graphormer checkpoint already exists: %s", best_path)
        prune_non_best_checkpoints(checkpoint_dir)
        return

    logger.info("Training Graphormer to generate best checkpoint...")
    _run_command(
        [
            sys.executable,
            "scripts/train_graphormer.py",
            "--dataset",
            dataset,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--device",
            device,
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--keep-best-only",
        ],
        cwd=root_dir,
    )

    if not best_path.exists():
        raise RuntimeError(f"Training finished but best checkpoint not found at: {best_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Helios startup assets")
    parser.add_argument("--force", action="store_true", help="Force rebuild/redownload assets")
    parser.add_argument("--skip-train", action="store_true", help="Skip Graphormer training")
    parser.add_argument("--train-dataset", default=os.getenv("HELIOS_TRAIN_DATASET", "ESOL"))
    parser.add_argument("--train-epochs", type=int, default=int(os.getenv("HELIOS_TRAIN_EPOCHS", "20")))
    parser.add_argument("--train-batch-size", type=int, default=int(os.getenv("HELIOS_TRAIN_BATCH_SIZE", "32")))
    parser.add_argument("--train-device", default=os.getenv("HELIOS_TRAIN_DEVICE", "cpu"))
    parser.add_argument("--llm-model", default=os.getenv("HELIOS_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("HELIOS_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    parser.add_argument("--max-rag-compounds", type=int, default=int(os.getenv("HELIOS_RAG_MAX_COMPOUNDS", "10000")))
    parser.add_argument("--max-rag-targets", type=int, default=int(os.getenv("HELIOS_RAG_MAX_TARGETS", "5000")))
    parser.add_argument("--max-rag-mechanisms", type=int, default=int(os.getenv("HELIOS_RAG_MAX_MECHANISMS", "5000")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]

    logger.info("=" * 72)
    logger.info("HELIOS STARTUP ASSET PREPARATION")
    logger.info("Repository root: %s", root_dir)
    logger.info("Force mode: %s", args.force)
    logger.info("=" * 72)

    config = _load_data_config(root_dir)
    ensure_moleculenet_datasets(root_dir)
    chembl_db_path = ensure_chembl_db(root_dir, config, force=args.force)
    ensure_rag_assets(
        root_dir=root_dir,
        chembl_db_path=chembl_db_path,
        force=args.force,
        max_compounds=args.max_rag_compounds,
        max_targets=args.max_rag_targets,
        max_mechanisms=args.max_rag_mechanisms,
        embedding_model=args.embedding_model,
    )
    ensure_model_downloads(args.embedding_model, args.llm_model)

    if not args.skip_train:
        ensure_graphormer_checkpoint(
            root_dir=root_dir,
            force=args.force,
            dataset=args.train_dataset,
            epochs=args.train_epochs,
            batch_size=args.train_batch_size,
            device=args.train_device,
        )
    else:
        logger.info("Skipping Graphormer training (--skip-train)")

    logger.info("✅ Startup assets are ready.")


if __name__ == "__main__":
    main()
