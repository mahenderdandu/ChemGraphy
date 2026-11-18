#!/usr/bin/env python3
"""Clean local, non-source artifacts before publishing Helios to GitHub.

Usage:
  python scripts/prepare_github_repo.py --dry-run
  python scripts/prepare_github_repo.py --apply
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def get_cleanup_targets(root: Path) -> list[Path]:
    return [
        root / "venv",
        root / ".venv",
        root / "logs",
        root / "outputs",
        root / "frontend" / "node_modules",
        root / "frontend" / "dist",
        root / "data" / "raw",
        root / "data" / "processed",
        root / "data" / "uploads",
        root / "data" / "rag",
        root / "models" / "checkpoints",
        root / "src" / "helios" / "__pycache__",
    ]


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare repository for GitHub publishing")
    parser.add_argument("--apply", action="store_true", help="Delete files instead of printing plan")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only (default)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    targets = get_cleanup_targets(root)

    mode_apply = args.apply
    if not args.apply and not args.dry_run:
        mode_apply = False

    print(f"[Helios] Repository root: {root}")
    print(f"[Helios] Mode: {'APPLY' if mode_apply else 'DRY-RUN'}")

    existing = [path for path in targets if path.exists()]
    if not existing:
        print("[Helios] Nothing to clean.")
        return

    print("[Helios] Targets:")
    for path in existing:
        print(f" - {path.relative_to(root)}")

    if not mode_apply:
        print("\n[Helios] No files deleted. Run with --apply to remove these paths.")
        return

    for path in existing:
        remove_path(path)

    print("\n[Helios] Cleanup complete.")


if __name__ == "__main__":
    main()
