#!/usr/bin/env python3
"""Fast, read-only repository and environment diagnostic."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def revision(relative: str) -> str | None:
    path = ROOT / relative
    if not path.is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def main() -> None:
    try:
        import torch
        import tokenizers
        import transformers

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        versions = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": gpu,
        }
    except Exception as error:
        versions = {"error": repr(error)}

    required_paths = [
        "src/vla_tricks",
        "configs/libero/config.yaml",
        "third_party/LIBERO",
        "third_party/vla-cache",
        "third_party/transformers-vla-cache",
    ]
    report = {
        "repository": str(ROOT),
        "python": sys.executable,
        "environment": versions,
        "imports": {
            name: importlib.util.find_spec(name) is not None
            for name in ("vla_tricks", "libero", "prismatic")
        },
        "paths": {path: (ROOT / path).exists() for path in required_paths},
        "libero_config": os.environ.get("LIBERO_CONFIG_PATH"),
        "third_party_revisions": {
            "LIBERO": revision("third_party/LIBERO"),
            "vla-cache": revision("third_party/vla-cache"),
            "transformers-vla-cache": revision("third_party/transformers-vla-cache"),
        },
    }
    print(json.dumps(report, indent=2))
    failed_paths = [path for path, exists in report["paths"].items() if not exists]
    failed_imports = [name for name, works in report["imports"].items() if not works]
    if "error" in versions or failed_paths or failed_imports:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

