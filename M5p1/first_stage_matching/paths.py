from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
M5P1_DIR = REPO_ROOT / "M5p1"
ML_DIR = REPO_ROOT / "NewMLSection"
CL2X2_DIR = REPO_ROOT / "2x2CLMatching"


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def configure_paths() -> None:
    """Install release-local import paths only."""
    project_root = REPO_ROOT.parent
    cleaned = []
    for entry in sys.path:
        if entry == "":
            cleaned.append(entry)
            continue
        try:
            path = Path(entry).resolve()
        except Exception:
            cleaned.append(entry)
            continue
        if _is_under(path, project_root) and not _is_under(path, REPO_ROOT):
            continue
        if path in {REPO_ROOT.resolve(), ML_DIR.resolve(), M5P1_DIR.resolve(), CL2X2_DIR.resolve()}:
            continue
        cleaned.append(entry)

    sys.path[:] = cleaned
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(1, str(ML_DIR))
    sys.path.insert(2, str(M5P1_DIR))


def import_from_path(module_name: str, file_path: str | os.PathLike):
    import importlib.util

    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


__all__ = [
    "REPO_ROOT",
    "M5P1_DIR",
    "ML_DIR",
    "CL2X2_DIR",
    "configure_paths",
    "import_from_path",
]
