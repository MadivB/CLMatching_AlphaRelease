"""Resolve external asset paths for v_alpha_test from paths.yaml.

This module is the single place that turns an entry in paths.yaml into a
real on-disk path used by the pipeline.  It also provides
``validate_required_assets`` which the front stage calls at startup to
fail fast with a friendly error when a required file is missing.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT


PATHS_YAML_NAME = "paths.yaml"


@dataclass
class ResolvedAsset:
    """One resolved entry from paths.yaml."""

    name: str
    path: str | None        # absolute path to a file that exists, or None
    candidates: list[str]   # all candidates we tried (already expanded to absolute)
    required: bool
    download_url: str       # may be empty
    download_command: str   # may be empty


def _yaml_load(yaml_path: Path) -> dict[str, Any]:
    """Try PyYAML; fall back to a tiny hand-rolled loader so the repo
    doesn't grow a hard dep just to read its own config."""
    try:
        import yaml  # type: ignore
        with open(yaml_path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return _fallback_yaml_load(yaml_path)


def _fallback_yaml_load(yaml_path: Path) -> dict[str, Any]:
    """Minimal YAML loader for paths.yaml only -- supports nested mappings,
    list items, scalar strings, comments, and blank lines.  Block scalars
    (``|`` / ``>``) are read verbatim until indentation drops.

    This is intentionally not a general YAML parser; it only handles what
    paths.yaml uses.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]
    pending_block: list[str] | None = None
    pending_block_indent = -1
    pending_block_target: tuple[dict[str, Any], str] | None = None

    def _coerce(val: str) -> Any:
        s = val.strip()
        if s == "":
            return ""
        if s.lower() in ("true", "yes"):
            return True
        if s.lower() in ("false", "no"):
            return False
        if s == "null":
            return None
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except ValueError:
            return s

    with open(yaml_path) as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if pending_block is not None:
                if not stripped or indent > pending_block_indent:
                    pending_block.append(line[pending_block_indent + 2:] if line.strip() else "")
                    continue
                else:
                    target_dict, target_key = pending_block_target
                    target_dict[target_key] = "\n".join(pending_block).rstrip()
                    pending_block = None
                    pending_block_target = None

            if not stripped or stripped.startswith("#"):
                continue

            while stack and indent <= stack[-1][0] and len(stack) > 1:
                stack.pop()
            container = stack[-1][1]

            if stripped.startswith("- "):
                item = stripped[2:].strip()
                if not isinstance(container, list):
                    raise ValueError(f"unexpected list item under non-list: {raw_line!r}")
                if ":" in item and not (item.startswith('"') or item.startswith("'")):
                    key, _, val = item.partition(":")
                    sub: dict[str, Any] = {}
                    sub[key.strip()] = _coerce(val) if val.strip() else sub
                    container.append(sub)
                    if not val.strip():
                        new_sub: dict[str, Any] = {}
                        sub[key.strip()] = new_sub
                        stack.append((indent, new_sub))
                else:
                    container.append(_coerce(item))
                continue

            if ":" in stripped:
                key, _, val = stripped.partition(":")
                key = key.strip()
                val = val.strip()
                if not isinstance(container, dict):
                    raise ValueError(f"unexpected key under non-dict: {raw_line!r}")
                if val == "":
                    new: dict[str, Any] = {}
                    container[key] = new
                    stack.append((indent, new))
                elif val == "|" or val == ">":
                    pending_block = []
                    pending_block_indent = indent
                    pending_block_target = (container, key)
                elif val == "[]":
                    container[key] = []
                else:
                    container[key] = _coerce(val)
                    if val.startswith("-"):
                        pass
            elif stripped == "[]":
                pass
            else:
                # bare scalar -- ignore
                continue

    if pending_block is not None and pending_block_target is not None:
        target_dict, target_key = pending_block_target
        target_dict[target_key] = "\n".join(pending_block).rstrip()

    # Special-case: turn `path_candidates:` followed by `-` items into a list.
    def _coerce_path_candidates(d: Any) -> None:
        if not isinstance(d, dict):
            return
        for k, v in list(d.items()):
            if k == "path_candidates" and isinstance(v, dict) and not v:
                d[k] = []
            _coerce_path_candidates(v)
    _coerce_path_candidates(root)
    return root


def find_paths_yaml(start: Path | None = None) -> Path:
    """Locate paths.yaml.  Search order:
    1. ``$V_ALPHA_TEST_PATHS_YAML`` env var (if set).
    2. ``REPO_ROOT/paths.yaml``.
    3. Walk up from ``start`` (default: this file) until found.
    """
    env = os.environ.get("V_ALPHA_TEST_PATHS_YAML")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    here = Path(REPO_ROOT) / PATHS_YAML_NAME
    if here.exists():
        return here
    cur = (start or Path(__file__).resolve()).parent
    for _ in range(8):
        candidate = cur / PATHS_YAML_NAME
        if candidate.exists():
            return candidate
        cur = cur.parent
    return here  # may not exist; resolver will fall back to defaults


def _expand_path(raw: str) -> str:
    """Expand ~ and turn relative paths into absolute (relative to REPO_ROOT)."""
    if not raw:
        return raw
    p = Path(os.path.expanduser(raw))
    if p.is_absolute():
        return str(p)
    return str(Path(REPO_ROOT) / p)


def resolve_asset(name: str, defaults: dict[str, Any] | None = None) -> ResolvedAsset:
    """Resolve one asset by name from paths.yaml.

    ``defaults`` provides a fallback dict if paths.yaml or the named
    entry is missing.  Useful for back-compat when a release predates a
    new asset entry.
    """
    yaml_path = find_paths_yaml()
    cfg: dict[str, Any] = {}
    if yaml_path.exists():
        try:
            cfg = _yaml_load(yaml_path)
        except Exception as exc:  # pragma: no cover - paths.yaml is user-edited
            print(
                f"WARN: failed to parse {yaml_path}: {exc}; using defaults",
                file=sys.stderr,
            )
    assets_block = (cfg.get("assets") or {}).get(name) or (defaults or {})
    if not assets_block:
        return ResolvedAsset(
            name=name, path=None, candidates=[], required=True,
            download_url="", download_command="",
        )

    raw_candidates: list[str] = []
    if "path_candidates" in assets_block and assets_block["path_candidates"]:
        raw_candidates = [str(c) for c in assets_block["path_candidates"]]
    elif "path" in assets_block and assets_block["path"]:
        raw_candidates = [str(assets_block["path"])]

    candidates = [_expand_path(c) for c in raw_candidates]
    chosen: str | None = None
    for c in candidates:
        if Path(c).exists():
            chosen = c
            break

    download_block = assets_block.get("download") or {}
    return ResolvedAsset(
        name=name,
        path=chosen,
        candidates=candidates,
        required=bool(assets_block.get("required", True)),
        download_url=str(download_block.get("url", "") or ""),
        download_command=str(download_block.get("command", "") or "").strip(),
    )


def resolve_input_data_dir(default: str | None = None) -> str | None:
    """Read input_data.default_data_dir from paths.yaml (or return default)."""
    yaml_path = find_paths_yaml()
    if not yaml_path.exists():
        return default
    try:
        cfg = _yaml_load(yaml_path)
    except Exception:
        return default
    return (cfg.get("input_data") or {}).get("default_data_dir", default)


def validate_required_assets(names: list[str]) -> list[ResolvedAsset]:
    """Return the list of REQUIRED assets that are missing.

    The caller (typically ``load_first_stage_models``) raises a helpful
    error if the returned list is non-empty.
    """
    missing: list[ResolvedAsset] = []
    for name in names:
        r = resolve_asset(name)
        if r.required and r.path is None:
            missing.append(r)
    return missing


def format_missing_assets_error(missing: list[ResolvedAsset]) -> str:
    """Build the human-friendly multi-line error message."""
    yaml_path = find_paths_yaml()
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("v_alpha_test: required asset(s) not found.")
    lines.append("=" * 72)
    lines.append(f"paths.yaml in use : {yaml_path}")
    lines.append("")
    for r in missing:
        lines.append(f"  * '{r.name}': missing.")
        if r.candidates:
            lines.append("    Looked here (first existing wins):")
            for c in r.candidates:
                lines.append(f"      - {c}")
        else:
            lines.append("    No path was configured in paths.yaml.")
        if r.download_url:
            lines.append(f"    Download URL : {r.download_url}")
        if r.download_command:
            lines.append("    Download command (run from the v_alpha_test repo root):")
            for sub in r.download_command.splitlines():
                lines.append(f"      {sub}")
        lines.append("")
    lines.append("How to fix:")
    lines.append(f"  1) Run the download command(s) above, OR")
    lines.append(f"  2) Edit {yaml_path} and change the 'path:' entry to point at")
    lines.append("     an existing file on your machine.")
    lines.append("=" * 72)
    return "\n".join(lines)


class MissingAssetError(FileNotFoundError):
    """Raised when a required asset is missing."""


def require_assets_or_explain(names: list[str]) -> dict[str, str]:
    """Validate required assets; on failure raise ``MissingAssetError``
    with a helpful multi-line message.  On success return a dict of
    {name: resolved_path}."""
    missing = validate_required_assets(names)
    if missing:
        raise MissingAssetError(format_missing_assets_error(missing))
    return {n: resolve_asset(n).path for n in names if resolve_asset(n).path}


__all__ = [
    "MissingAssetError",
    "ResolvedAsset",
    "find_paths_yaml",
    "format_missing_assets_error",
    "require_assets_or_explain",
    "resolve_asset",
    "resolve_input_data_dir",
    "validate_required_assets",
]
