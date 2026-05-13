"""Validate that all required v_alpha_test assets are present.

Reads paths.yaml, resolves every entry under `assets:`, and prints a
friendly report.  Exit code 0 if every required asset is present,
non-zero otherwise.

Run:

    python scripts/check_install.py

If a required asset is missing the script prints the same multi-line
error that the pipeline would have raised (paths.yaml location, the
candidate paths checked, and the download command).
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(here))

    try:
        from M5p1.first_stage_matching.asset_resolver import (
            find_paths_yaml,
            format_missing_assets_error,
            resolve_asset,
            validate_required_assets,
        )
    except ImportError as exc:
        print(f"ERROR: cannot import asset_resolver from this repo: {exc}", file=sys.stderr)
        return 2

    yaml_path = find_paths_yaml()
    print(f"paths.yaml: {yaml_path}  (exists={yaml_path.exists()})")
    print()

    asset_names = ["perceiver_charge_light_relation", "pulse_template", "variance_prediction"]
    width = max(len(n) for n in asset_names)

    print("Asset status:")
    for name in asset_names:
        r = resolve_asset(name)
        if r.path is not None:
            tag = "OK"
            extra = f"path = {r.path}"
        elif r.required:
            tag = "MISSING (required)"
            extra = f"candidates: {r.candidates or '(none configured)'}"
        else:
            tag = "missing (optional)"
            extra = f"candidates: {r.candidates or '(none configured)'}; pipeline runs with fallback"
        print(f"  {name.ljust(width)} : [{tag}]")
        print(f"    {extra}")
        if r.required and r.path is None and r.download_command:
            print(f"    download command:")
            for line in r.download_command.splitlines():
                print(f"      {line}")
        print()

    missing = validate_required_assets(asset_names)
    if missing:
        print(format_missing_assets_error(missing))
        return 1

    print("All required assets are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
