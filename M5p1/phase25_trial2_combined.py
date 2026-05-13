"""Combined Trial2 entry point with the physical dChi2 light gate.

`v_test_trial2.ipynb` is the reference implementation.  This module keeps the
combined-notebook import path intact while using the same spatial rescue plus
the physical-loss dChi2 light algorithm from that notebook plus the dominant
multi-TPC track-label veto.  Accepted physical light moves use exact GPU
family-image updates for the affected old/new t0 families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from . import phase25_trial2 as t2
except Exception:  # pragma: no cover - direct notebook import fallback
    import phase25_trial2 as t2


@dataclass
class Trial2CombinedConfig(t2.Trial2Config):
    """Combined defaults matching the physical-loss dChi2 light cell."""

    light_skip_shower_tpcs: bool = False
    light_use_physical_chi2: bool = True
    light_use_rescue_branches: bool = False
    light_overflow_sigma: float = 3.0
    light_overflow_abs_adc: float = 400.0
    light_model_activity_adc: float = 400.0
    light_min_overflow_channels: int = 6
    phys_min_source_ofch_reduction: int = 8
    phys_min_dchi2_improvement: float = 5.0e2
    phys_min_dchi2_per_mev: float = 1.0e1
    phys_std_floor: float = 1.0e-6
    light_veto_multitpc_track: bool = True
    light_veto_track_min_tpcs: int = 4


Trial2Config = Trial2CombinedConfig


def run_trial2_combined_rescue_from_namespace(
    namespace: dict[str, Any],
    *,
    config: Trial2CombinedConfig | t2.Trial2Config | None = None,
    commit: bool | None = None,
) -> dict[str, Any]:
    """Run the exact Trial2 path through the combined notebook entry point."""
    cfg = config or Trial2CombinedConfig()
    return t2.run_trial2_phase25_from_namespace(namespace, config=cfg, commit=commit)


def run_trial2_phase25_from_namespace(
    namespace: dict[str, Any],
    *,
    config: Trial2CombinedConfig | t2.Trial2Config | None = None,
    commit: bool | None = None,
) -> dict[str, Any]:
    """Alias for compatibility with code that imports the combined package."""
    return run_trial2_combined_rescue_from_namespace(
        namespace,
        config=config,
        commit=commit,
    )


print_stage_truth_summary = t2.print_stage_truth_summary
print_trial2_light_acceptance_table = t2.print_trial2_light_acceptance_table
print_trial2_summary = t2.print_trial2_summary
