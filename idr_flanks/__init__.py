"""Build flanking IDRs that improve a binder's affinity for its target.

Given a structure of a binder-target complex, ``idr_flanks`` finds the region of
the target that a new flank on the binder could actually reach, then uses GOOSE
to design a disordered sequence chemically complementary to that region.

Typical use::

    from idr_flanks import build_flanked_binder, describe_chains

    print(describe_chains("complex.pdb"))          # which chain is which?
    result = build_flanked_binder("complex.pdb",
                                  binder_chain="B", target_chain="A",
                                  c_flank_length=30)
    print(result.final_sequence)
    print(result.summary())

The heavy dependencies (GOOSE, FINCHES, metapredict) are imported lazily, so
reading structures and analysing interfaces works without them.
"""

try:
    from ._version import __version__
except ImportError:  # running from a source tree without a built _version.py
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            __version__ = version("idr_flanks")
        except PackageNotFoundError:
            __version__ = "0+unknown"
    except ImportError:
        __version__ = "0+unknown"

from .interface import (
    InterfaceError,
    ProximalRegion,
    ProximalResidue,
    contact_map,
    end_to_end_distance,
    find_proximal_region,
    min_distances_to,
    parse_residue_spec,
    reach_radius,
    tether_contact_weight,
    tether_contact_weights,
)
from .io import (
    THREE_TO_ONE,
    Atom,
    Chain,
    Residue,
    Structure,
    StructureParseError,
    read_cif,
    read_pdb,
    read_structure,
)
from .sasa import (
    ATOMIC_RADII,
    MAX_SASA,
    relative_residue_sasa,
    residue_sasa,
    solvent_accessible_mask,
    sphere_points,
)


def __getattr__(name):
    """Expose the design/pipeline API without importing GOOSE at import time."""
    _design = {"DesignConfig", "DesignError", "DesignResult", "PRESETS",
               "design_flank", "score_flank", "epsilon_per_residue",
               "load_epsilon_model", "target_discriminability",
               "shell_epsilon_class", "shell_weighted_epsilon",
               "idr_amino_acid_frequencies", "context_disorder_class",
               "binder_competition_class", "avoidance_class",
               "flank_epsilons", "FlankEpsilonScores",
               "shell_weighted_epsilon"}
    _pipeline = {"FlankedBinder", "build_flanked_binder", "describe_chains",
                 "score_flanks"}
    if name in _design:
        from . import design
        return getattr(design, name)
    if name in _pipeline:
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    # structure reading
    "Atom",
    "Chain",
    "Residue",
    "Structure",
    "StructureParseError",
    "read_structure",
    "read_pdb",
    "read_cif",
    "THREE_TO_ONE",
    # surface
    "residue_sasa",
    "relative_residue_sasa",
    "solvent_accessible_mask",
    "sphere_points",
    "ATOMIC_RADII",
    "MAX_SASA",
    # interface
    "InterfaceError",
    "ProximalRegion",
    "ProximalResidue",
    "find_proximal_region",
    "contact_map",
    "min_distances_to",
    "parse_residue_spec",
    "reach_radius",
    "end_to_end_distance",
    "tether_contact_weight",
    "tether_contact_weights",
    # design (lazily imported)
    "DesignConfig",
    "DesignError",
    "DesignResult",
    "PRESETS",
    "design_flank",
    "score_flank",
    "epsilon_per_residue",
    "load_epsilon_model",
    "flank_epsilons",
    "FlankEpsilonScores",
    "target_discriminability",
    "shell_epsilon_class",
    "shell_weighted_epsilon",
    "idr_amino_acid_frequencies",
    "context_disorder_class",
    "binder_competition_class",
    "avoidance_class",
    # pipeline (lazily imported)
    "FlankedBinder",
    "build_flanked_binder",
    "describe_chains",
    "score_flanks",
]
