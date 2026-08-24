"""Design a disordered flank that is chemically complementary to a target patch.

Wraps GOOSE's :class:`~goose.optimize.SequenceOptimizer` with an objective
tuned for this problem, and reports diagnostics that matter for judging whether
a designed flank is actually usable.

Three findings shaped the default objective, each measured rather than assumed:

* Maximising attraction with no other constraint drives straight to
  poly-tryptophan. Such sequences score as attractive and (in isolation) as
  disordered, but they aggregate and stick to everything.
* Predicting the flank's disorder *in the context of the binder* rather than in
  isolation kills that pathology on its own: a poly-W flank scores 1.00
  fraction-disordered alone but 0.00 once fused to a folded binder.
* Capping the aromatic fraction at 0.10 costs only ~14% of the attainable
  attraction while flipping self-interaction from attractive (aggregation
  prone) to strongly repulsive (soluble).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "DesignConfig",
    "DesignResult",
    "DesignError",
    "PRESETS",
    "design_flank",
    "score_flank",
    "load_epsilon_model",
    "epsilon_per_residue",
    "target_discriminability",
    "shell_epsilon_class",
    "shell_weighted_epsilon",
    "idr_amino_acid_frequencies",
    "context_disorder_class",
    "binder_competition_class",
    "avoidance_class",
]


class DesignError(RuntimeError):
    """Raised when a flank cannot be designed."""


_AROMATICS = ("W", "F", "Y")

# Below this much attainable preference the two surfaces are indistinguishable
# and the guard is dropped; above it the requested margin is clamped to a
# fraction of what is attainable rather than abandoned.
# A single background draw is too noisy to warn on; averaging this many keeps
# the reference stable at negligible cost (about 0.3 ms per epsilon call).
_N_REFERENCE_DRAWS = 24

_MIN_USEFUL_HEADROOM = 0.01
_HEADROOM_FRACTION = 0.5

# Amino-acid background frequencies of disordered regions, used to build the
# neutral reference and the decoy panel that specificity is measured against.
# Values are the Swiss-Prot background rounded to three places; they are used
# only for reference sequences, never as a design constraint.
_BACKGROUND = {
    "A": 0.0826, "R": 0.0553, "N": 0.0406, "D": 0.0546, "C": 0.0138,
    "Q": 0.0393, "E": 0.0672, "G": 0.0708, "H": 0.0227, "I": 0.0591,
    "L": 0.0965, "K": 0.0580, "M": 0.0241, "F": 0.0386, "P": 0.0472,
    "S": 0.0661, "T": 0.0535, "W": 0.0110, "Y": 0.0292, "V": 0.0686,
}

_model_cache: Dict[str, Any] = {}
_idr_probs_cache: Optional[Dict[str, float]] = None

_ALIPHATICS = ("A", "I", "L", "M", "V")


def idr_amino_acid_frequencies() -> Dict[str, float]:
    """Amino-acid frequencies of real disordered regions.

    Taken from GOOSE (``goose.data.aa_list_probabilities.IDRProbs``), which
    derives them from the IDRs of eleven proteomes as called by metapredict V3.
    Falls back to a copy of those values if GOOSE is unavailable.

    Returns
    -------
    dict
        Residue letter to frequency, summing to 1.
    """
    global _idr_probs_cache
    if _idr_probs_cache is None:
        try:
            from goose.data.aa_list_probabilities import IDRProbs
            _idr_probs_cache = dict(IDRProbs)
        except ImportError:  # pragma: no cover - GOOSE normally present
            _idr_probs_cache = dict(_IDR_PROBS_FALLBACK)
    return _idr_probs_cache


# Verbatim copy of GOOSE's IDRProbs, used only if GOOSE cannot be imported.
_IDR_PROBS_FALLBACK = {
    "A": 0.06267987, "C": 0.01257745, "D": 0.05355162, "E": 0.07514886,
    "F": 0.02518486, "G": 0.06474394, "H": 0.02332285, "I": 0.03529763,
    "K": 0.06197211, "L": 0.06889960, "M": 0.02095355, "N": 0.05480975,
    "P": 0.08052735, "Q": 0.05207587, "R": 0.05253632, "S": 0.12155916,
    "T": 0.06380657, "V": 0.04648557, "W": 0.00592648, "Y": 0.01794059,
}


def load_epsilon_model(model: str = "mpipi"):
    """Load and cache a FINCHES frontend for epsilon calculations.

    Constructing a frontend is expensive, so the same instance is shared across
    every property and every design run in the process.

    Parameters
    ----------
    model : str
        ``"mpipi"`` or ``"calvados"``.

    Returns
    -------
    A FINCHES frontend instance.
    """
    key = str(model).lower()
    cached = _model_cache.get(key)
    if cached is not None:
        return cached
    if key == "mpipi":
        from finches.frontend.mpipi_frontend import Mpipi_frontend
        obj = Mpipi_frontend()
    elif key == "calvados":
        from finches.frontend.calvados_frontend import CALVADOS_frontend
        obj = CALVADOS_frontend()
    else:
        raise ValueError(
            f"Unsupported epsilon model {model!r}; expected 'mpipi' or 'calvados'.")
    _model_cache[key] = obj
    return obj


def epsilon_per_residue(sequence: str, target: str, model: str = "mpipi") -> float:
    """Interaction epsilon between ``sequence`` and ``target``, per residue.

    FINCHES epsilon scales with the length of the first sequence and is nearly
    invariant to the length of the second, so dividing by ``len(sequence)``
    gives a length-comparable number. Negative is attractive.

    Parameters
    ----------
    sequence : str
        The designed sequence.
    target : str
        The target patch.
    model : str
        Epsilon model name.

    Returns
    -------
    float
    """
    if not sequence:
        raise ValueError("sequence must be non-empty")
    if not target:
        raise ValueError("target must be non-empty")
    mf = load_epsilon_model(model)
    return float(mf.epsilon(sequence, target)) / len(sequence)


# ---------------------------------------------------------------------------
# a context-aware disorder property
# ---------------------------------------------------------------------------

def _make_context_disorder_class():
    """Build the ContextFractionDisorder class against GOOSE's base class.

    Deferred so importing :mod:`idr_flanks.design` does not require GOOSE until
    a design is actually requested.
    """
    from goose.backend.optimizer_properties import ConstraintType, CustomProperty
    import metapredict as meta

    class ContextFractionDisorder(CustomProperty):
        """Fraction of the designed segment predicted disordered *in context*.

        Disorder prediction is context dependent, and the flank will not exist
        on its own -- it will be fused to the binder. Scoring the flank in
        isolation lets through sequences that stop being disordered the moment
        they are attached, which is how a poly-aromatic flank can look
        perfectly disordered right up until it is used.

        Parameters
        ----------
        n_context, c_context : str
            Sequence that will precede and follow the designed segment.
        target_value : float
            Target fraction of the segment predicted disordered.
        disorder_cutoff : float
            Per-residue score above which a residue counts as disordered.
        """

        can_be_linear_profile = False
        # metapredict batches many sequences far faster than it handles them one
        # at a time, and GOOSE only uses the batch path when this is True.
        calculate_in_batch = True

        def __init__(self, n_context: str = "", c_context: str = "",
                     target_value: float = 1.0, weight: float = 1.0,
                     disorder_cutoff: float = 0.5,
                     constraint_type=ConstraintType.MINIMUM):
            self.n_context = n_context or ""
            self.c_context = c_context or ""
            self.disorder_cutoff = float(disorder_cutoff)
            super().__init__(target_value=target_value, weight=weight,
                             constraint_type=constraint_type)
            # Deliberately NOT setting tracking_property_name: GOOSE's
            # duplicate-name handling (optimize.py:665) renames using a
            # substring test against the class name, and a custom name that is
            # not a substring of the class name makes properties collide and
            # silently drop out of the objective.

        def get_init_args(self) -> dict:
            args = super().get_init_args()
            args.update({
                "n_context": self.n_context,
                "c_context": self.c_context,
                "disorder_cutoff": self.disorder_cutoff,
            })
            return args

        def _fraction(self, scores, seq_len: int) -> float:
            start = len(self.n_context)
            segment = np.asarray(scores[start:start + seq_len],
                                 dtype=np.float64)
            if segment.size == 0:
                return 0.0
            return float((segment > self.disorder_cutoff).sum() / segment.size)

        def calculate_raw_value(self, protein) -> float:
            seq = protein.sequence
            full = self.n_context + seq + self.c_context
            return self._fraction(meta.predict_disorder(full), len(seq))

        def calculate_raw_value_batch(self, proteins: list) -> list:
            seqs = [p.sequence for p in proteins]
            full = [self.n_context + s + self.c_context for s in seqs]
            # metapredict's batch call returns (sequence, scores) pairs; the
            # order matches the input.
            predicted = meta.predict_disorder(full)
            out = []
            for seq, entry in zip(seqs, predicted):
                scores = entry[1] if isinstance(entry, (list, tuple)) else entry
                out.append(self._fraction(scores, len(seq)))
            return out

    return ContextFractionDisorder


def _make_avoidance_class(class_name: str):
    """Build a sequence-avoidance property under a distinct class name.

    A distinct name per use matters: GOOSE keys properties by class name and
    renames duplicates, so giving the decoy panel and the binder interface
    their own classes keeps both objectives and both labels intact.
    """
    from goose.backend.optimizer_properties import ConstraintType, CustomProperty

    class _Avoidance(CustomProperty):
        """Mean per-residue epsilon against a panel of unrelated sequences.

        This is the term that separates "complementary to this patch" from
        "sticky to everything". Maximising attraction alone tends to produce a
        flank that attracts arbitrary sequence nearly as strongly as it
        attracts the intended target, which would bind off-target in a cell.
        Requiring the decoy epsilon to stay non-attractive lets the attraction
        objective run free without buying non-specific stickiness.

        Note that this failure mode is specific to aromatic-driven attraction.
        Charge-complementary designs are naturally selective, and this term
        leaves them alone.

        Parameters
        ----------
        decoys : sequence of str
            Sequences the flank should *not* be attracted to.
        model : object
            A FINCHES frontend.
        target_value : float
            Minimum acceptable mean epsilon against the panel, as a total (it
            is scaled by flank length like every other epsilon target).
        """

        can_be_linear_profile = False
        calculate_in_batch = False

        def __init__(self, decoys, model, target_value: float = 0.0,
                     weight: float = 1.0,
                     constraint_type=ConstraintType.MINIMUM):
            self.decoys = list(decoys)
            self._model = model
            super().__init__(target_value=target_value, weight=weight,
                             constraint_type=constraint_type)

        def get_init_args(self) -> dict:
            args = super().get_init_args()
            args.update({"decoys": self.decoys, "model": self._model})
            return args

        def calculate_raw_value(self, protein) -> float:
            seq = protein.sequence
            eps = self._model.epsilon
            return float(np.mean([eps(seq, d) for d in self.decoys]))

    _Avoidance.__name__ = class_name
    _Avoidance.__qualname__ = class_name
    return _Avoidance


def _make_shell_epsilon_class():
    """Build the ShellWeightedEpsilon property against GOOSE's base class."""
    from goose.backend.optimizer_properties import ConstraintType, CustomProperty

    class ShellWeightedEpsilon(CustomProperty):
        """Weighted-average epsilon over distance shells of the target patch.

        Raw value is ``sum(w_k * epsilon(flank, shell_k)) / sum(w_k)``, so it
        keeps the units and length-scaling of a plain epsilon while aiming the
        objective at the surface the flank actually reaches.

        A flat average over the whole patch is the wrong target: on every
        system tested roughly half the selected residues sit beyond 15 A of the
        anchor, where the tethered-chain monomer density is under a tenth of its
        value at contact, yet they contributed proportional to their count.
        Weighting cannot be done inside a single epsilon call because FINCHES
        takes a sequence, so it is done across a few calls instead.

        Parameters
        ----------
        shells : sequence of (str, float)
            ``(sequence, weight)`` per shell, from
            :meth:`~idr_flanks.interface.ProximalRegion.weighted_shells`.
        model : object
            A FINCHES frontend.
        """

        can_be_linear_profile = False
        calculate_in_batch = False

        def __init__(self, shells, model, target_value: float = 0.0,
                     weight: float = 1.0,
                     constraint_type=ConstraintType.MAXIMUM):
            self.shells = [(str(s), float(w)) for s, w in shells]
            total = sum(w for _, w in self.shells)
            if not self.shells or total <= 0:
                raise ValueError("shells must be non-empty with positive weight")
            self._norm = total
            self._model = model
            super().__init__(target_value=target_value, weight=weight,
                             constraint_type=constraint_type)

        def get_init_args(self) -> dict:
            args = super().get_init_args()
            args.update({"shells": self.shells, "model": self._model})
            return args

        def calculate_raw_value(self, protein) -> float:
            seq = protein.sequence
            eps = self._model.epsilon
            return float(sum(w * eps(seq, s) for s, w in self.shells)
                         / self._norm)

    return ShellWeightedEpsilon


_ShellWeightedEpsilon = None


def shell_epsilon_class():
    """The :class:`ShellWeightedEpsilon` class, for ``add_property``."""
    global _ShellWeightedEpsilon
    if _ShellWeightedEpsilon is None:
        _ShellWeightedEpsilon = _make_shell_epsilon_class()
    return _ShellWeightedEpsilon


def shell_weighted_epsilon(sequence: str, shells, model: str = "mpipi") -> float:
    """Weighted per-residue epsilon of ``sequence`` against distance shells."""
    if not sequence:
        raise ValueError("sequence must be non-empty")
    mf = load_epsilon_model(model)
    total = sum(w for _, w in shells)
    if total <= 0:
        raise ValueError("shell weights must sum to a positive value")
    return float(sum(w * mf.epsilon(sequence, s) for s, w in shells)
                 / total) / len(sequence)


def _make_competition_class():
    """Build the BinderCompetition property against GOOSE's base class."""
    from goose.backend.optimizer_properties import ConstraintType, CustomProperty

    class BinderCompetition(CustomProperty):
        """How much more the flank prefers the target over its own binder.

        Raw value is ``epsilon(flank, binder_interface) - epsilon(flank,
        target_patch)``, so a positive value means the flank is more attracted
        to the target than to the binder's target-binding surface, which is the
        condition for the flank to add affinity rather than compete for it.

        Expressed as a difference rather than an absolute ceiling on binder
        attraction on purpose: forbidding all binder attraction is far too
        strict when the two surfaces share chemistry, and it throws away the
        target attraction along with it.
        """

        can_be_linear_profile = False
        calculate_in_batch = False

        def __init__(self, binder_interface: str, target_patch: str, model,
                     target_value: float = 0.0, weight: float = 1.0,
                     constraint_type=ConstraintType.MINIMUM):
            self.binder_interface = binder_interface
            self.target_patch = target_patch
            self._model = model
            super().__init__(target_value=target_value, weight=weight,
                             constraint_type=constraint_type)

        def get_init_args(self) -> dict:
            args = super().get_init_args()
            args.update({"binder_interface": self.binder_interface,
                         "target_patch": self.target_patch,
                         "model": self._model})
            return args

        def calculate_raw_value(self, protein) -> float:
            seq = protein.sequence
            eps = self._model.epsilon
            return float(eps(seq, self.binder_interface)
                         - eps(seq, self.target_patch))

    return BinderCompetition


_BinderCompetition = None


def binder_competition_class():
    """The :class:`BinderCompetition` class, for ``add_property``."""
    global _BinderCompetition
    if _BinderCompetition is None:
        _BinderCompetition = _make_competition_class()
    return _BinderCompetition


_avoidance_classes: Dict[str, Any] = {}


def avoidance_class(class_name: str = "DecoyRepulsion"):
    """A sequence-avoidance property class, for ``add_property``."""
    cls = _avoidance_classes.get(class_name)
    if cls is None:
        cls = _make_avoidance_class(class_name)
        _avoidance_classes[class_name] = cls
    return cls


def decoy_repulsion_class():
    """Backwards-compatible alias for the decoy-panel avoidance property."""
    return avoidance_class("DecoyRepulsion")


def binder_avoidance_class():
    """Avoidance property for the binder's own target-binding surface."""
    return avoidance_class("BinderInterfaceAvoidance")


_ContextFractionDisorder = None


def ContextFractionDisorder(*args, **kwargs):  # noqa: N802 - factory shim
    """Instantiate the context-aware disorder property (see module docs)."""
    global _ContextFractionDisorder
    if _ContextFractionDisorder is None:
        _ContextFractionDisorder = _make_context_disorder_class()
    return _ContextFractionDisorder(*args, **kwargs)


def context_disorder_class():
    """The :class:`ContextFractionDisorder` class itself, for ``add_property``."""
    global _ContextFractionDisorder
    if _ContextFractionDisorder is None:
        _ContextFractionDisorder = _make_context_disorder_class()
    return _ContextFractionDisorder


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass
class DesignConfig:
    """Knobs controlling how a flank is designed.

    The defaults are the ``"balanced"`` preset. See :data:`PRESETS` for
    alternatives and :func:`design_flank` for usage.
    """

    # --- attraction ---
    epsilon_model: str = "mpipi"
    target_epsilon_per_residue: Optional[float] = None
    """Desired per-residue epsilon against the patch, as a ``maximum``
    constraint (so "at least this attractive"). Negative is attractive.
    ``None`` means "as attractive as the other constraints allow".

    Left unbounded by default because specificity is enforced directly by
    :attr:`max_decoy_epsilon_per_residue` instead. Capping potency would also
    penalise charge-complementary designs, which reach far stronger attraction
    than aromatic ones while staying selective."""
    epsilon_weight: float = 1.0

    # --- disorder ---
    disorder_target: float = 1.0
    disorder_cutoff: float = 0.5
    disorder_weight: float = 2.0
    """Weight on the disorder terms. 2.0 rather than 1.0 as cheap insurance:
    with the composition envelope on, in-context disorder lands at 0.97-1.00
    regardless, but without it a weight of 1.0 lets the epsilon term trade
    disorder away entirely at some seeds."""
    context_aware_disorder: bool = True
    """Score the flank's disorder while fused to the binder rather than alone.
    Strongly recommended: it is what stops aggregation-prone aromatic-rich
    designs from passing the disorder check."""

    # --- composition guardrails (hard constraints on mutation) ---
    composition_envelope: Optional[float] = 3.0
    """Ceiling on every residue at this multiple of its frequency in real
    disordered regions (see :func:`idr_amino_acid_frequencies`).

    Capping only aromatics does not remove the degeneracy, it relocates it: an
    unbounded attraction objective saturates on whichever residue is still
    legal, giving poly-glutamine or poly-aspartate instead of
    poly-tryptophan. Bounding all twenty residues is what prevents that.
    ``3.0`` keeps designs inside the composition range real IDRs occupy;
    values above ~4 let the single-residue pathology back in."""
    max_aromatic_fraction: Optional[float] = 0.10
    """Ceiling on the combined W+F+Y fraction, applied on top of the envelope.

    Only binds where it is tighter than the sum of the individual W, F and Y
    ceilings, which at the default multiplier is about 0.15. Real disordered
    regions average 0.05 aromatic, so 0.10 leaves headroom while still cutting
    in for longer flanks and whenever the envelope is widened."""
    max_aliphatic_fraction: Optional[float] = 0.25
    """Ceiling on the combined A+I+L+M+V fraction. Real disordered regions
    average 0.23, and the per-residue ceilings sum to about 0.70, so this is
    the binding constraint on aliphatics."""
    max_fraction_per_residue: Optional[Dict[str, float]] = None
    """Explicit per-residue ceilings, overriding the envelope for those
    residues."""
    extra_aa_fraction_ranges: Optional[Dict[Any, Any]] = None
    """Passed straight through to ``SequenceOptimizer(aa_fraction_ranges=...)``,
    merged over the ceilings above."""

    # --- do not compete with the target for the binder ---
    min_target_preference: Optional[float] = 0.05
    """Require the flank to be at least this much more attracted (per residue)
    to the target patch than to the binder's own target-binding surface.

    This is the most consequential guard in the package. A flank that likes the
    binder's interface competes with the target for it, so a "higher affinity"
    design can end up with *lower* net affinity. It is not hypothetical: given
    an acidic target patch and an acidic binder interface, the unguarded design
    attracted the binder's interface at -0.61 per residue versus -0.59 for the
    intended target -- it preferred the binder.

    A relative margin rather than an absolute ceiling on binder attraction:
    when the two surfaces share chemistry, forbidding binder attraction outright
    also destroys the target attraction. ``None`` disables the guard."""
    max_binder_epsilon_per_residue: Optional[float] = None
    """Optional hard ceiling on per-residue epsilon against the binder
    interface, on top of :attr:`min_target_preference`. Usually unnecessary and
    often over-restrictive; prefer the relative margin."""
    binder_weight: float = 2.0

    # --- specificity ---
    max_decoy_epsilon_per_residue: Optional[float] = None
    """Optionally require the flank's mean per-residue epsilon against a panel
    of unrelated sequences to be at least this value, i.e. not attractive to
    arbitrary sequence.

    Off by default because it is measurably redundant: the composition
    envelope already drives attraction to random sequence down to about -0.01
    per residue, and adding this term moved it only to -0.018 from -0.016 while
    making a run about 2.5x slower. The term is also not a substitute for the
    envelope -- used alone it still yields 30% aromatic flanks. Enable it as a
    belt-and-braces check (the ``"specific"`` preset does)."""
    n_decoys: int = 12
    """Size of the decoy panel used when the specificity term is enabled. Each
    decoy costs one extra epsilon evaluation per candidate."""
    decoy_weight: float = 1.0

    # --- solubility ---
    min_self_epsilon_per_residue: Optional[float] = None
    """If set, require the flank's self-interaction epsilon per residue to be
    at least this value (i.e. not self-attractive). ``0.0`` is a good choice
    when aggregation is the main worry."""
    self_epsilon_weight: float = 1.0

    min_context_disorder: float = 0.8
    """If the finished flank's in-context fraction-disordered falls below this,
    the design is retried once with the disorder weight tripled. Rarely
    triggers with the envelope on, and closes the hole when it is off. Set to
    ``0`` to disable the retry."""

    # --- optimizer ---
    max_iterations: int = 500
    num_starting_candidates: int = 200
    num_candidates: int = 5
    seed: Optional[int] = None
    """Seed for reproducible designs. Setting it also disables GOOSE's
    shuffling step and forces the pure-Python mutation path, both of which use
    RNGs a seed can actually control; GOOSE's default fast mutation path uses a
    C RNG seeded at import and ignores ``seed``."""
    verbose: bool = False

    def aa_fraction_ranges(self, length: Optional[int] = None
                           ) -> Optional[Dict[Any, Any]]:
        """Assemble the ``aa_fraction_ranges`` dict for the optimizer.

        GOOSE turns an upper fraction into a residue count with ``ceil``, so
        asking for ``0.15`` at length 30 would actually permit 5 residues
        (0.167). When ``length`` is known the bound is rounded down first, so
        the ceiling means what the caller asked for.

        Parameters
        ----------
        length : int, optional
            Flank length, used to make the ceilings exact.

        Returns
        -------
        dict or None
        """
        def bound(fraction: float, floor_at_one: bool = False) -> float:
            f = float(fraction)
            if length is None or length <= 0:
                return f
            max_count = math.floor(f * length)
            if floor_at_one:
                # Never let rounding ban a residue outright. Besides being
                # unrealistic, it makes the bounds infeasible for short flanks:
                # at length 5 the envelope's counts sum to 4, and GOOSE rejects
                # aa_fraction_ranges whose maximum counts cannot reach the
                # target length.
                max_count = max(1, max_count)
            # Nudge below the exact ratio: GOOSE recomputes ceil(high*length),
            # and in floating point ceil(0.14 * 50) is 8, not 7.
            return max(0.0, (max_count - 1e-9) / length)

        ranges: Dict[Any, Any] = {}
        if self.composition_envelope is not None:
            k = float(self.composition_envelope)
            for aa, freq in idr_amino_acid_frequencies().items():
                ranges[aa] = (0.0, bound(min(1.0, k * freq),
                                         floor_at_one=True))
        if self.max_aromatic_fraction is not None:
            ranges["".join(_AROMATICS)] = (0.0, bound(self.max_aromatic_fraction))
        if self.max_aliphatic_fraction is not None:
            ranges["".join(_ALIPHATICS)] = (0.0, bound(self.max_aliphatic_fraction))
        for aa, hi in (self.max_fraction_per_residue or {}).items():
            ranges[aa] = (0.0, bound(hi))
        if self.extra_aa_fraction_ranges:
            ranges.update(self.extra_aa_fraction_ranges)
        return ranges or None


PRESETS: Dict[str, Dict[str, Any]] = {
    "balanced": {},
    "aggressive": {
        # Chase affinity: wider envelope and a relaxed specificity floor.
        # Expect stickier flanks -- check the reported specificity.
        "composition_envelope": 4.0,
        "max_aromatic_fraction": 0.18,
        "max_aliphatic_fraction": 0.35,
        "max_iterations": 800,
    },
    "soluble": {
        # Prioritise staying in solution: tighter envelope, fewer aromatics,
        # and an explicit self-repulsion requirement.
        "composition_envelope": 2.5,
        "max_aromatic_fraction": 0.08,
        "min_self_epsilon_per_residue": 0.0,
        "max_iterations": 600,
    },
    "specific": {
        # Tighten the envelope and additionally require measured repulsion of
        # unrelated sequence. Slower, and rarely changes the answer.
        "composition_envelope": 2.5,
        "max_aromatic_fraction": 0.08,
        "max_decoy_epsilon_per_residue": 0.0,
        "max_iterations": 800,
    },
    "unconstrained": {
        # No guardrails at all: no composition envelope, no potency bound and
        # no specificity floor. Reproduces the degenerate poly-aromatic
        # behaviour; kept for comparison and for users imposing their own
        # constraints.
        "composition_envelope": None,
        "max_aromatic_fraction": None,
        "max_aliphatic_fraction": None,
        "max_decoy_epsilon_per_residue": None,
        "target_epsilon_per_residue": None,
    },
}


def _resolve_config(preset: Optional[str],
                    config: Optional[DesignConfig],
                    overrides: Dict[str, Any]) -> DesignConfig:
    if config is not None and preset is not None:
        raise ValueError("pass either preset or config, not both")
    if config is not None:
        base = config
    else:
        name = preset or "balanced"
        if name not in PRESETS:
            raise ValueError(
                f"Unknown preset {name!r}. Available: {sorted(PRESETS)}")
        base = DesignConfig(**PRESETS[name])
    if overrides:
        unknown = set(overrides) - set(base.__dataclass_fields__)
        if unknown:
            raise TypeError(
                f"Unknown design option(s): {sorted(unknown)}. "
                f"Valid options: {sorted(base.__dataclass_fields__)}")
        base = replace(base, **overrides)
    return base


# ---------------------------------------------------------------------------
# results and scoring
# ---------------------------------------------------------------------------

@dataclass
class DesignResult:
    """A designed flank plus the diagnostics needed to judge it."""

    sequence: str
    patch_sequence: str
    """The patch string actually optimised against. When proximity weighting is
    used this contains repeated residues, so it is longer than the selected
    region; :attr:`selected_patch_sequence` is the region as chosen."""
    config: DesignConfig
    epsilon_per_residue: float
    epsilon_total: float
    self_epsilon_per_residue: float
    fraction_disorder: float
    fraction_disorder_in_context: float
    aromatic_fraction: float
    fcr: float = float("nan")
    ncpr: float = float("nan")
    kappa: float = float("nan")
    hydropathy: float = float("nan")
    specificity_z: float = float("nan")
    specificity_delta: float = float("nan")
    decoy_epsilon_mean: float = float("nan")
    decoy_epsilon_sd: float = float("nan")
    epsilon_weighted: float = float("nan")
    """Reach-weighted epsilon: the average over distance shells that the
    objective actually optimises when shells are supplied."""
    epsilon_near_shell: float = float("nan")
    """Epsilon against the innermost shell only -- the surface the flank is
    most likely to touch."""
    reference_epsilon_per_residue: float = float("nan")
    reference_epsilon_sd: float = float("nan")
    binder_interface_sequence: str = ""
    epsilon_vs_binder_interface: float = float("nan")
    complexity: float = float("nan")
    selected_patch_sequence: str = ""
    """The target region as selected, when :attr:`patch_sequence` contains
    proximity-weighted repeats of it. Empty when no weighting was applied."""
    cross_reactivity: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.sequence

    @property
    def target_preference(self) -> float:
        """How much more the flank likes the target than the binder interface.

        Positive means the flank adds affinity; zero or negative means it
        competes with the target for the binder. This, not the raw sign of
        :attr:`epsilon_vs_binder_interface`, is the criterion that matters.
        """
        return self.epsilon_vs_binder_interface - self.epsilon_per_residue

    def summary(self) -> str:
        lines = [
            f"Designed flank ({len(self.sequence)} residues)",
            f"  sequence               : {self.sequence}",
            f"  target patch           : "
            + (self.selected_patch_sequence or self.patch_sequence),
            *( [f"  weighted patch         : {self.patch_sequence} "
                f"(residues near the attachment point repeated, so the design "
                f"favours them)"]
               if self.selected_patch_sequence
               and self.selected_patch_sequence != self.patch_sequence else [] ),
            f"  epsilon vs patch       : {self.epsilon_total:.2f} "
            f"({self.epsilon_per_residue:+.3f} per residue; negative is attractive)",
            *( [f"  reach-weighted epsilon : {self.epsilon_weighted:+.3f} per "
                f"residue (what the objective optimises); near shell only "
                f"{self.epsilon_near_shell:+.3f}"]
               if self.epsilon_weighted == self.epsilon_weighted else [] ),
            f"  neutral reference      : "
            f"{self.reference_epsilon_per_residue:+.3f} per residue"
            + (f" (sd {self.reference_epsilon_sd:.3f})"
               if self.reference_epsilon_sd == self.reference_epsilon_sd else ""),
            f"  self-epsilon           : {self.self_epsilon_per_residue:+.3f} per residue "
            f"({'self-attractive, aggregation risk' if self.self_epsilon_per_residue < 0 else 'self-repulsive, soluble'})",
            f"  fraction disordered    : {self.fraction_disorder:.2f} alone, "
            f"{self.fraction_disorder_in_context:.2f} fused to the binder",
            *( [f"  vs binder interface    : "
                f"{self.epsilon_vs_binder_interface:+.3f} per residue; "
                f"prefers the target by {self.target_preference:+.3f} "
                f"({'COMPETES with the target' if self.target_preference <= 0 else 'ok'})"]
               if self.target_preference == self.target_preference
               else [] ),
            f"  aromatic (WFY) fraction: {self.aromatic_fraction:.2f}"
            + (f", complexity {self.complexity:.2f}"
               if self.complexity == self.complexity else ""),
            f"  FCR / NCPR             : {self.fcr:.2f} / {self.ncpr:+.2f}"
            + (f" / kappa {self.kappa:.2f}"
               if self.kappa == self.kappa else " (kappa undefined: one charge sign)"),
        ]
        if self.specificity_delta == self.specificity_delta:
            lines.append(
                f"  selectivity (context)  : {self.specificity_delta:+.3f} per "
                f"residue more attracted to the patch than to random sequence; "
                f"random-sequence attraction {self.decoy_epsilon_mean:+.3f}. "
                f"The binder supplies target specificity, so the number that "
                f"matters here is the second one staying near zero.")
        if self.cross_reactivity:
            profile = ", ".join(f"{k}={v:+.2f}"
                                for k, v in self.cross_reactivity.items())
            lines.append(f"  cross-reactivity       : {profile}")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)


def _random_decoys(length: int, n: int = 40, seed: int = 0) -> List[str]:
    """Background-composition decoy patches.

    Specificity is measured against these rather than against chemical
    extremes: a panel containing poly-lysine and poly-tryptophan has a huge
    spread of epsilon values, and dividing by that spread hides real
    preference. What we want to know is whether the flank prefers this patch
    over an ordinary patch it might meet elsewhere in a proteome.
    """
    rng = np.random.default_rng(seed)
    letters = np.array(list(_BACKGROUND))
    probs = np.array([_BACKGROUND[a] for a in letters], dtype=np.float64)
    probs = probs / probs.sum()
    return ["".join(rng.choice(letters, size=length, p=probs))
            for _ in range(n)]


def _extreme_decoys(length: int) -> Dict[str, str]:
    """Chemically extreme patches, reported as a cross-reactivity profile."""
    return {
        "basic": ("KR" * length)[:length],
        "acidic": ("DE" * length)[:length],
        "polar": ("QN" * length)[:length],
        "flexible": ("GS" * length)[:length],
        "aliphatic": ("LV" * length)[:length],
        "aromatic": ("WFY" * length)[:length],
    }


# Chemically diverse probes used to test whether two surfaces can be told apart
# at all. Each is a homo- or di-polymer of one chemical class, so together they
# span the directions a designed flank could move in.
_DISCRIMINATION_PROBES = (
    "K", "R", "E", "D", "W", "Y", "F", "L", "I", "V",
    "Q", "N", "S", "T", "G", "P", "H", "M", "A", "C",
)


def target_discriminability(target_patch: str, binder_interface: str,
                            model: str = "mpipi",
                            probe_length: int = 25) -> float:
    """Best per-residue preference for the target over the binder achievable.

    A flank can only avoid competing with the target if some chemistry is more
    attracted to the target patch than to the binder's own interface. Scanning a
    panel of single-chemistry probes gives an UPPER BOUND on that gap: if even
    the best homopolymer cannot prefer the target, no design can, because the
    two surfaces are chemically indistinguishable to the interaction model.

    Read it as an upper bound and nothing more. The probes are homopolymers,
    which the composition envelope forbids, so a real design reaches only part
    of this figure -- measured at +0.239 achieved against a +0.716 bound on
    1YCR. It is therefore reliable for proving a guard *impossible* and
    optimistic about proving one attainable, which is why the achieved
    preference is also checked after the fact.

    Parameters
    ----------
    target_patch : str
        The target region the flank should complement.
    binder_interface : str
        The binder residues that contact the target.
    model : str
        Epsilon model name.
    probe_length : int
        Length of the probe sequences.

    Returns
    -------
    float
        ``max`` over probes of ``eps(probe, binder) - eps(probe, target)``, per
        residue. Positive means a preference for the target is attainable.
    """
    if not target_patch or not binder_interface:
        return float("inf")
    mf = load_epsilon_model(model)
    best = -float("inf")
    for aa in _DISCRIMINATION_PROBES:
        probe = aa * probe_length
        gap = (float(mf.epsilon(probe, binder_interface))
               - float(mf.epsilon(probe, target_patch))) / probe_length
        best = max(best, gap)
    return best


def _neutral_reference(length: int, seed: int = 0) -> str:
    """A background-composition sequence used as the neutral epsilon baseline."""
    rng = np.random.default_rng(seed)
    letters = np.array(list(_BACKGROUND))
    probs = np.array([_BACKGROUND[a] for a in letters], dtype=np.float64)
    probs = probs / probs.sum()
    return "".join(rng.choice(letters, size=length, p=probs))


def score_flank(sequence: str, patch_sequence: str,
                n_context: str = "", c_context: str = "",
                binder_interface: str = "",
                shells: Optional[Sequence] = None,
                model: str = "mpipi",
                disorder_cutoff: float = 0.5,
                specificity: bool = True,
                n_decoys: int = 40,
                seed: int = 0) -> Dict[str, float]:
    """Measure a flank: attraction, specificity, disorder, and composition.

    Parameters
    ----------
    sequence : str
        The flank to score.
    patch_sequence : str
        The target patch it was designed against.
    n_context, c_context : str
        Sequence preceding/following the flank in the final construct, used for
        the in-context disorder number.
    binder_interface : str
        The binder residues that contact the target. Epsilon against these is
        reported so competition with the target can be seen.
    model : str
        Epsilon model.
    disorder_cutoff : float
        Per-residue disorder threshold.
    specificity : bool
        Compute epsilon against a decoy panel and report a z-score. This is
        what distinguishes "complementary to this patch" from "sticky to
        everything".
    n_decoys : int
        Number of random background-composition decoys.
    seed : int
        Seed for the decoy panel, so scores are reproducible.

    Returns
    -------
    dict
    """
    import metapredict as meta
    from sparrow import Protein

    if not sequence:
        raise ValueError("sequence must be non-empty")
    mf = load_epsilon_model(model)
    L = len(sequence)

    eps_total = float(mf.epsilon(sequence, patch_sequence))
    out: Dict[str, float] = {
        "epsilon_total": eps_total,
        "epsilon_per_residue": eps_total / L,
        "self_epsilon_per_residue": float(mf.epsilon(sequence, sequence)) / L,
    }

    scores = np.asarray(meta.predict_disorder(sequence), dtype=np.float64)
    out["fraction_disorder"] = float((scores > disorder_cutoff).sum() / L)
    out["mean_disorder"] = float(scores.mean())

    if n_context or c_context:
        full = n_context + sequence + c_context
        ctx = np.asarray(meta.predict_disorder(full), dtype=np.float64)
        seg = ctx[len(n_context):len(n_context) + L]
        out["fraction_disorder_in_context"] = float(
            (seg > disorder_cutoff).sum() / L)
        out["mean_disorder_in_context"] = float(seg.mean())
    else:
        out["fraction_disorder_in_context"] = out["fraction_disorder"]
        out["mean_disorder_in_context"] = out["mean_disorder"]

    if binder_interface:
        out["epsilon_vs_binder_interface"] = float(
            mf.epsilon(sequence, binder_interface)) / L

    if shells:
        total = sum(w for _, w in shells)
        if total > 0:
            out["epsilon_weighted"] = float(
                sum(w * mf.epsilon(sequence, s) for s, w in shells)
                / total) / L
        # The innermost shell is the surface the flank actually touches.
        inner = shells[0][0]
        out["epsilon_near_shell"] = float(mf.epsilon(sequence, inner)) / L

    out["aromatic_fraction"] = sum(sequence.count(a) for a in _AROMATICS) / L
    try:
        p = Protein(sequence)
        out["complexity"] = float(p.complexity)
        out["fcr"] = float(p.FCR)
        out["ncpr"] = float(p.NCPR)
        # sparrow returns -1 when kappa is undefined, which happens whenever a
        # sequence carries only one sign of charge -- common for these designs.
        kappa = float(p.kappa)
        out["kappa"] = float("nan") if kappa < 0 else kappa
        out["hydropathy"] = float(p.hydrophobicity)
    except Exception:  # pragma: no cover - sparrow property edge cases
        pass

    # Average over many draws: a single background sequence is noisy enough
    # (sd ~0.037, a quarter of a typical design signal) that the
    # "no better than background" warning would otherwise be a coin flip.
    refs = [float(mf.epsilon(_neutral_reference(L, seed=seed + k),
                             patch_sequence)) / L
            for k in range(_N_REFERENCE_DRAWS)]
    out["reference_epsilon_per_residue"] = float(np.mean(refs))
    out["reference_epsilon_sd"] = float(np.std(refs))

    if specificity and n_decoys > 0:
        plen = max(len(patch_sequence), 10)
        vals = np.array([float(mf.epsilon(sequence, d)) / L
                         for d in _random_decoys(plen, n=n_decoys, seed=seed)])
        mu, sd = float(vals.mean()), float(vals.std())
        out["decoy_epsilon_mean"] = mu
        out["decoy_epsilon_sd"] = sd
        # Negative epsilon is attractive, so a value more negative than the
        # decoys means genuine preference. Reported so that positive is good.
        out["specificity_z"] = ((mu - out["epsilon_per_residue"]) / sd
                                if sd > 0 else float("nan"))
        out["specificity_delta"] = mu - out["epsilon_per_residue"]
        for name, decoy in _extreme_decoys(plen).items():
            out[f"epsilon_vs_{name}"] = float(mf.epsilon(sequence, decoy)) / L
    return out


# ---------------------------------------------------------------------------
# the design driver
# ---------------------------------------------------------------------------

def design_flank(patch_sequence: str,
                 length: int,
                 *,
                 n_context: str = "",
                 c_context: str = "",
                 selected_patch: Optional[str] = None,
                 binder_interface: str = "",
                 shells: Optional[Sequence] = None,
                 preset: Optional[str] = None,
                 config: Optional[DesignConfig] = None,
                 **overrides: Any) -> DesignResult:
    """Design a flank, retrying once if it comes out insufficiently disordered.

    See :func:`_design_flank_once` for the parameters; they are identical.
    """
    cfg = _resolve_config(preset, config, overrides)
    result = _design_flank_once(patch_sequence, length, n_context=n_context,
                                c_context=c_context,
                                selected_patch=selected_patch,
                                binder_interface=binder_interface,
                                shells=shells, config=cfg)

    # Retrying only helps if a disorder term is actually in play. Without any
    # sequence context the in-context number equals the isolated one, and a
    # second attempt would report a fix it did not make.
    has_context = bool(n_context or c_context)
    threshold = cfg.min_context_disorder
    if (threshold and has_context
            and result.fraction_disorder_in_context < threshold):
        # GOOSE balances objectives by normalised error, so a strong epsilon
        # term can trade disorder away. Escalate the disorder weight and try
        # once more, keeping whichever attempt is more disordered.
        retry_cfg = replace(cfg, disorder_weight=cfg.disorder_weight * 3.0)
        retry = _design_flank_once(patch_sequence, length, n_context=n_context,
                                   c_context=c_context,
                                   selected_patch=selected_patch,
                                   binder_interface=binder_interface,
                                   shells=shells, config=retry_cfg)
        better = (retry if retry.fraction_disorder_in_context
                  > result.fraction_disorder_in_context else result)
        better.warnings.insert(0, (
            f"the first attempt was only "
            f"{result.fraction_disorder_in_context:.0%} disordered in context, "
            f"so the design was retried with the disorder weight tripled "
            f"(now {better.fraction_disorder_in_context:.0%})."))
        return better
    return result


def _design_flank_once(patch_sequence: str,
                       length: int,
                       *,
                       n_context: str = "",
                       c_context: str = "",
                       selected_patch: Optional[str] = None,
                       binder_interface: str = "",
                       shells: Optional[Sequence] = None,
                       preset: Optional[str] = None,
                       config: Optional[DesignConfig] = None,
                       **overrides: Any) -> DesignResult:
    """Design a disordered flank complementary to ``patch_sequence``.

    Parameters
    ----------
    patch_sequence : str
        Concatenated target residues the flank should be attracted to, as
        produced by :meth:`~idr_flanks.interface.ProximalRegion.patch_sequence`.
    length : int
        Number of residues to design.
    n_context, c_context : str
        Sequence that will precede and follow the flank in the finished
        construct. For a flank on the binder's N-terminus, pass the binder
        sequence as ``c_context``; for a C-terminal flank pass it as
        ``n_context``. Used for context-aware disorder.
    selected_patch : str, optional
        The target region as selected, for reporting, when ``patch_sequence``
        contains proximity-weighted repeats of it.
    preset : str, optional
        One of :data:`PRESETS`. Defaults to ``"balanced"``.
    config : DesignConfig, optional
        A fully specified configuration, instead of a preset.
    **overrides
        Individual :class:`DesignConfig` fields to override.

    Returns
    -------
    DesignResult

    Raises
    ------
    DesignError
        If GOOSE is unavailable or the optimizer cannot produce a sequence.
    """
    if not patch_sequence:
        raise ValueError("patch_sequence must be non-empty")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    cfg = _resolve_config(preset, config, overrides)

    try:
        from goose.optimize import SequenceOptimizer
        from goose.backend.optimizer_properties import (
            FractionDisorder,
            MeanEpsilonWithTarget,
            MeanSelfEpsilon,
        )
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DesignError(
            "GOOSE is required to design flanks but could not be imported "
            f"({exc}). Install it from https://github.com/idptools/goose ."
        ) from None

    model = load_epsilon_model(cfg.epsilon_model)

    # SequenceOptimizer.__init__ seeds the global random and numpy RNGs, so the
    # snapshot has to be taken before it is constructed, not before run().
    rng_state = random.getstate()
    np_state = np.random.get_state()

    aa_ranges = cfg.aa_fraction_ranges(length)
    if cfg.seed is not None and not aa_ranges:
        # GOOSE's fast mutation path uses C rand(), seeded once at import, so
        # seed= has no effect there. Supplying any aa_fraction_ranges routes
        # mutation through the pure-Python RNG that seed= actually controls, so
        # a vacuous range is added to keep seeded runs reproducible.
        aa_ranges = {"G": (0.0, 1.0)}

    try:
        optimizer = SequenceOptimizer(
            target_length=length,
            max_iterations=cfg.max_iterations,
            num_starting_candidates=cfg.num_starting_candidates,
            num_candidates=cfg.num_candidates,
            aa_fraction_ranges=aa_ranges,
            seed=cfg.seed,
            verbose=cfg.verbose,
            # GOOSE stops as soon as every property is inside its tolerance,
            # which for a reachable epsilon target happens within a handful of
            # iterations and returns a barely-optimised sequence. Runs are
            # seconds long, so always take the full budget.
            enable_error_tolerance=False,
            enable_shuffling=cfg.seed is None,
        )
    except Exception as exc:
        # The constructor seeds the global RNGs, so restore them before the
        # error propagates.
        random.setstate(rng_state)
        np.random.set_state(np_state)
        raise DesignError(
            f"could not configure the optimizer for a flank of length "
            f"{length}: {exc}") from exc

    # --- disorder ---
    # The flank must be disordered on its own *and* once fused to the binder.
    # Requiring only the in-context number is gameable -- the optimizer will
    # happily find a segment that metapredict calls disordered only because of
    # its neighbours -- so both are enforced when a context is supplied.
    optimizer.add_property(
        FractionDisorder,
        target_value=cfg.disorder_target,
        disorder_cutoff=cfg.disorder_cutoff,
        weight=cfg.disorder_weight,
        constraint_type="minimum",
        tolerance=0.01,
    )
    if cfg.context_aware_disorder and (n_context or c_context):
        optimizer.add_property(
            context_disorder_class(),
            n_context=n_context,
            c_context=c_context,
            target_value=cfg.disorder_target,
            disorder_cutoff=cfg.disorder_cutoff,
            weight=cfg.disorder_weight,
            constraint_type="minimum",
            tolerance=0.01,
        )

    # --- attraction to the patch ---
    # Epsilon scales with flank length, so a per-residue target is converted to
    # a total. With no explicit target we set an unreachable one under a
    # MAXIMUM constraint, which makes the objective "as attractive as the other
    # constraints allow".
    if cfg.target_epsilon_per_residue is None:
        eps_target = -1000.0 * length
    else:
        eps_target = float(cfg.target_epsilon_per_residue) * length
    if shells:
        optimizer.add_property(
            shell_epsilon_class(),
            shells=shells,
            model=model,
            target_value=eps_target,
            weight=cfg.epsilon_weight,
            constraint_type="maximum",
            tolerance=0.1,
        )
    else:
        optimizer.add_property(
            MeanEpsilonWithTarget,
            target_sequence=patch_sequence,
            target_value=eps_target,
            weight=cfg.epsilon_weight,
            constraint_type="maximum",
            preloaded_model=model,
            tolerance=0.1,
        )

    # --- do not compete with the target for the binder's own interface ---
    # Check first whether the two surfaces can be told apart at all. If they
    # cannot, the constraint is unsatisfiable and adding it only degrades the
    # target attraction, so it is skipped and the caller is told why.
    infeasible_note = ""
    preference_target = (None if cfg.min_target_preference is None
                         else float(cfg.min_target_preference))
    if binder_interface and preference_target is not None:
        headroom = target_discriminability(patch_sequence, binder_interface,
                                           model=cfg.epsilon_model)
        if headroom <= _MIN_USEFUL_HEADROOM:
            preference_target = None
            infeasible_note = (
                f"the target patch and the binder's own interface are too "
                f"chemically alike to separate: the best achievable preference "
                f"for the target is {headroom:+.3f} per residue. Any flank "
                f"attracted to this target region will also be attracted to "
                f"the binder's interface and compete with the target. Consider "
                f"the other terminus, or a target region further from the "
                f"existing interface.")
        elif headroom < preference_target:
            # Asking for more than the chemistry allows used to disable the
            # guard outright, which let the design compete. Clamp to a fraction
            # of what is actually attainable and keep the guard active.
            clamped = _HEADROOM_FRACTION * headroom
            infeasible_note = (
                f"the requested target preference of "
                f"{preference_target:+.3f} per residue exceeds what these two "
                f"surfaces allow (best attainable {headroom:+.3f}); the "
                f"constraint was reduced to {clamped:+.3f} rather than dropped. "
                f"Competition with the target is still a risk here.")
            preference_target = clamped

    if binder_interface and preference_target is not None:
        optimizer.add_property(
            binder_competition_class(),
            binder_interface=binder_interface,
            target_patch=patch_sequence,
            model=model,
            target_value=preference_target * length,
            weight=cfg.binder_weight,
            constraint_type="minimum",
            tolerance=0.1,
        )
    if binder_interface and cfg.max_binder_epsilon_per_residue is not None:
        optimizer.add_property(
            binder_avoidance_class(),
            decoys=[binder_interface],
            model=model,
            target_value=float(cfg.max_binder_epsilon_per_residue) * length,
            weight=cfg.binder_weight,
            constraint_type="minimum",
            tolerance=0.1,
        )

    # --- specificity: stay non-attractive to unrelated sequence ---
    if cfg.max_decoy_epsilon_per_residue is not None and cfg.n_decoys > 0:
        decoys = _random_decoys(max(len(patch_sequence), 20),
                               n=cfg.n_decoys,
                               seed=0 if cfg.seed is None else cfg.seed)
        optimizer.add_property(
            decoy_repulsion_class(),
            decoys=decoys,
            model=model,
            target_value=float(cfg.max_decoy_epsilon_per_residue) * length,
            weight=cfg.decoy_weight,
            constraint_type="minimum",
            tolerance=0.1,
        )

    # --- optional self-repulsion, to discourage aggregation ---
    if cfg.min_self_epsilon_per_residue is not None:
        optimizer.add_property(
            MeanSelfEpsilon,
            target_value=float(cfg.min_self_epsilon_per_residue) * length,
            weight=cfg.self_epsilon_weight,
            constraint_type="minimum",
            preloaded_model=model,
            tolerance=0.1,
        )

    try:
        sequence = optimizer.run()
    except DesignError:
        raise
    except Exception as exc:
        raise DesignError(f"GOOSE optimization failed: {exc}") from exc
    finally:
        random.setstate(rng_state)
        np.random.set_state(np_state)
    if not sequence or len(sequence) != length:
        raise DesignError(
            f"GOOSE returned a sequence of length "
            f"{0 if not sequence else len(sequence)}, expected {length}.")

    scores = score_flank(sequence, patch_sequence,
                         n_context=n_context, c_context=c_context,
                         binder_interface=binder_interface,
                         shells=shells,
                         model=cfg.epsilon_model,
                         disorder_cutoff=cfg.disorder_cutoff,
                         seed=0 if cfg.seed is None else cfg.seed)

    warnings: List[str] = []
    if infeasible_note:
        warnings.append(infeasible_note)
    # Judge the region the user selected, not the repeated string handed to the
    # optimiser: proximity weighting lengthens the patch and would silence this.
    judged_patch = selected_patch or patch_sequence
    distinct_patch = len(set(judged_patch))
    if len(judged_patch) < 5:
        warnings.append(
            f"the target patch is only {len(judged_patch)} residue(s) long, "
            f"so the epsilon objective is dominated by a handful of residues "
            f"and this design should not be trusted.")
    elif distinct_patch <= 2:
        warnings.append(
            f"the target patch contains only {distinct_patch} distinct residue "
            f"type(s); there is very little chemistry to complement.")
    if scores["self_epsilon_per_residue"] < 0:
        warnings.append(
            "the flank is self-attractive (self-epsilon < 0), so it may "
            "aggregate; try preset='soluble' or a lower "
            "max_aromatic_fraction.")
    if scores["fraction_disorder_in_context"] < 0.8:
        warnings.append(
            f"only {scores['fraction_disorder_in_context']:.0%} of the flank "
            f"is predicted disordered once fused to the binder.")
    eps = scores["epsilon_per_residue"]
    if eps >= 0:
        warnings.append(
            f"the flank is net repelled from the target patch ({eps:+.3f} per "
            f"residue; negative is attractive), so the model does not predict "
            f"it will add any affinity. This patch may have no complementable "
            f"chemistry -- try the other terminus or a different region.")
    elif eps >= scores["reference_epsilon_per_residue"]:
        warnings.append(
            "the flank is no more attractive to the patch than a "
            "background-composition sequence; the patch may offer little to "
            "complement.")
    delta = scores.get("specificity_delta", float("nan"))
    if delta == delta and delta < 0:
        warnings.append(
            f"the flank is more attracted to random sequence than to the "
            f"intended patch ({delta:+.3f} per residue); the design is not "
            f"on-target.")
    binder_eps = scores.get("epsilon_vs_binder_interface", float("nan"))
    if binder_eps == binder_eps:
        preference = binder_eps - scores["epsilon_per_residue"]
        # Verify the margin was actually achieved. The feasibility probe is an
        # upper bound over homopolymers, so it can promise more than a
        # composition-constrained design can deliver; only the finished flank
        # settles it.
        requested = cfg.min_target_preference
        if (requested is not None and preference > 0
                and preference < float(requested)):
            warnings.append(
                f"the flank prefers the target by only {preference:+.3f} per "
                f"residue, short of the {float(requested):+.3f} requested. The "
                f"guard could not be fully satisfied, so competition with the "
                f"target remains a risk.")
        if preference <= 0:
            warnings.append(
                f"the flank prefers the binder's own target-binding surface "
                f"({binder_eps:+.3f} per residue) to the target "
                f"({scores['epsilon_per_residue']:+.3f}), so it will compete "
                f"with the target and is likely to reduce net affinity. The "
                f"two surfaces may be too chemically similar to tell apart.")
        elif preference < 0.02 and cfg.min_target_preference is None:
            warnings.append(
                f"the flank barely prefers the target ({preference:+.3f} per "
                f"residue) over the binder's own interface; competition is a "
                f"real risk.")

    # Deliberately no warning for modest specificity against random sequence.
    # The binder supplies the specificity; the flank's job is added avidity, so
    # a flank that is only mildly selective on its own is expected and fine.
    # Generic stickiness is a different matter and is still flagged below.
    decoy_mean = scores.get("decoy_epsilon_mean", float("nan"))
    if decoy_mean == decoy_mean and decoy_mean < -0.1:
        warnings.append(
            f"the flank is generically sticky: it attracts random sequence at "
            f"{decoy_mean:+.3f} per residue, which usually means off-target "
            f"binding.")

    return DesignResult(
        sequence=sequence,
        patch_sequence=patch_sequence,
        config=cfg,
        selected_patch_sequence=selected_patch or "",
        epsilon_per_residue=scores["epsilon_per_residue"],
        epsilon_total=scores["epsilon_total"],
        self_epsilon_per_residue=scores["self_epsilon_per_residue"],
        fraction_disorder=scores["fraction_disorder"],
        fraction_disorder_in_context=scores["fraction_disorder_in_context"],
        aromatic_fraction=scores["aromatic_fraction"],
        fcr=scores.get("fcr", float("nan")),
        ncpr=scores.get("ncpr", float("nan")),
        kappa=scores.get("kappa", float("nan")),
        hydropathy=scores.get("hydropathy", float("nan")),
        specificity_z=scores.get("specificity_z", float("nan")),
        specificity_delta=scores.get("specificity_delta", float("nan")),
        decoy_epsilon_mean=scores.get("decoy_epsilon_mean", float("nan")),
        decoy_epsilon_sd=scores.get("decoy_epsilon_sd", float("nan")),
        epsilon_weighted=scores.get("epsilon_weighted", float("nan")),
        epsilon_near_shell=scores.get("epsilon_near_shell", float("nan")),
        reference_epsilon_per_residue=scores["reference_epsilon_per_residue"],
        reference_epsilon_sd=scores.get("reference_epsilon_sd", float("nan")),
        binder_interface_sequence=binder_interface,
        epsilon_vs_binder_interface=scores.get(
            "epsilon_vs_binder_interface", float("nan")),
        complexity=scores.get("complexity", float("nan")),
        # binder_interface has its own reported line, so keep it out of the
        # generic cross-reactivity profile.
        cross_reactivity={k[len("epsilon_vs_"):]: v
                          for k, v in scores.items()
                          if k.startswith("epsilon_vs_")
                          and k != "epsilon_vs_binder_interface"},
        warnings=warnings,
    )
