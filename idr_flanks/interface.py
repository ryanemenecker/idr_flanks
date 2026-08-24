"""Find the region of a target chain that a new flank on a binder could reach.

Two ideas drive this module.

**Physical proximity.** Which target residues sit close to the binder, and in
particular close to the binder terminus where the new flank will be attached.
A flank on the binder's C-terminus can only reach target surface near that
terminus, so proximity is measured from a terminal *anchor*, not from the whole
binder.

**Sequence locality.** These are usually predicted structures. Predictors
routinely place a distant part of the target near the binder even though that
region would not really participate in binding. A target residue is therefore
only accepted if it is also close *in sequence* to the genuine interface, which
is identified independently from binder-wide contacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .io import Chain, Residue, Structure
from .sasa import relative_residue_sasa

__all__ = [
    "ProximalResidue",
    "ProximalRegion",
    "InterfaceError",
    "find_proximal_region",
    "contact_map",
    "reach_radius",
    "end_to_end_distance",
    "tether_contact_weight",
    "tether_contact_weights",
    "min_distances_to",
    "parse_residue_spec",
]


class InterfaceError(ValueError):
    """Raised when the requested interface cannot be analysed."""


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def _kdtree(points: np.ndarray):
    """Build a KD-tree if scipy is available, else return None."""
    if points.shape[0] == 0:
        return None
    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        return None
    return cKDTree(points)


def _nearest_distance(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Distance from each point in ``query`` to the closest point in ``reference``.

    Uses a KD-tree when scipy is present (O(n log n)); otherwise falls back to
    a chunked brute-force pass so memory stays bounded on large complexes.
    """
    if query.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if reference.shape[0] == 0:
        return np.full(query.shape[0], np.inf)

    tree = _kdtree(reference)
    if tree is not None:
        dist, _ = tree.query(query, k=1)
        return np.asarray(dist, dtype=np.float64)

    out = np.empty(query.shape[0], dtype=np.float64)
    chunk = max(1, int(4_000_000 // max(reference.shape[0], 1)))
    for start in range(0, query.shape[0], chunk):
        stop = start + chunk
        block = query[start:stop]
        d = np.linalg.norm(block[:, None, :] - reference[None, :, :], axis=-1)
        out[start:stop] = d.min(axis=1)
    return out


def min_distances_to(chain: Chain, reference: np.ndarray) -> np.ndarray:
    """Minimum heavy-atom distance from each residue of ``chain`` to ``reference``.

    Parameters
    ----------
    chain : Chain
        Chain whose residues are measured.
    reference : ndarray, shape (n, 3)
        Coordinates to measure against.

    Returns
    -------
    ndarray, shape (len(chain),)
        ``result[i]`` is the shortest distance between any heavy atom of
        ``chain[i]`` and any point in ``reference``. Residues with no heavy
        atoms give ``inf``.
    """
    coords, owner = chain.stacked_heavy_coords()
    per_atom = _nearest_distance(coords, np.asarray(reference, dtype=np.float64))
    out = np.full(len(chain), np.inf, dtype=np.float64)
    if per_atom.size:
        np.minimum.at(out, owner, per_atom)
    return out


def contact_map(chain_a: Chain, chain_b: Chain,
                cutoff: float = 5.0) -> np.ndarray:
    """Boolean matrix of residue-residue heavy-atom contacts within ``cutoff``.

    Parameters
    ----------
    chain_a, chain_b : Chain
    cutoff : float
        Heavy-atom distance cutoff in angstroms.

    Returns
    -------
    ndarray of bool, shape (len(chain_a), len(chain_b))
    """
    ca, oa = chain_a.stacked_heavy_coords()
    cb, ob = chain_b.stacked_heavy_coords()
    out = np.zeros((len(chain_a), len(chain_b)), dtype=bool)
    if ca.shape[0] == 0 or cb.shape[0] == 0:
        return out

    tree_a, tree_b = _kdtree(ca), _kdtree(cb)
    if tree_a is not None and tree_b is not None:
        pairs = tree_a.sparse_distance_matrix(tree_b, cutoff,
                                              output_type="ndarray")
        if pairs.size:
            out[oa[pairs["i"]], ob[pairs["j"]]] = True
        return out

    chunk = max(1, int(4_000_000 // max(cb.shape[0], 1)))
    for start in range(0, ca.shape[0], chunk):
        stop = start + chunk
        d = np.linalg.norm(ca[start:stop, None, :] - cb[None, :, :], axis=-1)
        hit = d <= cutoff
        ia, ib = np.nonzero(hit)
        out[oa[start:stop][ia], ob[ib]] = True
    return out


# ---------------------------------------------------------------------------
# how far can a disordered flank reach?
# ---------------------------------------------------------------------------

# Root-mean-square end-to-end distance of a disordered chain, Ree = PREFACTOR *
# N ** EXPONENT, in angstroms.
#
# Provenance, stated precisely because it is easy to overclaim: the exponent
# 0.52 and prefactor come from the empirical IDR radius-of-gyration scaling
# Rg ~ 2.54 * N**0.522 converted with Ree = sqrt(6) * Rg, which is the *ideal
# chain* relation. 0.52 is therefore an empirical fit to IDR ensembles, close to
# but not identical with the ideal-chain value of 0.5, and below the
# self-avoiding-walk value of ~0.588. Real IDRs sit between those limits and the
# exponent varies with sequence charge and hydrophobicity, so treat the radius
# as an order-of-magnitude guide, not a measurement.
_REACH_PREFACTOR_DOC = None
# Two heavy atoms can only occlude each other's solvent shell within roughly
# r_i + r_j + 2 * probe_radius; 7 A covers the largest protein pairing.
_OCCLUSION_RANGE = 7.0

_REACH_PREFACTOR = 6.2
_REACH_EXPONENT = 0.52


def tether_contact_weight(distance: float, flank_length: int,
                          prefactor: float = _REACH_PREFACTOR,
                          exponent: float = _REACH_EXPONENT) -> float:
    r"""Relative density of flank monomers at ``distance`` from the anchor.

    Summing the ideal-chain end distribution over every residue of the flank,

    .. math::

        w(d) = \sum_{i=1}^{N} \left(\frac{3}{2\pi R_i^2}\right)^{3/2}
               \exp\!\left(-\frac{3 d^2}{2 R_i^2}\right),
        \qquad R_i = \text{prefactor} \cdot i^{\text{exponent}}

    normalised to 1 at ``d = 0``. Residue *i* is tethered by *i* segments, so it
    has its own span :math:`R_i`; the sum is the local concentration of flank
    chemistry a target residue at that distance actually experiences.

    This replaces a linear taper, which over-weighted distant surface by a
    measured 2.4x at 10 A and 2.8x at 15 A. Derived from the same polymer
    relation the reach radius uses, not fitted.

    Parameters
    ----------
    distance : float
        Distance from the anchor, in angstroms.
    flank_length : int
        Number of residues in the flank.
    prefactor, exponent : float
        Polymer scaling parameters.

    Returns
    -------
    float
        Weight in ``(0, 1]``.
    """
    return float(tether_contact_weights(
        np.asarray([distance], dtype=np.float64), flank_length,
        prefactor, exponent)[0])


def tether_contact_weights(distances: np.ndarray, flank_length: int,
                           prefactor: float = _REACH_PREFACTOR,
                           exponent: float = _REACH_EXPONENT) -> np.ndarray:
    """Vectorised :func:`tether_contact_weight` over an array of distances."""
    if flank_length <= 0:
        raise ValueError(f"flank_length must be positive, got {flank_length}")
    d = np.asarray(distances, dtype=np.float64).reshape(-1)
    i = np.arange(1, int(flank_length) + 1, dtype=np.float64)
    r2 = (prefactor * i ** exponent) ** 2                      # (N,)
    amp = (3.0 / (2.0 * np.pi * r2)) ** 1.5                    # (N,)
    # (n_distances, N) -> sum over residues
    dens = (amp * np.exp(-1.5 * d[:, None] ** 2 / r2)).sum(axis=1)
    peak = amp.sum()
    return dens / peak if peak > 0 else np.zeros_like(dens)


def end_to_end_distance(flank_length: int,
                        prefactor: float = _REACH_PREFACTOR,
                        exponent: float = _REACH_EXPONENT) -> float:
    """RMS end-to-end distance of a disordered chain, in angstroms.

    ``Ree = prefactor * N ** exponent``. This is the span of the *whole* chain,
    which is why it is not used directly as the reach radius; see
    :func:`reach_radius`.
    """
    if flank_length <= 0:
        raise ValueError(f"flank_length must be positive, got {flank_length}")
    return prefactor * float(flank_length) ** exponent


def reach_radius(flank_length: int, prefactor: float = _REACH_PREFACTOR,
                 exponent: float = _REACH_EXPONENT,
                 minimum: float = 8.0,
                 maximum: Optional[float] = None) -> float:
    """Typical distance from the anchor at which a flank residue sits.

    Deliberately *not* the end-to-end distance. Only the last residue of the
    flank reaches ``Ree``; averaging the tethered-chain relation over all
    residues, ``mean_i(prefactor * i**exponent) = Ree / (1 + exponent)``, gives
    the distance at which flank residues are actually found. The factor is
    derived, not tuned.

    Using ``Ree`` itself as a hard, equal-weight cutoff treats a target residue
    at 33 A as being as relevant as one at 3 A, and it over-includes badly: on
    1YCR it selected the entire 85-residue target domain from *either* terminus,
    giving the two termini identical patches (Jaccard 1.00) and so removing the
    terminus choice the package exists to make. With this definition the same
    comparison gives Jaccard 0.44.

    Parameters
    ----------
    flank_length : int
        Number of residues in the flank, including any linker.
    prefactor, exponent : float
        Polymer scaling parameters; see the module source for provenance.
    minimum : float
        Floor on the returned radius, so a very short flank still sees the
        residues immediately around its attachment point.
    maximum : float, optional
        Optional ceiling on the returned radius.

    Returns
    -------
    float
        Radius in angstroms.
    """
    ree = end_to_end_distance(flank_length, prefactor, exponent)
    r = max(ree / (1.0 + exponent), minimum)
    if maximum is not None:
        r = min(r, maximum)
    return r


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class ProximalResidue:
    """One target residue selected as reachable by the flank."""

    residue: Residue
    anchor_distance: float
    binder_distance: float
    is_contact: bool
    relative_sasa: float = float("nan")
    weight: float = 1.0

    @property
    def seq_id(self) -> int:
        return self.residue.seq_id

    @property
    def one_letter(self) -> str:
        return self.residue.one_letter

    @property
    def label(self) -> str:
        return self.residue.label


@dataclass
class ProximalRegion:
    """The target region a flank should be designed to complement."""

    residues: List[ProximalResidue]
    target_chain_id: str
    binder_chain_id: str
    terminus: str
    anchor_label: str
    reach_radius: float
    contact_labels: List[str] = field(default_factory=list)
    excluded_labels: List[str] = field(default_factory=list)
    binder_interface_sequence: str = ""
    """The binder's own residues that contact the target. A flank attracted to
    these would compete with the target for the binder."""
    binder_interface_labels: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.residues)

    def __iter__(self):
        return iter(self.residues)

    def __bool__(self) -> bool:
        return bool(self.residues)

    @property
    def labels(self) -> List[str]:
        return [r.label for r in self.residues]

    @property
    def weights(self) -> np.ndarray:
        """Proximity weight of each selected residue, highest nearest the anchor."""
        return np.array([r.weight for r in self.residues], dtype=np.float64)

    @property
    def seq_ids(self) -> List[int]:
        return [r.seq_id for r in self.residues]

    def _runs(self) -> List[List[ProximalResidue]]:
        """Selected residues grouped into genuinely consecutive runs.

        Grouping is by adjacency in the target chain *and* a residue-number
        step of at most one. Both conditions are needed: residue numbers alone
        cannot distinguish 52, 52A and 52B (insertion codes repeat a number),
        while chain adjacency alone would join residues that sit either side of
        an unresolved gap.
        """
        runs: List[List[ProximalResidue]] = []
        current: List[ProximalResidue] = []
        for p in self.residues:
            if current:
                prev = current[-1].residue
                adjacent = p.residue.index == prev.index + 1
                near_in_number = 0 <= (p.residue.seq_id - prev.seq_id) <= 1
                if not (adjacent and near_in_number):
                    runs.append(current)
                    current = []
            current.append(p)
        if current:
            runs.append(current)
        return runs

    @property
    def spans(self) -> List[Tuple[int, int]]:
        """Consecutive runs of selected residues, as ``(first, last)`` numbers.

        With insertion codes a run can begin and end on the same number, so use
        :attr:`span_labels` when the exact residues matter.
        """
        return [(run[0].seq_id, run[-1].seq_id) for run in self._runs()]

    @property
    def span_labels(self) -> List[Tuple[str, str]]:
        """Consecutive runs as ``(first_label, last_label)``, insertion codes
        included."""
        return [(run[0].label, run[-1].label) for run in self._runs()]

    @property
    def span_sequences(self) -> List[str]:
        """One-letter sequence of each consecutive run, in order."""
        return ["".join(p.one_letter for p in run) for run in self._runs()]

    @property
    def patch_sequence(self) -> str:
        """Selected residues concatenated, in target-chain order.

        This is what gets handed to the epsilon calculation as the "target" the
        flank should be complementary to.
        """
        return "".join(p.one_letter for p in self.residues)

    def weighted_patch_sequence(self, max_copies: int = 3) -> str:
        """Patch sequence with residues near the anchor repeated.

        FINCHES epsilon behaves as a *mean* over the target residues -- the
        ratio of residue types drives the value and the absolute count barely
        matters -- so repeating a residue is an exact way to weight it. Copying
        the residues closest to the attachment point makes the design
        preferentially complement the surface the flank is most likely to
        touch, instead of averaging the whole reachable area equally.

        Parameters
        ----------
        max_copies : int
            Copies given to a residue sitting right at the anchor. Residues at
            the edge of the reach radius get one copy. ``1`` reproduces
            :attr:`patch_sequence`.

        Returns
        -------
        str
        """
        if max_copies < 1:
            raise ValueError(f"max_copies must be >= 1, got {max_copies}")
        if max_copies == 1:
            return self.patch_sequence
        out: List[str] = []
        for p in self.residues:
            copies = 1 + int(round(p.weight * (max_copies - 1)))
            out.append(p.one_letter * copies)
        return "".join(out)

    def _span_text(self) -> str:
        """Spans rendered for display, using labels if insertion codes occur."""
        if any(r.residue.ins_code.strip() for r in self.residues):
            return ", ".join(f"{a}-{b}" if a != b else a
                             for a, b in self.span_labels)
        return ", ".join(f"{a}-{b}" if a != b else str(a)
                         for a, b in self.spans)

    def weighted_shells(self, edges: Sequence[float] = (5.0, 10.0, 15.0)
                        ) -> List[Tuple[str, float]]:
        """Selected residues split into distance shells with tether weights.

        Returns ``[(sequence, weight), ...]``, one entry per non-empty shell,
        where ``weight`` is the summed tethered-chain contact weight of the
        residues in it. Handing these to the design step lets the objective be
        a weighted average over shells rather than a flat average over the whole
        patch, so surface the flank rarely touches stops dominating.

        Splitting into shells rather than reweighting residue-by-residue is
        deliberate: FINCHES epsilon takes a sequence, so real-valued weights can
        only be applied *between* epsilon calls, not within one. A handful of
        shells costs a handful of epsilon calls.

        Parameters
        ----------
        edges : sequence of float
            Upper bounds of the inner shells, in angstroms. The last shell
            covers everything beyond the final edge.

        Returns
        -------
        list of (str, float)
        """
        bounds = list(edges) + [float("inf")]
        shells: List[Tuple[str, float]] = []
        for lo, hi in zip([0.0] + list(edges), bounds):
            members = [p for p in self.residues
                       if lo <= p.anchor_distance < hi]
            if not members:
                continue
            weight = float(sum(p.weight for p in members))
            if weight <= 0:
                continue
            shells.append(("".join(p.one_letter for p in members), weight))
        return shells

    def summary(self) -> str:
        lines = [
            f"Proximal region on chain {self.target_chain_id!r} "
            f"for a flank on the {self.terminus}-terminus of chain "
            f"{self.binder_chain_id!r}",
            f"  anchor residue      : {self.anchor_label}",
            f"  reach radius        : {self.reach_radius:.1f} A",
            f"  residues selected   : {len(self.residues)}",
            f"  spans               : " + (self._span_text() or "none"),
            f"  patch sequence      : {self.patch_sequence or '(empty)'}",
            f"  interface contacts  : {len(self.contact_labels)}",
            f"  binder interface    : {self.binder_interface_sequence or '(none)'}"
            f" ({len(self.binder_interface_labels)} residues; the flank must "
            f"not compete for these)",
        ]
        rel = [r.relative_sasa for r in self.residues
               if r.relative_sasa == r.relative_sasa]
        if rel:
            lines.append(f"  relative SASA range : "
                         f"{min(rel):.2f} - {min(max(rel), 9.99):.2f}")
        if self.excluded_labels:
            lines.append(
                f"  excluded (sequence-distant): {len(self.excluded_labels)} "
                f"({', '.join(self.excluded_labels[:10])}"
                f"{', ...' if len(self.excluded_labels) > 10 else ''})"
            )
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# the main entry point
# ---------------------------------------------------------------------------

def parse_residue_spec(spec) -> "set":
    """Interpret a residue selection as a set of author residue numbers.

    Accepted forms, all in author numbering (the numbers a viewer shows):

    ==============================  ==========================================
    ``"1-100"``                     residues 1 through 100, inclusive
    ``"1-100,250-300"``             several ranges
    ``"1-100,150"``                 ranges and single residues mixed
    ``(1, 100)``                    one inclusive range
    ``[(1, 100), (250, 300)]``      several ranges
    ``[1, 100]``                    **a range**, 1 through 100
    ``[5, 12, 88]``                 those three residues individually
    ``range(1, 101)``               residues 1 through 100
    ==============================  ==========================================

    Note the one ambiguous case: a bare pair of integers is read as a *range*,
    since that is overwhelmingly what it is meant for. Write ``[[5], [12]]`` or
    ``"5,12"`` for exactly two individual residues. Whatever is parsed is
    reported back in the region notes, so the interpretation is never silent.

    Parameters
    ----------
    spec : str, int, or sequence
        The selection.

    Returns
    -------
    set of int
    """
    out: set = set()
    if spec is None:
        return out

    if isinstance(spec, str):
        for part in spec.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part[1:]:                     # allow a leading minus
                idx = part.index("-", 1)
                lo, hi = part[:idx], part[idx + 1:]
                try:
                    out.update(range(int(lo), int(hi) + 1))
                except ValueError:
                    raise ValueError(
                        f"could not read residue range {part!r}; expected "
                        f"something like '1-100'") from None
            else:
                try:
                    out.add(int(part))
                except ValueError:
                    raise ValueError(
                        f"could not read residue number {part!r}") from None
        return out

    if isinstance(spec, int):
        return {int(spec)}

    items = list(spec)
    # A bare pair of integers is a range -- the common intent.
    if len(items) == 2 and all(isinstance(x, int) for x in items):
        lo, hi = int(items[0]), int(items[1])
        if hi < lo:
            raise ValueError(
                f"residue range ({lo}, {hi}) runs backwards")
        return set(range(lo, hi + 1))

    for item in items:
        if isinstance(item, int):
            out.add(int(item))
        elif isinstance(item, str):
            out |= parse_residue_spec(item)
        else:
            pair = list(item)
            if len(pair) == 1:
                out.add(int(pair[0]))
            elif len(pair) == 2:
                lo, hi = int(pair[0]), int(pair[1])
                if hi < lo:
                    raise ValueError(
                        f"residue range ({lo}, {hi}) runs backwards")
                out.update(range(lo, hi + 1))
            else:
                raise ValueError(
                    f"cannot read {item!r} as a residue or a (start, end) "
                    f"range")
    return out


def _describe_residue_set(numbers: "set") -> str:
    """Render a residue set as compact ranges, e.g. ``1-100, 250-300``."""
    if not numbers:
        return "none"
    ordered = sorted(numbers)
    spans = []
    start = prev = ordered[0]
    for n in ordered[1:]:
        if n == prev + 1:
            prev = n
            continue
        spans.append((start, prev))
        start = prev = n
    spans.append((start, prev))
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in spans)


def _normalize_terminus(terminus: str) -> str:
    t = str(terminus).strip().lower()
    if t in ("n", "n-term", "nterm", "n_terminus", "n-terminus", "5", "5'"):
        return "N"
    if t in ("c", "c-term", "cterm", "c_terminus", "c-terminus", "3", "3'"):
        return "C"
    raise ValueError(
        f"terminus must be 'N' or 'C', got {terminus!r}"
    )


def _cluster_seq_ids(seq_ids: Sequence[int],
                     max_gap: int) -> List[List[int]]:
    """Group sorted residue numbers into clusters split by gaps > ``max_gap``."""
    clusters: List[List[int]] = []
    current: List[int] = []
    for i in sorted(seq_ids):
        if current and i - current[-1] > max_gap:
            clusters.append(current)
            current = []
        current.append(i)
    if current:
        clusters.append(current)
    return clusters


def find_proximal_region(
    structure: Structure,
    binder_chain: str,
    target_chain: str,
    terminus: str,
    flank_length: int,
    *,
    contact_cutoff: float = 5.0,
    anchor_residues: int = 1,
    radius: Optional[float] = None,
    radius_scale: float = 1.0,
    max_radius: Optional[float] = None,
    cluster_gap: int = 15,
    min_cluster_contacts: int = 3,
    sequence_window: int = 25,
    max_residues: Optional[int] = None,
    require_surface: bool = True,
    surface_threshold: float = 0.10,
    sasa_points: int = 480,
    trust_distal_occlusion: bool = False,
    exclude_target_residues=None,
    include_target_residues=None,
) -> ProximalRegion:
    """Select the target residues a new flank should be complementary to.

    Parameters
    ----------
    structure : Structure
        Parsed complex containing both chains.
    binder_chain, target_chain : str
        Author chain identifiers. The binder is the chain being extended.
    terminus : str
        ``"N"`` or ``"C"``: which end of the binder the flank is added to.
    flank_length : int
        Length of the flank, which sets how far it can reach.
    contact_cutoff : float
        Heavy-atom distance defining a genuine interface contact, used to
        locate the interface in *sequence* space.
    anchor_residues : int
        How many terminal binder residues form the attachment anchor. The
        default of 1 uses just the terminal residue.
    radius : float, optional
        Override the reach radius entirely, in angstroms.
    radius_scale : float
        Multiplier applied to the polymer-scaling reach radius. Values above 1
        widen the search.
    max_radius : float, optional
        Ceiling on the reach radius.
    cluster_gap : int
        Contacts separated by more than this many residues are treated as
        belonging to separate interface patches.
    min_cluster_contacts : int
        Interface patches with fewer contacts than this are discarded as
        prediction noise. If no patch qualifies, the largest is kept.
    sequence_window : int
        How many residues beyond an accepted interface patch a residue may sit
        and still be accepted. This is the sequence-locality filter.
    max_residues : int, optional
        Keep at most this many residues, closest to the anchor first.
    require_surface : bool
        Discard target residues that are not solvent-exposed in the complex. A
        flank cannot touch a buried residue, so this is on by default.
    surface_threshold : float
        Minimum relative sidechain accessibility to count as exposed. 0.10
        matches the FINCHES convention.
    sasa_points : int
        Test points per atom for the accessibility calculation. Lower is
        faster; the default is far more than surface classification needs.
    exclude_target_residues : str or sequence, optional
        Target residues to remove from consideration entirely, in author
        numbering. See :func:`parse_residue_spec` for the accepted forms;
        ``[1, 100]`` and ``"1-100"`` both mean the first hundred residues.

        Use this when you know a region cannot really be at the interface. The
        automatic sequence-locality filter only removes *small* spurious contact
        patches; a predictor that folds a whole terminus back onto the true
        binding site produces a large, self-consistent patch that survives it,
        and only you know it is wrong.

        Excluded residues take no part in locating the interface either, not
        just in the final selection -- otherwise the spurious patch would still
        define an accepted sequence window and let its neighbours through.
    include_target_residues : str or sequence, optional
        The complement: consider *only* these target residues. Often the more
        direct way to say the same thing, e.g. ``include_target_residues="400-"``
        is expressed as ``(400, last_residue)``.
    trust_distal_occlusion : bool
        Whether target residues that fail the sequence-locality test may still
        occlude solvent. Default ``False``: a predictor that drapes a distant
        part of the target over the region of interest would otherwise bury
        surface that is really available to the flank, and the region would be
        discarded for a reason that is an artefact. Set ``True`` for an
        experimental structure, where such packing is real.

    Returns
    -------
    ProximalRegion

    Raises
    ------
    InterfaceError
        If the chains are missing, identical, empty, or not in contact.
    """
    term = _normalize_terminus(terminus)

    if binder_chain == target_chain:
        raise InterfaceError(
            f"binder_chain and target_chain are both {binder_chain!r}; "
            f"they must be different chains."
        )
    for name, cid in (("binder_chain", binder_chain),
                      ("target_chain", target_chain)):
        if cid not in structure:
            raise InterfaceError(
                f"{name}={cid!r} is not in the structure. "
                f"Available chains: {structure.chain_ids}"
            )

    binder = structure[binder_chain]
    target = structure[target_chain]
    if len(binder) == 0 or len(target) == 0:
        raise InterfaceError(
            f"chain {binder_chain!r} has {len(binder)} residues and "
            f"chain {target_chain!r} has {len(target)}; both must be non-empty."
        )
    if flank_length <= 0:
        raise ValueError(f"flank_length must be positive, got {flank_length}")
    if anchor_residues < 1:
        raise ValueError(
            f"anchor_residues must be at least 1, got {anchor_residues}")
    if anchor_residues > len(binder):
        raise ValueError(
            f"anchor_residues={anchor_residues} exceeds the {len(binder)} "
            f"residues in chain {binder_chain!r}. Using the whole binder as the "
            f"anchor makes the two termini indistinguishable.")
    if max_residues is not None and max_residues < 1:
        raise ValueError(
            f"max_residues must be at least 1 if given, got {max_residues}")
    if contact_cutoff <= 0:
        raise ValueError(
            f"contact_cutoff must be positive, got {contact_cutoff}")
    if radius_scale <= 0:
        raise ValueError(
            f"radius_scale must be positive, got {radius_scale}")
    if sequence_window < 0:
        raise ValueError(
            f"sequence_window cannot be negative, got {sequence_window}")

    notes: List[str] = []

    # --- 1. the genuine interface, in sequence space ---
    binder_coords, _ = binder.stacked_heavy_coords()
    d_binder = min_distances_to(target, binder_coords)
    contact_idx = np.nonzero(d_binder <= contact_cutoff)[0]
    if contact_idx.size == 0:
        closest = float(np.min(d_binder)) if np.isfinite(d_binder).any() else float("inf")
        raise InterfaceError(
            f"chains {binder_chain!r} and {target_chain!r} make no heavy-atom "
            f"contact within {contact_cutoff} A (closest approach "
            f"{closest:.1f} A). Are the chain assignments right, and is this "
            f"really a complex?"
        )

    # User-declared eligibility, applied BEFORE the interface is located: a
    # mispredicted region must not get to define an accepted sequence window.
    excluded_spec = parse_residue_spec(exclude_target_residues)
    included_spec = parse_residue_spec(include_target_residues)

    def eligible(seq_id: int) -> bool:
        if included_spec and seq_id not in included_spec:
            return False
        return seq_id not in excluded_spec

    if excluded_spec or included_spec:
        present = {int(r.seq_id) for r in target.residues}
        if included_spec:
            missing = included_spec - present
            if len(missing) == len(included_spec):
                raise InterfaceError(
                    f"include_target_residues selects "
                    f"{_describe_residue_set(included_spec)}, but chain "
                    f"{target_chain!r} contains none of those residue numbers "
                    f"(it spans {target[0].seq_id}-{target[-1].seq_id}).")
            notes.append(
                f"restricted to target residues "
                f"{_describe_residue_set(included_spec & present)} as requested.")
        if excluded_spec:
            hit = excluded_spec & present
            notes.append(
                f"excluded target residues "
                f"{_describe_residue_set(excluded_spec)} as requested"
                + (f" ({len(hit)} of them present in this chain)"
                   if len(hit) != len(excluded_spec) else "")
                + ".")

    contact_idx = np.array(
        [i for i in contact_idx if eligible(int(target[i].seq_id))],
        dtype=np.int64)
    if contact_idx.size == 0:
        raise InterfaceError(
            f"after applying exclude_target_residues/include_target_residues, "
            f"no contact between chains {binder_chain!r} and {target_chain!r} "
            f"remains. The eligible region does not touch the binder -- check "
            f"the numbering, or widen the selection."
        )

    contact_seq_ids = [int(target[i].seq_id) for i in contact_idx]
    clusters = _cluster_seq_ids(contact_seq_ids, cluster_gap)
    kept = [c for c in clusters if len(c) >= min_cluster_contacts]
    if not kept:
        kept = [max(clusters, key=len)]
        notes.append(
            f"no interface patch reached min_cluster_contacts="
            f"{min_cluster_contacts}; kept the largest ({len(kept[0])} contacts)."
        )
    dropped = [c for c in clusters if c not in kept]
    if dropped:
        notes.append(
            f"discarded {len(dropped)} small interface patch(es) as prediction "
            f"noise: "
            + ", ".join(f"{c[0]}-{c[-1]} ({len(c)})" for c in dropped[:5])
        )
    if len(kept) > 1:
        notes.append(
            f"target presents {len(kept)} distinct interface patches: "
            + ", ".join(f"{c[0]}-{c[-1]}" for c in kept)
        )

    allowed_windows = [(c[0] - sequence_window, c[-1] + sequence_window)
                       for c in kept]

    def sequence_local(seq_id: int) -> bool:
        return any(lo <= seq_id <= hi for lo, hi in allowed_windows)

    # --- 2. the anchor: the binder terminus the flank attaches to ---
    anchor_res = (binder.residues[:anchor_residues] if term == "N"
                  else binder.residues[-anchor_residues:])
    anchor_blocks = [r.heavy_coords for r in anchor_res if r.heavy_coords.size]
    if not anchor_blocks:
        raise InterfaceError(
            f"terminal residue(s) of chain {binder_chain!r} have no heavy "
            f"atoms; cannot anchor the flank."
        )
    anchor_coords = np.vstack(anchor_blocks)
    anchor_label = (f"{anchor_res[0].label}"
                    if len(anchor_res) == 1
                    else f"{anchor_res[0].label}..{anchor_res[-1].label}")

    d_anchor = min_distances_to(target, anchor_coords)

    # --- 3. reachable from the anchor ---
    if radius is not None:
        r = float(radius)
        if r <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
    else:
        # Scale the polymer estimate first, then apply the ceiling, so
        # max_radius is a hard cap on the value actually used rather than an
        # intermediate that radius_scale could scale away from.
        r = reach_radius(flank_length) * float(radius_scale)
        if max_radius is not None:
            r = min(r, float(max_radius))

    # --- 4. solvent accessibility, measured in the complex ---
    # A flank can only interact with target surface, and it has to be surface
    # that is still exposed once the binder is bound.
    # Candidates are residues that are both within reach and sequence-local to
    # a trusted interface patch. The sequence filter is applied *before*
    # accessibility, not after, so a sequence-distant region cannot bury the
    # surface it was spuriously placed on top of.
    candidate_idx: List[int] = []
    excluded: List[str] = []
    n_ineligible = 0
    for i, res in enumerate(target.residues):
        if not np.isfinite(d_anchor[i]) or d_anchor[i] > r:
            continue
        if not eligible(int(res.seq_id)):
            n_ineligible += 1
            continue
        if sequence_local(int(res.seq_id)):
            candidate_idx.append(i)
        else:
            excluded.append(res.label)

    if n_ineligible:
        notes.append(
            f"{n_ineligible} residue(s) were within reach but ruled out by the "
            f"target-residue selection you supplied.")

    rel_sasa = np.full(len(target), np.nan)
    if require_surface and candidate_idx:
        other_chains = [res for cid, chain in structure.chains.items()
                        if cid != target_chain for res in chain.residues]
        # Distal target regions are the same prediction artefact the
        # sequence-locality filter exists to remove. Letting them occlude would
        # reintroduce it through the back door: a spuriously draped loop would
        # bury the very surface the flank could use, and the region would be
        # discarded for a reason that is not real.
        candidate_set = set(candidate_idx)
        occluders = [
            target.residues[i] for i in range(len(target))
            if i not in candidate_set
            and (trust_distal_occlusion
                 or (sequence_local(int(target.residues[i].seq_id))
                     and eligible(int(target.residues[i].seq_id))))]
        # Only candidates need an accessibility number, which also keeps the
        # Shrake-Rupley pass off the rest of a large chain.
        het_xyz, het_els = structure.heteroatoms()
        rel_sasa[candidate_idx] = relative_residue_sasa(
            [target.residues[i] for i in candidate_idx],
            context=other_chains + occluders,
            n_points=sasa_points,
            extra_coords=het_xyz, extra_elements=het_els)

    # --- 5. combine ---
    selected: List[ProximalResidue] = []
    n_buried = 0
    for i in candidate_idx:
        res = target.residues[i]
        if require_surface and not (rel_sasa[i] > surface_threshold):
            n_buried += 1
            continue
        selected.append(ProximalResidue(
            residue=res,
            anchor_distance=float(d_anchor[i]),
            binder_distance=float(d_binder[i]),
            is_contact=bool(d_binder[i] <= contact_cutoff),
            relative_sasa=float(rel_sasa[i]),
        ))

    if n_buried:
        notes.append(f"{n_buried} residue(s) within reach were buried "
                     f"(relative SASA <= {surface_threshold}) and excluded.")
    if require_surface and not trust_distal_occlusion and candidate_idx:
        # Count only the sequence-distant residues that were close enough to a
        # candidate to actually have occluded it. Counting every distant
        # residue in the chain would report hundreds on a large target and
        # imply far more was discarded than really was.
        cand_set = set(candidate_idx)
        distal = [target.residues[i] for i in range(len(target))
                  if i not in cand_set
                  and not sequence_local(int(target.residues[i].seq_id))]
        n_distal = 0
        if distal:
            blocks = [target.residues[i].heavy_coords for i in candidate_idx]
            cand_coords = (np.vstack([b for b in blocks if b.size])
                           if any(b.size for b in blocks)
                           else np.empty((0, 3)))
            if cand_coords.shape[0]:
                for res in distal:
                    hc = res.heavy_coords
                    if hc.size and _nearest_distance(
                            hc, cand_coords).min() <= _OCCLUSION_RANGE:
                        n_distal += 1
        if n_distal:
            notes.append(
                f"{n_distal} sequence-distant target residue(s) were close "
                f"enough to bury reachable surface but were excluded from the "
                f"accessibility calculation, so a predicted misplacement "
                f"cannot hide a usable region. Pass "
                f"trust_distal_occlusion=True for an experimental structure.")

    if max_residues is not None and len(selected) > max_residues:
        by_distance = sorted(selected, key=lambda p: p.anchor_distance)
        keep = {id(p) for p in by_distance[:max_residues]}
        dropped_n = len(selected) - max_residues
        selected = [p for p in selected if id(p) in keep]
        notes.append(f"kept the {max_residues} residues closest to the anchor "
                     f"(dropped {dropped_n}).")

    selected.sort(key=lambda p: (p.residue.seq_id, p.residue.ins_code))

    # Proximity weight: the tethered-chain monomer density at that distance,
    # normalised to 1 at the anchor. A linear taper to zero at the reach radius
    # over-weights distant surface several-fold.
    if selected:
        weights = tether_contact_weights(
            np.array([p.anchor_distance for p in selected]), flank_length)
        for p, w in zip(selected, weights):
            p.weight = float(w)

    if not selected:
        raise InterfaceError(
            f"no target residue on chain {target_chain!r} is both within "
            f"{r:.1f} A of the {term}-terminal anchor {anchor_label} and within "
            f"{sequence_window} residues of the interface. The flank probably "
            f"cannot reach the interface from this terminus -- try the other "
            f"terminus, a longer flank, or a larger radius."
        )

    # Unresolved residues in the TARGET matter too: whatever they would have
    # covered now reads as exposed, so a residue can be selected only because
    # its neighbours are missing from the model.
    target_breaks = target.chain_breaks()
    if target_breaks and selected:
        chosen = {p.seq_id for p in selected}
        near = [(a, b) for a, b in target_breaks
                if any(a - sequence_window <= s <= b + sequence_window
                       for s in chosen)]
        if near:
            notes.append(
                f"target chain {target_chain!r} has unresolved break(s) near "
                f"the selected region ("
                + ", ".join(f"{a}->{b}" for a, b in near[:5])
                + "). Surface those missing residues would have covered reads "
                  "as exposed here, so some selected residues may not really "
                  "be accessible in the intact protein.")

    if len(selected) < 5:
        notes.append(
            f"only {len(selected)} residue(s) survived the filters. A patch "
            f"this small gives the design almost nothing to complement, and "
            f"the resulting epsilon is dominated by those few residues. "
            f"Consider a longer flank, the other terminus, a larger radius, or "
            f"a looser surface_threshold."
        )

    if excluded:
        notes.append(
            f"{len(excluded)} residue(s) were near the anchor in space but "
            f"sequence-distant from the interface, so were excluded as likely "
            f"prediction artefacts."
        )

    # The binder's own target-contacting residues, so the design step can be
    # told what not to compete with.
    target_coords, _ = target.stacked_heavy_coords()
    d_target = min_distances_to(binder, target_coords)
    binder_iface = [binder.residues[i]
                    for i in np.nonzero(d_target <= contact_cutoff)[0]]

    return ProximalRegion(
        residues=selected,
        target_chain_id=target_chain,
        binder_chain_id=binder_chain,
        terminus=term,
        anchor_label=anchor_label,
        reach_radius=r,
        contact_labels=[target[i].label for i in contact_idx],
        excluded_labels=excluded,
        binder_interface_sequence="".join(r.one_letter for r in binder_iface),
        binder_interface_labels=[r.label for r in binder_iface],
        notes=notes,
    )
