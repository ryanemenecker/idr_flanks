"""Solvent-accessible surface area by the Shrake-Rupley algorithm.

A designed flank can only interact with target residues that are actually
exposed, so surface accessibility is a hard requirement for selecting the
region a flank should complement.

Implemented here rather than taken from mdtraj so the package stays light; the
atomic radii, probe radius, and per-residue reference areas match the
conventions used by ``finches``/mdtraj so numbers are comparable across tools.
The test suite checks agreement with ``mdtraj.shrake_rupley`` directly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .io import Residue

__all__ = [
    "ATOMIC_RADII",
    "MAX_SASA",
    "residue_sasa",
    "relative_residue_sasa",
    "solvent_accessible_mask",
    "sphere_points",
]

# Van der Waals radii in angstroms. Every element that occurs in a protein
# heavy atom -- C, N, O, S, Se, P -- is numerically identical to mdtraj's
# ``_ATOMIC_RADII`` (which stores them in nanometres), which is what makes the
# cross-check against ``mdtraj.shrake_rupley`` meaningful. The metal-ion entries
# are Bondi van der Waals radii and deliberately differ from mdtraj's ionic
# radii (Na, Mg, K, Ca, Cl); they are unreachable in practice because ions are
# not parsed as amino-acid residues, and are kept only so a caller
# constructing residues by hand gets a sane default.
ATOMIC_RADII: Dict[str, float] = {
    "H": 1.20, "D": 1.20, "He": 1.40,
    "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "Na": 2.27, "Mg": 1.73, "P": 1.80, "S": 1.80, "Cl": 1.75,
    "K": 2.75, "Ca": 2.31, "Se": 1.90, "Br": 1.85, "I": 1.98,
    "Fe": 2.00, "Zn": 1.39, "Cu": 1.40, "Mn": 2.00, "Ni": 1.63,
    "Co": 2.00, "Cd": 1.58, "Hg": 1.55,
}
_DEFAULT_RADIUS = 1.70

# Maximum sidechain and backbone SASA per residue type, in square angstroms,
# from all-atom simulations of GXG tripeptides. Reproduced from FINCHES
# (finches.utils.folded_domain_utils.MAX_SASA_DATA) so relative accessibilities
# computed here are comparable with its folded-domain machinery. Entry is
# ``(sidechain, backbone)``. Glycine's sidechain area is zero; see
# :func:`relative_residue_sasa` for how that is handled here, which differs
# from FINCHES on purpose.
MAX_SASA: Dict[str, Tuple[float, float]] = {
    "A": (75.81871795654297, 76.07605743408203),
    "C": (115.40644836425781, 67.8772201538086),
    "D": (130.25582885742188, 71.82710266113281),
    "E": (161.79856872558594, 68.05746459960938),
    "F": (209.38710021972656, 65.9827880859375),
    "G": (0.0, 114.97527313232422),
    "H": (180.81494140625, 67.50666809082031),
    "I": (172.7196502685547, 60.34464645385742),
    "K": (205.8575897216797, 68.71156311035156),
    "L": (172.03604125976562, 64.51246643066406),
    "M": (184.76600646972656, 67.78076934814453),
    "N": (142.74412536621094, 66.804931640625),
    "P": (134.29147338867188, 55.83909606933594),
    "Q": (173.3262939453125, 66.60184478759766),
    "R": (236.48756408691406, 66.73487854003906),
    "S": (95.87133026123047, 72.87202453613281),
    "T": (130.9214324951172, 64.21310424804688),
    "V": (143.11781311035156, 61.72962188720703),
    "W": (254.5694122314453, 64.3099136352539),
    "Y": (222.518310546875, 71.86695098876953),
}

_SPHERE_CACHE: Dict[int, np.ndarray] = {}


def sphere_points(n: int = 480) -> np.ndarray:
    """``n`` approximately equidistant unit vectors on a sphere.

    Uses the golden-spiral (Fibonacci) construction, which gives a more even
    distribution than random sampling for the same point count.

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    ndarray, shape (n, 3)
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    cached = _SPHERE_CACHE.get(n)
    if cached is not None:
        return cached
    idx = np.arange(n, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * idx / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = idx * np.pi * (1.0 + 5.0 ** 0.5)
    pts = np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=1)
    # The array is cached and handed to every caller, so freeze it: a caller
    # mutating it in place would silently corrupt every later calculation.
    pts.flags.writeable = False
    _SPHERE_CACHE[n] = pts
    return pts


def _atom_arrays(residues: Sequence[Residue]
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten residues into ``(coords, radii, owner)`` arrays."""
    coords: List[np.ndarray] = []
    radii: List[float] = []
    owner: List[int] = []
    for i, res in enumerate(residues):
        for atom in res.atoms:
            if atom.is_hydrogen:
                continue
            coords.append(atom.xyz)
            radii.append(ATOMIC_RADII.get(atom.element, _DEFAULT_RADIUS))
            owner.append(i)
    if not coords:
        return (np.empty((0, 3)), np.empty(0), np.empty(0, dtype=np.int64))
    return (np.asarray(coords, dtype=np.float64),
            np.asarray(radii, dtype=np.float64),
            np.asarray(owner, dtype=np.int64))


def residue_sasa(residues: Sequence[Residue],
                 context: Optional[Sequence[Residue]] = None,
                 probe_radius: float = 1.4,
                 n_points: int = 480,
                 extra_coords: Optional[np.ndarray] = None,
                 extra_elements: Optional[Sequence[str]] = None) -> np.ndarray:
    """Per-residue solvent-accessible surface area in square angstroms.

    Parameters
    ----------
    residues : sequence of Residue
        Residues to compute accessibility for.
    context : sequence of Residue, optional
        Additional residues that occlude solvent but whose own area is not
        reported. Pass the other chains of a complex here to get accessibility
        *in the complex* rather than for an isolated chain. Residues that also
        appear in ``residues`` are ignored, so overlapping inputs cannot make a
        residue occlude itself.
    probe_radius : float
        Solvent probe radius in angstroms. 1.4 approximates a water molecule.
    n_points : int
        Test points per atom. Higher is more accurate and slower; 480 is
        accurate to well under a percent for surface classification.
    extra_coords : ndarray, optional
        ``(n, 3)`` coordinates of additional occluding atoms that are not part
        of any residue -- nucleic acids, glycans, cofactors. Surface they cover
        is not solvent-accessible.
    extra_elements : sequence of str, optional
        Element symbols for ``extra_coords``; carbon is assumed if omitted.

    Returns
    -------
    ndarray, shape (len(residues),)
        SASA of each residue.
    """
    coords, radii, owner = _atom_arrays(residues)
    n_res = len(residues)
    out = np.zeros(n_res, dtype=np.float64)
    if coords.shape[0] == 0:
        return out

    if context:
        # Drop any residue that is also in ``residues``: duplicating its atoms
        # would let it occlude itself. Identity comparison, so a caller passing
        # overlapping chain slices is handled without comparing coordinates.
        seen = {id(r) for r in residues}
        extra = [r for r in context if id(r) not in seen]
        c_coords, c_radii, _ = _atom_arrays(extra)
    else:
        c_coords = np.empty((0, 3))
        c_radii = np.empty(0)

    if extra_coords is not None and len(extra_coords):
        e_coords = np.asarray(extra_coords, dtype=np.float64).reshape(-1, 3)
        if extra_elements is None:
            e_radii = np.full(e_coords.shape[0], _DEFAULT_RADIUS)
        else:
            e_radii = np.array(
                [ATOMIC_RADII.get(el, _DEFAULT_RADIUS)
                 for el in extra_elements], dtype=np.float64)
            if e_radii.shape[0] != e_coords.shape[0]:
                raise ValueError(
                    f"extra_elements has {e_radii.shape[0]} entries but "
                    f"extra_coords has {e_coords.shape[0]} rows")
        c_coords = np.vstack([c_coords, e_coords]) if c_coords.shape[0] else e_coords
        c_radii = np.concatenate([c_radii, e_radii]) if c_radii.size else e_radii

    # All atoms that can occlude: the residues of interest plus the context.
    all_coords = np.vstack([coords, c_coords]) if c_coords.shape[0] else coords
    all_radii = np.concatenate([radii, c_radii]) if c_radii.size else radii

    probe = float(probe_radius)
    expanded = all_radii + probe
    unit = sphere_points(n_points)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(all_coords)
        max_reach = float(expanded.max()) + float(expanded.max())
        neighbor_lists = tree.query_ball_point(coords, max_reach)
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        neighbor_lists = [np.arange(all_coords.shape[0])] * coords.shape[0]

    for i in range(coords.shape[0]):
        ri = radii[i] + probe
        neigh = np.asarray(neighbor_lists[i], dtype=np.int64)
        # Drop self and any atom too far to possibly occlude this sphere.
        neigh = neigh[neigh != i]
        if neigh.size:
            d = np.linalg.norm(all_coords[neigh] - coords[i], axis=1)
            keep = d < (ri + expanded[neigh])
            neigh = neigh[keep]

        if neigh.size == 0:
            out[owner[i]] += 4.0 * np.pi * ri * ri
            continue

        test = coords[i] + unit * ri                      # (n_points, 3)
        nc = all_coords[neigh]                            # (n_neigh, 3)
        nr = expanded[neigh]                              # (n_neigh,)
        # A test point is buried if it falls inside any neighbour's probe
        # sphere. Compare squared distances to avoid the sqrt.
        d2 = ((test[:, None, :] - nc[None, :, :]) ** 2).sum(axis=-1)
        buried = (d2 < (nr * nr)[None, :]).any(axis=1)
        frac = 1.0 - buried.mean()
        out[owner[i]] += 4.0 * np.pi * ri * ri * frac

    return out


def relative_residue_sasa(residues: Sequence[Residue],
                          context: Optional[Sequence[Residue]] = None,
                          probe_radius: float = 1.4,
                          n_points: int = 480,
                          mode: str = "sidechain",
                          extra_coords: Optional[np.ndarray] = None,
                          extra_elements: Optional[Sequence[str]] = None
                          ) -> np.ndarray:
    """Per-residue SASA as a fraction of that residue type's maximum.

    Parameters
    ----------
    residues, context, probe_radius, n_points
        As for :func:`residue_sasa`.
    mode : str
        ``"sidechain"`` normalises by the sidechain reference area (the
        convention FINCHES uses, under which glycine is always accessible);
        ``"total"`` normalises by sidechain plus backbone.

    Returns
    -------
    ndarray, shape (len(residues),)
        Relative accessibility, always finite.

    Notes
    -----
    Glycine has no sidechain, so its sidechain reference area is zero. FINCHES
    handles that by treating every glycine as accessible; here it is normalised
    by its backbone reference area instead, which keeps the value finite and
    lets a genuinely buried glycine be recognised as buried.
    """
    if mode not in ("sidechain", "total"):
        raise ValueError(
            f"mode must be 'sidechain' or 'total', got {mode!r}")
    absolute = residue_sasa(residues, context=context,
                            probe_radius=probe_radius, n_points=n_points,
                            extra_coords=extra_coords,
                            extra_elements=extra_elements)
    ref = np.empty(len(residues), dtype=np.float64)
    for i, res in enumerate(residues):
        side, back = MAX_SASA.get(res.one_letter, MAX_SASA["A"])
        if mode == "total":
            ref[i] = side + back
        else:
            ref[i] = side if side > 0 else back
    return absolute / ref


def solvent_accessible_mask(residues: Sequence[Residue],
                            context: Optional[Sequence[Residue]] = None,
                            threshold: float = 0.10,
                            probe_radius: float = 1.4,
                            n_points: int = 480,
                            mode: str = "sidechain",
                            extra_coords: Optional[np.ndarray] = None,
                            extra_elements: Optional[Sequence[str]] = None
                            ) -> np.ndarray:
    """Boolean mask of residues exposed above ``threshold`` relative SASA.

    Parameters
    ----------
    residues, context, probe_radius, n_points, mode
        As for :func:`relative_residue_sasa`.
    threshold : float
        Minimum relative accessibility to count as surface. 0.10 matches the
        FINCHES default.

    Returns
    -------
    ndarray of bool, shape (len(residues),)
    """
    rel = relative_residue_sasa(residues, context=context,
                                probe_radius=probe_radius,
                                n_points=n_points, mode=mode,
                                extra_coords=extra_coords,
                                extra_elements=extra_elements)
    return rel > threshold
