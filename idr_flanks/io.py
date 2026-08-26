"""Dependency-free readers for PDB and mmCIF/PDBx structure files.

Returns a small object model (:class:`Structure` / :class:`Chain` / :class:`Residue`)
with coordinates held as numpy arrays so downstream distance work is vectorized.
Only numpy is required -- deliberately no Biopython/mdtraj dependency.
"""

from __future__ import annotations

import gzip
import os
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

__all__ = [
    "Atom",
    "Residue",
    "Chain",
    "Structure",
    "read_structure",
    "read_pdb",
    "read_cif",
    "THREE_TO_ONE",
    "StructureParseError",
]


class StructureParseError(ValueError):
    """Raised when a structure file cannot be parsed."""


# The 20 standard residues plus the modified residues common enough in the PDB
# that silently dropping them would corrupt a sequence. Selenomethionine (MSE)
# is by far the most important: it is deposited as HETATM, so a parser that
# ignores HETATM loses every SeMet position.
_STANDARD: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Modified residues mapped to the parent amino acid they are derived from.
# Rationale: every entry here is a chemically modified form of a standard
# residue whose side chain still occupies the same position, so mapping to the
# parent gives the right answer for both sequence reconstruction and the
# coarse-grained chemistry used downstream.
_MODIFIED: Dict[str, str] = {
    "MSE": "M",   # selenomethionine
    "SEC": "C",   # selenocysteine
    "PYL": "K",   # pyrrolysine
    "HYP": "P",   # 4-hydroxyproline
    "HIC": "H", "MHS": "H", "NEP": "H", "HIP": "H", "HSD": "H",
    "HSE": "H", "HSP": "H",
    "SEP": "S",   # phosphoserine
    "TPO": "T",   # phosphothreonine
    "PTR": "Y",   # phosphotyrosine
    "TYS": "Y",   # sulfotyrosine
    "CSO": "C", "CSD": "C", "CME": "C", "CMT": "C", "OCS": "C",
    "CSS": "C", "CSX": "C", "SMC": "C", "CAS": "C", "CYM": "C",
    "MLY": "K", "MLZ": "K", "M3L": "K", "KCX": "K", "LLP": "K",
    "ALY": "K", "M2L": "K",
    "MEN": "N",
    "MED": "M", "FME": "M", "CXM": "M",
    "AAR": "R", "AGM": "R", "MMO": "R",
    "DAL": "A", "AIB": "A", "ABA": "A", "MAA": "A",
    "CGU": "E", "PCA": "Q", "GMA": "E",
    "TRO": "W", "TRQ": "W", "TPQ": "Y",
    "PHI": "F", "PFF": "F", "DAH": "F",
    "IAS": "D", "BFD": "D", "PHD": "D",
    "LED": "L", "MLE": "L", "NLE": "L",
    "IIL": "I", "DIL": "I",
    "DVA": "V", "MVA": "V",
    "DPR": "P", "PRS": "P",
    "SAC": "S", "SVA": "S", "MIS": "S",
    "DTH": "T", "ALO": "T",
    "GL3": "G", "GLZ": "G", "SAR": "G", "MPQ": "G",
}

THREE_TO_ONE: Dict[str, str] = {**_STANDARD, **_MODIFIED}

# Residue names that are never part of a polypeptide chain. Anything not in
# THREE_TO_ONE is skipped anyway; this set exists so we can distinguish
# "solvent/ligand, expected" from "unrecognized residue, worth warning about".
_SOLVENT = frozenset({
    "HOH", "DOD", "WAT", "H2O", "SOL", "TIP3", "TIP4",
    "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "FE2", "CU", "CU1",
    "CO", "NI", "CD", "HG", "BR", "IOD", "F", "SO4", "PO4", "GOL",
    "EDO", "PEG", "DMS", "ACT", "MPD", "TRS", "EPE", "IMD", "NH4",
    "CIT", "FMT", "ACY", "BME", "DTT", "UNX", "UNL",
})

# Nucleic acid residues -- recognised so we can report a helpful error rather
# than silently returning an empty chain for a protein/DNA complex.
_NUCLEIC = frozenset({
    "A", "C", "G", "T", "U", "I",
    "DA", "DC", "DG", "DT", "DU", "DI",
    "RA", "RC", "RG", "RU",
    "ADE", "CYT", "GUA", "THY", "URA",
})

# Element symbols of every heavy atom name that occurs in the 20 standard
# residues plus selenomethionine. Needed because an atom *name* alone is
# ambiguous: "CA" is the alpha carbon in a residue but calcium as a free ion,
# and "SE" is selenium. Files that omit the element column (many MD and
# modelling tools do) have to be resolved from the name.
_PROTEIN_ATOM_ELEMENTS: Dict[str, str] = {
    # backbone and terminal oxygens
    "N": "N", "CA": "C", "C": "C", "O": "O",
    "OXT": "O", "OT1": "O", "OT2": "O", "OT": "O",
    # carbons
    "CB": "C", "CG": "C", "CG1": "C", "CG2": "C",
    "CD": "C", "CD1": "C", "CD2": "C",
    "CE": "C", "CE1": "C", "CE2": "C", "CE3": "C",
    "CZ": "C", "CZ2": "C", "CZ3": "C", "CH2": "C",
    # nitrogens
    "ND": "N", "ND1": "N", "ND2": "N",
    "NE": "N", "NE1": "N", "NE2": "N",
    "NZ": "N", "NH1": "N", "NH2": "N",
    # oxygens
    "OD": "O", "OD1": "O", "OD2": "O",
    "OE": "O", "OE1": "O", "OE2": "O",
    "OG": "O", "OG1": "O", "OH": "O",
    # sulfur, and selenium in selenomethionine / selenocysteine
    "SD": "S", "SG": "S", "SE": "Se",
}

# Two-letter element symbols that legitimately appear in structure files.
_TWO_LETTER_ELEMENTS = frozenset({
    "SE", "FE", "ZN", "MG", "MN", "CU", "NA", "CL", "CA", "CD", "NI",
    "CO", "BR", "HG", "LI", "BE", "AL", "SI", "AR", "CR", "GA", "GE",
    "AS", "KR", "RB", "SR", "MO", "AG", "SN", "SB", "TE", "BA", "PT",
    "AU", "PB", "CS", "XE", "RU", "RH", "PD", "IR", "OS", "TL", "BI",
})


def _normalise_het_name(resname: str) -> str:
    """Strip trailing charge notation so MD-style ion names match _SOLVENT.

    Simulation tools write ``NA+``, ``CL-`` and ``MG2+`` where the wwPDB writes
    ``NA``, ``CL`` and ``MG``; without this they survive as occluders.
    """
    n = resname.strip().upper()
    return n.rstrip("+-0123456789") or n


def _element_from_pdb_columns(field: str) -> str:
    """Infer an element from the raw 4-character PDB atom-name field.

    The legacy PDB format right-justifies the element symbol in columns 13-14,
    so a leading space means a one-letter element (``" CA "`` is the alpha
    carbon) while a letter in column 13 means a two-letter symbol (``"SE  "``
    is selenium, ``"CA  "`` is calcium). A leading digit marks hydrogen naming
    such as ``"1HB "``.
    """
    if not field:
        return ""
    padded = field.ljust(4)
    c0, c1 = padded[0], padded[1]

    if c0.isdigit():
        # Hydrogen-style name: "1HB ", "2HD1".
        return c1.upper() if c1.isalpha() else ""

    if c0 == " ":
        return c1.upper() if c1.isalpha() else ""

    # Column 13 occupied: a candidate two-letter symbol, but only when nothing
    # trails it. Without that check the gamma hydrogens of Val/Leu/Ile/Thr
    # ("HG11", "HG21", ...) read as mercury and survive as heavy atoms in files
    # that omit the element column. A genuine two-letter element fills columns
    # 13-14 and leaves 15-16 blank ("SE  ", "FE  ").
    if c1.isalpha() and padded[2:].strip() == "":
        pair = (c0 + c1).upper()
        if pair in _TWO_LETTER_ELEMENTS:
            return pair.capitalize()
    return c0.upper()


def _element_from_name(name: str) -> str:
    """Infer an element from a bare atom name with no column information.

    Used for mmCIF files that omit ``type_symbol``. Standard protein atom
    names are resolved from a table because they are ambiguous on their own;
    anything else falls back to a two-letter element check.
    """
    n = name.strip().upper()
    if not n:
        return ""
    known = _PROTEIN_ATOM_ELEMENTS.get(n)
    if known is not None:
        return known
    # Inside an amino acid every name beginning with H or D is a hydrogen:
    # "HG" is the cysteine/serine gamma hydrogen, not mercury, and "HG11" is a
    # valine gamma hydrogen. Free metal ions are not amino acids, so they never
    # reach this function.
    if n[0] in ("H", "D"):
        return n[0]
    # Strip a leading digit from hydrogen-style names.
    if n[0].isdigit():
        return n[1].upper() if len(n) > 1 and n[1].isalpha() else ""
    # Only an exactly-two-character name can be a two-letter element; "HG11" is
    # a hydrogen, not mercury.
    if len(n) == 2 and n[1].isalpha() and n in _TWO_LETTER_ELEMENTS:
        return n.capitalize()
    return n[0]


class Atom:
    """A single atom.

    ``xyz`` is a length-3 float64 numpy array.
    """

    __slots__ = ("name", "element", "xyz", "altloc", "occupancy",
                 "bfactor", "is_hetatm", "serial")

    def __init__(self, name: str, element: str, xyz: np.ndarray,
                 altloc: str = "", occupancy: float = 1.0,
                 bfactor: float = 0.0, is_hetatm: bool = False,
                 serial: int = 0):
        self.name = name
        self.element = element
        self.xyz = xyz
        self.altloc = altloc
        self.occupancy = occupancy
        self.bfactor = bfactor
        self.is_hetatm = is_hetatm
        self.serial = serial

    @property
    def is_hydrogen(self) -> bool:
        return self.element == "H" or self.element == "D"

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Atom {self.name} {self.element} {tuple(np.round(self.xyz, 2))}>"


class Residue:
    """A residue: one monomer of a chain, identified by (seq_id, ins_code)."""

    __slots__ = ("resname", "seq_id", "ins_code", "chain_id", "atoms",
                 "index", "_heavy_coords")

    def __init__(self, resname: str, seq_id: int, ins_code: str,
                 chain_id: str, index: int = -1):
        self.resname = resname
        self.seq_id = seq_id
        self.ins_code = ins_code
        self.chain_id = chain_id
        self.atoms: List[Atom] = []
        self.index = index
        self._heavy_coords: Optional[np.ndarray] = None

    @property
    def one_letter(self) -> str:
        return THREE_TO_ONE.get(self.resname, "X")

    @property
    def label(self) -> str:
        """Author-facing residue label, e.g. ``"A:52A"`` or ``"A:100"``."""
        return f"{self.chain_id}:{self.seq_id}{self.ins_code.strip()}"

    @property
    def heavy_coords(self) -> np.ndarray:
        """``(n_heavy_atoms, 3)`` float64 array of non-hydrogen coordinates."""
        if self._heavy_coords is None:
            coords = [a.xyz for a in self.atoms if not a.is_hydrogen]
            if coords:
                self._heavy_coords = np.asarray(coords, dtype=np.float64)
            else:
                self._heavy_coords = np.empty((0, 3), dtype=np.float64)
        return self._heavy_coords

    def atom(self, name: str) -> Optional[Atom]:
        for a in self.atoms:
            if a.name == name:
                return a
        return None

    @property
    def ca(self) -> Optional[Atom]:
        return self.atom("CA")

    @property
    def cb_or_ca(self) -> Optional[Atom]:
        """CB, falling back to CA (glycine, or a truncated side chain)."""
        return self.atom("CB") or self.atom("CA")

    @property
    def centroid(self) -> np.ndarray:
        hc = self.heavy_coords
        if hc.shape[0] == 0:
            return np.full(3, np.nan)
        return hc.mean(axis=0)

    def __len__(self) -> int:
        return len(self.atoms)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Residue {self.resname} {self.label} ({len(self.atoms)} atoms)>"


class Chain:
    """An ordered collection of residues sharing an author chain identifier."""

    __slots__ = ("chain_id", "residues", "full_sequence", "_coords_cache")

    def __init__(self, chain_id: str):
        self.chain_id = chain_id
        self.residues: List[Residue] = []
        self.full_sequence: str = ""
        """The complete chain as deposited (from SEQRES or ``_entity_poly``),
        including residues that were never resolved. Empty when the file does
        not declare it."""
        self._coords_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @property
    def sequence(self) -> str:
        """One-letter sequence in the order residues appear in the file."""
        return "".join(r.one_letter for r in self.residues)

    @property
    def seq_ids(self) -> np.ndarray:
        return np.fromiter((r.seq_id for r in self.residues),
                           dtype=np.int64, count=len(self.residues))

    def stacked_heavy_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """All heavy-atom coordinates of the chain, plus a residue-owner index.

        Returns ``(coords, owner)`` where ``coords`` is ``(n_atoms, 3)`` and
        ``owner[i]`` is the index into :attr:`residues` of the residue owning
        atom ``i``. Cached, because the interface search reuses it.
        """
        if self._coords_cache is None:
            blocks = [r.heavy_coords for r in self.residues]
            owners = [np.full(b.shape[0], i, dtype=np.int64)
                      for i, b in enumerate(blocks)]
            if blocks:
                coords = np.vstack(blocks) if any(b.size for b in blocks) \
                    else np.empty((0, 3), dtype=np.float64)
                owner = np.concatenate(owners) if owners \
                    else np.empty(0, dtype=np.int64)
            else:
                coords = np.empty((0, 3), dtype=np.float64)
                owner = np.empty(0, dtype=np.int64)
            self._coords_cache = (coords, owner)
        return self._coords_cache

    def numbering_gaps(self) -> List[Tuple[int, int]]:
        """Jumps in the author numbering, as ``(last_before, first_after)``.

        A jump does **not** necessarily mean residues are missing. Antibody
        Kabat/Chothia numbering deliberately skips numbers so that equivalent
        positions align across sequences, and such chains are physically
        continuous. Use :meth:`chain_breaks` for actual discontinuities.
        """
        out: List[Tuple[int, int]] = []
        for prev, nxt in zip(self.residues, self.residues[1:]):
            if nxt.seq_id - prev.seq_id > 1:
                out.append((prev.seq_id, nxt.seq_id))
        return out

    def unresolved_termini(self, probe: int = 6) -> Tuple[int, int]:
        """Residues missing from the N- and C-terminus, as ``(n_missing, c_missing)``.

        Requires :attr:`full_sequence`; returns ``(0, 0)`` when the file does
        not declare one. Terminal truncation is invisible to
        :meth:`chain_breaks`, which can only compare residues that are present,
        yet it is the common case: both chains of PDB 1YCR are truncated at both
        ends, and the terminus is exactly where a flank gets attached.

        Parameters
        ----------
        probe : int
            How many resolved residues to match when locating the resolved
            stretch inside the full sequence.

        Returns
        -------
        (int, int)
        """
        full = self.full_sequence
        resolved = self.sequence
        if not full or not resolved:
            return (0, 0)
        head = resolved[:probe]
        tail = resolved[-probe:]
        start = full.find(head)
        end = full.rfind(tail)
        if start < 0 or end < 0:
            return (0, 0)
        return (start, len(full) - (end + len(tail)))

    def chain_breaks(self, peptide_bond_max: float = 2.0,
                     ca_max: float = 4.5) -> List[Tuple[int, int]]:
        """Numbering jumps where the backbone is genuinely discontinuous.

        Decided geometrically rather than from numbering, because a numbering
        jump is often just a convention: on an antibody heavy chain every one of
        its Kabat jumps spans an ordinary peptide bond (C-N about 1.33 A). A
        break is reported only when consecutive residues are too far apart to be
        bonded, which is what actually means residues are unresolved.

        Parameters
        ----------
        peptide_bond_max : float
            Longest C-N distance still counted as a peptide bond, in angstroms.
        ca_max : float
            Fallback CA-CA ceiling, used when a C or N atom is absent.
            Consecutive alpha carbons sit about 3.8 A apart.

        Returns
        -------
        list of (int, int)
            ``(last_before, first_after)`` for each genuine break.
        """
        out: List[Tuple[int, int]] = []
        for prev, nxt in zip(self.residues, self.residues[1:]):
            if nxt.seq_id - prev.seq_id <= 1:
                continue
            c, n = prev.atom("C"), nxt.atom("N")
            if c is not None and n is not None:
                bonded = float(np.linalg.norm(c.xyz - n.xyz)) <= peptide_bond_max
            elif prev.ca is not None and nxt.ca is not None:
                bonded = float(
                    np.linalg.norm(prev.ca.xyz - nxt.ca.xyz)) <= ca_max
            else:
                bonded = False
            if not bonded:
                out.append((prev.seq_id, nxt.seq_id))
        return out

    def residue_by_seq_id(self, seq_id: int,
                          ins_code: str = "") -> Optional[Residue]:
        for r in self.residues:
            if r.seq_id == seq_id and r.ins_code.strip() == ins_code.strip():
                return r
        return None

    def __len__(self) -> int:
        return len(self.residues)

    def __iter__(self) -> Iterator[Residue]:
        return iter(self.residues)

    def __getitem__(self, i):
        return self.residues[i]

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (f"<Chain {self.chain_id!r}: {len(self.residues)} residues, "
                f"seq={self.sequence[:24]}{'...' if len(self.residues) > 24 else ''}>")


class Structure:
    """Parsed structure: an ordered mapping of chain id to :class:`Chain`."""

    __slots__ = ("chains", "path", "format", "model", "skipped_residues",
                 "warnings", "plddt_from_bfactor",
                 "_hetero_xyz", "_hetero_elements")

    def __init__(self, path: str = "", fmt: str = "", model: Optional[int] = None):
        self.chains: "Dict[str, Chain]" = {}
        self.path = path
        self.format = fmt
        self.model = model
        # resname -> count, for residues we could not interpret as amino acids
        self.skipped_residues: Dict[str, int] = {}
        # non-fatal problems worth surfacing to the caller
        self.warnings: List[str] = []
        # True when the B-factor column holds per-residue pLDDT rather than a
        # crystallographic B-factor -- i.e. a predicted structure. Set only on
        # an explicit declaration (mmCIF _ma_qa_metric or an AlphaFold REMARK),
        # never guessed from magnitude, since crystal B-factors also fall in
        # [0, 100].
        self.plddt_from_bfactor: bool = False
        # Heavy atoms of residues that are not amino acids (nucleic acids,
        # glycans, cofactors). They are not part of any chain, but they do
        # occlude solvent, so their coordinates are kept.
        self._hetero_xyz: List[np.ndarray] = []
        self._hetero_elements: List[str] = []

    @property
    def chain_ids(self) -> List[str]:
        return list(self.chains.keys())

    def heteroatoms(self) -> Tuple[np.ndarray, List[str]]:
        """Heavy atoms of non-amino-acid residues, as ``(coords, elements)``.

        Nucleic acids, glycans and cofactors are not returned as chains, but
        they occupy space: target surface covered by bound DNA or a glycan would
        otherwise be reported as solvent-exposed and offered up as somewhere a
        flank could bind. Solvent and simple ions are excluded.
        """
        if not self._hetero_xyz:
            return np.empty((0, 3), dtype=np.float64), []
        return (np.asarray(self._hetero_xyz, dtype=np.float64),
                list(self._hetero_elements))

    def __len__(self) -> int:
        return len(self.chains)

    def __iter__(self) -> Iterator[Chain]:
        return iter(self.chains.values())

    def __contains__(self, chain_id: str) -> bool:
        return chain_id in self.chains

    def __getitem__(self, chain_id: str) -> Chain:
        try:
            return self.chains[chain_id]
        except KeyError:
            raise KeyError(
                f"No chain {chain_id!r} in {self.path or 'structure'}. "
                f"Available chains: {self.chain_ids}"
            ) from None

    def get_chain(self, chain_id: str) -> Chain:
        return self[chain_id]

    def summary(self) -> str:
        lines = [f"Structure: {os.path.basename(self.path) or '<memory>'} "
                 f"({self.format}, model {self.model})"]
        for ch in self.chains.values():
            first = ch.residues[0].label if ch.residues else "-"
            last = ch.residues[-1].label if ch.residues else "-"
            lines.append(f"  chain {ch.chain_id!r}: {len(ch):4d} residues "
                         f"[{first} .. {last}]")
            lines.append(f"      {ch.sequence}")
        if self.skipped_residues:
            top = sorted(self.skipped_residues.items(),
                         key=lambda kv: -kv[1])[:8]
            lines.append("  skipped (non-amino-acid): "
                         + ", ".join(f"{k}x{v}" for k, v in top))
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Structure {os.path.basename(self.path)!r} chains={self.chain_ids}>"


# ---------------------------------------------------------------------------
# shared assembly logic
# ---------------------------------------------------------------------------

class _Builder:
    """Accumulates parsed atom records into a :class:`Structure`.

    Handles altloc resolution and residue grouping identically for both file
    formats so the two parsers cannot drift apart.
    """

    def __init__(self, path: str, fmt: str, model: Optional[int],
                 keep_altloc: str = "occupancy"):
        self.struct = Structure(path=path, fmt=fmt, model=model)
        self.keep_altloc = keep_altloc
        # (chain_id, seq_id, ins_code) -> Residue
        self._index: Dict[Tuple[str, int, str], Residue] = {}
        # (chain_id, seq_id, ins_code, atom_name) -> Atom, for altloc contests
        self._atom_index: Dict[Tuple[str, int, str, str], Atom] = {}
        # residue keys already counted in skipped_residues
        self._skipped_seen: set = set()
        # residue positions with two different residue names, and positions
        # whose numbering appears to collide
        self._microhet: set = set()
        self._collisions: set = set()

    def note_skipped(self, resname: str,
                     xyz: Optional[np.ndarray] = None,
                     element: str = "",
                     residue_key: Optional[tuple] = None) -> None:
        if _normalise_het_name(resname) in _SOLVENT:
            return
        # Count residues, not atoms: the field is named and displayed as a
        # residue count, so one haem must not read as 43.
        key = (resname,) if residue_key is None else residue_key
        if key not in self._skipped_seen:
            self._skipped_seen.add(key)
            self.struct.skipped_residues[resname] = \
                self.struct.skipped_residues.get(resname, 0) + 1
        # Keep the coordinates: these atoms still block solvent even though
        # they are not part of a polypeptide chain.
        if xyz is not None and element not in ("H", "D"):
            self.struct._hetero_xyz.append(xyz)
            self.struct._hetero_elements.append(element)

    def add(self, chain_id: str, resname: str, seq_id: int, ins_code: str,
            atom_name: str, element: str, xyz: np.ndarray, altloc: str,
            occupancy: float, bfactor: float, is_hetatm: bool,
            serial: int) -> None:
        rkey = (chain_id, seq_id, ins_code)
        res = self._index.get(rkey)
        if res is None:
            chain = self.struct.chains.get(chain_id)
            if chain is None:
                chain = Chain(chain_id)
                self.struct.chains[chain_id] = chain
            res = Residue(resname, seq_id, ins_code, chain_id,
                          index=len(chain.residues))
            chain.residues.append(res)
            self._index[rkey] = res

        # Microheterogeneity: two different residue names can share one
        # position. Keep whichever name arrived first and ignore atoms of the
        # other, rather than splicing them into one chimeric residue.
        if resname != res.resname:
            self._microhet.add((chain_id, seq_id, ins_code))
            return

        # A residue whose atoms have all been seen already means the numbering
        # collides -- a chain whose numbering restarts, say. Silently folding
        # the second copy into the first drops its atoms, so record it.
        if atom_name in {a.name for a in res.atoms} and altloc == "":
            self._collisions.add((chain_id, seq_id, ins_code))

        akey = (chain_id, seq_id, ins_code, atom_name)
        existing = self._atom_index.get(akey)
        atom = Atom(atom_name, element, xyz, altloc, occupancy, bfactor,
                    is_hetatm, serial)
        if existing is None:
            res.atoms.append(atom)
            self._atom_index[akey] = atom
            return

        # Duplicate atom name in the same residue: an alternate location.
        # Keep the higher-occupancy copy, breaking ties by altloc order so the
        # result is deterministic regardless of file ordering.
        if self.keep_altloc == "first":
            return
        better = (occupancy > existing.occupancy or
                  (occupancy == existing.occupancy and
                   altloc and existing.altloc and altloc < existing.altloc))
        if better:
            res.atoms[res.atoms.index(existing)] = atom
            self._atom_index[akey] = atom

    def finish(self) -> Structure:
        # Drop chains with no interpretable residues, repair out-of-order
        # residues, and renumber indices so Residue.index is always a valid
        # position in Chain.residues.
        for cid in list(self.struct.chains):
            chain = self.struct.chains[cid]
            chain.residues = [r for r in chain.residues if r.atoms]
            if not chain.residues:
                del self.struct.chains[cid]
                continue

            # Modified residues are written as HETATM, and some files emit
            # every HETATM after every ATOM. That would splice them onto the
            # end of the chain and corrupt the sequence, so put residues back
            # into numbering order when file order is not already sorted.
            keys = [(r.seq_id, r.ins_code.strip()) for r in chain.residues]
            if keys != sorted(keys):
                # Residue keys are unique by construction (they are the
                # dict keys used to group atoms), so sorting is unambiguous.
                chain.residues = sorted(
                    chain.residues,
                    key=lambda r: (r.seq_id, r.ins_code.strip()))
                self.struct.warnings.append(
                    f"chain {cid!r}: residues were not in numbering order in "
                    f"the file (usually HETATM records written after ATOM "
                    f"records); reordered by residue number."
                )

            for i, r in enumerate(chain.residues):
                r.index = i
            chain._coords_cache = None

        if self._microhet:
            self.struct.warnings.append(
                f"{len(self._microhet)} residue position(s) carry more than one "
                f"residue name (microheterogeneity); the first name seen was "
                f"kept and the alternatives dropped: "
                + ", ".join(f"{c}:{s}{i.strip()}"
                            for c, s, i in sorted(self._microhet)[:5]))
        if self._collisions:
            self.struct.warnings.append(
                f"{len(self._collisions)} residue position(s) appear more than "
                f"once with the same identity, so numbering collides and the "
                f"later atoms were discarded: "
                + ", ".join(f"{c}:{s}{i.strip()}"
                            for c, s, i in sorted(self._collisions)[:5])
                + ". Check the chain numbering.")
        return self.struct


def _open_text(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


# ---------------------------------------------------------------------------
# PDB
# ---------------------------------------------------------------------------

def read_pdb(path: str, model: Optional[int] = None,
             keep_altloc: str = "occupancy") -> Structure:
    """Parse a legacy-format PDB file.

    Parameters
    ----------
    path : str
        Path to a ``.pdb`` (optionally ``.gz``) file.
    model : int, optional
        1-based model number to read. Default (``None``) reads the first model.
    keep_altloc : str
        ``"occupancy"`` keeps the highest-occupancy alternate location;
        ``"first"`` keeps whichever appears first in the file.

    Returns
    -------
    Structure
    """
    path = os.fspath(path)
    builder = _Builder(path, "pdb", model, keep_altloc)
    # None means "whichever model comes first", not "model 1": NMR ensembles and
    # some predictors number their first model 0, or start at something else.
    wanted: Optional[int] = None if model is None else int(model)
    current_model = 1
    seen_model_record = False
    found_wanted = False
    n_atom_records = 0
    seqres: Dict[str, List[str]] = {}
    seen_plddt_header = False

    with _open_text(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec.startswith("REMARK") and "PLDDT" in line.upper():
                seen_plddt_header = True
                continue
            if rec == "SEQRES":
                # SEQRES gives the chain as deposited, including residues that
                # were never resolved. Columns: 12 chain id, 20-70 residue
                # names in four-character fields.
                cid = line[11] if len(line) > 11 else " "
                names = line[19:70].split()
                seqres.setdefault(cid.strip() or " ", []).extend(names)
                continue
            if rec == "MODEL ":
                seen_model_record = True
                try:
                    current_model = int(line[10:14])
                except ValueError:
                    current_model += 1
                if wanted is None:
                    wanted = current_model
                continue
            if rec == "ENDMDL":
                if seen_model_record and current_model == wanted:
                    break
                continue
            if rec != "ATOM  " and rec != "HETATM":
                continue
            if seen_model_record and current_model != wanted:
                continue
            found_wanted = True
            n_atom_records += 1

            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            if resname not in THREE_TO_ONE:
                raw = line[76:78].strip() if len(line) >= 78 else ""
                el = raw.capitalize() if raw.isalpha() else ""
                if not el:
                    el = _element_from_pdb_columns(line[12:16])
                try:
                    het_xyz = np.array(
                        (float(line[30:38]), float(line[38:46]),
                         float(line[46:54])), dtype=np.float64)
                except ValueError:
                    het_xyz = None
                builder.note_skipped(
                    resname, het_xyz, el,
                    residue_key=(resname, line[21], line[22:27]))
                continue

            # Columns 77-78 are optional and legacy writers put charges or
            # other junk there. Believe them only if they look like an element
            # symbol; otherwise a hydrogen named "HB1" is kept as a heavy atom
            # of element "1".
            raw_element = line[76:78].strip() if len(line) >= 78 else ""
            element = raw_element.capitalize() if raw_element.isalpha() else ""
            if not element:
                element = _element_from_pdb_columns(line[12:16])
            if element in ("H", "D"):
                continue

            try:
                xyz = np.array((float(line[30:38]), float(line[38:46]),
                                float(line[46:54])), dtype=np.float64)
            except ValueError:
                raise StructureParseError(
                    f"{path}: malformed coordinates in line:\n{line.rstrip()}"
                ) from None

            chain_id = line[21].strip() or " "
            try:
                seq_id = int(line[22:26])
            except ValueError:
                raise StructureParseError(
                    f"{path}: malformed residue number in line:\n{line.rstrip()}"
                ) from None
            ins_code = line[26] if len(line) > 26 else " "
            altloc = line[16].strip() if len(line) > 16 else ""

            occ_txt = line[54:60].strip() if len(line) >= 60 else ""
            bf_txt = line[60:66].strip() if len(line) >= 66 else ""
            try:
                occupancy = float(occ_txt) if occ_txt else 1.0
            except ValueError:
                occupancy = 1.0
            try:
                bfactor = float(bf_txt) if bf_txt else 0.0
            except ValueError:
                bfactor = 0.0
            try:
                serial = int(line[6:11])
            except ValueError:
                serial = 0

            builder.add(chain_id, resname, seq_id, ins_code, atom_name,
                        element, xyz, altloc, occupancy, bfactor,
                        rec == "HETATM", serial)

    if seen_model_record and not found_wanted:
        raise StructureParseError(
            f"{path}: model {wanted} not found in file."
        )
    if model is not None and not seen_model_record and int(model) != 1:
        raise StructureParseError(
            f"{path}: model={model} was requested but the file contains no "
            f"MODEL records, so it holds a single unnumbered model."
        )

    struct = builder.finish()
    struct.model = wanted
    struct.plddt_from_bfactor = seen_plddt_header
    for cid, names in seqres.items():
        chain = struct.chains.get(cid)
        if chain is not None:
            chain.full_sequence = "".join(
                THREE_TO_ONE.get(n.upper(), "X") for n in names)
    if not struct.chains:
        if n_atom_records == 0:
            raise StructureParseError(
                f"{path}: no ATOM or HETATM records found -- this does not "
                f"look like a PDB file."
            )
        raise StructureParseError(
            f"{path}: found {n_atom_records} atom record(s) but no amino-acid "
            f"residues. Skipped residue names: "
            f"{sorted(struct.skipped_residues)[:10]}. Is this a nucleic-acid "
            f"or ligand-only structure?"
        )
    return struct


# ---------------------------------------------------------------------------
# mmCIF / PDBx
# ---------------------------------------------------------------------------

def _cif_tokenize_line(line: str) -> List[Tuple[str, bool]]:
    """Split one mmCIF data line into tokens, honouring quoting rules.

    A quote character only opens a quoted string at the start of a token, and
    only closes it when the matching quote is followed by whitespace or EOL.
    That rule is what lets values like ``O5'`` and ``a "quoted" word`` parse.
    """
    tokens: List[Tuple[str, bool]] = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c in " \t":
            i += 1
            continue
        if c in "'\"":
            quote = c
            i += 1
            start = i
            while i < n:
                if line[i] == quote and (i + 1 >= n or line[i + 1] in " \t"):
                    break
                i += 1
            # Flagged as quoted so a value like '_entity' is never mistaken for
            # a tag and treated as the end of the loop.
            tokens.append((line[start:i], True))
            i += 1
            continue
        if c == "#":
            break  # comment to end of line
        start = i
        while i < n and line[i] not in " \t":
            i += 1
        tokens.append((line[start:i], False))
    return tokens


def _cif_token_stream(fh: Iterable[str]) -> Iterator[Tuple[str, bool]]:
    """Yield ``(token, is_literal)`` for every value/tag token in a CIF file.

    ``is_literal`` marks tokens that came from a semicolon-delimited multi-line
    block or from a quoted string. Either can legally contain characters -- a
    leading underscore, or the word ``loop_`` -- that would otherwise look like
    structure, so the caller must treat them as opaque values.
    """
    in_block = False
    block: List[str] = []
    for raw in fh:
        line = raw.rstrip("\n").rstrip("\r")
        if in_block:
            if line.startswith(";"):
                yield ("\n".join(block), True)
                in_block = False
                block = []
                # Anything after the closing semicolon on the same line is a
                # normal token sequence.
                rest = line[1:]
                if rest.strip():
                    for tok, quoted in _cif_tokenize_line(rest):
                        yield (tok, quoted)
            else:
                block.append(line)
            continue
        if line.startswith(";"):
            in_block = True
            block = [line[1:]]
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for tok, quoted in _cif_tokenize_line(line):
            yield (tok, quoted)
    if in_block:
        # Unterminated text block; emit what we have rather than losing it.
        yield ("\n".join(block), True)


_CIF_RESERVED = frozenset({"loop_", "stop_", "global_"})


def _cif_is_tag(token: str, was_block: bool) -> bool:
    return (not was_block) and token.startswith("_")


def _cif_is_terminator(token: str, was_block: bool) -> bool:
    """True if this token ends the current loop's value rows."""
    if was_block:
        return False
    if token.startswith("_"):
        return True
    low = token.lower()
    return low in _CIF_RESERVED or low.startswith(("data_", "save_"))


def _iter_atom_site_rows(fh: Iterable[str]) -> Iterator[Tuple[Dict[str, int], List[str]]]:
    """Yield ``(column_index, row)`` for each row of the ``_atom_site`` loop.

    Walks the token stream with a single-token pushback so that the token which
    terminates one loop (often the next ``loop_``) is not swallowed while
    skipping over loops we do not care about.
    """
    stream = _cif_token_stream(fh)
    pushback: Optional[Tuple[str, bool]] = None

    def next_tok() -> Optional[Tuple[str, bool]]:
        nonlocal pushback
        if pushback is not None:
            tok, pushback = pushback, None
            return tok
        return next(stream, None)

    while True:
        tok = next_tok()
        if tok is None:
            return
        if tok[1] or tok[0].lower() != "loop_":
            continue

        # ---- this loop's tag header ----
        tags: List[str] = []
        first_value: Optional[Tuple[str, bool]] = None
        while True:
            t = next_tok()
            if t is None:
                break
            if _cif_is_tag(*t):
                tags.append(t[0])
            else:
                first_value = t
                break

        if not tags:
            pushback = first_value
            continue

        if not tags[0].startswith("_atom_site."):
            # Skip this loop's value rows without materialising them, then hand
            # the terminating token back to the outer loop.
            t = first_value
            while t is not None and not _cif_is_terminator(*t):
                t = next_tok()
            pushback = t
            continue

        col = {t[len("_atom_site."):]: i for i, t in enumerate(tags)}
        width = len(tags)
        row: List[str] = []
        t = first_value
        while t is not None and not _cif_is_terminator(*t):
            row.append(t[0])
            if len(row) == width:
                yield col, row
                row = []
            t = next_tok()
        if row:
            # Truncated final row: pad so column lookups stay in range.
            yield col, row + [""] * (width - len(row))
        return


def _cif_entity_poly(path: str) -> Dict[str, str]:
    """Full deposited sequences per author chain, from ``_entity_poly``.

    Returns ``{chain_id: one_letter_sequence}``. Empty when the file omits the
    category. Read in a separate pass because ``_entity_poly`` precedes
    ``_atom_site`` and the atom-site reader stops as soon as it is done.
    """
    out: Dict[str, str] = {}
    with _open_text(path) as fh:
        stream = _cif_token_stream(fh)
        pending: Optional[Tuple[str, bool]] = None

        def nxt():
            nonlocal pending
            if pending is not None:
                t, pending = pending, None
                return t
            return next(stream, None)

        while True:
            tok = nxt()
            if tok is None:
                break
            name, blk = tok
            # Single-entity files use the item form rather than a loop.
            if not blk and name == "_entity_poly.pdbx_seq_one_letter_code_can":
                value = nxt()
                strand = None
                for _ in range(40):
                    t = nxt()
                    if t is None:
                        break
                    if not t[1] and t[0] == "_entity_poly.pdbx_strand_id":
                        strand = nxt()
                        break
                if value is not None and strand is not None:
                    seq = "".join(value[0].split())
                    for cid in strand[0].replace(",", " ").split():
                        out[cid] = seq
                continue
            if blk or name.lower() != "loop_":
                continue
            tags: List[str] = []
            first: Optional[Tuple[str, bool]] = None
            while True:
                t = nxt()
                if t is None:
                    break
                if _cif_is_tag(*t):
                    tags.append(t[0])
                else:
                    first = t
                    break
            if not tags or not tags[0].startswith("_entity_poly."):
                pending = first
                continue
            col = {t[len("_entity_poly."):]: i for i, t in enumerate(tags)}
            if "pdbx_seq_one_letter_code_can" not in col or \
                    "pdbx_strand_id" not in col:
                continue
            width = len(tags)
            row: List[str] = []
            t = first
            while t is not None and not _cif_is_terminator(*t):
                row.append(t[0])
                if len(row) == width:
                    seq = "".join(row[col["pdbx_seq_one_letter_code_can"]].split())
                    strands = row[col["pdbx_strand_id"]]
                    for cid in strands.replace(",", " ").split():
                        out[cid] = seq
                    row = []
                t = nxt()
            break
    return out


def _cif_declares_plddt(path: str) -> bool:
    """True if the mmCIF declares pLDDT as a model-quality metric.

    Looks for an ``_ma_qa_metric`` row whose type/name is pLDDT. That is how
    AlphaFold-DB and AF3 mmCIF files mark the B-factor column as confidence.
    """
    # The tag (_ma_qa_metric.name) and the value (pLDDT) sit on separate lines
    # in the loop, so require both to appear anywhere in the header region
    # rather than on one line. Stop at the atom records.
    saw_qa_metric = False
    saw_plddt = False
    try:
        with _open_text(path) as fh:
            for _ in range(50000):
                line = fh.readline()
                if not line:
                    break
                low = line.lower()
                if low.startswith(("atom ", "hetatm")):
                    break
                if "_ma_qa_metric" in low:
                    saw_qa_metric = True
                if "plddt" in low:
                    saw_plddt = True
                if saw_qa_metric and saw_plddt:
                    return True
    except OSError:
        return False
    return saw_qa_metric and saw_plddt


def read_cif(path: str, model: Optional[int] = None,
             keep_altloc: str = "occupancy",
             prefer_auth: bool = True) -> Structure:
    """Parse an mmCIF/PDBx file.

    Only the ``_atom_site`` loop is required. Author identifiers
    (``auth_asym_id`` / ``auth_seq_id``) are preferred because those are the
    chain letters and residue numbers users read off a paper or a viewer;
    ``label_*`` is used as a fallback when the author tags are absent (some
    structure predictors omit them).

    Parameters
    ----------
    path : str
        Path to a ``.cif``/``.mmcif`` (optionally ``.gz``) file.
    model : int, optional
        ``pdbx_PDB_model_num`` to read. Default reads the first model present.
    keep_altloc : str
        ``"occupancy"`` or ``"first"``.
    prefer_auth : bool
        Prefer author chain/residue identifiers over label identifiers.

    Returns
    -------
    Structure
    """
    path = os.fspath(path)
    builder = _Builder(path, "cif", model, keep_altloc)
    wanted_model: Optional[int] = None if model is None else int(model)
    n_rows = 0
    n_used = 0
    fallback_seq = 0
    fallback_seen: set = set()
    fallback_resname: Optional[str] = None
    models_seen: set = set()

    with _open_text(path) as fh:
        for col, row in _iter_atom_site_rows(fh):
            n_rows += 1

            def get(name: str, default: str = "") -> str:
                idx = col.get(name)
                if idx is None or idx >= len(row):
                    return default
                v = row[idx]
                return default if v in (".", "?", "") else v

            if "pdbx_PDB_model_num" in col:
                m_txt = get("pdbx_PDB_model_num")
                if m_txt:
                    try:
                        m = int(m_txt)
                    except ValueError:
                        m = 1
                    models_seen.add(m)
                    if wanted_model is None:
                        wanted_model = m
                    if m != wanted_model:
                        continue

            resname = ""
            if prefer_auth:
                resname = get("auth_comp_id")
            if not resname:
                resname = get("label_comp_id")
            resname = resname.strip().upper()
            if resname not in THREE_TO_ONE:
                raw = get("type_symbol").strip()
                el = raw.capitalize() if raw.isalpha() else ""
                if not el:
                    # Infer, so a ligand hydrogen is not retained as a
                    # carbon-sized occluder.
                    het_name = (get("auth_atom_id") or get("label_atom_id"))
                    el = _element_from_name(het_name.strip().strip('"').strip("'"))
                try:
                    het_xyz = np.array((float(row[col["Cartn_x"]]),
                                        float(row[col["Cartn_y"]]),
                                        float(row[col["Cartn_z"]])),
                                       dtype=np.float64)
                except (ValueError, KeyError, IndexError):
                    het_xyz = None
                builder.note_skipped(
                    resname, het_xyz, el,
                    residue_key=(resname, chain_id, seq_txt, ins_code))
                continue

            atom_name = ""
            if prefer_auth:
                atom_name = get("auth_atom_id")
            if not atom_name:
                atom_name = get("label_atom_id")
            atom_name = atom_name.strip().strip('"').strip("'")

            raw_element = get("type_symbol").strip()
            element = raw_element.capitalize() if raw_element.isalpha() else ""
            if not element:
                element = _element_from_name(atom_name)
            if element in ("H", "D"):
                continue

            try:
                xyz = np.array((float(row[col["Cartn_x"]]),
                                float(row[col["Cartn_y"]]),
                                float(row[col["Cartn_z"]])), dtype=np.float64)
            except (ValueError, KeyError, IndexError):
                raise StructureParseError(
                    f"{path}: malformed or missing coordinates in _atom_site "
                    f"row {n_rows}."
                ) from None

            chain_id = ""
            if prefer_auth:
                chain_id = get("auth_asym_id").strip()
            if not chain_id:
                chain_id = get("label_asym_id").strip() or " "

            ins_code = get("pdbx_PDB_ins_code").strip() or " "
            altloc = get("label_alt_id").strip()

            seq_txt = ""
            if prefer_auth:
                seq_txt = get("auth_seq_id").strip()
            if not seq_txt:
                seq_txt = get("label_seq_id").strip()
            try:
                seq_id = int(seq_txt)
            except ValueError:
                # An amino acid with no usable number: some writers emit '.'
                # for label_seq_id. Number by residue rather than by atom -- a
                # per-row counter would turn every atom into its own residue.
                # A new residue is signalled by the residue name changing or by
                # an atom name repeating within the current one.
                # Identity must not include altloc: the standard
                # sidechain-only alternate-conformation pattern would otherwise
                # split one residue into several and corrupt the sequence.
                if (resname != fallback_resname
                        or (atom_name, altloc) in fallback_seen):
                    fallback_seq += 1
                    fallback_seen = set()
                    fallback_resname = resname
                fallback_seen.add((atom_name, altloc))
                seq_id = fallback_seq

            try:
                occupancy = float(get("occupancy", "1.0"))
            except ValueError:
                occupancy = 1.0
            try:
                bfactor = float(get("B_iso_or_equiv", "0.0"))
            except ValueError:
                bfactor = 0.0
            try:
                serial = int(get("id", "0"))
            except ValueError:
                serial = 0

            group = get("group_PDB", "ATOM").upper()
            builder.add(chain_id, resname, seq_id, ins_code, atom_name,
                        element, xyz, altloc, occupancy, bfactor,
                        group == "HETATM", serial)
            n_used += 1

    if n_rows == 0:
        raise StructureParseError(
            f"{path}: no _atom_site loop found -- is this really an mmCIF file?"
        )
    if model is not None and models_seen and int(model) not in models_seen:
        raise StructureParseError(
            f"{path}: model {model} not found; the file contains model(s) "
            f"{sorted(models_seen)[:12]}."
        )
    builder.struct.model = wanted_model
    struct = builder.finish()
    struct.plddt_from_bfactor = _cif_declares_plddt(path)
    try:
        for cid, seq in _cif_entity_poly(path).items():
            chain = struct.chains.get(cid)
            if chain is not None:
                chain.full_sequence = seq
    except Exception:  # pragma: no cover - header parsing must never be fatal
        pass
    if not struct.chains:
        raise StructureParseError(
            f"{path}: parsed {n_rows} _atom_site row(s) but found no amino-acid "
            f"residues. Skipped residue names: "
            f"{sorted(struct.skipped_residues)[:10]}. Is this a nucleic-acid "
            f"or ligand-only structure?"
        )
    return struct


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def read_structure(path: str, model: Optional[int] = None,
                   keep_altloc: str = "occupancy",
                   fmt: Optional[str] = None) -> Structure:
    """Read a ``.pdb`` or ``.cif`` file, dispatching on extension or content.

    Parameters
    ----------
    path : str
        Path to the structure file. ``.gz`` is handled transparently.
    model : int, optional
        Model to read; default is the first model in the file.
    keep_altloc : str
        ``"occupancy"`` (default) or ``"first"``.
    fmt : str, optional
        Force ``"pdb"`` or ``"cif"`` instead of sniffing.

    Returns
    -------
    Structure
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such structure file: {path}")

    if fmt is None:
        low = path.lower()
        if low.endswith(".gz"):
            low = low[:-3]
        if low.endswith((".cif", ".mmcif", ".pdbx")):
            fmt = "cif"
        elif low.endswith((".pdb", ".ent", ".pdb1")):
            fmt = "pdb"
        else:
            fmt = _sniff_format(path)

    if fmt == "cif":
        return read_cif(path, model=model, keep_altloc=keep_altloc)
    if fmt == "pdb":
        return read_pdb(path, model=model, keep_altloc=keep_altloc)
    raise ValueError(f"Unknown structure format {fmt!r}; expected 'pdb' or 'cif'.")


def _sniff_format(path: str) -> str:
    """Guess the format by looking for format-defining tokens near the top."""
    path = os.fspath(path)
    with _open_text(path) as fh:
        for _ in range(400):
            line = fh.readline()
            if not line:
                break
            s = line.lstrip()
            if s.startswith(("data_", "loop_", "_atom_site.", "_entry.")):
                return "cif"
            if line[:6] in ("ATOM  ", "HETATM") or line.startswith(
                    ("HEADER", "CRYST1", "MODEL ", "SEQRES", "REMARK")):
                return "pdb"
    raise StructureParseError(
        f"{path}: could not determine whether this is a PDB or mmCIF file."
    )
