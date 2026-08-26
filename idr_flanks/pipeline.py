"""End-to-end: structure in, flanked binder sequence out."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .design import DesignConfig, DesignResult, design_flank
from .interface import InterfaceError, ProximalRegion, find_proximal_region
from .io import Structure, read_structure

__all__ = ["FlankedBinder", "build_flanked_binder", "describe_chains",
           "score_flanks"]


# Keyword arguments understood by find_proximal_region, so callers can pass
# them through build_flanked_binder without a second nested dict.
# Derived from find_proximal_region's own signature rather than hand-listed, so
# a parameter added there is routed automatically instead of being rejected as
# an unknown design option.
def _interface_keys() -> frozenset:
    import inspect
    params = set(inspect.signature(find_proximal_region).parameters)
    return frozenset(params - {"structure", "binder_chain", "target_chain",
                               "terminus", "flank_length"})


_INTERFACE_KEYS = _interface_keys()


@dataclass
class FlankedBinder:
    """The designed construct plus everything needed to justify it."""

    binder_sequence: str
    final_sequence: str
    binder_chain: str
    target_chain: str
    n_flank: str = ""
    c_flank: str = ""
    linker: str = ""
    """Flexible linker inserted between each flank and the binder, if any."""
    binder_full_sequence: str = ""
    """The binder chain as deposited, including unresolved residues. Splice the
    designed flank onto this rather than onto :attr:`binder_sequence` when the
    two differ."""
    regions: Dict[str, ProximalRegion] = field(default_factory=dict)
    designs: Dict[str, DesignResult] = field(default_factory=dict)
    structure: Optional[Structure] = None
    structure_warnings: List[str] = field(default_factory=list)
    """Problems with the input structure itself, such as unresolved residues in
    the binder chain, which mean the returned sequence is not the full binder."""

    def __str__(self) -> str:
        return self.final_sequence

    def __len__(self) -> int:
        return len(self.final_sequence)

    @property
    def added_residues(self) -> int:
        n_linkers = bool(self.n_flank) + bool(self.c_flank)
        return len(self.n_flank) + len(self.c_flank) + n_linkers * len(self.linker)

    @property
    def warnings(self) -> List[str]:
        """Everything worth surfacing, from all three stages.

        Region notes are included: without them ``--quiet`` and every
        programmatic caller would be blind to the whole interface stage,
        including the small-patch and unresolved-target-break notices.
        """
        out: List[str] = list(self.structure_warnings)
        for term, region in self.regions.items():
            for note in region.notes:
                out.append(f"{term}-terminal region: {note}")
        for term, design in self.designs.items():
            for w in design.warnings:
                out.append(f"{term}-terminal flank: {w}")
        return out

    def annotated_sequence(self) -> str:
        """Final sequence with flanks bracketed and linkers in parentheses."""
        parts = []
        if self.n_flank:
            parts.append(f"[{self.n_flank}]")
            if self.linker:
                parts.append(f"({self.linker})")
        parts.append(self.binder_sequence)
        if self.c_flank:
            if self.linker:
                parts.append(f"({self.linker})")
            parts.append(f"[{self.c_flank}]")
        return "".join(parts)

    def fasta(self, name: str = "flanked_binder", width: int = 60) -> str:
        """The final sequence as a FASTA record."""
        header = (f">{name} binder_chain={self.binder_chain} "
                  f"target_chain={self.target_chain} "
                  f"n_flank={len(self.n_flank)} c_flank={len(self.c_flank)}"
                  + (f" linker={self.linker}" if self.linker else ""))
        body = "\n".join(self.final_sequence[i:i + width]
                         for i in range(0, len(self.final_sequence), width))
        return f"{header}\n{body}"

    def summary(self) -> str:
        lines = [
            "=" * 72,
            f"Flanked binder: chain {self.binder_chain!r} extended against "
            f"target chain {self.target_chain!r}",
            "=" * 72,
            f"original binder ({len(self.binder_sequence)} aa resolved"
            + (f", {len(self.binder_full_sequence)} aa deposited)"
               if self.binder_full_sequence
               and len(self.binder_full_sequence) != len(self.binder_sequence)
               else ")"),
            f"  {self.binder_sequence}",
        ]
        for w in self.structure_warnings:
            lines.append(f"  WARNING: {w}")
        for term in ("N", "C"):
            if term not in self.designs:
                continue
            lines.append("")
            lines.append("-" * 72)
            lines.append(f"{term}-terminal flank")
            lines.append("-" * 72)
            lines.append(self.regions[term].summary())
            lines.append("")
            lines.append(self.designs[term].summary())
        lines += [
            "",
            "=" * 72,
            f"final construct ({len(self.final_sequence)} aa, "
            f"+{self.added_residues}):",
            f"  {self.annotated_sequence()}",
            "=" * 72,
        ]
        return "\n".join(lines)


def describe_chains(structure: Union[str, "os.PathLike", Structure]) -> str:
    """Summarise the chains in a structure, to help pick binder and target.

    Parameters
    ----------
    structure : str or Structure
        Path to a structure file, or an already-parsed structure.

    Returns
    -------
    str
    """
    struct = (structure if isinstance(structure, Structure)
              else read_structure(structure))
    return struct.summary()


def build_flanked_binder(
    structure: Union[str, "os.PathLike", Structure],
    binder_chain: str,
    target_chain: str,
    *,
    n_flank_length: int = 0,
    c_flank_length: int = 0,
    linker_length: int = 0,
    linker_sequence: Optional[str] = None,
    model: Optional[int] = None,
    preset: Optional[str] = None,
    config: Optional[DesignConfig] = None,
    patch_weighting: int = 1,
    reach_weighted: bool = True,
    shell_edges: Sequence[float] = (5.0, 10.0, 15.0),
    auto_detect_region: bool = False,
    interface_options: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> FlankedBinder:
    """Design flanking IDRs for a binder against its target.

    Parameters
    ----------
    structure : str or Structure
        Path to a ``.pdb``/``.cif`` file, or an already-parsed structure.
    binder_chain : str
        Author chain id of the binder -- the chain that gets extended.
    target_chain : str
        Author chain id of the target.
    n_flank_length, c_flank_length : int
        Residues to add at each terminus. At least one must be positive.
    linker_length : int
        Length of a flexible linker inserted between each flank and the binder.
        A designed flank is chemically loaded on purpose, and placing it hard
        against the binder risks perturbing how the binder folds; a short
        GS linker buys separation. The linker also lengthens the tether, so the
        reach used to pick target residues accounts for it.
    linker_sequence : str, optional
        Explicit linker sequence. Defaults to a ``GS`` repeat trimmed to
        ``linker_length``. Ignored when ``linker_length`` is 0.
    model : int, optional
        Structure model to read, when the file has several.
    preset : str, optional
        Design preset; see :data:`~idr_flanks.design.PRESETS`.
    config : DesignConfig, optional
        Full design configuration instead of a preset.
    reach_weighted : bool
        Aim the attraction objective at a weighted average over distance shells
        rather than flatly at the whole patch. On by default: roughly half the
        selected residues typically sit beyond 15 A of the anchor, where the
        tethered-chain monomer density is under a tenth of its contact value,
        yet a flat patch lets them contribute in proportion to their count.
        Costs one epsilon call per shell per candidate.
    shell_edges : sequence of float
        Distance-shell boundaries in angstroms, used when ``reach_weighted``.
    patch_weighting : int
        Repeat target residues near the attachment point up to this many times,
        so the design preferentially complements the surface the flank is most
        likely to touch. ``1`` (the default) weights every reachable residue
        equally. See
        :meth:`~idr_flanks.interface.ProximalRegion.weighted_patch_sequence`.
    auto_detect_region : bool
        Allow the target region to be chosen automatically from binder-target
        contacts. Off by default: you must say which target residues the flank
        should complement, via ``target_residues`` (or
        ``include_target_residues``).

        This is deliberate. On predicted structures the automatic interface
        detection cannot reliably tell a real epitope from an artifact -- a
        mispredicted region docked onto the binder is spatially,
        compositionally and in buried area indistinguishable from a genuine
        second epitope -- so designing against an auto-detected region can
        silently aim the flank at the wrong surface. Explore the interface first
        with :func:`find_proximal_region` or ``idr-flanks contacts`` (which show
        the ranked, artifact-flagged patch report), decide which region is real,
        then pass it here. Set ``auto_detect_region=True`` only for
        well-behaved experimental structures where you trust the interface.
    interface_options : dict, optional
        Extra keyword arguments for
        :func:`~idr_flanks.interface.find_proximal_region`.
    **overrides
        Either :class:`~idr_flanks.design.DesignConfig` fields or
        ``find_proximal_region`` parameters; they are routed automatically.

    Returns
    -------
    FlankedBinder

    Raises
    ------
    ValueError
        If a flank length is negative, or neither is positive.
    InterfaceError
        If a chain id is absent, both ids are the same, or the chains do not
        form an analysable interface.
    """
    if n_flank_length < 0 or c_flank_length < 0:
        raise ValueError(
            f"flank lengths cannot be negative (got n={n_flank_length}, "
            f"c={c_flank_length})")
    if n_flank_length == 0 and c_flank_length == 0:
        raise ValueError(
            "at least one of n_flank_length or c_flank_length must be "
            "positive -- otherwise there is nothing to design.")
    if linker_length < 0:
        raise ValueError(
            f"linker_length cannot be negative, got {linker_length}")

    if linker_sequence is not None and not linker_length:
        raise ValueError(
            "linker_sequence was given but linker_length is 0, so no linker "
            "would be inserted. Set linker_length to len(linker_sequence).")
    if linker_length:
        if linker_sequence is None:
            linker = ("GS" * (linker_length // 2 + 1))[:linker_length]
        else:
            linker = str(linker_sequence).strip().upper()
            if len(linker) != linker_length:
                raise ValueError(
                    f"linker_sequence has length {len(linker)} but "
                    f"linker_length is {linker_length}; give one or the other.")
            unknown = sorted(set(linker) - set("ACDEFGHIKLMNPQRSTVWY"))
            if unknown:
                raise ValueError(
                    f"linker_sequence contains non-amino-acid character(s): "
                    f"{unknown}")
    else:
        linker = ""

    if isinstance(structure, Structure):
        if model is not None:
            raise ValueError(
                "model= only applies when a file path is given; the structure "
                "passed in has already been parsed. Read it with "
                "read_structure(path, model=...) instead.")
        struct = structure
    else:
        struct = read_structure(structure, model=model)

    # Validate chains here so a bad id is an InterfaceError with the available
    # chains listed, rather than a bare KeyError from the structure lookup.
    for name, cid in (("binder_chain", binder_chain),
                      ("target_chain", target_chain)):
        if cid not in struct:
            raise InterfaceError(
                f"{name}={cid!r} is not in the structure. "
                f"Available chains: {struct.chain_ids}")
    if binder_chain == target_chain:
        raise InterfaceError(
            f"binder_chain and target_chain are both {binder_chain!r}; "
            f"they must be different chains.")

    # Route stray keyword arguments to whichever stage understands them.
    iface_kwargs: Dict[str, Any] = dict(interface_options or {})
    design_overrides: Dict[str, Any] = {}
    for key, value in overrides.items():
        if key in _INTERFACE_KEYS:
            if key in iface_kwargs and iface_kwargs[key] != value:
                raise ValueError(
                    f"{key!r} was given both in interface_options "
                    f"({iface_kwargs[key]!r}) and as a keyword argument "
                    f"({value!r}); pass it once.")
            iface_kwargs[key] = value
        else:
            design_overrides[key] = value

    if not auto_detect_region and not (
            iface_kwargs.get("target_residues")
            or iface_kwargs.get("include_target_residues")):
        raise ValueError(
            f"no target region was specified. You must say which residues of "
            f"target chain {target_chain!r} the flank should complement, e.g. "
            f"target_residues=[250, 280]. Automatic interface detection is not "
            f"trustworthy on predicted structures -- a mispredicted region "
            f"docked onto the binder is indistinguishable from a real epitope "
            f"-- so it is not used by default. Explore the interface first with "
            f"describe_chains / find_proximal_region (or `idr-flanks contacts`), "
            f"which rank the interface patches and flag likely artifacts, then "
            f"pass the region you trust. To opt into automatic detection anyway "
            f"(e.g. for a reliable experimental structure), pass "
            f"auto_detect_region=True.")

    binder = struct[binder_chain]
    binder_seq = binder.sequence
    if not binder_seq:
        raise ValueError(
            f"binder chain {binder_chain!r} has no interpretable residues.")

    # A structure only contains the residues that were resolved. If the binder
    # chain has numbering gaps, its sequence here is the resolved residues
    # spliced together, so the returned construct is NOT the real binder plus a
    # flank -- it is missing whatever the structure is missing.
    structure_warnings: List[str] = list(struct.warnings)
    # Geometric breaks, not numbering jumps: antibody Kabat numbering skips
    # numbers by convention on a chain that is physically continuous, and
    # warning about those would be wrong every time.
    # Terminal truncation is invisible to chain_breaks (it can only compare
    # residues that are present) yet it is the common case, and the terminus is
    # exactly where the flank attaches. Both chains of 1YCR are truncated.
    n_missing, c_missing = binder.unresolved_termini()
    for term, missing, length in (("N", n_missing, n_flank_length),
                                  ("C", c_missing, c_flank_length)):
        if missing and length > 0:
            structure_warnings.append(
                f"the {term}-terminus of binder chain {binder_chain!r} is "
                f"truncated: {missing} residue(s) present in the deposited "
                f"sequence were not resolved, so the flank would be attached "
                f"{missing} residue(s) from where you expect and the anchor "
                f"used for the reach calculation is the wrong atom. Graft the "
                f"designed flank onto your full-length sequence instead."
            )
        elif missing:
            structure_warnings.append(
                f"the {term}-terminus of binder chain {binder_chain!r} is "
                f"truncated by {missing} unresolved residue(s); this does not "
                f"affect the flank you asked for.")

    breaks = binder.chain_breaks()
    if breaks:
        missing = sum(b - a - 1 for a, b in breaks)
        structure_warnings.append(
            f"binder chain {binder_chain!r} has {len(breaks)} unresolved "
            f"break(s) ({', '.join(f'{a}->{b}' for a, b in breaks[:5])}), so at "
            f"least {missing} residue(s) present in the real protein are absent "
            f"from the structure. The returned sequence contains only the "
            f"resolved residues; splice the designed flank onto your own "
            f"full-length binder sequence rather than using it verbatim."
        )

    regions: Dict[str, ProximalRegion] = {}
    designs: Dict[str, DesignResult] = {}
    n_flank = ""
    c_flank = ""

    # Design the N-terminal flank first so the C-terminal design can see it as
    # context; disorder prediction depends on the whole construct.
    for term, length in (("N", n_flank_length), ("C", c_flank_length)):
        if length <= 0:
            continue

        region = find_proximal_region(
            struct,
            binder_chain=binder_chain,
            target_chain=target_chain,
            terminus=term,
            # The linker is part of the tether, so it extends how far the flank
            # can reach from the attachment point.
            flank_length=length + linker_length,
            **iface_kwargs,
        )
        regions[term] = region

        # Contexts describe the finished construct, since predicted disorder
        # depends on the whole thing.
        if term == "N":
            n_context = ""
            c_context = linker + binder_seq + (linker + c_flank if c_flank else "")
        else:
            n_context = ((n_flank + linker) if n_flank else "") + binder_seq + linker
            c_context = ""

        shells = (region.weighted_shells(shell_edges)
                  if reach_weighted else None)

        result = design_flank(
            region.weighted_patch_sequence(patch_weighting),
            length,
            shells=shells,
            n_context=n_context,
            c_context=c_context,
            selected_patch=region.patch_sequence,
            binder_interface=region.binder_interface_sequence,
            epitope=region.epitope_sequence,
            preset=preset,
            config=config,
            **design_overrides,
        )
        designs[term] = result
        if term == "N":
            n_flank = result.sequence
        else:
            c_flank = result.sequence

    final = ((n_flank + linker if n_flank else "")
             + binder_seq
             + (linker + c_flank if c_flank else ""))

    return FlankedBinder(
        structure_warnings=structure_warnings,
        binder_sequence=binder_seq,
        binder_full_sequence=binder.full_sequence,
        linker=linker,
        final_sequence=final,
        binder_chain=binder_chain,
        target_chain=target_chain,
        n_flank=n_flank,
        c_flank=c_flank,
        regions=regions,
        designs=designs,
        structure=struct,
    )


def score_flanks(
    flanks,
    structure: Union[str, "os.PathLike", Structure],
    *,
    binder_chain: str,
    target_chain: str,
    terminus: str,
    flank_length: Optional[int] = None,
    model: str = "mpipi",
    reach_weighted: bool = True,
    shell_edges: Sequence[float] = (5.0, 10.0, 15.0),
    patch_weighting: int = 1,
    interface_options: Optional[Dict[str, Any]] = None,
    **overrides: Any,
):
    """Recompute epsilon vs patch / epitope / binder interface for existing flanks.

    Reconstructs the target references from the structure exactly as a design run
    would, then scores each supplied flank sequence against them -- so you can go
    back and evaluate flanks made earlier (or by another method) without
    rerunning the design.

    To reproduce the numbers a particular design reported, pass the SAME
    ``target_residues`` (and ``flank_length``, ``linker_length`` via
    ``flank_length``, terminus, cutoffs) you designed with; the patch, epitope
    and binder interface all depend on them.

    Parameters
    ----------
    flanks : str, sequence of str, or mapping of name -> str
        The flank sequence(s) to score.
    structure : str, path, or Structure
        The binder-target complex.
    binder_chain, target_chain : str
        Author chain ids.
    terminus : str
        ``"N"`` or ``"C"`` -- the end the flank is attached to (sets the anchor
        and hence the reach-based patch).
    flank_length : int, optional
        Flank length used to define reach. Defaults to the length of the first
        flank. If you designed with a linker, add its length here so the reach
        (and therefore the patch) matches.
    model : str
        Epsilon model.
    reach_weighted, shell_edges, patch_weighting
        Must match the design run for the patch to be identical; defaults match
        :func:`build_flanked_binder`.
    interface_options, **overrides
        Passed to :func:`~idr_flanks.interface.find_proximal_region` (e.g.
        ``target_residues``, ``contact_cutoff``). Pass what you designed with.

    Returns
    -------
    FlankEpsilonScores, list, or dict
        One result per flank, matching the shape of ``flanks`` (a bare string
        returns a single result; a mapping returns ``{name: result}``).

    Examples
    --------
    >>> score_flanks("EPQDNGPYD...", "complex.pdb", binder_chain="A",
    ...              target_chain="B", terminus="C", target_residues=[250, 280])
    """
    from .design import flank_epsilons
    from .interface import find_proximal_region

    if isinstance(flanks, Mapping):
        items = list(flanks.items())
    elif isinstance(flanks, str):
        items = [(None, flanks)]
    else:
        items = [(None, f) for f in flanks]
    if not items:
        raise ValueError("no flanks were given")

    struct = (structure if isinstance(structure, Structure)
              else read_structure(structure))

    length = flank_length if flank_length is not None else len(items[0][1])
    if length <= 0:
        raise ValueError("flank_length must be positive")

    iface_kwargs: Dict[str, Any] = dict(interface_options or {})
    iface_kwargs.update(overrides)

    region = find_proximal_region(
        struct, binder_chain=binder_chain, target_chain=target_chain,
        terminus=terminus, flank_length=length, **iface_kwargs)

    patch = region.weighted_patch_sequence(patch_weighting)
    shells = region.weighted_shells(shell_edges) if reach_weighted else None

    results = {}
    ordered = []
    for name, seq in items:
        scored = flank_epsilons(
            seq, patch=patch, epitope=region.epitope_sequence,
            binder_interface=region.binder_interface_sequence,
            shells=shells, model=model)
        ordered.append(scored)
        if name is not None:
            results[name] = scored

    if isinstance(flanks, Mapping):
        return results
    if isinstance(flanks, str):
        return ordered[0]
    return ordered
