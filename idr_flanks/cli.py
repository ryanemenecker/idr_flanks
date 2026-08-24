"""Command-line interface for :mod:`idr_flanks`."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idr-flanks",
        description="Design flanking IDRs that help a binder bind its target.",
    )
    parser.add_argument("--version", action="version",
                        version=f"idr-flanks {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser(
        "info", help="list the chains in a structure file")
    info.add_argument("structure", help="path to a .pdb or .cif file")
    info.add_argument("--model", type=int, default=None,
                      help="model to read (default: the first)")

    contacts = sub.add_parser(
        "contacts",
        help="show the target region a flank could reach, without designing")
    design = sub.add_parser(
        "design", help="design flanking IDRs and print the final sequence")

    for p in (contacts, design):
        p.add_argument("structure", help="path to a .pdb or .cif file")
        p.add_argument("-b", "--binder", required=True,
                       help="chain id of the binder (the chain to extend)")
        p.add_argument("-t", "--target", required=True,
                       help="chain id of the target")
        p.add_argument("--model", type=int, default=None,
                       help="model to read (default: the first)")
        p.add_argument("--contact-cutoff", type=float, default=5.0,
                       help="heavy-atom contact distance in A (default: 5.0)")
        p.add_argument("--radius", type=float, default=None,
                       help="override the reach radius in A")
        p.add_argument("--radius-scale", type=float, default=1.0,
                       help="multiply the reach radius (default: 1.0)")
        p.add_argument("--sequence-window", type=int, default=25,
                       help="residues beyond the interface still accepted "
                            "(default: 25)")
        p.add_argument("--cluster-gap", type=int, default=15,
                       help="sequence gap that splits interface patches "
                            "(default: 15)")
        p.add_argument("--surface-threshold", type=float, default=0.10,
                       help="minimum relative SASA to count as surface "
                            "(default: 0.10)")
        p.add_argument("--no-surface-filter", action="store_true",
                       help="keep buried target residues too (not advised)")
        p.add_argument("--max-residues", type=int, default=None,
                       help="keep only this many residues, nearest first")
        p.add_argument("--exclude-target", metavar="SPEC", default=None,
                       help="target residues to rule out, in author numbering, "
                            "e.g. '1-100' or '1-100,250-300'. Use when a "
                            "predictor has folded a region onto the real "
                            "binding site.")
        p.add_argument("--include-target", metavar="SPEC", default=None,
                       help="consider only these target residues, same format")
        p.add_argument("--trust-distal-occlusion", action="store_true",
                       help="let sequence-distant target regions occlude "
                            "solvent (appropriate for experimental, not "
                            "predicted, structures)")

    contacts.add_argument("-n", "--n-flank", type=int, default=0,
                          help="length of a hypothetical N-terminal flank")
    contacts.add_argument("-c", "--c-flank", type=int, default=0,
                          help="length of a hypothetical C-terminal flank")
    contacts.add_argument("--linker", type=int, default=0, metavar="N",
                          help="assume an N-residue linker, which lengthens "
                               "the tether and so widens the reach")

    design.add_argument("-n", "--n-flank", type=int, default=0,
                        help="residues to add at the binder N-terminus")
    design.add_argument("-c", "--c-flank", type=int, default=0,
                        help="residues to add at the binder C-terminus")
    design.add_argument("--preset", default="balanced",
                        help="design preset (default: balanced)")
    design.add_argument("--iterations", type=int, default=None,
                        help="GOOSE optimizer iterations")
    design.add_argument("--seed", type=int, default=None,
                        help="random seed, for reproducible designs")
    design.add_argument("--patch-weighting", type=int, default=1,
                        help="repeat target residues near the attachment "
                             "point up to N times (default: 1, no weighting)")
    design.add_argument("--target-epsilon", type=float, default=None,
                        help="desired per-residue epsilon (negative is "
                             "attractive); default is 'as attractive as "
                             "possible'")
    design.add_argument("--max-aromatic", type=float, default=None,
                        help="ceiling on the W+F+Y fraction")
    design.add_argument("--no-reach-weighting", action="store_true",
                        help="aim the objective flatly at the whole patch "
                             "instead of weighting by tethered-chain reach")
    design.add_argument("--linker", type=int, default=0, metavar="N",
                        help="insert an N-residue GS linker between each flank "
                             "and the binder")
    design.add_argument("--min-target-preference", type=float, default=None,
                        help="how much more the flank must prefer the target "
                             "over the binder's own interface, per residue "
                             "(default 0.05; 0 disables)")
    design.add_argument("--fasta", metavar="PATH", default=None,
                        help="also write the final sequence as FASTA")
    design.add_argument("--quiet", action="store_true",
                        help="print only the final sequence")
    return parser


def _interface_kwargs(args: argparse.Namespace) -> dict:
    return {
        "contact_cutoff": args.contact_cutoff,
        "radius": args.radius,
        "radius_scale": args.radius_scale,
        "sequence_window": args.sequence_window,
        "cluster_gap": args.cluster_gap,
        "surface_threshold": args.surface_threshold,
        "require_surface": not args.no_surface_filter,
        "max_residues": args.max_residues,
        "trust_distal_occlusion": args.trust_distal_occlusion,
        "exclude_target_residues": args.exclude_target,
        "include_target_residues": args.include_target,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Imported lazily so `--help` stays fast; neither module needs GOOSE.
    # Both must be imported before the try block, since the except clause
    # names them.
    from .interface import InterfaceError, find_proximal_region
    from .io import StructureParseError, read_structure

    try:
        if args.command == "info":
            struct = read_structure(args.structure, model=args.model)
            print(struct.summary())
            return 0

        if args.command == "contacts":
            if args.n_flank <= 0 and args.c_flank <= 0:
                parser.error("give --n-flank and/or --c-flank a positive length")
            struct = read_structure(args.structure, model=args.model)
            for term, length in (("N", args.n_flank), ("C", args.c_flank)):
                if length <= 0:
                    continue
                region = find_proximal_region(
                    struct, binder_chain=args.binder,
                    target_chain=args.target, terminus=term,
                    flank_length=length + args.linker,
                    **_interface_kwargs(args))
                print(region.summary())
                print()
            return 0

        if args.command == "design":
            if args.n_flank <= 0 and args.c_flank <= 0:
                parser.error("give --n-flank and/or --c-flank a positive length")
            from .pipeline import build_flanked_binder

            overrides = {}
            if args.iterations is not None:
                overrides["max_iterations"] = args.iterations
            if args.seed is not None:
                overrides["seed"] = args.seed
            if args.target_epsilon is not None:
                overrides["target_epsilon_per_residue"] = args.target_epsilon
            if args.max_aromatic is not None:
                overrides["max_aromatic_fraction"] = args.max_aromatic
            if args.min_target_preference is not None:
                overrides["min_target_preference"] = (
                    None if args.min_target_preference == 0
                    else args.min_target_preference)

            result = build_flanked_binder(
                args.structure,
                binder_chain=args.binder,
                target_chain=args.target,
                n_flank_length=args.n_flank,
                c_flank_length=args.c_flank,
                model=args.model,
                preset=args.preset,
                linker_length=args.linker,
                reach_weighted=not args.no_reach_weighting,
                patch_weighting=args.patch_weighting,
                interface_options=_interface_kwargs(args),
                **overrides,
            )
            if args.quiet:
                print(result.final_sequence)
                # Warnings still go to stderr: a scripted caller must not be
                # handed a construct that is missing binder residues, or that
                # competes with the target, with no indication anywhere.
                for warning in result.warnings:
                    print(f"warning: {warning}", file=sys.stderr)
            else:
                print(result.summary())
            if args.fasta:
                with open(args.fasta, "w") as fh:
                    fh.write(result.fasta() + "\n")
                if not args.quiet:
                    print(f"\nwrote {args.fasta}")
            return 0

    except (StructureParseError, InterfaceError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        # Covers FileNotFoundError, PermissionError, IsADirectoryError and
        # gzip.BadGzipFile, all of which a user can hit with a bad path.
        if exc.strerror:
            name = getattr(exc, "filename", None) or args.structure
            print(f"error: could not read {name}: {exc.strerror}",
                  file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyError as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 1

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
