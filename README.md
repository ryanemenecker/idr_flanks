# idr_flanks

Design flanking intrinsically disordered regions (IDRs) that improve a binder's
affinity for its target.

Given a structure of a binder–target complex, `idr_flanks`:

1. reads the structure (`.pdb` or `.cif`),
2. finds the region of the **target** that a new flank on the **binder** could
   actually reach,
3. uses [GOOSE](https://github.com/idptools/goose) to design a disordered
   sequence that is chemically complementary to that region,
4. returns the extended binder sequence, with the numbers you need to judge it.

The idea: a disordered tail that is electrostatically and chemically
complementary to the target surface next to the binding site adds avidity
without disturbing the folded interface.

## Install

```bash
pip install -e .
```

Requires `numpy`, `scipy`, and the idptools stack (`goose`, `finches`,
`sparrow`, `metapredict`). Structure reading and interface analysis work with
just `numpy`/`scipy`; GOOSE and FINCHES are imported only when you design.

## Quick start

```python
from idr_flanks import build_flanked_binder, describe_chains

# Which chain is which?
print(describe_chains("complex.pdb"))

result = build_flanked_binder(
    "complex.pdb",
    binder_chain="B",      # the chain to extend
    target_chain="A",      # the chain to bind better
    c_flank_length=30,     # residues to add at the binder's C-terminus
)

print(result.final_sequence)
print(result.summary())    # everything below, explained
```

From the command line:

```bash
idr-flanks info complex.pdb
```

```bash
idr-flanks design complex.pdb -b B -t A -c 30 --seed 1
```

`idr-flanks contacts` shows which target residues were selected without running
the (slower) design step.

## How the target region is chosen

A flank is a tethered polymer, so it can only interact with target surface that
is *near where it is attached*, *exposed*, and *plausibly part of the real
interface*. Three filters, in order:

**Reach from the attachment point.** Proximity is measured from an *anchor* —
the binder's N- or C-terminal residue — not from the whole binder. A C-terminal
flank cannot reach surface near the binder's N-terminus. The radius is the
root-mean-square end-to-end distance of a disordered chain,
`Ree = 6.2 · N^0.52` Å (≈20 Å at 10 residues, 36 Å at 30, 68 Å at 100) — the
*typical* span, not the fully extended `3.5 · N`.

**Solvent accessibility.** A flank cannot touch a buried residue. Per-residue
SASA is computed with a Shrake–Rupley implementation in this package (validated
against `mdtraj.shrake_rupley`: correlation 0.99994, mean absolute difference
0.4 Å²) *in the context of the complex*, and residues below 10% relative
accessibility are dropped. This matters a lot: on 1YCR, including buried core
residues flips an acidic probe's epsilon against the patch from −5.2 (attractive)
to +0.7 (repulsive).

Non-protein components occlude too. Nucleic acids, glycans and cofactors are not
returned as chains, but their heavy atoms are kept and used as occluders, so
surface they cover is not offered up as somewhere a flank could bind. On
haemoglobin the haem reduces the accessibility of 25 chain-A residues and pushes
10 of them below the surface threshold.

**Sequence locality.** These are usually *predicted* structures, and predictors
routinely place a sequence-distant part of the target next to the binder. The
real interface is located independently from binder-wide contacts, grouped into
patches (splitting at gaps > `cluster_gap` residues), patches with fewer than
`min_cluster_contacts` contacts are discarded as noise, and a residue is only
kept if it sits within `sequence_window` residues of a surviving patch. Anything
excluded this way is reported, not silently dropped.

The selected residues are concatenated into a *patch sequence*. Concatenating
non-contiguous residues is safe here, tested directly: designing against a
randomly shuffled patch yields a flank that is just as attractive to the
*original* patch, within 0.0–1.4% across three real patches. So the outcome
depends on which residues were selected, not on their order.

(Epsilon itself is not perfectly order-blind — Mpipi weights aliphatic and
charged residues by their sequence neighbours, so an aliphatic-rich probe can
shift by up to ~0.015 per residue on shuffling. That is small next to a typical
design signal of 0.15, and it washes out of the design outcome entirely.)

Gap-filling to make spans contiguous is **not** done, because it drags in buried
core residues and inverts the chemistry.

## How the flank is designed

GOOSE's `SequenceOptimizer` optimises the flank (and only the flank — the binder
is never mutated) against:

- **Attraction to the patch** — FINCHES epsilon, `maximum` constraint. Negative
  epsilon is attractive. Unbounded by default: as attractive as the other
  constraints allow.
- **Disorder, twice** — fraction predicted disordered by metapredict, both for
  the flank alone *and* for the flank fused to the binder. Both are needed.
  Scoring the flank in isolation is not a weak proxy for the real thing, it is
  actively misleading: a poly-aromatic flank scores 1.00 disordered alone and
  **0.00** once fused to a folded binder. Requiring only the in-context number
  is also gameable, so both are enforced.
- **A composition envelope** — every residue capped at 3× its frequency in real
  IDRs (from GOOSE's `IDRProbs`, derived from the disordered regions of eleven
  proteomes), plus group caps on W+F+Y (0.15) and A+I+L+M+V (0.30).

### Why the composition envelope matters

Maximising attraction with no compositional constraint goes straight to
poly-tryptophan. Such a flank looks excellent by both obvious metrics — strongly
attractive, fully "disordered" in isolation — and is useless: it aggregates and
sticks to everything.

Capping aromatics alone does not fix this, it *relocates* it: the optimizer
saturates on whichever residue is still legal, giving poly-aspartate instead.
Bounding all twenty residues is what actually works. Measured on a 30-residue
flank against an MDM2 surface patch (same seed and iteration budget throughout):

| | epsilon/res | vs random seq | self-epsilon | max single residue | complexity |
|---|---|---|---|---|---|
| no guardrails | −0.373 | **−0.272** | **−0.45** (aggregates) | W 0.67 | 0.28 |
| aromatic cap only | −0.305 | +0.018 | +2.03 | **D 0.67** | **0.28** |
| full envelope | −0.172 | −0.020 | +0.20 (soluble) | P 0.23 | 0.65 |

Two separate failures, and they need separate fixes.

*Stickiness and aggregation* are the aromatic problem. The unguarded flank
attracts **random** sequence at −0.272 per residue — nearly as strongly as it
attracts its intended target — and is self-attractive (−0.45). That is not
affinity, it is glue. Capping aromatics fixes both outright (+0.018 against
random, +2.03 self).

*Implausibility* is not fixed by the aromatic cap. `DDWDDDDDDDDDEEDDDWEDDDEE…`
is soluble and selective but is 67% aspartate with a sequence complexity of
0.28, far outside the 0.59–0.77 range of natural IDRs. It is a polyelectrolyte,
not a disordered region. The full envelope is what brings composition back into
the natural range (max single residue 0.23, complexity 0.65).

The envelope costs roughly half the nominal epsilon. Most of what it gives up
was never on-target affinity in the first place.

An explicit anti-stickiness term is available
(`max_decoy_epsilon_per_residue`, used by the `specific` preset) but is off by
default: it is measurably redundant once the envelope is in place, and roughly
2.5× the runtime.

## Presets

| preset | intent |
|---|---|
| `balanced` | default: envelope 3×, W+F+Y ≤ 0.15, unbounded attraction |
| `soluble` | tighter envelope, few aromatics, explicit self-repulsion |
| `specific` | tighter envelope plus a measured anti-stickiness constraint |
| `aggressive` | wider envelope, chases affinity; check the reported numbers |
| `unconstrained` | no guardrails — reproduces the degenerate behaviour, for comparison |

```python
result = build_flanked_binder("complex.pdb", binder_chain="B",
                             target_chain="A", c_flank_length=30,
                             preset="soluble")
```

Any individual knob can be overridden directly
(`max_aromatic_fraction=0.05`, `composition_envelope=2.0`, `max_residues=15`,
`radius=18.0`, …); arguments are routed to the interface or design stage
automatically.

## Reading the output

Every design reports:

- **epsilon per residue** against the patch (negative = attractive), next to a
  background-composition reference so you can see how much is real signal;
- **specificity** — how much more attracted the flank is to the patch than to
  random sequence, plus its raw attraction to random sequence. A flank with
  strong epsilon *and* strong random-sequence attraction is sticky, not potent;
- **cross-reactivity** against basic / acidic / polar / flexible / aliphatic /
  aromatic decoys;
- **self-epsilon** — positive means self-repulsive and soluble, negative flags
  aggregation risk;
- **fraction disordered**, alone and fused to the binder;
- composition: aromatic fraction, FCR, NCPR, κ.

Warnings are raised automatically for self-attraction, low in-context disorder,
attraction no better than background, low specificity, and generic stickiness.

## Honest limitations

- **Everything is a prediction.** Epsilon (FINCHES/Mpipi) is a mean-field
  coarse-grained interaction model, and metapredict is a disorder predictor.
  Neither is a binding free energy. These designs are hypotheses to test, not
  validated binders.
- **Hydrophobic target patches are intrinsically hard.** A hydrophobic patch
  attracts almost everything, so no configuration made a flank against one
  measurably more selective than a random sequence. Charge-complementary patches
  are the opposite: against an acidic patch the design reached −0.65 per residue
  and was more selective than all 1500 random controls. Check the reported
  specificity before trusting a design.
- **Reach is a statistical estimate.** `Ree = 6.2 · N^0.52` is an ensemble
  average for a generic disordered chain. It ignores excluded volume from the
  target and any residual structure.
- **Small target domains saturate.** If the target is small relative to the
  flank's reach, most of its exposed surface is selected and the "patch" becomes
  the whole surface. Use `max_residues` or an explicit `radius` to focus, and
  read the reported span list.
- **Only the first model is read** from multi-model files unless you pass
  `model=`. Alternate locations are resolved by occupancy.
- **Unresolved residues are unresolved.** A structure contains only what was
  modelled. If the binder chain has a genuine backbone break, the returned
  sequence is the resolved residues spliced together, not the real protein — the
  package warns, and you should graft the designed flank onto your own
  full-length sequence. Numbering jumps that are merely conventional (antibody
  Kabat/Chothia numbering) are correctly *not* reported, since the check is
  geometric.
- **Adding a charged tail changes the whole molecule** — solubility, expression,
  pI, and possibly the binder's own fold. Nothing here models that.
- `seed=` gives reproducible designs, but it also switches GOOSE to its
  pure-Python mutation path and disables shuffling; unseeded runs use a faster
  path whose RNG a seed cannot reach.

## Testing

```bash
python -m pytest idr_flanks/tests -q
```

Structure-reading and interface tests need only numpy/scipy. Design and
pipeline tests need GOOSE and are skipped if it is missing. The SASA test
compares against `mdtraj` when available.

## Copyright

Copyright (c) 2026, Ryan Emenecker WUSM
