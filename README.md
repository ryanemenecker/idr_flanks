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
flank cannot reach surface near the binder's N-terminus.

The radius is the typical distance at which a flank *residue* sits from the
anchor, not the end-to-end span of the whole flank. Only the last residue
reaches `Ree`; averaging the tethered-chain relation over all residues gives
`Ree / (1 + ν)`, which for `Ree = 6.2 · N^0.52` is ≈13 Å at 10 residues, 24 Å at
30, 45 Å at 100. The factor is derived, not tuned.

Using `Ree` itself as an equal-weight cutoff was measurably wrong: on 1YCR it
selected the entire 85-residue target domain from *either* terminus, so both
termini produced identical patches (Jaccard 1.00) and the terminus choice — the
whole point of anchoring — did nothing. With the residue-averaged radius the two
patches differ (Jaccard 0.42, 34 vs 47 residues), and the more focused patch
gives a *stronger* per-residue signal (−0.156 vs −0.131).

The provenance is worth stating precisely, because it is easy to overclaim: the
exponent 0.52 and the prefactor come from the empirical IDR scaling
`Rg ≈ 2.54 · N^0.522` converted with `Ree = √6 · Rg`, which is the *ideal-chain*
relation. So 0.52 is an empirical fit close to the ideal-chain 0.5, below the
self-avoiding-walk 0.588, and it varies with sequence charge and hydrophobicity.
Treat the radius as an order-of-magnitude guide.

**Solvent accessibility.** A flank cannot touch a buried residue. Per-residue
SASA is computed with a Shrake–Rupley implementation in this package (validated
against `mdtraj.shrake_rupley`: at the default 480 sample points, correlation
0.99986 and mean absolute difference 0.56 Å²; 0.99994 and 0.40 Å² at 960) *in the context of the complex*, and residues below 10% relative
accessibility are dropped. This matters a lot: on 1YCR, including buried core
residues flips an acidic probe's epsilon against the patch from −5.2 (attractive)
to +0.7 (repulsive).

Non-protein components occlude too. Nucleic acids, glycans and cofactors are not
returned as chains, but their heavy atoms are kept and used as occluders, so
surface they cover is not offered up as somewhere a flank could bind. On
haemoglobin the haem reduces the accessibility of 25 chain-A residues and pushes
10 of them below the surface threshold.

Sequence-distant target regions, however, are **excluded** from the occluder set
by default. They are the same prediction artefact the sequence-locality filter
exists to remove, and letting them occlude reintroduces it: a spuriously draped
loop buries the very surface the flank could have used, and the region is
discarded for a reason that is not real. On a constructed case where a distant
region is draped over the reachable surface, trusting the drape collapses seven
usable residues to one; ignoring it keeps all seven. Pass
`trust_distal_occlusion=True` for an experimental structure, where such packing
is genuine. On 1YCR, where nothing is sequence-distant, the setting changes
nothing.

**Your own knowledge of the system.** The automatic filters below remove
*small* spurious contact patches. They cannot help when a predictor folds a
whole terminus back onto the real binding site: that produces a large,
self-consistent patch indistinguishable from the genuine one. Only you know it
is wrong, so say so:

```python
result = build_flanked_binder("complex.pdb", binder_chain="B",
                             target_chain="A", c_flank_length=30,
                             exclude_target_residues=[1, 100])   # or "1-100"
```

`include_target_residues` is the complement — consider only those residues.
Both take author numbering, and accept `"1-100"`, `"1-100,250-300"`,
`(1, 100)`, `[(1, 100), (250, 300)]`, `[5, 12, 88]`, or `range(1, 101)`. Note
that a bare pair of integers means a *range*: `[1, 100]` is the first hundred
residues, not residues 1 and 100. Whatever is parsed is echoed back in the
region notes.

This applies **before** the interface is located, not just to the final
selection — otherwise the mispredicted patch would still define an accepted
sequence window and let its neighbours through. Excluded residues also stop
occluding, on the same reasoning as `trust_distal_occlusion`.

It matters more than dilution: on a reproduction where a basic N-terminal
region was mispredicted onto an acidic C-terminal binding site, the unrestricted
patch was 67% wrong region (32 K against 16 E) and the resulting flank came out
acidic and **repelled** from the true site (+0.832 per residue). Excluding
`[1, 100]` flipped it to basic and strongly attracted (−0.757). A sign error,
silent apart from a "target presents 2 distinct interface patches" note.

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
- **Reach weighting** — the objective is a weighted average over distance
  shells, not a flat average over the whole patch (see below).
- **A composition envelope** — every residue capped at 3× its frequency in real
  IDRs (from GOOSE's `IDRProbs`, derived from the disordered regions of eleven
  proteomes), plus group caps on W+F+Y and A+I+L+M+V.
- **Not competing with the target for the binder** — the flank must prefer the
  target patch to the binder's *own* target-binding surface by a margin.

### Reach weighting, and what it did and did not fix

Roughly half the selected residues typically sit beyond 15 Å of the anchor,
where the tethered-chain monomer density is under a tenth of its contact value —
yet a flat patch let them contribute in proportion to their count. The residue
weight is now the tethered-chain monomer density,
`w(d) = Σᵢ (3/2πRᵢ²)^{3/2} exp(−3d²/2Rᵢ²)` with `Rᵢ = 6.2·i^0.52`, normalised to
1 at contact. It is derived from the same polymer relation as the reach radius,
not fitted. The linear taper it replaces over-weighted distant surface by a
measured 2.4× at 10 Å and 2.9× at 15 Å.

The objective is then a weighted average over distance shells rather than one
epsilon call against the concatenated patch — weights can only be applied
*between* epsilon calls, since FINCHES takes a sequence. On 1YCR this improves
attraction to the innermost shell by about 20% (−0.086 → −0.104 per residue) for
a ~3% cost in whole-patch epsilon. Disable with `reach_weighted=False`.

**It did not make the designs target-specific, and that is the more important
result.** A cross-design matrix over six complexes (1YCR C and N, 1DFJ, 3HHR,
1BRS, 1FCC), scoring every flank against every patch with the same
reach-weighted metric:

| objective | diagonal wins | mean rank of the correct flank | gap to best rival |
|---|---|---|---|
| flat patch | 3 / 6 | 1.83 / 6 | +0.004 (tied) |
| reach-weighted | 3 / 6 | 1.50 / 6 | −0.004 (marginal) |

So the geometry was *not* the cause of interchangeability. The limit is the
interaction model: FINCHES epsilon is very nearly a function of composition alone
(shuffling a patch moves it ≤1.3%), so once the patch's net chemistry is fixed
the optimal composition is determined and any sequence with that composition
scores the same. Two designs against different targets came out as distinct
sequences that scored identically to three decimals.

What the method *does* do is pick the right chemical class: mean epsilon on the
intended patch is −0.30 versus −0.09 for flanks designed against a different
class of patch, a 3.3× advantage. Read the package as selecting complementary
chemistry for a target, not a target-unique sequence.

### Why the anti-competition guard matters

If the flank likes the surface of the binder that grips the target, it competes
with the target for it, and a "higher affinity" design ends up with *lower* net
affinity. This is not hypothetical. Given an acidic target patch and an acidic
binder interface, the unguarded design came out at −0.599 per residue against
the target and **−0.624 against the binder's own interface** — it preferred the
binder.

The constraint is relative ("prefer the target by at least 0.05 per residue")
rather than an absolute ban on binder attraction, because when the two surfaces
share chemistry an absolute ban destroys the target attraction along with it
(−0.587 → +0.016 in that case).

Before applying it, the package checks whether the two surfaces can be told
apart at all, by scanning single-chemistry probes for the best achievable
preference. On 1YCR there is +0.716 per residue of headroom and the guard costs
nothing — the same sequence comes out either way. On two chemically identical
surfaces the headroom is +0.005, the constraint is unsatisfiable, and you are
told so plainly instead of being handed a flank that competes. When the
requested margin merely *exceeds* the headroom, the constraint is reduced to
half of what is attainable rather than dropped — dropping it silently was
measurably worse than keeping a weaker version.

The 0.05 default is calibrated on six real complexes (1YCR, 1DFJ, 3HHR, 1BRS,
1FCC, 2P1M):

| margin | competing designs | mean target-affinity cost |
|---|---|---|
| none | **2 / 6** | — |
| 0.02 | 0 / 6 | +0.013 |
| 0.05 | 0 / 6 | +0.019 |

Two of six real systems produce a competing flank with no guard — barstar
against barnase at −0.168 preference, and protein G against Fc at −0.028. This
is not a corner case. A margin of 0.05 eliminates it across all six for about
0.02 per residue of target affinity.

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

### Why not just require high complexity?

Sequence complexity looks like it should subsume all of this — a high-entropy
sequence cannot be poly-anything. It does not work, measured:

| constraint | achieved complexity | WFY | self-epsilon |
|---|---|---|---|
| none | 0.33 | 0.30 | +1.15 |
| `Complexity ≥ 0.70` | 0.55 | **0.70** | **−0.41** |
| `Complexity ≥ 0.90` | 0.50 | 0.63 | −0.02 |
| composition envelope | **0.66** | 0.10 | +0.19 |

Two problems. It is a *soft* objective, so GOOSE trades it against epsilon and
never reaches the target — asking for 0.90 delivered 0.50. And it is blind to
residue *identity*: `WQWFWPYYFWWWYDWWDWEYNFWDDFWWWD` uses eight residue types,
so its entropy is respectable, and it is 70% aromatic and self-attractive.
Requiring complexity actually made the flank *stickier* than requiring nothing.

The envelope reaches complexity 0.66 without being asked, because bounding every
residue's abundance forces diversity as a side effect. Adding `Complexity` on
top of the envelope changes nothing measurable, so it is not used — complexity
is reported as a diagnostic instead.

An explicit anti-stickiness term is available
(`max_decoy_epsilon_per_residue`, used by the `specific` preset) but is off by
default: it is measurably redundant once the envelope is in place, and roughly
2.5× the runtime.

## Optional linker

A designed flank is chemically loaded by construction, and butting it straight
against the binder risks perturbing how the binder folds. `linker_length=N`
inserts an `N`-residue GS linker between each flank and the binder:

```python
result = build_flanked_binder("complex.pdb", binder_chain="B",
                             target_chain="A", c_flank_length=25,
                             linker_length=6)
# ETFSDLWKLLPEN(GSGSGS)[DPNPPEQHQDEWEYNENDNQPPEDP]
```

The linker is part of the tether, so it is included when working out how far the
flank can reach, and it forms part of the sequence context used for the
in-context disorder check. Supply `linker_sequence` for something other than GS.

## Presets

| preset | intent |
|---|---|
| `balanced` | default: envelope 3×, W+F+Y ≤ 0.10, unbounded attraction |
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
- **selectivity** — how much more attracted the flank is to the patch than to
  random sequence, plus its raw attraction to random sequence. Read the second
  number. The binder already supplies target specificity; the flank's job is
  added avidity, so a modest selectivity margin is expected and is not flagged.
  What does matter is the flank not attracting *everything*: strong epsilon
  together with strong random-sequence attraction is stickiness, not potency;
- **attraction to the binder's own interface** — positive means repelled, which
  is what you want. Negative means it competes with the target;
- **cross-reactivity** against basic / acidic / polar / flexible / aliphatic /
  aromatic decoys;
- **self-epsilon** — positive means self-repulsive and soluble, negative flags
  aggregation risk;
- **fraction disordered**, alone and fused to the binder;
- composition: aromatic fraction, FCR, NCPR, κ.

Warnings are raised automatically for self-attraction, low in-context disorder,
net repulsion from the patch, attraction no better than background (measured
against 24 background draws, not one, since a single draw has a standard
deviation of about a quarter of a typical design signal), generic stickiness, a
patch too small to design against, unresolved residues in either chain, and —
most importantly — the flank preferring the binder's own interface to the
target.

`FlankedBinder.warnings` collects all three stages, so `--quiet` and
programmatic callers see the interface-stage notes too, not just the design
ones.

## Honest limitations

- **Everything is a prediction.** Epsilon (FINCHES/Mpipi) is a mean-field
  coarse-grained interaction model, and metapredict is a disorder predictor.
  Neither is a binding free energy. These designs are hypotheses to test, not
  validated binders.
- **Designs are chemistry-class specific, not target specific.** See the
  cross-design matrix above. Flanks are largely interchangeable among targets
  presenting the same net chemistry, and reach weighting did not change that.
  Do not present a design as bespoke to one target without testing it against a
  same-class control.
- **Hydrophobic target patches are intrinsically hard.** A hydrophobic patch
  attracts almost everything, so no configuration made a flank against one
  measurably more selective than a random sequence. Charge-complementary patches
  are the opposite: against an acidic patch the design reached −0.65 per residue
  and was more selective than all 1500 random controls. Check the reported
  specificity before trusting a design.
- **Reach is a statistical estimate.** The radius is an ensemble average for a
  generic disordered chain. It ignores excluded volume from the target, the
  direction the binder terminus points (a flank cannot reach backwards through
  the binder), the flank's own excluded volume, and any residual structure. It
  is also an equal-weight cutoff: every residue inside the radius counts the
  same, which is why proximity weighting is offered as an option.
- **Small target domains saturate.** If the target is small relative to the
  flank's reach, most of its exposed surface is selected and the "patch" becomes
  the whole surface. Use `max_residues` or an explicit `radius` to focus, and
  read the reported span list.
- **Only the first model is read** from multi-model files unless you pass
  `model=`. Alternate locations are resolved by occupancy.
- **Unresolved residues are unresolved.** A structure contains only what was
  modelled, and this matters more than it looks. The package reads SEQRES (or
  `_entity_poly` in mmCIF) so it knows the deposited sequence, and warns about
  two distinct problems:
  - **Internal breaks**, detected geometrically from a missing peptide bond, so
    conventional numbering jumps (antibody Kabat/Chothia) are correctly *not*
    reported.
  - **Truncated termini**, which a break check structurally cannot see, because
    it can only compare residues that are present. This is the common case and
    the dangerous one: the terminus is exactly where the flank attaches. Both
    chains of the bundled 1YCR example are truncated — MDM2 by 8 and 16
    residues, and the p53 binder by 2 at its N-terminus — so an N-terminal flank
    there would be grafted two residues from where you expect, and the reach
    would be measured from the wrong atom. The warning says so, and
    `binder_full_sequence` gives you the sequence to graft onto instead.
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
