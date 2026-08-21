User Guide
==========

This guide explains what ``idr_flanks`` computes, why each step is designed the
way it is, and how to read the output critically.

The idea
--------

A binder and its target form an interface. Immediately around that interface is
target surface the binder does not touch. If a disordered tail is attached to
the binder and that tail is chemically complementary to the nearby surface, it
can add favourable interaction without perturbing the folded interface. This
package picks the surface a tail could reach and designs a sequence for it.

Stage 1: reading the structure
------------------------------

``idr_flanks.io`` reads legacy PDB and mmCIF with no dependency beyond numpy.

* Author identifiers are used throughout. In mmCIF that means ``auth_asym_id``
  and ``auth_seq_id``, falling back to ``label_*`` when a predictor omits them.
  Chain ``"A"`` and residue ``54`` mean what a viewer or a paper says they mean.
* A residue's identity is ``(chain, seq_id, ins_code)``, so insertion codes
  (antibody numbering) give distinct residues.
* Modified residues are mapped to their parent amino acid. Selenomethionine
  matters most: it is deposited as ``HETATM``, so ignoring HETATM records would
  silently delete every SeMet position.
* Alternate locations are resolved by occupancy, ties broken by altloc letter so
  results do not depend on file ordering.
* Only the first model is read unless ``model=`` is given.
* Residues written out of numbering order (some writers emit all HETATM records
  after all ATOM records) are put back in order, and a warning is recorded on
  ``Structure.warnings``.

Elements come from columns 77–78 when present. When absent, they are inferred
from the atom-name columns rather than the stripped name, because the name alone
is ambiguous: ``" CA "`` is a backbone alpha carbon while ``"CA  "`` is a
calcium ion.

Stage 2: choosing the target region
-----------------------------------

Three filters, applied in order.

**Reach from the attachment point.** The flank is tethered at one terminus, so
proximity is measured from that *anchor* residue, not from the binder as a
whole. The radius is the typical distance at which a flank *residue* sits from
the anchor, obtained by averaging the tethered-chain relation over all residues:

.. math::

    R_{ee} = 6.2 \cdot N^{0.52}\ \text{\AA}, \qquad
    r_{\text{reach}} = \frac{R_{ee}}{1 + \nu}

giving about 13 Å for 10 residues, 19 Å for 20, 24 Å for 30 and 45 Å for 100.
The factor is derived rather than tuned.

Using :math:`R_{ee}` directly as an equal-weight cutoff was measurably wrong:
only the flank's last residue reaches that far, and on 1YCR it selected the
whole 85-residue target domain from either terminus, giving identical patches
(Jaccard 1.00) and nullifying the terminus choice. The residue-averaged radius
separates them (Jaccard 0.42) and yields a stronger per-residue signal.

Provenance, stated precisely: the exponent and prefactor come from the empirical
IDR scaling :math:`R_g \approx 2.54 \cdot N^{0.522}` converted with
:math:`R_{ee} = \sqrt{6} R_g`, the *ideal-chain* relation. 0.52 is therefore an
empirical fit sitting between the ideal-chain 0.5 and the self-avoiding-walk
0.588, and it varies with sequence composition. Override with ``radius=`` or
scale with ``radius_scale=``.

**Solvent accessibility.** A tail cannot touch a buried residue. Per-residue
SASA is computed with a Shrake–Rupley implementation in
:mod:`idr_flanks.sasa`, in the context of the whole complex, and residues below
``surface_threshold`` (default 0.10 relative to the residue type's maximum) are
dropped.

This filter is not cosmetic. On 1YCR, taking the reachable residues *without*
it and testing an acidic probe sequence against them gives epsilon −5.2
(attractive); filling the same span in contiguously, which drags in the
hydrophobic core, gives +0.7 (repulsive). The chemistry of a protein's core is
not the chemistry of its surface.

Glycine is a special case: it has no sidechain, so its sidechain reference area
is zero. FINCHES treats every glycine as accessible; here glycine is normalised
by its backbone reference instead, which keeps the value finite and lets a
genuinely occluded glycine be recognised as buried.

Non-protein components occlude as well. Nucleic acids, glycans and cofactors are
not returned as chains -- they are not polypeptide -- but their heavy atoms are
retained via :meth:`~idr_flanks.io.Structure.heteroatoms` and passed to the
accessibility calculation. Without that, DNA-covered or glycan-covered target
surface would read as exposed and be offered up as a binding site. Solvent and
simple ions are excluded.

Accessibility is computed only for residues that are within reach, with the rest
of the chain kept as occluding context. That is bit-identical to computing the
whole chain and about six times faster on a 1000-residue target, where the
Shrake-Rupley pass would otherwise dominate the call.

Sequence-distant target regions are also excluded from the *occluder* set by
default, not just from selection. Letting them occlude would reintroduce the
artefact they were filtered out for: a spuriously draped loop buries the surface
the flank could have used, and the region is discarded for a reason that is not
real. On a constructed case, trusting such a drape collapses seven usable
residues to one, while ignoring it keeps all seven. Pass
``trust_distal_occlusion=True`` for an experimental structure, where the packing
is genuine; on 1YCR, which has nothing sequence-distant, it changes nothing.

**Sequence locality.** Predicted structures place sequence-distant parts of the
target next to the binder more often than real ones do, and those contacts are
usually artifacts. So:

1. the genuine interface is located from binder-wide heavy-atom contacts within
   ``contact_cutoff`` (default 5.0 Å);
2. contacting residues are grouped into patches, splitting wherever the sequence
   gap exceeds ``cluster_gap`` (default 15);
3. patches with fewer than ``min_cluster_contacts`` contacts (default 3) are
   discarded as noise;
4. a reachable residue is kept only if it lies within ``sequence_window``
   (default 25) residues of a surviving patch.

Genuinely bipartite epitopes survive, because every sufficiently large patch is
kept and reported. Everything discarded is listed on the result, so the filter
is auditable rather than silent.

The patch sequence
^^^^^^^^^^^^^^^^^^

Selected residues are concatenated. That is safe, and it was tested the direct
way rather than argued: design a flank against a randomly shuffled patch, then
score it against the *original* patch. Across three real patches the resulting
attraction varied by 0.0-1.4% of the signal, so what matters is which residues
were selected, not their order.

Epsilon itself is not strictly order-blind. Mpipi weights aliphatic and charged
residues according to their sequence neighbours, so shuffling a patch can move
epsilon by up to about 0.015 per residue for an aliphatic-rich probe -- large in
relative terms when epsilon is already near zero, but small against a typical
design signal of 0.15, and invisible in the final design.

Gaps are deliberately *not* filled in to make spans contiguous, for the reason
given above.

``ProximalRegion.weighted_patch_sequence(n)`` repeats residues near the anchor
up to ``n`` times. This works as an exact weighting because FINCHES epsilon
behaves as a mean over the target residues: the *ratio* of residue types drives
the value and the absolute count barely does (a 2:1 mixture scores −12.70 at 15
residues and −12.84 at 30).

An optional linker
^^^^^^^^^^^^^^^^^^

``linker_length=N`` inserts an ``N``-residue GS linker between each flank and
the binder. A designed flank is chemically loaded by construction, and placing
it hard against the binder risks perturbing how the binder folds. The linker
counts as part of the tether, so it is included when computing reach, and it
forms part of the sequence context for the in-context disorder check. Pass
``linker_sequence`` for something other than GS.

Stage 3: designing the flank
----------------------------

GOOSE's ``SequenceOptimizer`` mutates the flank only; the binder is never part
of the optimised string. The objective:

Attraction
^^^^^^^^^^

``MeanEpsilonWithTarget`` with a ``maximum`` constraint. Negative epsilon is
attractive. Because epsilon scales linearly with the length of the sequence
being designed, targets are always expressed per residue and multiplied by the
flank length internally. By default the target is unreachable, which makes the
term "as attractive as the other constraints allow".

Disorder, measured twice
^^^^^^^^^^^^^^^^^^^^^^^^

Both the isolated flank and the flank fused to the binder must be predicted
disordered.

Scoring the flank alone is not merely a weak proxy — for aromatic-rich
sequences it is anti-correlated with the truth. The flank
``WWYDWWWWWWWEFWWYDWEDWWWEYDDDEW`` scores fraction-disorder 1.00 in isolation
and 0.00 fused to a folded binder. Requiring only the in-context number is also
gameable, since the optimizer will find a segment that scores well only because
of its neighbours. Requiring both closes each loophole.

In-context disorder is evaluated by a ``CustomProperty`` that rebuilds the
construct before predicting, with batching enabled; GOOSE has no native support
for fixed sequence context.

The composition envelope
^^^^^^^^^^^^^^^^^^^^^^^^

Every residue is capped at ``composition_envelope`` (default 3.0) times its
frequency in real disordered regions, taken from GOOSE's ``IDRProbs`` (the IDRs
of eleven proteomes as called by metapredict V3). Group caps are applied on top:
W+F+Y at 0.10 and A+I+L+M+V at 0.25.

This exists because an unbounded attraction objective saturates on whichever
residue is most attractive and still legal. With no constraint that is
tryptophan; cap aromatics only and it becomes aspartate. See the README for the
measured comparison. Two distinct failure modes are involved: aromatics cause
stickiness and aggregation, and any single-residue saturation causes
implausibility. Bounding all twenty residues addresses both.

GOOSE converts a fraction bound into a residue count with ``ceil``, so the
requested cap is rounded down to an exact count first. Without that, a 0.15 cap
at length 50 admits 8 aromatics rather than 7.

Not competing with the binder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The flank must prefer the target patch to the binder's *own* target-binding
surface, by ``min_target_preference`` per residue (default 0.05).

This is the guard that most directly protects the point of the package. A flank
attracted to the surface that grips the target competes with the target for it,
so a design with better nominal affinity can have worse net affinity. Measured
on a constructed case with an acidic target patch and an acidic binder
interface, the unguarded design scored -0.599 per residue against the target and
-0.624 against the binder -- it preferred the binder.

The constraint is a *relative* margin, not an absolute ceiling on binder
attraction, because when the two surfaces share chemistry a ceiling destroys the
target attraction too (-0.587 to +0.016 in that case).

:func:`~idr_flanks.design.target_discriminability` checks feasibility first, by
scanning single-chemistry probes for the largest attainable preference. With
+0.716 per residue of headroom on 1YCR the guard is free -- the same sequence
comes out with and without it. With +0.005 on two chemically identical surfaces
the constraint is unsatisfiable, so it is skipped and reported rather than
silently degrading the design. When the requested margin only *exceeds* the
headroom it is reduced to half of what is attainable rather than dropped;
dropping it was measurably worse, and let a design compete on protein G / Fc.

The 0.05 default is calibrated across six real complexes (1YCR, 1DFJ, 3HHR,
1BRS, 1FCC, 2P1M). With no guard, two of the six produce a flank that prefers
the binder -- barstar against barnase at -0.168 per residue, protein G against
Fc at -0.028. A margin of 0.02 or 0.05 eliminates competition in all six, at a
mean target-affinity cost of 0.013 and 0.019 per residue respectively. Margins
of 0.10 and 0.20 were no better, because they exceed the attainable headroom on
the hardest system.

Why not just require high complexity?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because it is soft and identity-blind, both measured. GOOSE trades a
``Complexity`` target against epsilon and never reaches it (asking for 0.90
yielded 0.50), and entropy does not care *which* residues supply the diversity:
``WQWFWPYYFWWWYDWWDWEYNFWDDFWWWD`` has eight residue types and is 70% aromatic
with self-epsilon -0.41. Requiring complexity produced a stickier flank than
requiring nothing at all. The composition envelope reaches complexity 0.66
without being asked, and adding ``Complexity`` on top of it changes nothing, so
complexity is reported as a diagnostic rather than optimised.

Optional terms
^^^^^^^^^^^^^^

``min_self_epsilon_per_residue`` requires the flank to be self-repulsive.
``max_decoy_epsilon_per_residue`` requires it to be non-attractive to a panel of
background-composition sequences. Both are off by default, because the envelope
already delivers self-epsilon around +0.2 and random-sequence attraction around
−0.02. The specificity term in particular is redundant once the envelope is in
place, and roughly 2.5× the runtime.

Reading the diagnostics
-----------------------

``epsilon per residue``
    Attraction to the patch; negative is attractive. Compare it against the
    reported ``neutral reference``, the same quantity for a
    background-composition sequence.

``selectivity``
    How much more attracted the flank is to the patch than to random sequence,
    with the raw random-sequence attraction alongside it. Read the second
    number. The binder already provides target specificity -- the flank exists
    to add avidity -- so a modest margin here is expected and is deliberately
    not warned about. What does matter is the flank not attracting *everything*:
    an unguarded design can show epsilon -0.37 while attracting random sequence
    at -0.27, and almost all of that apparent affinity is non-specific.

``vs binder interface``
    Epsilon against the binder's own target-binding surface. Positive means
    repelled, which is what you want. Negative means the flank competes with the
    target and may reduce net affinity.

``cross-reactivity``
    Epsilon against basic, acidic, polar, flexible, aliphatic and aromatic
    decoys. An acidic flank should show strong attraction to basic and
    repulsion from acidic. If a flank attracts everything, distrust it.

``self-epsilon``
    Positive means self-repulsive and therefore soluble. Negative flags
    aggregation risk.

``fraction disordered``
    Reported alone and fused to the binder. Trust the fused number.

Warnings are emitted automatically for self-attraction, in-context disorder
below 0.8, attraction no better than background, generic stickiness, a patch
too small to design against, and the flank preferring the binder's own
interface. Modest selectivity against random sequence is deliberately *not*
warned about; see above for why.

When to distrust a design
-------------------------

* **Hydrophobic target patches.** A hydrophobic patch attracts nearly
  everything, and no configuration tested produced a flank against one that was
  measurably more selective than a random sequence. Charge-complementary patches
  behave far better: against an acidic patch a design reached −0.65 per residue
  and beat all 1500 random controls on selectivity.
* **Saturated patches.** If the target is small relative to the flank's reach,
  most of its exposed surface is selected and "the patch" becomes "the whole
  surface", diluting the design signal. Read the reported span list; use
  ``max_residues`` or an explicit ``radius`` to focus.
* **A flank that prefers the binder.** If the reported ``vs binder interface``
  number is below the target epsilon, the flank competes. Sometimes the two
  surfaces simply cannot be told apart, in which case the honest answer is that
  a flank on that terminus will not help.
* **Anything the warnings flag.**

Remember what the models are. FINCHES epsilon is a mean-field coarse-grained
interaction scale and metapredict is a disorder predictor; neither is a binding
free energy. A designed flank is a hypothesis to test experimentally.

Reproducibility
---------------

``seed=`` makes designs reproducible, and ``idr_flanks`` restores the caller's
``random`` and ``numpy`` RNG state afterwards so designing does not perturb
your own random stream. Setting a seed also switches GOOSE to its pure-Python
mutation path and disables shuffling, because its default fast path uses a C
RNG seeded at import that a Python seed cannot reach. Unseeded runs are faster
but not reproducible.
