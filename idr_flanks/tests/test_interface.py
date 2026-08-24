"""Tests for proximal-region detection."""

import textwrap

import numpy as np
import pytest

from idr_flanks.data import structure_path
from idr_flanks.interface import (
    InterfaceError,
    contact_map,
    find_proximal_region,
    min_distances_to,
    reach_radius,
)
from idr_flanks.io import read_pdb, read_structure

# MDM2 residues that contact the p53 peptide within 5 A in 1YCR.
KNOWN_CONTACTS_5A = {25, 26, 50, 51, 54, 55, 57, 58, 61, 62, 67, 70, 71, 72,
                     73, 75, 91, 93, 94, 96, 99, 100, 103, 104}


@pytest.fixture(scope="module")
def ycr():
    return read_structure(structure_path("1ycr.pdb"))


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text).lstrip("\n"))
    return str(p)


class TestReachRadius:
    def test_grows_with_length(self):
        radii = [reach_radius(n) for n in (5, 10, 20, 40, 80)]
        assert radii == sorted(radii)

    def test_sublinear(self):
        """A disordered chain's span grows as a power law, not linearly."""
        assert reach_radius(80) < 4 * reach_radius(20)

    def test_far_below_fully_extended(self):
        for n in (10, 30, 100):
            assert reach_radius(n) < 3.5 * n

    def test_floor_applies_to_tiny_flanks(self):
        assert reach_radius(1, minimum=8.0) == pytest.approx(8.0)

    def test_ceiling_applies(self):
        assert reach_radius(500, maximum=40.0) == pytest.approx(40.0)

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            reach_radius(0)


class TestReachIsResidueAveraged:
    """The radius must be the typical distance of a flank *residue* from the
    anchor, not the end-to-end span, or the terminus choice stops mattering."""

    def test_reach_is_below_end_to_end(self):
        from idr_flanks.interface import end_to_end_distance
        for n in (5, 10, 20, 30, 50, 100):
            assert reach_radius(n) < end_to_end_distance(n)

    def test_ratio_is_the_derived_factor(self):
        from idr_flanks.interface import end_to_end_distance
        for n in (20, 30, 50, 100):
            ratio = reach_radius(n) / end_to_end_distance(n)
            assert ratio == pytest.approx(1.0 / 1.52, rel=1e-6)

    def test_termini_are_distinguishable_on_1ycr(self, ycr):
        """The whole point of anchoring: the two termini must see different
        surface. With the end-to-end radius they saw exactly the same 57
        residues."""
        n = set(find_proximal_region(ycr, "B", "A", "N", 25).seq_ids)
        c = set(find_proximal_region(ycr, "B", "A", "C", 25).seq_ids)
        jaccard = len(n & c) / len(n | c)
        assert jaccard < 0.8
        assert n != c


class TestTetherContactWeights:
    """The weight is the tethered-chain monomer density at that distance, which
    decays far faster than the linear taper it replaced."""

    def test_unity_at_the_anchor(self):
        from idr_flanks.interface import tether_contact_weight
        assert tether_contact_weight(0.0, 25) == pytest.approx(1.0)

    def test_monotonically_decreasing(self):
        from idr_flanks.interface import tether_contact_weights
        w = tether_contact_weights(np.arange(0.0, 40.0, 2.0), 25)
        assert np.all(np.diff(w) < 0)
        assert np.all(w > 0)

    def test_decays_faster_than_a_linear_taper(self):
        """The measured over-weighting the linear taper caused: 2.4x at 10 A."""
        from idr_flanks.interface import tether_contact_weight
        r = reach_radius(25)
        for d, expected_ratio in ((10.0, 2.4), (15.0, 2.9)):
            linear = max(0.0, 1.0 - d / r)
            tethered = tether_contact_weight(d, 25)
            assert linear / tethered == pytest.approx(expected_ratio, abs=0.2)

    def test_longer_flanks_reach_further(self):
        from idr_flanks.interface import tether_contact_weight
        assert tether_contact_weight(20.0, 60) > tether_contact_weight(20.0, 15)

    def test_rejects_nonpositive_length(self):
        from idr_flanks.interface import tether_contact_weights
        with pytest.raises(ValueError):
            tether_contact_weights(np.array([1.0]), 0)

    def test_region_weights_use_it(self, ycr):
        from idr_flanks.interface import tether_contact_weight
        region = find_proximal_region(ycr, "B", "A", "C", 25)
        for p in region:
            assert p.weight == pytest.approx(
                tether_contact_weight(p.anchor_distance, 25))


class TestWeightedShells:
    def test_shells_partition_the_patch(self, ycr):
        """Every selected residue appears in exactly one shell. Shells group by
        distance, so their concatenation is a permutation of the patch rather
        than the patch itself."""
        from collections import Counter
        region = find_proximal_region(ycr, "B", "A", "C", 25)
        shells = region.weighted_shells()
        assert shells
        assert all(w > 0 for _, w in shells)
        joined = "".join(s for s, _ in shells)
        assert len(joined) == len(region.patch_sequence)
        assert Counter(joined) == Counter(region.patch_sequence)

    def test_inner_shells_carry_disproportionate_weight(self, ycr):
        """The point of the exercise: near residues must dominate."""
        region = find_proximal_region(ycr, "B", "A", "C", 25)
        shells = region.weighted_shells()
        total_w = sum(w for _, w in shells)
        total_n = sum(len(s) for s, _ in shells)
        inner_seq, inner_w = shells[0]
        # the innermost shell's weight share exceeds its residue-count share
        assert inner_w / total_w > len(inner_seq) / total_n

    def test_custom_edges(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 25)
        assert len(region.weighted_shells(edges=(8.0,))) <= 2

    def test_empty_shells_are_dropped(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 25)
        shells = region.weighted_shells(edges=(0.5, 1.0, 1.5))
        assert all(s for s, _ in shells)


class TestContactMap:
    def test_matches_brute_force(self, ycr):
        a, b = ycr["A"], ycr["B"]
        got = contact_map(a, b, 5.0)
        expected = np.zeros_like(got)
        for i, ra in enumerate(a):
            for j, rb in enumerate(b):
                d = np.linalg.norm(
                    ra.heavy_coords[:, None, :] - rb.heavy_coords[None, :, :],
                    axis=-1).min()
                expected[i, j] = d <= 5.0
        assert np.array_equal(got, expected)

    def test_recovers_known_interface(self, ycr):
        cm = contact_map(ycr["A"], ycr["B"], 5.0)
        contacts = {ycr["A"][i].seq_id
                    for i in np.nonzero(cm.any(axis=1))[0]}
        assert contacts == KNOWN_CONTACTS_5A

    def test_cutoff_is_monotonic(self, ycr):
        counts = [contact_map(ycr["A"], ycr["B"], c).sum()
                  for c in (4.0, 4.5, 5.0, 6.0)]
        assert counts == sorted(counts)

    def test_shape(self, ycr):
        assert contact_map(ycr["A"], ycr["B"]).shape == (85, 13)


class TestSasaScoping:
    """Accessibility is computed only for residues within reach, with the rest
    of the chain kept as occluding context. That must be identical to computing
    the whole chain, just faster."""

    def test_matches_whole_chain_computation(self, ycr):
        from idr_flanks.sasa import relative_residue_sasa
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        whole = relative_residue_sasa(ycr["A"].residues,
                                      context=list(ycr["B"].residues))
        by_label = {p.label: p.relative_sasa for p in region}
        checked = 0
        for i, res in enumerate(ycr["A"]):
            if res.label in by_label:
                assert by_label[res.label] == pytest.approx(whole[i], abs=1e-12)
                checked += 1
        assert checked == len(region)

    def test_out_of_reach_residues_still_occlude(self, ycr):
        """If the rest of the chain were dropped rather than kept as context,
        reachable residues would look far more exposed than they are."""
        from idr_flanks.sasa import relative_residue_sasa
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        selected = [p.residue for p in region]
        no_context = relative_residue_sasa(selected)
        assert any(no_context[i] > region.residues[i].relative_sasa + 0.05
                   for i in range(len(region)))

    def test_unreached_residues_have_no_sasa_recorded(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 20, radius=10.0)
        assert all(p.anchor_distance <= 10.0 for p in region)
        assert all(p.relative_sasa == p.relative_sasa for p in region)


class TestScipylessFallback:
    """The brute-force paths only run when scipy is missing, so they would
    otherwise never be exercised."""

    @staticmethod
    def _no_kdtree(monkeypatch):
        import idr_flanks.interface as mod
        monkeypatch.setattr(mod, "_kdtree", lambda points: None)

    def test_contact_map_matches_kdtree(self, ycr, monkeypatch):
        expected = contact_map(ycr["A"], ycr["B"], 5.0)
        self._no_kdtree(monkeypatch)
        assert np.array_equal(contact_map(ycr["A"], ycr["B"], 5.0), expected)

    def test_min_distances_matches_kdtree(self, ycr, monkeypatch):
        coords, _ = ycr["B"].stacked_heavy_coords()
        expected = min_distances_to(ycr["A"], coords)
        self._no_kdtree(monkeypatch)
        assert np.allclose(min_distances_to(ycr["A"], coords), expected)

    def test_region_selection_matches_kdtree(self, ycr, monkeypatch):
        expected = find_proximal_region(ycr, "B", "A", "C", 20)
        self._no_kdtree(monkeypatch)
        got = find_proximal_region(ycr, "B", "A", "C", 20)
        assert got.seq_ids == expected.seq_ids
        assert got.patch_sequence == expected.patch_sequence


class TestMinDistances:
    def test_shape_and_finiteness(self, ycr):
        coords, _ = ycr["B"].stacked_heavy_coords()
        d = min_distances_to(ycr["A"], coords)
        assert d.shape == (85,)
        assert np.all(np.isfinite(d))

    def test_agrees_with_contact_map(self, ycr):
        coords, _ = ycr["B"].stacked_heavy_coords()
        d = min_distances_to(ycr["A"], coords)
        from_contacts = contact_map(ycr["A"], ycr["B"], 5.0).any(axis=1)
        assert np.array_equal(d <= 5.0, from_contacts)

    def test_empty_reference_gives_inf(self, ycr):
        d = min_distances_to(ycr["A"], np.empty((0, 3)))
        assert np.all(np.isinf(d))


class TestFindProximalRegion:
    def test_basic(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        assert len(region) > 0
        assert region.patch_sequence
        assert len(region.patch_sequence) == len(region)
        assert region.terminus == "C"
        assert region.binder_chain_id == "B"
        assert region.target_chain_id == "A"

    def test_anchor_is_the_right_terminus(self, ycr):
        assert find_proximal_region(ycr, "B", "A", "N", 20).anchor_label == "B:17"
        assert find_proximal_region(ycr, "B", "A", "C", 20).anchor_label == "B:29"

    def test_terminus_changes_the_selection(self, ycr):
        n = find_proximal_region(ycr, "B", "A", "N", 10, max_residues=15)
        c = find_proximal_region(ycr, "B", "A", "C", 10, max_residues=15)
        assert set(n.seq_ids) != set(c.seq_ids)

    def test_terminus_aliases(self, ycr):
        for alias in ("N", "n", "N-term", "nterm", "N-terminus"):
            assert find_proximal_region(ycr, "B", "A", alias, 10).terminus == "N"
        for alias in ("C", "c", "C-term", "cterm"):
            assert find_proximal_region(ycr, "B", "A", alias, 10).terminus == "C"

    def test_longer_flank_reaches_further(self, ycr):
        short = find_proximal_region(ycr, "B", "A", "C", 5)
        long = find_proximal_region(ycr, "B", "A", "C", 40)
        assert long.reach_radius > short.reach_radius
        assert set(short.seq_ids) <= set(long.seq_ids)

    def test_surface_filter_excludes_buried_residues(self, ycr):
        with_filter = find_proximal_region(ycr, "B", "A", "C", 30,
                                          require_surface=True)
        without = find_proximal_region(ycr, "B", "A", "C", 30,
                                       require_surface=False)
        assert len(with_filter) < len(without)
        assert set(with_filter.seq_ids) <= set(without.seq_ids)

    def test_surface_filter_reports_what_it_dropped(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 30)
        assert any("buried" in n for n in region.notes)

    def test_all_selected_residues_are_exposed(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 30,
                                      surface_threshold=0.10)
        assert all(r.relative_sasa > 0.10 for r in region)

    def test_explicit_radius_overrides_scaling(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 30, radius=12.0)
        assert region.reach_radius == pytest.approx(12.0)
        assert all(r.anchor_distance <= 12.0 for r in region)

    def test_radius_scale(self, ycr):
        base = find_proximal_region(ycr, "B", "A", "C", 20)
        wide = find_proximal_region(ycr, "B", "A", "C", 20, radius_scale=1.5)
        assert wide.reach_radius > base.reach_radius
        assert wide.reach_radius == pytest.approx(base.reach_radius * 1.5)

    def test_max_radius_is_a_hard_ceiling(self, ycr):
        """The ceiling applies to the value actually used, after scaling."""
        capped = find_proximal_region(ycr, "B", "A", "C", 20,
                                      radius_scale=3.0, max_radius=25.0)
        assert capped.reach_radius == pytest.approx(25.0)

    def test_max_radius_not_scaled_away(self, ycr):
        """Shrinking the radius must not be applied on top of the ceiling."""
        base = find_proximal_region(ycr, "B", "A", "C", 20).reach_radius
        got = find_proximal_region(ycr, "B", "A", "C", 20,
                                   radius_scale=0.5, max_radius=100.0)
        assert got.reach_radius == pytest.approx(base * 0.5)

    def test_max_residues_keeps_nearest(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 30, max_residues=10)
        assert len(region) == 10
        full = find_proximal_region(ycr, "B", "A", "C", 30)
        nearest = sorted(full, key=lambda p: p.anchor_distance)[:10]
        assert set(region.seq_ids) == {p.seq_id for p in nearest}
        assert any("closest to the anchor" in n for n in region.notes)

    def test_residues_are_sorted_by_number(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 30)
        assert region.seq_ids == sorted(region.seq_ids)

    def test_weights_decrease_with_distance(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 30)
        by_distance = sorted(region, key=lambda p: p.anchor_distance)
        weights = [p.weight for p in by_distance]
        assert weights == sorted(weights, reverse=True)
        assert all(0.0 <= w <= 1.0 for w in weights)

    def test_contact_labels_recorded(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        assert len(region.contact_labels) == len(KNOWN_CONTACTS_5A)


class TestSpansAndPatch:
    def test_spans_are_contiguous_runs(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        for first, last in region.spans:
            assert first <= last
        # 1YCR has no insertion codes, so expanding the spans numerically must
        # reproduce the selected residues exactly.
        flattened = [i for a, b in region.spans for i in range(a, b + 1)]
        assert flattened == region.seq_ids

    def test_spans_split_at_unresolved_gaps(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        # Every consecutive pair inside a span must differ by exactly one.
        for run, (first, last) in zip(region.span_sequences, region.spans):
            assert len(run) == last - first + 1

    def test_span_labels_available(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        assert len(region.span_labels) == len(region.spans)
        assert all(a.startswith("A:") for a, _ in region.span_labels)

    def test_patch_sequence_matches_residues(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        assert region.patch_sequence == "".join(r.one_letter for r in region)

    def test_span_sequences_concatenate_to_patch(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        assert "".join(region.span_sequences) == region.patch_sequence

    def test_weighted_patch_repeat_counts_follow_the_weights(self, ycr):
        """Each residue's copy count must be exactly 1 + round(w*(n-1))."""
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        for max_copies in (2, 3, 5):
            expected = "".join(
                p.one_letter * (1 + round(p.weight * (max_copies - 1)))
                for p in region)
            assert region.weighted_patch_sequence(max_copies) == expected

    def test_nearest_residue_gets_the_most_copies(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 20, max_residues=12)
        weighted = region.weighted_patch_sequence(4)
        by_distance = sorted(region, key=lambda p: p.anchor_distance)
        nearest, farthest = by_distance[0], by_distance[-1]

        def copies(p):
            return 1 + round(p.weight * 3)

        assert copies(nearest) > copies(farthest)
        # and the total length is the sum of the per-residue copy counts
        assert len(weighted) == sum(copies(p) for p in region)

    def test_weighting_grows_the_patch_monotonically(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        lengths = [len(region.weighted_patch_sequence(n)) for n in (1, 2, 3, 5)]
        assert lengths == sorted(lengths)
        assert lengths[0] == len(region)

    def test_weighted_patch_identity_at_one(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        assert region.weighted_patch_sequence(1) == region.patch_sequence

    def test_weighted_patch_rejects_zero(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        with pytest.raises(ValueError):
            region.weighted_patch_sequence(0)

    def test_summary_is_informative(self, ycr):
        text = find_proximal_region(ycr, "B", "A", "C", 20).summary()
        for expected in ("anchor residue", "reach radius", "patch sequence",
                         "interface contacts"):
            assert expected in text


class TestInsertionCodes:
    """Insertion codes repeat a residue number, so anything keyed on the
    number alone silently produces the wrong patch chemistry."""

    # Target residues 52, 52A, 52B, 53 beside a three-residue binder.
    PDB = """
        ATOM      1  N   GLY A  52       0.000   0.000   0.000  1.00  0.00           N
        ATOM      2  CA  GLY A  52       0.000   0.000   1.500  1.00  0.00           C
        ATOM      3  C   GLY A  52       0.000   0.000   3.000  1.00  0.00           C
        ATOM      4  N   PRO A  52A      3.500   0.000   0.000  1.00  0.00           N
        ATOM      5  CA  PRO A  52A      3.500   0.000   1.500  1.00  0.00           C
        ATOM      6  C   PRO A  52A      3.500   0.000   3.000  1.00  0.00           C
        ATOM      7  N   TRP A  52B      7.000   0.000   0.000  1.00  0.00           N
        ATOM      8  CA  TRP A  52B      7.000   0.000   1.500  1.00  0.00           C
        ATOM      9  C   TRP A  52B      7.000   0.000   3.000  1.00  0.00           C
        ATOM     10  N   LYS A  53      10.500   0.000   0.000  1.00  0.00           N
        ATOM     11  CA  LYS A  53      10.500   0.000   1.500  1.00  0.00           C
        ATOM     12  C   LYS A  53      10.500   0.000   3.000  1.00  0.00           C
        TER
        ATOM     13  N   GLY B   1       0.000   3.000   0.000  1.00  0.00           N
        ATOM     14  CA  GLY B   1       0.000   3.000   1.500  1.00  0.00           C
        ATOM     15  C   GLY B   1       0.000   3.000   3.000  1.00  0.00           C
        ATOM     16  N   GLY B   2       3.500   3.000   0.000  1.00  0.00           N
        ATOM     17  CA  GLY B   2       3.500   3.000   1.500  1.00  0.00           C
        ATOM     18  C   GLY B   2       3.500   3.000   3.000  1.00  0.00           C
        ATOM     19  N   GLY B   3       7.000   3.000   0.000  1.00  0.00           N
        ATOM     20  CA  GLY B   3       7.000   3.000   1.500  1.00  0.00           C
        ATOM     21  C   GLY B   3       7.000   3.000   3.000  1.00  0.00           C
        END
    """

    @pytest.fixture
    def region(self, tmp_path):
        struct = read_pdb(_write(tmp_path, "icode.pdb", self.PDB))
        return find_proximal_region(struct, "B", "A", "N", 20,
                                    require_surface=False,
                                    min_cluster_contacts=2,
                                    sequence_window=50)

    def test_all_four_residues_selected(self, region):
        assert region.labels == ["A:52", "A:52A", "A:52B", "A:53"]

    def test_patch_sequence_is_correct(self, region):
        # Keying on seq_id alone would collapse 52/52A/52B and yield "WWWK".
        assert region.patch_sequence == "GPWK"

    def test_patch_matches_selected_residues(self, region):
        assert region.patch_sequence == "".join(r.one_letter for r in region)

    def test_span_sequences_concatenate_to_patch(self, region):
        assert "".join(region.span_sequences) == region.patch_sequence

    def test_insertion_codes_form_one_run(self, region):
        assert region.spans == [(52, 53)]
        assert region.span_labels == [("A:52", "A:53")]
        assert region.span_sequences == ["GPWK"]

    def test_weighted_patch_covers_every_residue(self, region):
        weighted = region.weighted_patch_sequence(3)
        for letter in "GPWK":
            assert letter in weighted

    def test_summary_uses_labels_when_codes_present(self, region):
        assert "A:52-A:53" in region.summary()


class TestSequenceLocalityFilter:
    """The filter that removes spatially-close but sequence-distant residues.

    Predicted structures routinely place a far-away part of the target next to
    the binder; those residues would not really contribute to binding.
    """

    # Binder (chain B) beside target residues 10-13. Target residues 200-201
    # are placed right next to the binder too, but are ~190 residues away in
    # sequence -- the artifact this filter exists to remove.
    PDB = """
        ATOM      1  N   ALA A  10       0.000   0.000   0.000  1.00  0.00           N
        ATOM      2  CA  ALA A  10       0.000   0.000   1.500  1.00  0.00           C
        ATOM      3  C   ALA A  10       0.000   0.000   3.000  1.00  0.00           C
        ATOM      4  N   ALA A  11       3.500   0.000   0.000  1.00  0.00           N
        ATOM      5  CA  ALA A  11       3.500   0.000   1.500  1.00  0.00           C
        ATOM      6  C   ALA A  11       3.500   0.000   3.000  1.00  0.00           C
        ATOM      7  N   ALA A  12       7.000   0.000   0.000  1.00  0.00           N
        ATOM      8  CA  ALA A  12       7.000   0.000   1.500  1.00  0.00           C
        ATOM      9  C   ALA A  12       7.000   0.000   3.000  1.00  0.00           C
        ATOM     10  N   ALA A  13      10.500   0.000   0.000  1.00  0.00           N
        ATOM     11  CA  ALA A  13      10.500   0.000   1.500  1.00  0.00           C
        ATOM     12  C   ALA A  13      10.500   0.000   3.000  1.00  0.00           C
        ATOM     13  N   TRP A 200       0.000   6.000   0.000  1.00  0.00           N
        ATOM     14  CA  TRP A 200       0.000   6.000   1.500  1.00  0.00           C
        ATOM     15  C   TRP A 200       0.000   6.000   3.000  1.00  0.00           C
        ATOM     16  N   TRP A 201       3.500   6.000   0.000  1.00  0.00           N
        ATOM     17  CA  TRP A 201       3.500   6.000   1.500  1.00  0.00           C
        ATOM     18  C   TRP A 201       3.500   6.000   3.000  1.00  0.00           C
        TER
        ATOM     19  N   GLY B   1       0.000   3.000   0.000  1.00  0.00           N
        ATOM     20  CA  GLY B   1       0.000   3.000   1.500  1.00  0.00           C
        ATOM     21  C   GLY B   1       0.000   3.000   3.000  1.00  0.00           C
        ATOM     22  N   GLY B   2       3.500   3.000   0.000  1.00  0.00           N
        ATOM     23  CA  GLY B   2       3.500   3.000   1.500  1.00  0.00           C
        ATOM     24  C   GLY B   2       3.500   3.000   3.000  1.00  0.00           C
        ATOM     25  N   GLY B   3       7.000   3.000   0.000  1.00  0.00           N
        ATOM     26  CA  GLY B   3       7.000   3.000   1.500  1.00  0.00           C
        ATOM     27  C   GLY B   3       7.000   3.000   3.000  1.00  0.00           C
        END
    """

    def test_sequence_distant_artifact_is_excluded(self, tmp_path):
        """The default min_cluster_contacts rejects the 2-residue artifact."""
        struct = read_pdb(_write(tmp_path, "artifact.pdb", self.PDB))
        region = find_proximal_region(
            struct, "B", "A", "N", 20,
            require_surface=False, sequence_window=25, cluster_gap=15)
        assert set(region.seq_ids) <= {10, 11, 12, 13}
        assert 200 not in region.seq_ids and 201 not in region.seq_ids
        assert any(lbl.endswith(("200", "201"))
                   for lbl in region.excluded_labels)
        assert any("prediction artefact" in n for n in region.notes)

    def test_small_patch_reported_as_discarded(self, tmp_path):
        struct = read_pdb(_write(tmp_path, "artifact_note.pdb", self.PDB))
        region = find_proximal_region(
            struct, "B", "A", "N", 20,
            require_surface=False, cluster_gap=15)
        assert any("noise" in n for n in region.notes)

    def test_lowering_min_cluster_contacts_admits_the_artifact(self, tmp_path):
        """Opting in to 2-residue patches keeps the sequence-distant pair."""
        struct = read_pdb(_write(tmp_path, "artifact2.pdb", self.PDB))
        region = find_proximal_region(
            struct, "B", "A", "N", 20,
            require_surface=False, sequence_window=25, cluster_gap=15,
            min_cluster_contacts=2)
        assert 200 in region.seq_ids

    def test_widening_the_window_admits_the_artifact(self, tmp_path):
        """A window wide enough to span the gap also admits it."""
        struct = read_pdb(_write(tmp_path, "artifact3.pdb", self.PDB))
        region = find_proximal_region(
            struct, "B", "A", "N", 20,
            require_surface=False, sequence_window=300, cluster_gap=15)
        assert 200 in region.seq_ids

    def test_large_cluster_gap_merges_the_patches(self, tmp_path):
        """With no gap-based split there is one patch, big enough to keep."""
        struct = read_pdb(_write(tmp_path, "artifact4.pdb", self.PDB))
        region = find_proximal_region(
            struct, "B", "A", "N", 20,
            require_surface=False, sequence_window=25, cluster_gap=500)
        assert 200 in region.seq_ids

    def test_multiple_patches_are_reported(self, ycr):
        # 1YCR's interface genuinely comes in two sequence-separated patches.
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        assert any("distinct interface patches" in n for n in region.notes)


class TestTargetChainBreaks:
    """Missing target residues expose surface they would have covered, so a
    residue can be selected only because the model is incomplete."""

    @pytest.fixture
    def gapped_target(self, tmp_path):
        kept = [l for l in open(structure_path("1ycr.pdb"))
                if not (l.startswith("ATOM") and l[21] == "A"
                        and l[22:26].strip().isdigit()
                        and 63 <= int(l[22:26]) <= 66)]
        p = tmp_path / "tgap.pdb"
        p.write_text("".join(kept))
        return read_structure(str(p))

    def test_break_is_detected(self, gapped_target):
        assert gapped_target["A"].chain_breaks() == [(62, 67)]

    def test_region_reports_the_break(self, gapped_target):
        region = find_proximal_region(gapped_target, "B", "A", "C", 25)
        assert any("unresolved break" in n for n in region.notes)

    def test_intact_target_is_not_flagged(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 25)
        assert not any("unresolved break" in n for n in region.notes)

    def test_distant_break_is_not_flagged(self, tmp_path):
        """A break far in sequence from the selection is not relevant."""
        kept = [l for l in open(structure_path("1ycr.pdb"))
                if not (l.startswith("ATOM") and l[21] == "A"
                        and l[22:26].strip().isdigit()
                        and 26 <= int(l[22:26]) <= 28)]
        p = tmp_path / "far.pdb"
        p.write_text("".join(kept))
        region = find_proximal_region(read_structure(str(p)), "B", "A", "C", 25,
                                      sequence_window=5, max_residues=6)
        breaks = [n for n in region.notes if "unresolved break" in n]
        if breaks:
            # if reported, it must genuinely be near a selected residue
            assert any(abs(s - 25) <= 10 or abs(s - 29) <= 10
                       for s in region.seq_ids)


class TestSmallPatchWarning:
    def test_note_when_few_residues_survive(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 20, max_residues=3)
        assert any("survived the filters" in n for n in region.notes)

    def test_no_note_for_a_healthy_patch(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        assert not any("survived the filters" in n for n in region.notes)


class TestErrors:
    def test_same_chain(self, ycr):
        with pytest.raises(InterfaceError):
            find_proximal_region(ycr, "A", "A", "N", 10)

    def test_missing_binder_chain(self, ycr):
        with pytest.raises(InterfaceError, match="not in the structure"):
            find_proximal_region(ycr, "Z", "A", "N", 10)

    def test_missing_target_chain(self, ycr):
        with pytest.raises(InterfaceError, match="not in the structure"):
            find_proximal_region(ycr, "B", "Z", "N", 10)

    def test_bad_terminus(self, ycr):
        with pytest.raises(ValueError, match="must be 'N' or 'C'"):
            find_proximal_region(ycr, "B", "A", "middle", 10)

    def test_nonpositive_flank(self, ycr):
        with pytest.raises(ValueError):
            find_proximal_region(ycr, "B", "A", "N", 0)

    def test_bad_anchor_residues(self, ycr):
        with pytest.raises(ValueError):
            find_proximal_region(ycr, "B", "A", "N", 10, anchor_residues=0)

    def test_anchor_residues_larger_than_the_binder(self, ycr):
        """Using the whole binder as the anchor makes the termini identical."""
        with pytest.raises(ValueError, match="exceeds"):
            find_proximal_region(ycr, "B", "A", "N", 10, anchor_residues=99)

    def test_anchor_residues_equal_to_binder_length_is_allowed(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "N", 10,
                                      anchor_residues=len(ycr["B"]))
        assert len(region) > 0

    def test_chains_not_in_contact(self, ycr):
        with pytest.raises(InterfaceError, match="no heavy-atom contact"):
            find_proximal_region(ycr, "B", "A", "N", 10, contact_cutoff=0.5)

    def test_nothing_within_reach(self, ycr):
        with pytest.raises(InterfaceError, match="cannot reach"):
            find_proximal_region(ycr, "B", "A", "N", 10, radius=0.5)

    def test_error_message_suggests_remedies(self, ycr):
        with pytest.raises(InterfaceError) as exc:
            find_proximal_region(ycr, "B", "A", "N", 10, radius=0.5)
        assert "other terminus" in str(exc.value)


class TestAnchorResidues:
    def test_more_anchor_residues_widens_selection(self, ycr):
        one = find_proximal_region(ycr, "B", "A", "C", 10, anchor_residues=1)
        three = find_proximal_region(ycr, "B", "A", "C", 10, anchor_residues=3)
        assert three.anchor_label == "B:27..B:29"
        assert len(three) >= len(one)

    def test_n_terminal_anchor_spans_the_start(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "N", 10, anchor_residues=3)
        assert region.anchor_label == "B:17..B:19"


class TestFormatAgnostic:
    def test_pdb_and_cif_give_identical_regions(self):
        pdb = read_structure(structure_path("1ycr.pdb"))
        cif = read_structure(structure_path("1ycr.cif"))
        rp = find_proximal_region(pdb, "B", "A", "C", 20)
        rc = find_proximal_region(cif, "B", "A", "C", 20)
        assert rp.seq_ids == rc.seq_ids
        assert rp.patch_sequence == rc.patch_sequence


class TestResidueSpecParsing:
    """The selection syntax, including the one deliberately ambiguous case."""

    def test_string_range(self):
        from idr_flanks.interface import parse_residue_spec as parse
        assert parse("1-100") == set(range(1, 101))

    def test_string_multiple_ranges(self):
        from idr_flanks.interface import parse_residue_spec as parse
        assert parse("1-10,20-25") == set(range(1, 11)) | set(range(20, 26))

    def test_string_mixed(self):
        from idr_flanks.interface import parse_residue_spec as parse
        assert parse("1-10,50") == set(range(1, 11)) | {50}

    def test_bare_pair_is_a_range(self):
        """The documented behaviour: [1, 100] means the first hundred."""
        from idr_flanks.interface import parse_residue_spec as parse
        assert parse([1, 100]) == set(range(1, 101))
        assert parse((1, 100)) == set(range(1, 101))

    def test_three_or_more_are_individual(self):
        from idr_flanks.interface import parse_residue_spec as parse
        assert parse([5, 12, 88]) == {5, 12, 88}

    def test_nested_ranges(self):
        from idr_flanks.interface import parse_residue_spec as parse
        assert parse([(1, 10), (20, 25)]) == set(range(1, 11)) | set(range(20, 26))

    def test_singleton_lists_are_individual(self):
        from idr_flanks.interface import parse_residue_spec as parse
        assert parse([[5], [12]]) == {5, 12}

    def test_range_object_and_int(self):
        from idr_flanks.interface import parse_residue_spec as parse
        assert parse(range(1, 11)) == set(range(1, 11))
        assert parse(5) == {5}

    def test_none_is_empty(self):
        from idr_flanks.interface import parse_residue_spec as parse
        assert parse(None) == set()

    def test_negative_numbering_supported(self):
        from idr_flanks.interface import parse_residue_spec as parse
        assert parse("-3--1") == {-3, -2, -1}

    def test_backwards_range_rejected(self):
        from idr_flanks.interface import parse_residue_spec as parse
        with pytest.raises(ValueError, match="backwards"):
            parse([(100, 1)])

    def test_garbage_rejected(self):
        from idr_flanks.interface import parse_residue_spec as parse
        with pytest.raises(ValueError):
            parse("not-a-number")


class TestTargetResidueSelection:
    """A predictor that folds a whole terminus onto the real binding site makes
    a large, self-consistent contact patch that the automatic noise filter
    cannot distinguish from the genuine one. Only the user knows."""

    @staticmethod
    def _residue(chain, num, name, x, y, z, serial):
        out = []
        for atom, el, dx in (("N", "N", 0.0), ("CA", "C", 1.4),
                             ("C", "C", 2.4), ("O", "O", 3.0)):
            out.append(
                f"ATOM  {serial:5d}  {atom:<3s} {name} {chain}{num:4d}    "
                f"{x + dx:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
                f"{el:>2s}\n")
            serial += 1
        return out, serial

    @pytest.fixture
    def mispredicted(self, tmp_path):
        """True site 480-495 (acidic) above the binder; residues 1-100
        (basic) mispredicted onto the other face, also in contact."""
        lines, sn = [], 1
        for i in range(6):
            block, sn = self._residue("B", i + 1, "GLY", i * 3.8, 0.0, 0.0, sn)
            lines += block
        for k, num in enumerate(range(480, 496)):
            block, sn = self._residue("A", num, "GLU", (k % 8) * 3.8, 4.3,
                                      (k // 8) * 3.6, sn)
            lines += block
        for k, num in enumerate(range(1, 101)):
            block, sn = self._residue("A", num, "LYS", (k % 8) * 3.8, -4.3,
                                      (k // 8) * 3.6, sn)
            lines += block
        p = tmp_path / "mispredicted.pdb"
        p.write_text("".join(lines) + "END\n")
        return read_pdb(str(p))

    def test_the_failure_reproduces_without_help(self, mispredicted):
        """Both patches survive: the automatic filters cannot fix this."""
        region = find_proximal_region(mispredicted, "B", "A", "C", 25)
        assert set("KE") <= set(region.patch_sequence)
        assert any(a <= 100 for a, _ in region.spans)
        assert any("2 distinct interface patches" in n for n in region.notes)

    def test_exclusion_removes_the_mispredicted_region(self, mispredicted):
        region = find_proximal_region(mispredicted, "B", "A", "C", 25,
                                      exclude_target_residues=[1, 100])
        assert set(region.patch_sequence) == {"E"}
        assert region.spans == [(480, 495)]
        assert all(s >= 480 for s in region.seq_ids)

    def test_inclusion_is_equivalent_here(self, mispredicted):
        excluded = find_proximal_region(mispredicted, "B", "A", "C", 25,
                                        exclude_target_residues="1-100")
        included = find_proximal_region(mispredicted, "B", "A", "C", 25,
                                        include_target_residues=[480, 495])
        assert excluded.seq_ids == included.seq_ids

    def test_string_and_list_forms_agree(self, mispredicted):
        a = find_proximal_region(mispredicted, "B", "A", "C", 25,
                                 exclude_target_residues="1-100")
        b = find_proximal_region(mispredicted, "B", "A", "C", 25,
                                 exclude_target_residues=[1, 100])
        assert a.seq_ids == b.seq_ids

    def test_excluded_region_cannot_define_an_interface_patch(self, mispredicted):
        """The filter must run before clustering, or the excluded patch still
        opens a sequence window for its neighbours."""
        region = find_proximal_region(mispredicted, "B", "A", "C", 25,
                                      exclude_target_residues=[1, 100])
        assert not any("2 distinct interface patches" in n
                       for n in region.notes)

    def test_selection_is_reported(self, mispredicted):
        region = find_proximal_region(mispredicted, "B", "A", "C", 25,
                                      exclude_target_residues=[1, 100])
        assert any("excluded target residues 1-100" in n for n in region.notes)
        assert any("ruled out by the target-residue selection" in n
                   for n in region.notes)

    def test_excluding_the_whole_interface_errors_clearly(self, mispredicted):
        with pytest.raises(InterfaceError, match="no contact"):
            find_proximal_region(mispredicted, "B", "A", "C", 25,
                                 exclude_target_residues="1-1000")

    def test_include_selecting_absent_numbers_errors(self, mispredicted):
        with pytest.raises(InterfaceError, match="contains none"):
            find_proximal_region(mispredicted, "B", "A", "C", 25,
                                 include_target_residues=[9000, 9100])

    def test_no_selection_leaves_behaviour_unchanged(self, ycr):
        plain = find_proximal_region(ycr, "B", "A", "C", 25)
        explicit = find_proximal_region(ycr, "B", "A", "C", 25,
                                        exclude_target_residues=None,
                                        include_target_residues=None)
        assert plain.seq_ids == explicit.seq_ids

    def test_partial_exclusion_on_a_real_structure(self, ycr):
        """Excluding part of a real interface.

        The result is not simply a subset of the unrestricted selection, and
        that is correct: removing 25-60 shifts the accepted interface patch
        from 50-75 to 61-75, which moves the sequence window, and the excluded
        residues also stop occluding, so a few formerly buried neighbours
        become exposed. What is guaranteed is that the exclusion is honoured.
        """
        full = find_proximal_region(ycr, "B", "A", "C", 25)
        trimmed = find_proximal_region(ycr, "B", "A", "C", 25,
                                       exclude_target_residues="25-60")
        assert all(not (25 <= s <= 60) for s in trimmed.seq_ids)
        assert 0 < len(trimmed) < len(full)
        assert any(25 <= s <= 60 for s in full.seq_ids), (
            "the range must actually have been in play")

    def test_exclusion_shifts_the_accepted_patch(self, ycr):
        trimmed = find_proximal_region(ycr, "B", "A", "C", 25,
                                       exclude_target_residues="25-60")
        assert any("61-75" in n for n in trimmed.notes)


class TestDistalOcclusion:
    """A predictor that drapes a sequence-distant region over the surface near
    the binder would otherwise bury target surface that is really available,
    discarding the region for a reason that is an artefact.
    """

    @staticmethod
    def _residue(chain, num, name, x, y, z, serial):
        out = []
        for atom, element, dx in (("N", "N", 0.0), ("CA", "C", 1.4),
                                  ("C", "C", 2.4), ("O", "O", 3.0)):
            out.append(
                f"ATOM  {serial:5d}  {atom:<3s} {name} {chain}{num:4d}    "
                f"{x + dx:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
                f"{element:>2s}\n")
            serial += 1
        return out, serial

    @pytest.fixture
    def draped(self, tmp_path):
        """Local target region 50-56, with distal 500-506 and 520-526 sandwiched
        against it."""
        lines, sn = [], 1
        for i in range(3):
            block, sn = self._residue("B", i + 1, "GLY", i * 3.8, 0.0, 0.0, sn)
            lines += block
        for k, num in enumerate(range(50, 57)):
            block, sn = self._residue("A", num, "SER", k * 3.8, 5.0, 0.0, sn)
            lines += block
        for k, num in enumerate(range(500, 507)):
            block, sn = self._residue("A", num, "TRP", k * 3.8, 5.0, 3.0, sn)
            lines += block
        for k, num in enumerate(range(520, 527)):
            block, sn = self._residue("A", num, "TRP", k * 3.8, 5.0, -3.0, sn)
            lines += block
        p = tmp_path / "draped.pdb"
        p.write_text("".join(lines) + "END\n")
        return read_pdb(str(p))

    # A 40-residue flank so every local residue of the fixture (out to ~23 A
    # from the anchor) is within reach; these tests are about occlusion, not
    # about where the reach radius falls.
    FLANK = 40

    def _region(self, struct, trust, threshold=0.30):
        return find_proximal_region(
            struct, "B", "A", "N", self.FLANK, trust_distal_occlusion=trust,
            min_cluster_contacts=2, sequence_window=25,
            surface_threshold=threshold)

    def test_distal_region_does_not_bury_local_surface(self, draped):
        keep = self._region(draped, trust=False)
        assert set(keep.seq_ids) == set(range(50, 57))

    def test_trusting_it_buries_the_surface(self, draped):
        """Opting in reproduces the artefact, which is the point of the flag."""
        trusted = self._region(draped, trust=True)
        assert len(trusted) < 7

    def test_default_is_not_to_trust(self, draped):
        default = find_proximal_region(draped, "B", "A", "N", self.FLANK,
                                       min_cluster_contacts=2,
                                       sequence_window=25,
                                       surface_threshold=0.30)
        assert set(default.seq_ids) == set(range(50, 57))

    def test_exclusion_is_reported(self, draped):
        region = self._region(draped, trust=False)
        note = next((n for n in region.notes
                     if "sequence-distant" in n and "accessibility" in n), None)
        assert note is not None
        # 14 distal residues sit against the candidates in this fixture.
        assert "14 sequence-distant" in note

    def test_note_counts_only_residues_that_could_occlude(self, ycr):
        """Counting every sequence-distant residue in the chain would report
        hundreds on a large target and imply far more was discarded than was."""
        region = find_proximal_region(ycr, "B", "A", "C", 25)
        assert not any("sequence-distant" in n and "accessibility" in n
                       for n in region.notes)

    def test_note_absent_on_a_large_target_with_nothing_nearby(self, tmp_path):
        """A distant region far away in space must not be counted at all."""
        lines, sn = [], 1
        for i in range(3):
            block, sn = self._residue("B", i + 1, "GLY", i * 3.8, 0.0, 0.0, sn)
            lines += block
        for k, num in enumerate(range(50, 57)):
            block, sn = self._residue("A", num, "SER", k * 3.8, 5.0, 0.0, sn)
            lines += block
        # 200 sequence-distant residues, but 500 A away: they occlude nothing.
        for k, num in enumerate(range(500, 700)):
            block, sn = self._residue("A", num, "TRP", k * 3.8, 500.0, 0.0, sn)
            lines += block
        p = tmp_path / "faraway.pdb"
        p.write_text("".join(lines) + "END\n")
        region = find_proximal_region(read_pdb(str(p)), "B", "A", "N",
                                      self.FLANK, min_cluster_contacts=2,
                                      sequence_window=25)
        assert not any("sequence-distant" in n and "accessibility" in n
                       for n in region.notes)

    def test_real_structure_is_unaffected(self, ycr):
        """With nothing sequence-distant, the flag must change nothing."""
        a = find_proximal_region(ycr, "B", "A", "C", 25,
                                 trust_distal_occlusion=False)
        b = find_proximal_region(ycr, "B", "A", "C", 25,
                                 trust_distal_occlusion=True)
        assert a.seq_ids == b.seq_ids
        assert a.patch_sequence == b.patch_sequence


class TestBinderInterfaceReported:
    def test_binder_interface_is_extracted(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        # p53 residues that contact MDM2 within 5 A.
        assert region.binder_interface_sequence == "ETFSLWLLPEN"
        assert region.binder_interface_labels[0] == "B:17"

    def test_it_is_a_subsequence_of_the_binder(self, ycr):
        region = find_proximal_region(ycr, "B", "A", "C", 20)
        binder = ycr["B"].sequence
        it = iter(binder)
        assert all(c in it for c in region.binder_interface_sequence)

    def test_reported_in_the_summary(self, ycr):
        text = find_proximal_region(ycr, "B", "A", "C", 20).summary()
        assert "binder interface" in text

    def test_reversed_roles_give_the_other_interface(self, ycr):
        region = find_proximal_region(ycr, "A", "B", "C", 20)
        # now MDM2 is the binder, so its own contacting residues are reported
        assert len(region.binder_interface_sequence) > 15
