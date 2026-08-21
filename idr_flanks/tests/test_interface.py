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
