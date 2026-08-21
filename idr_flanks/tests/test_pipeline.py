"""End-to-end tests: structure file in, flanked binder sequence out."""

import textwrap

import pytest

goose = pytest.importorskip("goose", reason="GOOSE is needed to design flanks")

from idr_flanks.data import structure_path  # noqa: E402
from idr_flanks.interface import InterfaceError  # noqa: E402
from idr_flanks.io import read_structure  # noqa: E402
from idr_flanks.pipeline import (  # noqa: E402
    FlankedBinder,
    build_flanked_binder,
    describe_chains,
)

P53 = "ETFSDLWKLLPEN"
MDM2 = ("ETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGD"
        "LFGVPSFSVKEHRKIYTMIYRNLVV")
FAST = dict(max_iterations=120, num_starting_candidates=60, seed=17)


@pytest.fixture(scope="module")
def pdb_path():
    return structure_path("1ycr.pdb")


@pytest.fixture(scope="module")
def c_only(pdb_path):
    return build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                c_flank_length=18, **FAST)


class TestDescribeChains:
    def test_from_path(self, pdb_path):
        text = describe_chains(pdb_path)
        assert "chain 'A'" in text and "chain 'B'" in text
        assert P53 in text

    def test_from_structure(self, pdb_path):
        assert describe_chains(read_structure(pdb_path)).startswith("Structure:")


class TestSingleFlank:
    def test_c_terminal_flank_is_appended(self, c_only):
        assert c_only.final_sequence.startswith(P53)
        assert c_only.final_sequence == P53 + c_only.c_flank
        assert len(c_only.c_flank) == 18
        assert c_only.n_flank == ""

    def test_binder_sequence_is_untouched(self, c_only):
        assert c_only.binder_sequence == P53
        assert P53 in c_only.final_sequence

    def test_n_terminal_flank_is_prepended(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 n_flank_length=18, **FAST)
        assert r.final_sequence.endswith(P53)
        assert r.final_sequence == r.n_flank + P53
        assert r.c_flank == ""

    def test_only_the_requested_region_is_analysed(self, c_only):
        assert set(c_only.regions) == {"C"}
        assert set(c_only.designs) == {"C"}


class TestBothFlanks:
    def test_both_are_added(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 n_flank_length=12, c_flank_length=15, **FAST)
        assert len(r.n_flank) == 12
        assert len(r.c_flank) == 15
        assert r.final_sequence == r.n_flank + P53 + r.c_flank
        assert len(r) == len(P53) + 27
        assert r.added_residues == 27
        assert set(r.regions) == {"N", "C"}

    def test_the_two_flanks_differ(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 n_flank_length=20, c_flank_length=20,
                                 max_residues=12, **FAST)
        # They target different surfaces, so they should not come out identical.
        assert r.n_flank != r.c_flank

    def test_c_flank_sees_the_n_flank_as_context(self, pdb_path, monkeypatch):
        """The contexts handed to each design must match the real construct.

        Disorder is context dependent, so a C-terminal flank must be scored
        against n_flank + binder, not the bare binder. Recorded directly rather
        than inferred, since the wrong context still produces a valid-looking
        sequence.
        """
        import idr_flanks.pipeline as mod

        seen = []
        real = mod.design_flank

        def spy(patch, length, **kwargs):
            seen.append((length, kwargs.get("n_context", ""),
                         kwargs.get("c_context", "")))
            return real(patch, length, **kwargs)

        monkeypatch.setattr(mod, "design_flank", spy)
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 n_flank_length=12, c_flank_length=15, **FAST)

        assert len(seen) == 2
        (n_len, n_pre, n_post), (c_len, c_pre, c_post) = seen
        # N-terminal flank: nothing before it, the binder after it.
        assert n_len == 12
        assert n_pre == ""
        assert n_post == P53
        # C-terminal flank: the finished N-flank plus the binder before it.
        assert c_len == 15
        assert c_pre == r.n_flank + P53
        assert c_post == ""

    def test_single_c_flank_context_is_just_the_binder(self, pdb_path, monkeypatch):
        import idr_flanks.pipeline as mod
        seen = []
        real = mod.design_flank

        def spy(patch, length, **kwargs):
            seen.append((kwargs.get("n_context", ""), kwargs.get("c_context", "")))
            return real(patch, length, **kwargs)

        monkeypatch.setattr(mod, "design_flank", spy)
        build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                             c_flank_length=10, **FAST)
        assert seen == [(P53, "")]


class TestOutputs:
    def test_annotated_sequence_brackets_the_flanks(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 n_flank_length=10, c_flank_length=10, **FAST)
        text = r.annotated_sequence()
        assert text == f"[{r.n_flank}]{P53}[{r.c_flank}]"

    def test_fasta_is_wellformed(self, c_only):
        fasta = c_only.fasta(name="test")
        lines = fasta.splitlines()
        assert lines[0].startswith(">test")
        assert "binder_chain=B" in lines[0] and "target_chain=A" in lines[0]
        assert "".join(lines[1:]) == c_only.final_sequence

    def test_fasta_wraps(self, c_only):
        lines = c_only.fasta(width=10).splitlines()[1:]
        assert all(len(x) <= 10 for x in lines)

    def test_str_is_the_final_sequence(self, c_only):
        assert str(c_only) == c_only.final_sequence

    def test_summary_covers_both_stages(self, c_only):
        text = c_only.summary()
        assert "Proximal region" in text
        assert "Designed flank" in text
        assert "final construct" in text
        assert c_only.final_sequence in text.replace("[", "").replace("]", "")

    def test_warnings_are_namespaced_by_stage(self, pdb_path):
        """Region and design warnings carry a terminus prefix; structure
        warnings deliberately do not, since they describe the input file."""
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=20, preset="unconstrained",
                                 **FAST)
        assert r.warnings, "1YCR is truncated, so warnings are expected"
        structure = set(r.structure_warnings)
        staged = [w for w in r.warnings if w not in structure]
        assert staged, "expected at least one region or design warning"
        for w in staged:
            assert w.startswith(("N-terminal ", "C-terminal ")), w

    def test_structure_warnings_are_not_prefixed(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=20, **FAST)
        assert r.structure_warnings
        for w in r.structure_warnings:
            assert not w.startswith(("N-terminal ", "C-terminal "))
            assert w in r.warnings

    def test_structure_is_retained(self, c_only):
        assert c_only.structure is not None
        assert c_only.structure.chain_ids == ["A", "B"]


class TestArgumentRouting:
    def test_interface_options_are_routed(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=15, max_residues=8, **FAST)
        assert len(r.regions["C"]) == 8

    def test_design_options_are_routed(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=15, max_aromatic_fraction=0.0,
                                 **FAST)
        assert sum(r.c_flank.count(a) for a in "WFY") == 0

    def test_explicit_interface_options_dict(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=15,
                                 interface_options={"max_residues": 6},
                                 **FAST)
        assert len(r.regions["C"]) == 6

    def test_patch_weighting_changes_the_patch(self, pdb_path):
        plain = build_flanked_binder(pdb_path, binder_chain="B",
                                     target_chain="A", c_flank_length=12,
                                     patch_weighting=1, **FAST)
        weighted = build_flanked_binder(pdb_path, binder_chain="B",
                                        target_chain="A", c_flank_length=12,
                                        patch_weighting=3, **FAST)
        assert (weighted.designs["C"].patch_sequence
                != plain.designs["C"].patch_sequence)
        assert (len(weighted.designs["C"].patch_sequence)
                > len(plain.designs["C"].patch_sequence))

    def test_weighted_patch_is_labelled_not_conflated(self, pdb_path):
        """The user must see the region they selected, not the repeated string
        that was optimised against."""
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=12, patch_weighting=3,
                                 max_residues=8, **FAST)
        design = r.designs["C"]
        region = r.regions["C"]
        assert design.selected_patch_sequence == region.patch_sequence
        assert len(design.patch_sequence) > len(region.patch_sequence)
        text = design.summary()
        assert f"target patch           : {region.patch_sequence}" in text
        assert "weighted patch" in text

    def test_unweighted_run_has_no_weighted_line(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=12, patch_weighting=1,
                                 max_residues=8, **FAST)
        design = r.designs["C"]
        assert design.patch_sequence == r.regions["C"].patch_sequence
        assert "weighted patch" not in design.summary()

    def test_routing_set_matches_the_real_signature(self):
        """Derived, not hand-listed: a parameter added to find_proximal_region
        must route automatically instead of being rejected downstream."""
        import inspect
        from idr_flanks.interface import find_proximal_region
        from idr_flanks.pipeline import _INTERFACE_KEYS
        params = set(inspect.signature(find_proximal_region).parameters)
        params -= {"structure", "binder_chain", "target_chain", "terminus",
                   "flank_length"}
        assert params == set(_INTERFACE_KEYS)

    def test_no_ambiguous_option_names(self):
        """No name may mean one thing to the interface and another to design."""
        from idr_flanks.design import DesignConfig
        from idr_flanks.pipeline import _INTERFACE_KEYS
        assert not (set(DesignConfig.__dataclass_fields__) & set(_INTERFACE_KEYS))

    def test_every_interface_parameter_is_accepted(self, pdb_path):
        """Each routed parameter must be usable through the Python API."""
        from idr_flanks.pipeline import _INTERFACE_KEYS
        samples = {
            "contact_cutoff": 5.5, "anchor_residues": 2, "radius": 20.0,
            "radius_scale": 1.2, "max_radius": 40.0, "cluster_gap": 12,
            "min_cluster_contacts": 2, "sequence_window": 30,
            "max_residues": 10, "require_surface": True,
            "surface_threshold": 0.12, "sasa_points": 120,
            "trust_distal_occlusion": True,
        }
        assert set(samples) == set(_INTERFACE_KEYS), (
            "update this test's sample values for new parameters")
        for name, value in samples.items():
            build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=6, max_iterations=30,
                                 num_starting_candidates=15, seed=1,
                                 **{name: value})

    def test_unknown_option_is_rejected(self, pdb_path):
        with pytest.raises(TypeError):
            build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=10, bogus_option=1)


class TestFormatAgnostic:
    def test_pdb_and_cif_give_the_same_result(self):
        kwargs = dict(binder_chain="B", target_chain="A", c_flank_length=15,
                      **FAST)
        a = build_flanked_binder(structure_path("1ycr.pdb"), **kwargs)
        b = build_flanked_binder(structure_path("1ycr.cif"), **kwargs)
        assert a.regions["C"].patch_sequence == b.regions["C"].patch_sequence
        assert a.final_sequence == b.final_sequence

    def test_accepts_a_parsed_structure(self, pdb_path):
        struct = read_structure(pdb_path)
        r = build_flanked_binder(struct, binder_chain="B", target_chain="A",
                                 c_flank_length=12, **FAST)
        assert r.structure is struct


class TestReversedRoles:
    def test_mdm2_can_be_the_binder(self, pdb_path):
        """Either chain may be the binder; nothing hard-codes the roles."""
        r = build_flanked_binder(pdb_path, binder_chain="A", target_chain="B",
                                 c_flank_length=12, **FAST)
        assert r.binder_sequence == MDM2
        assert r.final_sequence.startswith(MDM2)
        assert r.target_chain == "B"


class TestInputHandling:
    def test_accepts_a_pathlib_path(self, pdb_path):
        import pathlib
        r = build_flanked_binder(pathlib.Path(pdb_path), binder_chain="B",
                                 target_chain="A", c_flank_length=10, **FAST)
        assert r.final_sequence.startswith(P53)

    def test_describe_chains_accepts_a_path(self, pdb_path):
        import pathlib
        assert "chain 'A'" in describe_chains(pathlib.Path(pdb_path))

    def test_model_rejected_for_a_parsed_structure(self, pdb_path):
        struct = read_structure(pdb_path)
        with pytest.raises(ValueError, match="already been parsed"):
            build_flanked_binder(struct, binder_chain="B", target_chain="A",
                                 c_flank_length=10, model=1)

    def test_conflicting_option_is_rejected(self, pdb_path):
        with pytest.raises(ValueError, match="pass it once"):
            build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=10,
                                 interface_options={"max_residues": 5},
                                 max_residues=9, **FAST)

    def test_identical_duplicate_option_is_allowed(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=10,
                                 interface_options={"max_residues": 6},
                                 max_residues=6, **FAST)
        assert len(r.regions["C"]) == 6


class TestUnresolvedBinderResidues:
    """A structure holds only resolved residues, so a gap in the binder means
    the returned construct is not the real binder plus a flank."""

    @pytest.fixture
    def gapped(self, tmp_path, pdb_path):
        kept = [l for l in open(pdb_path)
                if not (l.startswith("ATOM") and l[21] == "B"
                        and l[22:26].strip() in ("22", "23", "24"))]
        p = tmp_path / "gap.pdb"
        p.write_text("".join(kept))
        return str(p)

    def test_break_is_detected(self, gapped):
        s = read_structure(gapped)
        assert s["B"].numbering_gaps() == [(21, 25)]
        assert s["B"].chain_breaks() == [(21, 25)]
        assert s["B"].sequence == "ETFSDLLPEN"

    def test_pipeline_warns_loudly(self, gapped):
        r = build_flanked_binder(gapped, binder_chain="B", target_chain="A",
                                 c_flank_length=10, **FAST)
        assert any("unresolved break" in w for w in r.warnings)
        assert any("unresolved break" in w for w in r.structure_warnings)
        assert "WARNING" in r.summary()

    def test_intact_structure_is_not_flagged(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=10, **FAST)
        assert not any("unresolved break" in w for w in r.warnings)

    def test_no_breaks_reported_for_contiguous_chain(self, pdb_path):
        s = read_structure(pdb_path)
        assert s["A"].chain_breaks() == []
        assert s["B"].chain_breaks() == []

    def test_conventional_numbering_jumps_are_not_breaks(self, tmp_path):
        """Antibody Kabat numbering skips numbers on a continuous chain.

        Reporting those as unresolved residues would tell every antibody user
        their construct is a deletion mutant. Only a missing peptide bond
        counts.
        """
        # Residues 10 and 13 are numbered three apart but properly bonded.
        pdb = """
            ATOM      1  N   ALA A  10       0.000   0.000   0.000  1.00  0.00           N
            ATOM      2  CA  ALA A  10       1.458   0.000   0.000  1.00  0.00           C
            ATOM      3  C   ALA A  10       2.009   1.420   0.000  1.00  0.00           C
            ATOM      4  N   GLY A  13       3.339   1.510   0.000  1.00  0.00           N
            ATOM      5  CA  GLY A  13       3.990   2.810   0.000  1.00  0.00           C
            ATOM      6  C   GLY A  13       5.500   2.700   0.000  1.00  0.00           C
            END
        """
        p = tmp_path / "kabat.pdb"
        p.write_text(textwrap.dedent(pdb).lstrip("\n"))
        s = read_structure(str(p))
        assert s["A"].numbering_gaps() == [(10, 13)]
        assert s["A"].chain_breaks() == []

    def test_real_antibody_numbering_is_not_flagged(self):
        """1IGY's heavy chains carry 22 Kabat jumps and no real break."""
        import os
        path = os.path.join(
            "/private/tmp/claude-501/-Users-ryanemenecker-Desktop-lab-packages"
            "-idr-flanks/883622ff-384d-409d-b210-6fca7aa740aa/scratchpad/edge",
            "1IGY.pdb")
        if not os.path.isfile(path):
            pytest.skip("1IGY not available locally")
        s = read_structure(path)
        assert len(s["B"].numbering_gaps()) > 10
        assert s["B"].chain_breaks() == []


class TestTruncatedTerminus:
    """The terminus is where the flank attaches, so truncation there means the
    anchor is the wrong atom."""

    def test_attachment_terminus_truncation_warns_loudly(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 n_flank_length=12, **FAST)
        assert any("N-terminus" in w and "truncated" in w
                   and "wrong atom" in w for w in r.warnings)

    def test_other_terminus_truncation_is_noted_as_harmless(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=12, **FAST)
        assert any("does not affect the flank you asked for" in w
                   for w in r.warnings)

    def test_full_sequence_is_reported(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=12, **FAST)
        assert r.binder_full_sequence == "SQ" + P53
        assert r.binder_sequence == P53
        assert "deposited" in r.summary()


class TestRegionNotesReachWarnings:
    def test_region_notes_are_surfaced(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=12, **FAST)
        notes = r.regions["C"].notes
        assert notes, "expected the region to report something"
        for note in notes:
            assert f"C-terminal region: {note}" in r.warnings

    def test_small_patch_note_is_visible_programmatically(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=12, max_residues=3, **FAST)
        assert any("survived the filters" in w for w in r.warnings)


class TestErrors:
    def test_no_flank_requested(self, pdb_path):
        with pytest.raises(ValueError, match="must be positive"):
            build_flanked_binder(pdb_path, binder_chain="B", target_chain="A")

    def test_negative_flank(self, pdb_path):
        with pytest.raises(ValueError, match="cannot be negative"):
            build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=-5)

    def test_missing_chain(self, pdb_path):
        with pytest.raises(InterfaceError):
            build_flanked_binder(pdb_path, binder_chain="Q", target_chain="A",
                                 c_flank_length=10)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            build_flanked_binder("/nope/missing.pdb", binder_chain="B",
                                 target_chain="A", c_flank_length=10)


class TestReproducibility:
    def test_same_seed_same_construct(self, pdb_path):
        kwargs = dict(binder_chain="B", target_chain="A", c_flank_length=15,
                      max_iterations=120, num_starting_candidates=60, seed=5)
        a = build_flanked_binder(pdb_path, **kwargs)
        b = build_flanked_binder(pdb_path, **kwargs)
        assert a.final_sequence == b.final_sequence


class TestLinker:
    """A designed flank is chemically loaded on purpose; a short flexible linker
    keeps it off the binder so it is less likely to perturb folding."""

    def test_linker_is_inserted_on_the_c_side(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=10, linker_length=4, **FAST)
        assert r.linker == "GSGS"
        assert r.final_sequence == P53 + "GSGS" + r.c_flank

    def test_linker_is_inserted_on_the_n_side(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 n_flank_length=10, linker_length=4, **FAST)
        assert r.final_sequence == r.n_flank + "GSGS" + P53

    def test_both_sides_get_a_linker(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 n_flank_length=8, c_flank_length=8,
                                 linker_length=4, **FAST)
        assert r.final_sequence == (r.n_flank + "GSGS" + P53 + "GSGS"
                                    + r.c_flank)
        assert r.added_residues == 8 + 8 + 8

    def test_odd_linker_length(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=8, linker_length=5, **FAST)
        assert r.linker == "GSGSG"
        assert len(r.linker) == 5

    def test_explicit_linker_sequence(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=8, linker_length=6,
                                 linker_sequence="GGSGGS", **FAST)
        assert r.linker == "GGSGGS"
        assert r.final_sequence == P53 + "GGSGGS" + r.c_flank

    def test_mismatched_linker_sequence_rejected(self, pdb_path):
        with pytest.raises(ValueError, match="linker_length"):
            build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=8, linker_length=4,
                                 linker_sequence="GGSGGS", **FAST)

    def test_linker_sequence_without_length_rejected(self, pdb_path):
        with pytest.raises(ValueError, match="linker_length is 0"):
            build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=8, linker_sequence="GSGS")

    def test_non_amino_acid_linker_rejected(self, pdb_path):
        with pytest.raises(ValueError, match="non-amino-acid"):
            build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=8, linker_length=4,
                                 linker_sequence="GZGS")

    def test_linker_sequence_is_normalised(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=8, linker_length=4,
                                 linker_sequence="ggss", **FAST)
        assert r.linker == "GGSS"

    def test_negative_linker_rejected(self, pdb_path):
        with pytest.raises(ValueError, match="cannot be negative"):
            build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=8, linker_length=-2)

    def test_no_linker_by_default(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=8, **FAST)
        assert r.linker == ""
        assert r.final_sequence == P53 + r.c_flank

    def test_annotated_sequence_shows_the_linker(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=8, linker_length=4, **FAST)
        assert r.annotated_sequence() == f"{P53}(GSGS)[{r.c_flank}]"

    def test_linker_extends_the_reach(self, pdb_path):
        """The linker is part of the tether, so the flank can reach further."""
        near = build_flanked_binder(pdb_path, binder_chain="B",
                                    target_chain="A", c_flank_length=6,
                                    linker_length=0, **FAST)
        far = build_flanked_binder(pdb_path, binder_chain="B",
                                   target_chain="A", c_flank_length=6,
                                   linker_length=20, **FAST)
        assert far.regions["C"].reach_radius > near.regions["C"].reach_radius

    def test_linker_is_part_of_the_disorder_context(self, pdb_path, monkeypatch):
        import idr_flanks.pipeline as mod
        seen = []
        real = mod.design_flank

        def spy(patch, length, **kwargs):
            seen.append((kwargs.get("n_context", ""), kwargs.get("c_context", "")))
            return real(patch, length, **kwargs)

        monkeypatch.setattr(mod, "design_flank", spy)
        build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                             c_flank_length=8, linker_length=4, **FAST)
        # A C-terminal flank sits after binder + linker, so that is its context.
        assert seen == [(P53 + "GSGS", "")]

    def test_fasta_records_the_linker(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=8, linker_length=4, **FAST)
        assert "linker=GSGS" in r.fasta()


class TestBinderInterfacePassedThrough:
    def test_design_receives_the_binder_interface(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=12, **FAST)
        design = r.designs["C"]
        assert design.binder_interface_sequence == \
               r.regions["C"].binder_interface_sequence
        assert design.epsilon_vs_binder_interface == \
               design.epsilon_vs_binder_interface

    def test_flank_does_not_prefer_the_binder(self, pdb_path):
        r = build_flanked_binder(pdb_path, binder_chain="B", target_chain="A",
                                 c_flank_length=20, seed=3,
                                 max_iterations=300,
                                 num_starting_candidates=120)
        design = r.designs["C"]
        assert (design.epsilon_vs_binder_interface
                > design.epsilon_per_residue)


class TestPublicApiSurface:
    """The package's exports must stay in step with the modules' own __all__,
    so a newly added public function cannot end up unreachable."""

    def test_no_broken_exports(self):
        import idr_flanks
        assert [n for n in idr_flanks.__all__
                if not hasattr(idr_flanks, n)] == []

    def test_every_module_public_name_is_reachable(self):
        import idr_flanks
        from idr_flanks import design, interface, io, pipeline, sasa
        for module in (io, sasa, interface, design, pipeline):
            missing = [n for n in getattr(module, "__all__", [])
                       if not hasattr(idr_flanks, n)]
            assert missing == [], f"{module.__name__}: {missing}"

    def test_feasibility_check_is_public(self):
        """A user needs to test whether a terminus is worth designing for
        before paying for a design run."""
        import idr_flanks
        assert callable(idr_flanks.target_discriminability)

    def test_reference_data_is_public(self):
        import idr_flanks
        assert set(idr_flanks.idr_amino_acid_frequencies()) == set(
            "ACDEFGHIKLMNPQRSTVWY")
        assert idr_flanks.THREE_TO_ONE["MSE"] == "M"
        assert idr_flanks.ATOMIC_RADII["C"] == pytest.approx(1.70)

    def test_import_does_not_pull_in_goose(self):
        """Structure work must not require the heavy optional stack."""
        import subprocess
        import sys as _sys
        out = subprocess.run(
            [_sys.executable, "-c",
             "import sys, idr_flanks; "
             "print('goose' in sys.modules, 'metapredict' in sys.modules)"],
            capture_output=True, text=True, cwd=".")
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "False False"
