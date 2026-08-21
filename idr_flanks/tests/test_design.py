"""Tests for flank design.

These exercise GOOSE, FINCHES, and metapredict, so they are slower than the
rest of the suite. Iteration counts are kept low deliberately: the point is to
pin down behaviour and guardrails, not to produce publication-grade designs.
"""

import math
import random

import numpy as np
import pytest

goose = pytest.importorskip("goose", reason="GOOSE is needed to design flanks")

from idr_flanks.design import (  # noqa: E402
    PRESETS,
    DesignConfig,
    DesignError,
    context_disorder_class,
    design_flank,
    epsilon_per_residue,
    load_epsilon_model,
    score_flank,
)

# A real proximal patch taken from 1YCR: the MDM2 surface a flank on the p53
# peptide's N-terminus can reach. Basic and polar, so a complementary flank
# should come out acidic.
PATCH = "VKFYGQMTKRYDEKQQHIYSNGVPSSKEHRKYT"
BINDER = "ETFSDLWKLLPEN"
# MDM2, standing in for a folded binder when the point is sequence context.
FOLDED_BINDER = ("ETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCS"
                 "NDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV")
# An actual flank produced by maximising attraction with no guardrails: it
# reads as fully disordered alone and fully ordered once fused.
AROMATIC_TRAP = "WWYDWWWWWWWEFWWYDWEDWWWEYDDDEW"
FAST = dict(max_iterations=120, num_starting_candidates=60)


@pytest.fixture(scope="module")
def model():
    return load_epsilon_model("mpipi")


class TestEpsilonHelpers:
    def test_sign_convention(self, model):
        """Negative epsilon is attractive. Everything downstream assumes it."""
        assert epsilon_per_residue("K" * 20, "E" * 20) < 0
        assert epsilon_per_residue("K" * 20, "K" * 20) > 0

    def test_per_residue_is_length_comparable(self, model):
        """Raw epsilon scales with the designed sequence's length."""
        a = epsilon_per_residue("E" * 20, "K" * 20)
        b = epsilon_per_residue("E" * 80, "K" * 20)
        assert a == pytest.approx(b, rel=0.02)

    def test_model_is_cached(self):
        assert load_epsilon_model("mpipi") is load_epsilon_model("mpipi")

    def test_unknown_model_rejected(self):
        with pytest.raises(ValueError):
            load_epsilon_model("not-a-model")

    def test_empty_inputs_rejected(self):
        with pytest.raises(ValueError):
            epsilon_per_residue("", PATCH)
        with pytest.raises(ValueError):
            epsilon_per_residue("EEEE", "")


class TestScoreFlank:
    def test_reports_expected_keys(self):
        s = score_flank("EEQDDQQQQWDEEEQWDDQQ", PATCH, c_context=BINDER)
        for key in ("epsilon_total", "epsilon_per_residue",
                    "self_epsilon_per_residue", "fraction_disorder",
                    "fraction_disorder_in_context", "aromatic_fraction",
                    "specificity_z", "specificity_delta",
                    "reference_epsilon_per_residue"):
            assert key in s

    def test_aromatic_fraction_is_right(self):
        s = score_flank("WWFFYYGGGGGGGGGGGGGG", PATCH)
        assert s["aromatic_fraction"] == pytest.approx(6 / 20)

    def test_context_changes_disorder(self):
        """Disorder prediction depends on neighbours, which is why the flank is
        scored both alone and fused.

        AROMATIC_TRAP is a real sequence produced by maximising attraction with
        no composition guardrails. On its own metapredict calls every residue
        disordered; fused to a folded binder it calls none of them disordered.
        Scoring only the isolated flank would let this design through.
        """
        s = score_flank(AROMATIC_TRAP, PATCH, c_context=FOLDED_BINDER)
        assert s["fraction_disorder"] == pytest.approx(1.0)
        assert s["fraction_disorder_in_context"] == pytest.approx(0.0)

    def test_context_effect_is_side_independent(self):
        n_side = score_flank(AROMATIC_TRAP, PATCH, n_context=FOLDED_BINDER)
        c_side = score_flank(AROMATIC_TRAP, PATCH, c_context=FOLDED_BINDER)
        assert n_side["fraction_disorder_in_context"] < 0.2
        assert c_side["fraction_disorder_in_context"] < 0.2

    def test_no_context_means_both_agree(self):
        s = score_flank("EEQDDQQQQWDEEEQWDDQQ", PATCH)
        assert s["fraction_disorder"] == s["fraction_disorder_in_context"]

    def test_kappa_undefined_is_nan_not_sentinel(self):
        """sparrow returns -1 for undefined kappa; that must not leak out."""
        s = score_flank("D" * 20, PATCH)
        assert not (s.get("kappa", float("nan")) == -1.0)

    def test_specificity_is_reproducible(self):
        a = score_flank("EEQDDQQQQWDEEEQWDDQQ", PATCH, seed=3)
        b = score_flank("EEQDDQQQQWDEEEQWDDQQ", PATCH, seed=3)
        assert a["specificity_z"] == b["specificity_z"]

    def test_cross_reactivity_reported(self):
        s = score_flank("D" * 20, PATCH)
        assert s["epsilon_vs_basic"] < s["epsilon_vs_acidic"]

    def test_generic_stickiness_is_detected(self):
        """A poly-aromatic sequence attracts random sequence, an acidic one
        much less. This is what separates real complementarity from stickiness."""
        sticky = score_flank("W" * 30, PATCH)
        picky = score_flank("D" * 30, PATCH)
        assert sticky["decoy_epsilon_mean"] < picky["decoy_epsilon_mean"]

    def test_rejects_empty_sequence(self):
        with pytest.raises(ValueError):
            score_flank("", PATCH)


class TestConfig:
    def test_presets_exist(self):
        assert {"balanced", "aggressive", "soluble",
                "unconstrained"} <= set(PRESETS)

    def test_every_preset_is_constructible(self):
        for name, kwargs in PRESETS.items():
            cfg = DesignConfig(**kwargs)
            assert isinstance(cfg, DesignConfig), name

    def test_default_caps_aromatics(self):
        ranges = DesignConfig().aa_fraction_ranges(30)
        assert "WFY" in ranges

    def test_unconstrained_has_no_ranges(self):
        assert DesignConfig(**PRESETS["unconstrained"]).aa_fraction_ranges(30) is None

    def test_bounds_are_exact_despite_ceil(self):
        """GOOSE turns fractions into counts with ceil, so the requested cap
        has to be rounded down first or it leaks an extra residue."""
        for length in (10, 13, 20, 30, 50, 80):
            for cap in (0.05, 0.10, 0.15, 0.25):
                cfg = DesignConfig(max_aromatic_fraction=cap)
                high = cfg.aa_fraction_ranges(length)["WFY"][1]
                assert math.ceil(high * length) == math.floor(cap * length)

    def test_unknown_option_rejected(self):
        with pytest.raises(TypeError, match="Unknown design option"):
            design_flank(PATCH, 10, not_a_real_option=1)

    def test_preset_and_config_conflict(self):
        with pytest.raises(ValueError, match="either preset or config"):
            design_flank(PATCH, 10, preset="balanced", config=DesignConfig())

    def test_unknown_preset(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            design_flank(PATCH, 10, preset="nope")


class TestCompositionEnvelope:
    """The envelope is the package's main guardrail, so pin its behaviour."""

    def test_frequencies_are_a_distribution(self):
        from idr_flanks.design import idr_amino_acid_frequencies
        freqs = idr_amino_acid_frequencies()
        assert set(freqs) == set("ACDEFGHIKLMNPQRSTVWY")
        assert sum(freqs.values()) == pytest.approx(1.0, abs=1e-6)

    def test_matches_goose_reference(self):
        """We must be using GOOSE's real IDR composition, not a guess."""
        from goose.data.aa_list_probabilities import IDRProbs
        from idr_flanks.design import idr_amino_acid_frequencies
        assert idr_amino_acid_frequencies() == pytest.approx(dict(IDRProbs))

    def test_envelope_caps_every_residue(self):
        ranges = DesignConfig().aa_fraction_ranges(30)
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            assert aa in ranges, aa

    def test_envelope_scales_with_multiplier(self):
        tight = DesignConfig(composition_envelope=2.0).aa_fraction_ranges(100)
        loose = DesignConfig(composition_envelope=4.0).aa_fraction_ranges(100)
        assert tight["S"][1] < loose["S"][1]

    def test_envelope_can_be_disabled(self):
        cfg = DesignConfig(composition_envelope=None, max_aromatic_fraction=None,
                           max_aliphatic_fraction=None)
        assert cfg.aa_fraction_ranges(30) is None

    def test_group_caps_present(self):
        ranges = DesignConfig().aa_fraction_ranges(30)
        assert "WFY" in ranges
        assert "AILMV" in ranges

    def test_no_single_residue_dominates(self):
        """Capping only aromatics relocates the pathology to another residue,
        so the envelope has to bound all twenty."""
        from collections import Counter
        r = design_flank(PATCH, 30, c_context=BINDER, seed=7,
                         max_iterations=300, num_starting_candidates=100)
        top = Counter(r.sequence).most_common(1)[0][1]
        assert top / 30 <= 0.30

    def test_envelope_beats_aromatic_cap_alone_on_dominance(self):
        from collections import Counter

        def dominance(**kw):
            r = design_flank(PATCH, 30, c_context=BINDER, seed=7,
                             max_iterations=300, num_starting_candidates=100,
                             **kw)
            return Counter(r.sequence).most_common(1)[0][1] / 30

        with_envelope = dominance()
        aromatic_only = dominance(composition_envelope=None,
                                  max_aliphatic_fraction=None,
                                  max_aromatic_fraction=0.10)
        assert with_envelope < aromatic_only

    def test_envelope_keeps_the_flank_soluble(self):
        r = design_flank(PATCH, 30, c_context=BINDER, seed=7,
                         max_iterations=300, num_starting_candidates=100)
        assert r.self_epsilon_per_residue > 0

    def test_envelope_keeps_attraction_on_target(self):
        """Without the envelope the flank attracts random sequence strongly."""
        kw = dict(max_iterations=300, num_starting_candidates=100, seed=7)
        tight = design_flank(PATCH, 30, c_context=BINDER, **kw)
        loose = design_flank(PATCH, 30, c_context=BINDER,
                             preset="unconstrained", **kw)
        assert tight.decoy_epsilon_mean > loose.decoy_epsilon_mean


class TestSpecificityTerm:
    def test_off_by_default(self):
        assert DesignConfig().max_decoy_epsilon_per_residue is None

    def test_specific_preset_enables_it(self):
        cfg = DesignConfig(**PRESETS["specific"])
        assert cfg.max_decoy_epsilon_per_residue == 0.0

    def test_enabling_it_keeps_decoy_attraction_near_zero(self):
        r = design_flank(PATCH, 25, c_context=BINDER, seed=7,
                         max_decoy_epsilon_per_residue=0.0,
                         max_iterations=200, num_starting_candidates=60)
        assert r.decoy_epsilon_mean > -0.05

    def test_property_computes_the_panel_mean(self):
        import numpy as np
        from sparrow import Protein
        from idr_flanks.design import decoy_repulsion_class
        model = load_epsilon_model("mpipi")
        decoys = ["GSGSGSGSGS", "KEKEKEKEKE", "QNQNQNQNQN"]
        prop = decoy_repulsion_class()(decoys=decoys, model=model,
                                       target_value=0.0)
        seq = "DDDDEEEEDDDDEEEE"
        expected = np.mean([model.epsilon(seq, d) for d in decoys])
        assert prop.calculate_raw_value(Protein(seq)) == pytest.approx(expected)


class TestDesignFlank:
    def test_returns_requested_length(self):
        r = design_flank(PATCH, 24, c_context=BINDER, seed=1, **FAST)
        assert len(r.sequence) == 24
        assert set(r.sequence) <= set("ACDEFGHIKLMNPQRSTVWY")

    def test_is_attractive_to_the_patch(self):
        r = design_flank(PATCH, 24, c_context=BINDER, seed=1, **FAST)
        assert r.epsilon_per_residue < 0
        # and more attractive than a background-composition sequence
        assert r.epsilon_per_residue < r.reference_epsilon_per_residue

    def test_complements_a_basic_patch_with_acid(self):
        """The whole premise: chemistry should be complementary."""
        r = design_flank("KRKRKRKKRRKKRKRKKR", 24, c_context=BINDER,
                         seed=1, **FAST)
        acidic = sum(r.sequence.count(a) for a in "DE")
        basic = sum(r.sequence.count(a) for a in "KR")
        assert acidic > basic

    def test_complements_an_acidic_patch_with_base(self):
        r = design_flank("DEDEEDDEEDDEEDDEED", 24, c_context=BINDER,
                         seed=1, **FAST)
        acidic = sum(r.sequence.count(a) for a in "DE")
        basic = sum(r.sequence.count(a) for a in "KR")
        assert basic > acidic

    def test_is_disordered(self):
        r = design_flank(PATCH, 24, c_context=BINDER, seed=1, **FAST)
        assert r.fraction_disorder >= 0.8
        assert r.fraction_disorder_in_context >= 0.8

    def test_respects_the_aromatic_cap(self):
        for cap in (0.0, 0.05, 0.10):
            r = design_flank(PATCH, 30, c_context=BINDER, seed=2,
                             max_aromatic_fraction=cap, **FAST)
            n = sum(r.sequence.count(a) for a in "WFY")
            assert n <= math.floor(cap * 30), (cap, r.sequence)

    def test_default_is_not_self_attractive(self):
        """The default guardrails should give a soluble, non-aggregating flank."""
        r = design_flank(PATCH, 30, c_context=BINDER, seed=4, **FAST)
        assert r.self_epsilon_per_residue > 0

    def test_unconstrained_is_more_aromatic_than_default(self):
        tight = design_flank(PATCH, 30, c_context=BINDER, seed=4,
                             preset="balanced", **FAST)
        loose = design_flank(PATCH, 30, c_context=BINDER, seed=4,
                             preset="unconstrained", **FAST)
        assert loose.aromatic_fraction >= tight.aromatic_fraction

    def test_reachable_target_is_met(self):
        r = design_flank(PATCH, 30, c_context=BINDER, seed=5,
                         target_epsilon_per_residue=-0.10,
                         max_iterations=400, num_starting_candidates=100)
        # A MAXIMUM constraint: at least as attractive as asked for.
        assert r.epsilon_per_residue <= -0.10 + 0.02

    def test_seed_is_reproducible(self):
        a = design_flank(PATCH, 20, c_context=BINDER, seed=9, **FAST)
        b = design_flank(PATCH, 20, c_context=BINDER, seed=9, **FAST)
        assert a.sequence == b.sequence

    def test_different_seeds_differ(self):
        a = design_flank(PATCH, 30, c_context=BINDER, seed=1, **FAST)
        b = design_flank(PATCH, 30, c_context=BINDER, seed=2, **FAST)
        assert a.sequence != b.sequence

    def test_caller_rng_state_is_restored(self):
        random.seed(4321)
        expected = [random.random() for _ in range(3)]
        random.seed(4321)
        design_flank(PATCH, 15, c_context=BINDER, seed=11, **FAST)
        assert [random.random() for _ in range(3)] == expected

    def test_numpy_rng_state_is_restored(self):
        np.random.seed(4321)
        expected = np.random.rand(3).copy()
        np.random.seed(4321)
        design_flank(PATCH, 15, c_context=BINDER, seed=11, **FAST)
        assert np.allclose(np.random.rand(3), expected)

    def test_summary_mentions_the_essentials(self):
        r = design_flank(PATCH, 20, c_context=BINDER, seed=1, **FAST)
        text = r.summary()
        for expected in ("epsilon vs patch", "self-epsilon",
                         "fraction disordered", "specificity"):
            assert expected in text

    def test_str_is_the_sequence(self):
        r = design_flank(PATCH, 20, c_context=BINDER, seed=1, **FAST)
        assert str(r) == r.sequence

    def test_rejects_empty_patch(self):
        with pytest.raises(ValueError):
            design_flank("", 20)

    def test_rejects_nonpositive_length(self):
        with pytest.raises(ValueError):
            design_flank(PATCH, 0)


class TestContextDisorderProperty:
    def test_supports_batching(self):
        """Batching is what keeps context-aware disorder affordable."""
        cls = context_disorder_class()
        assert cls.calculate_in_batch is True

    def test_batch_matches_single(self):
        from sparrow import Protein
        cls = context_disorder_class()
        prop = cls(n_context="", c_context=BINDER, target_value=1.0)
        seqs = ["EEQDDQQQQWDEEEQWDDQQ", "W" * 20, "GS" * 10]
        proteins = [Protein(s) for s in seqs]
        batch = prop.calculate_raw_value_batch(proteins)
        single = [prop.calculate_raw_value(p) for p in proteins]
        assert batch == pytest.approx(single)

    def test_scores_the_segment_only(self):
        """The property must report the flank's disorder, not the construct's."""
        from sparrow import Protein
        cls = context_disorder_class()
        flank = "GS" * 10
        # A long ordered context would drag a whole-construct fraction far
        # below 1.0; scoring only the segment keeps it high.
        prop = cls(c_context=FOLDED_BINDER, target_value=1.0)
        assert prop.calculate_raw_value(Protein(flank)) > 0.9

    def test_reports_the_context_penalty(self):
        from sparrow import Protein
        cls = context_disorder_class()
        bare = cls(target_value=1.0)
        fused = cls(c_context=FOLDED_BINDER, target_value=1.0)
        p = Protein(AROMATIC_TRAP)
        assert bare.calculate_raw_value(p) == pytest.approx(1.0)
        assert fused.calculate_raw_value(p) == pytest.approx(0.0)

    def test_granularity_is_one_over_length(self):
        """The value must be a fraction of the flank, so its resolution is 1/L."""
        from sparrow import Protein
        cls = context_disorder_class()
        prop = cls(c_context="LLVILLVAILVLLAVILLVAILV", target_value=1.0)
        value = prop.calculate_raw_value(Protein("GS" * 10))
        assert (value * 20) == pytest.approx(round(value * 20))

    def test_class_name_is_preserved(self):
        """GOOSE keys properties by class name; a custom tracking name would
        make properties collide and silently drop out of the objective."""
        cls = context_disorder_class()
        prop = cls(c_context=BINDER, target_value=1.0)
        assert prop.tracking_property_name == cls.__name__


class TestWarnings:
    def test_warns_about_a_tiny_patch(self):
        """A two-residue patch cannot support a meaningful design."""
        r = design_flank("RN", 15, c_context=BINDER, seed=3, **FAST)
        assert any("only 2 residue" in w for w in r.warnings)

    def test_warns_about_low_patch_diversity(self):
        r = design_flank("KKKKKKKKKK", 15, c_context=BINDER, seed=3, **FAST)
        assert any("distinct residue" in w for w in r.warnings)

    def test_no_tiny_patch_warning_for_a_real_patch(self):
        r = design_flank(PATCH, 20, c_context=BINDER, seed=3, **FAST)
        assert not any("residue(s) long" in w for w in r.warnings)

    def test_warns_about_generic_stickiness(self):
        """An unconstrained design should be flagged if it goes sticky."""
        r = design_flank(PATCH, 30, c_context=BINDER, seed=3,
                         preset="unconstrained", max_iterations=300,
                         num_starting_candidates=100)
        if r.self_epsilon_per_residue < 0 or r.decoy_epsilon_mean < -0.1:
            assert r.warnings

    def test_no_spurious_warnings_on_a_good_design(self):
        r = design_flank(PATCH, 30, c_context=BINDER, seed=4,
                         max_iterations=400, num_starting_candidates=100)
        assert not any("aggregate" in w for w in r.warnings)
