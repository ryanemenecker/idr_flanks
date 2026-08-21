"""Tests for the Shrake-Rupley solvent-accessibility implementation."""

import numpy as np
import pytest

from idr_flanks.data import structure_path
from idr_flanks.io import read_structure
from idr_flanks.sasa import (
    ATOMIC_RADII,
    MAX_SASA,
    relative_residue_sasa,
    residue_sasa,
    solvent_accessible_mask,
    sphere_points,
)

# Residues lining the p53-binding cleft of MDM2. These are the residues the
# p53 peptide buries, and are documented as the hydrophobic cleft in the 1YCR
# literature.
MDM2_CLEFT = {54, 62, 72, 93, 96, 100}


@pytest.fixture(scope="module")
def complex_():
    return read_structure(structure_path("1ycr.pdb"))


@pytest.fixture(scope="module")
def chains(complex_):
    return list(complex_["A"].residues), list(complex_["B"].residues)


class TestSpherePoints:
    def test_unit_vectors(self):
        pts = sphere_points(500)
        assert pts.shape == (500, 3)
        assert np.allclose(np.linalg.norm(pts, axis=1), 1.0)

    def test_roughly_centred(self):
        # An even distribution has a centroid near the origin.
        pts = sphere_points(2000)
        assert np.linalg.norm(pts.mean(axis=0)) < 0.05

    def test_cached(self):
        assert sphere_points(64) is sphere_points(64)

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            sphere_points(0)


class TestRadii:
    def test_matches_mdtraj_values(self):
        # mdtraj stores these in nanometres; ours are angstroms.
        mdtraj_nm = {"C": 0.17, "N": 0.155, "O": 0.152,
                     "S": 0.18, "Se": 0.19, "H": 0.12, "P": 0.18}
        for element, nm in mdtraj_nm.items():
            assert ATOMIC_RADII[element] == pytest.approx(nm * 10.0)

    def test_max_sasa_covers_twenty(self):
        assert set(MAX_SASA) == set("ACDEFGHIKLMNPQRSTVWY")

    def test_glycine_has_no_sidechain_area(self):
        assert MAX_SASA["G"][0] == 0.0


class TestAgainstMdtraj:
    """The implementation is only trustworthy if it reproduces a reference."""

    def test_matches_mdtraj_per_residue(self, complex_):
        md = pytest.importorskip("mdtraj")
        path = structure_path("1ycr.pdb")
        traj = md.load(path)
        traj = traj.atom_slice(traj.top.select("protein and not element H"))
        reference = md.shrake_rupley(traj, mode="residue", probe_radius=0.14,
                                    n_sphere_points=960)[0] * 100.0

        residues = [r for ch in complex_ for r in ch]
        assert len(residues) == len(reference)
        mine = residue_sasa(residues, n_points=960)

        # Both are Monte-Carlo-style point samplings of the same surface, so
        # they agree closely but not exactly.
        assert np.corrcoef(mine, reference)[0, 1] > 0.999
        assert np.abs(mine - reference).mean() < 2.0
        assert np.abs(mine - reference).max() < 12.0


class TestBuriedSurface:
    def test_binder_buries_target_surface(self, chains):
        a, b = chains
        isolated = residue_sasa(a)
        in_complex = residue_sasa(a, context=b)
        # Context can only ever occlude, never expose.
        assert np.all(in_complex <= isolated + 1e-6)
        buried = (isolated - in_complex).sum()
        # The 1YCR interface buries several hundred square angstroms per side.
        assert 300 < buried < 1200

    def test_most_buried_residues_are_the_known_cleft(self, chains):
        a, b = chains
        buried = residue_sasa(a) - residue_sasa(a, context=b)
        top = {a[i].seq_id for i in np.argsort(-buried)[:8]}
        # The known cleft residues should dominate the buried list.
        assert len(top & MDM2_CLEFT) >= 5

    def test_context_does_not_change_reported_length(self, chains):
        a, b = chains
        assert residue_sasa(a, context=b).shape == (len(a),)

    def test_empty_input(self):
        assert residue_sasa([]).shape == (0,)


class TestRelativeSasa:
    def test_all_finite(self, chains):
        a, b = chains
        rel = relative_residue_sasa(a, context=b)
        assert np.all(np.isfinite(rel))

    def test_glycine_normalised_by_backbone(self, chains):
        """Pin the exact deviation from FINCHES: glycine is divided by its
        BACKBONE reference, since its sidechain reference is zero."""
        a, _ = chains
        absolute = residue_sasa(a)
        rel = relative_residue_sasa(a)
        gly = [i for i, r in enumerate(a) if r.resname == "GLY"]
        assert gly, "1YCR chain A should contain glycines"
        backbone_ref = MAX_SASA["G"][1]
        for i in gly:
            assert rel[i] == pytest.approx(absolute[i] / backbone_ref)
        assert MAX_SASA["G"][0] == 0.0, "sidechain reference must be zero"

    def test_glycine_can_be_buried(self, chains):
        """Glycine must be able to read as buried, not always as surface.

        FINCHES treats every glycine as accessible because its sidechain
        reference area is zero; normalising by the backbone reference instead
        lets a genuinely occluded glycine be recognised. G58 of MDM2 sits at
        the p53 interface: exposed in the free protein, buried once bound.
        """
        a, b = chains
        alone = relative_residue_sasa(a)
        bound = relative_residue_sasa(a, context=b)
        idx = next(i for i, r in enumerate(a) if r.seq_id == 58)
        assert a[idx].resname == "GLY"
        assert alone[idx] > 0.10
        assert bound[idx] < 0.01

    def test_core_residues_read_as_buried(self, chains):
        a, _ = chains
        rel = relative_residue_sasa(a)
        # A folded 85-residue domain has a real hydrophobic core.
        assert (rel < 0.05).sum() >= 5
        assert rel.min() < 0.01

    def test_total_mode_is_strictly_smaller(self, chains):
        """A larger normalising reference must give strictly smaller ratios,
        so the two modes cannot be the same calculation."""
        a, _ = chains
        side = relative_residue_sasa(a, mode="sidechain")
        total = relative_residue_sasa(a, mode="total")
        non_gly = [i for i, r in enumerate(a)
                   if r.resname != "GLY" and side[i] > 0]
        assert non_gly
        assert np.all(total[non_gly] < side[non_gly])

    def test_total_mode_divides_by_sidechain_plus_backbone(self, chains):
        a, _ = chains
        absolute = residue_sasa(a)
        total = relative_residue_sasa(a, mode="total")
        for i, res in enumerate(a[:12]):
            side, back = MAX_SASA[res.one_letter]
            assert total[i] == pytest.approx(absolute[i] / (side + back))

    def test_rejects_bad_mode(self, chains):
        a, _ = chains
        with pytest.raises(ValueError):
            relative_residue_sasa(a, mode="nonsense")


class TestProbeRadius:
    """probe_radius must actually reach the calculation."""

    def test_probe_radius_changes_the_result(self, chains):
        """The parameter must reach the calculation.

        Total SASA is not monotonic in probe radius for a packed protein: a
        small probe follows every crevice (large area), a mid-sized probe is
        excluded from them (minimum near 1.4-2.0 A), and a very large probe
        envelops the whole molecule in a smooth surface that grows again.
        """
        a, _ = chains
        areas = {p: residue_sasa(a, probe_radius=p, n_points=480).sum()
                 for p in (0.0, 0.7, 1.4, 2.0, 3.0)}
        assert len(set(round(v, 3) for v in areas.values())) == len(areas)
        # A zero-radius probe reaches into crevices a water-sized one cannot.
        assert areas[0.0] > areas[1.4] * 1.5
        assert areas[0.7] > areas[1.4]

    def test_matches_analytic_sphere_for_a_lone_atom(self):
        """With one atom the area is exactly 4*pi*(r+probe)^2."""
        from idr_flanks.io import Atom, Residue
        for probe in (0.0, 1.4, 2.5):
            res = Residue("ALA", 1, " ", "A")
            res.atoms.append(Atom("CA", "C", np.array([0.0, 0.0, 0.0])))
            got = residue_sasa([res], probe_radius=probe, n_points=2000)[0]
            expected = 4 * np.pi * (ATOMIC_RADII["C"] + probe) ** 2
            assert got == pytest.approx(expected, rel=1e-9)

    def test_matches_mdtraj_at_a_nondefault_probe(self, complex_):
        md = pytest.importorskip("mdtraj")
        traj = md.load(structure_path("1ycr.pdb"))
        traj = traj.atom_slice(traj.top.select("protein and not element H"))
        reference = md.shrake_rupley(traj, mode="residue", probe_radius=0.20,
                                    n_sphere_points=960)[0] * 100.0
        residues = [r for ch in complex_ for r in ch]
        mine = residue_sasa(residues, probe_radius=2.0, n_points=960)
        assert np.corrcoef(mine, reference)[0, 1] > 0.999
        assert np.abs(mine - reference).mean() < 3.0


class TestPointSamplingAccuracy:
    """Validate the sampled area against a closed form with real occlusion,
    not just the zero-neighbour short-circuit."""

    @staticmethod
    def _two_atom_analytic(d, r1, r2):
        """Exposed area of two overlapping spheres of radii r1, r2 at separation d."""
        if d >= r1 + r2:
            return 4 * np.pi * (r1 ** 2 + r2 ** 2)
        # Spherical cap heights removed from each sphere by the intersection.
        h1 = r1 - (d * d + r1 * r1 - r2 * r2) / (2 * d)
        h2 = r2 - (d * d + r2 * r2 - r1 * r1) / (2 * d)
        return (4 * np.pi * r1 ** 2 - 2 * np.pi * r1 * h1
                + 4 * np.pi * r2 ** 2 - 2 * np.pi * r2 * h2)

    def test_two_overlapping_atoms(self):
        from idr_flanks.io import Atom, Residue
        probe = 1.4
        r = ATOMIC_RADII["C"] + probe
        for d in (2.0, 3.0, 4.0, 5.0):
            res = Residue("ALA", 1, " ", "A")
            res.atoms.append(Atom("CA", "C", np.array([0.0, 0.0, 0.0])))
            res.atoms.append(Atom("CB", "C", np.array([d, 0.0, 0.0])))
            got = residue_sasa([res], probe_radius=probe, n_points=8000)[0]
            expected = self._two_atom_analytic(d, r, r)
            assert got == pytest.approx(expected, rel=0.02), d

    def test_fully_enclosed_atom_has_no_area(self):
        """A tiny atom inside a large one contributes nothing."""
        from idr_flanks.io import Atom, Residue
        res = Residue("ALA", 1, " ", "A")
        res.atoms.append(Atom("SE", "Se", np.array([0.0, 0.0, 0.0])))
        res.atoms.append(Atom("C", "C", np.array([0.0, 0.0, 0.0])))
        alone = Residue("ALA", 1, " ", "A")
        alone.atoms.append(Atom("SE", "Se", np.array([0.0, 0.0, 0.0])))
        both = residue_sasa([res], n_points=4000)[0]
        just_se = residue_sasa([alone], n_points=4000)[0]
        # The coincident smaller carbon adds nothing outside the selenium.
        assert both == pytest.approx(just_se, rel=0.02)


class TestSphereCache:
    def test_returned_array_is_immutable(self):
        pts = sphere_points(128)
        with pytest.raises(ValueError):
            pts[0, 0] = 99.0


class TestSelfOcclusion:
    def test_overlapping_context_is_ignored(self, chains):
        """Passing the same residues as context must not bury them."""
        a, _ = chains
        assert np.allclose(residue_sasa(a, context=a), residue_sasa(a))

    def test_partial_overlap_ignored(self, chains):
        a, b = chains
        both = residue_sasa(a, context=list(b) + list(a))
        just_b = residue_sasa(a, context=b)
        assert np.allclose(both, just_b)

    def test_genuine_context_still_occludes(self, chains):
        a, b = chains
        assert residue_sasa(a, context=b).sum() < residue_sasa(a).sum()


class TestAccessibleMask:
    def test_threshold_is_monotonic(self, chains):
        a, b = chains
        counts = [solvent_accessible_mask(a, context=b, threshold=t).sum()
                  for t in (0.05, 0.10, 0.20, 0.40)]
        assert counts == sorted(counts, reverse=True)

    def test_some_but_not_all_residues_are_surface(self, chains):
        a, b = chains
        mask = solvent_accessible_mask(a, context=b, threshold=0.10)
        assert 0 < mask.sum() < len(a)

    def test_binding_makes_cleft_residues_less_accessible(self, chains):
        a, b = chains
        alone = solvent_accessible_mask(a, threshold=0.25)
        bound = solvent_accessible_mask(a, context=b, threshold=0.25)
        assert bound.sum() <= alone.sum()

    def test_dtype_is_bool(self, chains):
        a, _ = chains
        assert solvent_accessible_mask(a).dtype == np.bool_


class TestPointCountConvergence:
    def test_more_points_converge(self, chains):
        a, _ = chains
        coarse = residue_sasa(a, n_points=120)
        fine = residue_sasa(a, n_points=960)
        finer = residue_sasa(a, n_points=1920)
        # The finer pair should agree better than the coarse pair does.
        assert (np.abs(fine - finer).mean()
                < np.abs(coarse - finer).mean())

    def test_classification_is_stable_across_point_counts(self, chains):
        a, b = chains
        m1 = solvent_accessible_mask(a, context=b, n_points=240)
        m2 = solvent_accessible_mask(a, context=b, n_points=960)
        # Surface classification should barely move with sampling density.
        assert (m1 == m2).mean() > 0.95


class TestHeteroatomOcclusion:
    """Nucleic acids, glycans and cofactors are not chains, but they take up
    space. Surface they cover is not available to a flank."""

    def test_extra_coords_reduce_accessibility(self, chains):
        a, _ = chains
        base = residue_sasa(a[:20], n_points=240)
        # A wall of atoms right against the first residue.
        centre = a[0].centroid
        blockers = centre + np.array([[3.5, 0.0, 0.0], [0.0, 3.5, 0.0],
                                      [0.0, 0.0, 3.5], [-3.5, 0.0, 0.0]])
        blocked = residue_sasa(a[:20], n_points=240, extra_coords=blockers,
                               extra_elements=["C"] * 4)
        assert blocked[0] < base[0]
        assert np.all(blocked <= base + 1e-9)

    def test_default_element_is_assumed_when_omitted(self, chains):
        a, _ = chains
        centre = a[0].centroid + np.array([[3.5, 0.0, 0.0]])
        with_el = residue_sasa(a[:5], n_points=240, extra_coords=centre,
                               extra_elements=["C"])
        without = residue_sasa(a[:5], n_points=240, extra_coords=centre)
        assert np.allclose(with_el, without)

    def test_mismatched_element_count_is_rejected(self, chains):
        a, _ = chains
        with pytest.raises(ValueError, match="extra_elements"):
            residue_sasa(a[:5], extra_coords=np.zeros((3, 3)),
                         extra_elements=["C"])

    def test_empty_extra_coords_is_a_noop(self, chains):
        a, _ = chains
        base = residue_sasa(a[:10], n_points=240)
        assert np.allclose(
            residue_sasa(a[:10], n_points=240,
                         extra_coords=np.empty((0, 3))), base)

    def test_threaded_through_relative_and_mask(self, chains):
        a, _ = chains
        blockers = a[0].centroid + np.array([[3.5, 0.0, 0.0]])
        rel_a = relative_residue_sasa(a[:10], n_points=240)
        rel_b = relative_residue_sasa(a[:10], n_points=240,
                                      extra_coords=blockers,
                                      extra_elements=["C"])
        assert rel_b[0] < rel_a[0]
        m = solvent_accessible_mask(a[:10], n_points=240,
                                    extra_coords=blockers,
                                    extra_elements=["C"])
        assert m.dtype == np.bool_


class TestScipylessFallback:
    def test_matches_kdtree_result(self, chains, monkeypatch):
        """The brute-force neighbour search must agree with the KD-tree one."""
        import builtins
        a, _ = chains
        residues = a[:15]
        expected = residue_sasa(residues, n_points=240)

        real_import = builtins.__import__

        def no_scipy_spatial(name, *args, **kwargs):
            if name == "scipy.spatial":
                raise ImportError("forced for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_scipy_spatial)
        assert np.allclose(residue_sasa(residues, n_points=240), expected)


class TestIsolatedAtomLimit:
    def test_lone_residue_area_matches_analytic_spheres(self):
        """A residue with no neighbours must give the full sphere areas."""
        from idr_flanks.io import Atom, Residue

        res = Residue("GLY", 1, " ", "A")
        # Two atoms placed far apart so neither occludes the other.
        res.atoms.append(Atom("CA", "C", np.array([0.0, 0.0, 0.0])))
        res.atoms.append(Atom("N", "N", np.array([500.0, 0.0, 0.0])))
        got = residue_sasa([res], n_points=2000)[0]
        expected = (4 * np.pi * (ATOMIC_RADII["C"] + 1.4) ** 2
                    + 4 * np.pi * (ATOMIC_RADII["N"] + 1.4) ** 2)
        assert got == pytest.approx(expected, rel=1e-6)
