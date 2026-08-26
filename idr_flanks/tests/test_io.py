"""Tests for the PDB / mmCIF readers."""

import os
import textwrap

import numpy as np
import pytest

from idr_flanks.data import available_structures, structure_path
from idr_flanks.io import (
    THREE_TO_ONE,
    StructureParseError,
    read_cif,
    read_pdb,
    read_structure,
)

# 1YCR: chain A is MDM2 (25-109), chain B the p53 TAD peptide (17-29).
MDM2 = ("ETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGD"
        "LFGVPSFSVKEHRKIYTMIYRNLVV")
P53 = "ETFSDLWKLLPEN"


@pytest.fixture(scope="module")
def pdb():
    return read_structure(structure_path("1ycr.pdb"))


@pytest.fixture(scope="module")
def cif():
    return read_structure(structure_path("1ycr.cif"))


class TestPackagedData:
    def test_structures_available(self):
        assert "1ycr.pdb" in available_structures()
        assert "1ycr.cif" in available_structures()

    def test_missing_structure_raises(self):
        with pytest.raises(FileNotFoundError):
            structure_path("does_not_exist.pdb")


class TestReadPdb:
    def test_chains(self, pdb):
        assert pdb.chain_ids == ["A", "B"]

    def test_sequences(self, pdb):
        assert pdb["A"].sequence == MDM2
        assert pdb["B"].sequence == P53

    def test_residue_counts(self, pdb):
        assert len(pdb["A"]) == 85
        assert len(pdb["B"]) == 13

    def test_author_numbering_preserved(self, pdb):
        assert pdb["A"][0].seq_id == 25
        assert pdb["A"][-1].seq_id == 109
        assert pdb["B"][0].seq_id == 17
        assert pdb["B"][-1].seq_id == 29

    def test_numbering_is_monotonic(self, pdb):
        for chain in pdb:
            ids = [r.seq_id for r in chain]
            assert ids == sorted(ids)

    def test_no_hydrogens(self, pdb):
        for chain in pdb:
            for res in chain:
                assert all(not a.is_hydrogen for a in res.atoms)

    def test_element_column_takes_precedence_over_the_name(self, tmp_path):
        """Columns 77-78 must actually be read, and must win.

        Pinned with a case where the two disagree: the atom-name field reads
        " CA " (an alpha carbon) while columns 77-78 say CA (calcium). If the
        column were ignored, the element would come out as carbon.
        """
        # Built by column position: the element field is columns 77-78, and a
        # hand-aligned literal is one space out.
        lines = [
            "HETATM    1  CA  UNK X 999      0.000   0.000   0.000  1.00  0.00".ljust(76) + "CA",
            "ATOM      2  CA  ALA A   1      5.000   0.000   0.000  1.00  0.00".ljust(76) + " C",
            "END",
        ]
        path = tmp_path / "prec.pdb"
        path.write_text("\n".join(lines) + "\n")
        s = read_pdb(str(path))
        # the amino acid's CA stays carbon
        assert s["A"][0].atom("CA").element == "C"
        # and the calcium HETATM was read as calcium, hence not an amino acid
        assert s.skipped_residues.get("UNK") == 1
        assert s.heteroatoms()[1] == ["Ca"]

    def test_no_duplicate_atom_names_within_residue(self, pdb):
        for chain in pdb:
            for res in chain:
                names = [a.name for a in res.atoms]
                assert len(names) == len(set(names)), res.label

    def test_heavy_coords_shape(self, pdb):
        res = pdb["A"][0]
        assert res.heavy_coords.shape == (len(res.atoms), 3)
        assert res.heavy_coords.dtype == np.float64

    def test_stacked_coords_owner_index(self, pdb):
        coords, owner = pdb["A"].stacked_heavy_coords()
        assert coords.shape[0] == owner.shape[0]
        assert coords.shape[0] == sum(len(r.atoms) for r in pdb["A"])
        # every owner index maps back to a real residue
        assert owner.max() == len(pdb["A"]) - 1
        # the atoms of residue 0 are the first block
        n0 = len(pdb["A"][0].atoms)
        assert np.all(owner[:n0] == 0)
        assert np.allclose(coords[:n0], pdb["A"][0].heavy_coords)

    def test_stacked_coords_cached(self, pdb):
        a = pdb["A"].stacked_heavy_coords()
        b = pdb["A"].stacked_heavy_coords()
        assert a[0] is b[0]

    def test_residue_index_matches_position(self, pdb):
        for chain in pdb:
            for i, res in enumerate(chain):
                assert res.index == i

    def test_ca_and_cb_lookup(self, pdb):
        res = pdb["A"].residue_by_seq_id(54)  # LEU54, part of the p53 cleft
        assert res is not None
        assert res.resname == "LEU"
        assert res.ca is not None
        assert res.cb_or_ca is not None and res.cb_or_ca.name == "CB"

    def test_glycine_falls_back_to_ca(self, pdb):
        gly = next(r for r in pdb["A"] if r.resname == "GLY")
        assert gly.atom("CB") is None
        assert gly.cb_or_ca.name == "CA"

    def test_residue_labels(self, pdb):
        assert pdb["A"][0].label == "A:25"
        assert pdb["B"][-1].label == "B:29"

    def test_missing_chain_error_lists_options(self, pdb):
        with pytest.raises(KeyError) as exc:
            pdb["Z"]
        assert "A" in str(exc.value) and "B" in str(exc.value)

    def test_centroid_is_mean_of_heavy_atoms(self, pdb):
        res = pdb["A"][0]
        assert np.allclose(res.centroid, res.heavy_coords.mean(axis=0))


class TestReadCif:
    def test_matches_pdb_exactly(self, pdb, cif):
        assert cif.chain_ids == pdb.chain_ids
        for cid in pdb.chain_ids:
            assert cif[cid].sequence == pdb[cid].sequence
            assert len(cif[cid]) == len(pdb[cid])
            assert [(r.seq_id, r.ins_code) for r in cif[cid]] == \
                   [(r.seq_id, r.ins_code) for r in pdb[cid]]

    def test_coordinates_match_pdb(self, pdb, cif):
        for cid in pdb.chain_ids:
            for rp, rc in zip(pdb[cid], cif[cid]):
                assert np.allclose(rp.heavy_coords, rc.heavy_coords, atol=1e-3)

    def test_uses_author_chain_ids(self, cif):
        # label_asym_id and auth_asym_id happen to agree for 1YCR, but the
        # reader must report the author identifiers.
        assert cif.chain_ids == ["A", "B"]

    def test_model_recorded(self, cif):
        assert cif.model == 1

    def test_format_recorded(self, pdb, cif):
        assert pdb.format == "pdb"
        assert cif.format == "cif"


class TestFormatDispatch:
    def test_forced_format(self):
        s = read_structure(structure_path("1ycr.pdb"), fmt="pdb")
        assert s["B"].sequence == P53

    def test_sniffs_pdb_without_extension(self, tmp_path):
        target = tmp_path / "noext"
        target.write_bytes(open(structure_path("1ycr.pdb"), "rb").read())
        assert read_structure(str(target))["B"].sequence == P53

    def test_sniffs_cif_without_extension(self, tmp_path):
        target = tmp_path / "noext_cif"
        target.write_bytes(open(structure_path("1ycr.cif"), "rb").read())
        assert read_structure(str(target))["B"].sequence == P53

    def test_gzip_roundtrip(self, tmp_path):
        import gzip
        gz = tmp_path / "1ycr.pdb.gz"
        with gzip.open(gz, "wt") as fh:
            fh.write(open(structure_path("1ycr.pdb")).read())
        assert read_structure(str(gz))["B"].sequence == P53

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            read_structure("/nonexistent/path/foo.pdb")

    def test_unknown_format_rejected(self):
        with pytest.raises(ValueError):
            read_structure(structure_path("1ycr.pdb"), fmt="xyz")

    def test_unsniffable_file(self, tmp_path):
        junk = tmp_path / "junk.txt"
        junk.write_text("this is not a structure\n" * 5)
        with pytest.raises(StructureParseError):
            read_structure(str(junk))

    def test_cif_without_atom_site(self, tmp_path):
        f = tmp_path / "empty.cif"
        f.write_text("data_TEST\n_entry.id TEST\n#\n")
        with pytest.raises(StructureParseError):
            read_cif(str(f))


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text).lstrip("\n"))
    return str(p)


class TestPdbColumnHandling:
    """The PDB format is column-oriented; these pin the exact field offsets."""

    PDB = """
        ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00 20.00           N
        ATOM      2  CA  ALA A   1      11.639   6.399  -5.147  1.00 20.00           C
        ATOM      3  C   ALA A   1      12.985   5.735  -4.978  1.00 20.00           C
        ATOM      4  O   ALA A   1      13.960   6.199  -5.571  1.00 20.00           O
        ATOM      5  CB  ALA A   1      11.750   7.907  -4.912  1.00 20.00           C
        ATOM      6  N   GLY B   7      13.048   4.618  -4.253  1.00 20.00           N
        ATOM      7  CA  GLY B   7      14.264   3.859  -3.980  1.00 20.00           C
        END
    """

    def test_fields(self, tmp_path):
        s = read_pdb(_write(tmp_path, "cols.pdb", self.PDB))
        assert s.chain_ids == ["A", "B"]
        assert s["A"].sequence == "A"
        assert s["B"].sequence == "G"
        res = s["A"][0]
        assert res.resname == "ALA"
        assert res.seq_id == 1
        assert len(res.atoms) == 5
        n = res.atom("N")
        assert np.allclose(n.xyz, [11.104, 6.134, -6.504])
        assert n.element == "N"
        assert n.occupancy == pytest.approx(1.00)
        assert n.bfactor == pytest.approx(20.00)
        assert n.serial == 1

    def test_negative_residue_numbers(self, tmp_path):
        pdb = """
            ATOM      1  CA  MET A  -2      0.000   0.000   0.000  1.00  0.00           C
            ATOM      2  CA  ALA A  -1      3.800   0.000   0.000  1.00  0.00           C
            ATOM      3  CA  GLY A   0      7.600   0.000   0.000  1.00  0.00           C
            END
        """
        s = read_pdb(_write(tmp_path, "neg.pdb", pdb))
        assert [r.seq_id for r in s["A"]] == [-2, -1, 0]
        assert s["A"].sequence == "MAG"

    def test_non_alphabetic_element_column_is_not_trusted(self, tmp_path):
        """Legacy writers put charges and other junk in columns 77-78.

        Trusting a digit there gave atoms element "1" and let a hydrogen named
        "HB1" survive as a heavy atom, corrupting every distance and area.
        """
        pdb = """
            ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00 20.00           1
            ATOM      2  CA  ALA A   1      11.639   6.399  -5.147  1.00 20.00           2
            ATOM      3 HB1  ALA A   1      11.750   7.907  -4.912  1.00 20.00           1
            ATOM      4  CB  ALA A   1      11.751   7.908  -4.913  1.00 20.00           C
            END
        """
        s = read_pdb(_write(tmp_path, "digit.pdb", pdb))
        assert [(a.name, a.element) for a in s["A"][0].atoms] == [
            ("N", "N"), ("CA", "C"), ("CB", "C")]

    def test_four_character_hydrogen_names_are_not_mercury(self, tmp_path):
        """"HG11" is a gamma hydrogen, not mercury.

        Without the element column, treating columns 13-14 as a two-letter
        symbol reads every Val/Leu/Ile/Thr gamma hydrogen as Hg, which keeps
        them as heavy atoms and corrupts every distance and area downstream.
        """
        pdb = """
            ATOM      1  N   VAL A   1      11.104   6.134  -6.504  1.00 20.00
            ATOM      2  CA  VAL A   1      11.639   6.399  -5.147  1.00 20.00
            ATOM      3  CG1 VAL A   1      12.639   7.399  -5.147  1.00 20.00
            ATOM      4 HG11 VAL A   1      13.000   7.500  -5.100  1.00 20.00
            ATOM      5 HG12 VAL A   1      13.100   7.600  -5.200  1.00 20.00
            ATOM      6 HG21 VAL A   1      13.200   7.700  -5.300  1.00 20.00
            END
        """
        s = read_pdb(_write(tmp_path, "hg.pdb", pdb))
        assert [a.name for a in s["A"][0].atoms] == ["N", "CA", "CG1"]

    def test_real_two_letter_element_still_recognised(self, tmp_path):
        pdb = """
            ATOM      1  CA  MSE A   1      11.639   6.399  -5.147  1.00 20.00
            HETATM    2 SE   MSE A   1      12.639   7.399  -5.147  1.00 20.00
            END
        """
        s = read_pdb(_write(tmp_path, "se.pdb", pdb))
        assert s["A"][0].atom("SE").element == "Se"

    def test_element_inferred_when_column_absent(self, tmp_path):
        # Truncated lines with no element column -- common from MD tools.
        # Selenium is written with column 13 occupied ("SE  "), which is what
        # real depositions do; a right-justified " SE " would mean sulfur.
        pdb = """
            ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00 20.00
            ATOM      2  CA  ALA A   1      11.639   6.399  -5.147  1.00 20.00
            HETATM    3 SE   MSE A   2      12.639   7.399  -5.147  1.00 20.00
            END
        """
        s = read_pdb(_write(tmp_path, "noelem.pdb", pdb))
        assert s["A"][0].atom("N").element == "N"
        assert s["A"][0].atom("CA").element == "C"
        assert s["A"][1].atom("SE").element == "Se"

    def test_atom_name_columns_disambiguate_element(self, tmp_path):
        # " CA " is the alpha carbon; "CA  " in column 13 is a calcium ion.
        # "SE  " is the selenium of selenomethionine, not a sulfur.
        pdb = """
            ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00
            ATOM      2  SD  MET A   2       3.800   0.000   0.000  1.00  0.00
            HETATM    3 SE   MSE A   3       7.600   0.000   0.000  1.00  0.00
            HETATM    4 CA    CA A 400      50.000  50.000  50.000  1.00  0.00
            END
        """
        s = read_pdb(_write(tmp_path, "elem.pdb", pdb))
        assert s["A"][0].atom("CA").element == "C"
        assert s["A"][1].atom("SD").element == "S"
        assert s["A"][2].atom("SE").element == "Se"
        # the free calcium ion is not an amino acid, so it is skipped entirely
        assert s["A"].sequence == "AMM"

    def test_hydrogens_dropped_including_numeric_names(self, tmp_path):
        pdb = """
            ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00 20.00           N
            ATOM      2  CA  ALA A   1      11.639   6.399  -5.147  1.00 20.00           C
            ATOM      3 1HB  ALA A   1      11.750   7.907  -4.912  1.00 20.00           H
            ATOM      4 2HB  ALA A   1      11.751   7.908  -4.913  1.00 20.00
            END
        """
        s = read_pdb(_write(tmp_path, "h.pdb", pdb))
        assert [a.name for a in s["A"][0].atoms] == ["N", "CA"]

    def test_solvent_and_ligands_skipped(self, tmp_path):
        pdb = """
            ATOM      1  CA  ALA A   1      11.639   6.399  -5.147  1.00 20.00           C
            HETATM    2  O   HOH A 100      20.000  20.000  20.000  1.00 20.00           O
            HETATM    3 FE   HEM A 200      30.000  30.000  30.000  1.00 20.00          FE
            END
        """
        s = read_pdb(_write(tmp_path, "solv.pdb", pdb))
        assert s["A"].sequence == "A"
        assert len(s["A"]) == 1
        # water is expected and not reported; an unknown ligand is reported
        assert "HOH" not in s.skipped_residues
        assert s.skipped_residues.get("HEM") == 1

    def test_blank_chain_id(self, tmp_path):
        pdb = """
            ATOM      1  CA  ALA     1      11.639   6.399  -5.147  1.00 20.00           C
            ATOM      2  CA  GLY     2      15.639   6.399  -5.147  1.00 20.00           C
            END
        """
        s = read_pdb(_write(tmp_path, "blank.pdb", pdb))
        assert s.chain_ids == [" "]
        assert s[" "].sequence == "AG"

    def test_malformed_coordinates_raise(self, tmp_path):
        pdb = """
            ATOM      1  CA  ALA A   1      abcdefg   6.399  -5.147  1.00 20.00           C
            END
        """
        with pytest.raises(StructureParseError):
            read_pdb(_write(tmp_path, "bad.pdb", pdb))

    def test_malformed_residue_number_raises(self, tmp_path):
        pdb = """
            ATOM      1  CA  ALA A   x      11.639   6.399  -5.147  1.00 20.00           C
            END
        """
        with pytest.raises(StructureParseError):
            read_pdb(_write(tmp_path, "badnum.pdb", pdb))


class TestInsertionCodes:
    PDB = """
        ATOM      1  CA  GLY A  52      0.000   0.000   0.000  1.00  0.00           C
        ATOM      2  CA  PRO A  52A     3.800   0.000   0.000  1.00  0.00           C
        ATOM      3  CA  SER A  52B     7.600   0.000   0.000  1.00  0.00           C
        ATOM      4  CA  ALA A  53     11.400   0.000   0.000  1.00  0.00           C
        END
    """

    def test_insertion_codes_are_distinct_residues(self, tmp_path):
        s = read_pdb(_write(tmp_path, "ins.pdb", self.PDB))
        assert len(s["A"]) == 4
        assert s["A"].sequence == "GPSA"
        assert [r.label for r in s["A"]] == ["A:52", "A:52A", "A:52B", "A:53"]

    def test_lookup_by_seq_id_and_ins_code(self, tmp_path):
        s = read_pdb(_write(tmp_path, "ins2.pdb", self.PDB))
        assert s["A"].residue_by_seq_id(52).resname == "GLY"
        assert s["A"].residue_by_seq_id(52, "A").resname == "PRO"
        assert s["A"].residue_by_seq_id(52, "B").resname == "SER"
        assert s["A"].residue_by_seq_id(99) is None

    def test_insertion_codes_sort_after_bare_number(self, tmp_path):
        # Same residues, shuffled in the file: ordering must be repaired.
        shuffled = """
            ATOM      4  CA  ALA A  53     11.400   0.000   0.000  1.00  0.00           C
            ATOM      2  CA  PRO A  52A     3.800   0.000   0.000  1.00  0.00           C
            ATOM      1  CA  GLY A  52      0.000   0.000   0.000  1.00  0.00           C
            ATOM      3  CA  SER A  52B     7.600   0.000   0.000  1.00  0.00           C
            END
        """
        s = read_pdb(_write(tmp_path, "ins3.pdb", shuffled))
        assert [r.label for r in s["A"]] == ["A:52", "A:52A", "A:52B", "A:53"]
        assert s["A"].sequence == "GPSA"
        assert s.warnings


class TestAltLoc:
    PDB = """
        ATOM      1  N  ATHR A   1       3.265 -14.107  16.877  0.87  4.71           N
        ATOM      2  N  BTHR A   1       4.046 -14.111  17.614  0.25  9.35           N
        ATOM      3  CA ATHR A   1       4.047 -12.839  16.901  0.30  3.80           C
        ATOM      4  CA BTHR A   1       4.261 -12.797  17.017  0.70  4.94           C
        END
    """

    def test_highest_occupancy_wins(self, tmp_path):
        s = read_pdb(_write(tmp_path, "alt.pdb", self.PDB))
        res = s["A"][0]
        assert len(res.atoms) == 2
        assert res.atom("N").altloc == "A"      # 0.87 > 0.25
        assert res.atom("CA").altloc == "B"     # 0.70 > 0.30

    def test_keep_first_mode(self, tmp_path):
        s = read_pdb(_write(tmp_path, "alt2.pdb", self.PDB),
                     keep_altloc="first")
        res = s["A"][0]
        assert res.atom("N").altloc == "A"
        assert res.atom("CA").altloc == "A"     # first in file, despite 0.30

    def test_tie_broken_deterministically(self, tmp_path):
        """Equal occupancy must resolve the same way in either file order."""
        b_first = """
            ATOM      1  CA BALA A   1       1.000   0.000   0.000  0.50  0.00           C
            ATOM      2  CA AALA A   1       2.000   0.000   0.000  0.50  0.00           C
            END
        """
        a_first = """
            ATOM      1  CA AALA A   1       2.000   0.000   0.000  0.50  0.00           C
            ATOM      2  CA BALA A   1       1.000   0.000   0.000  0.50  0.00           C
            END
        """
        one = read_pdb(_write(tmp_path, "tie1.pdb", b_first))["A"][0].atom("CA")
        two = read_pdb(_write(tmp_path, "tie2.pdb", a_first))["A"][0].atom("CA")
        assert one.altloc == two.altloc == "A"
        assert np.allclose(one.xyz, two.xyz)

    def test_microheterogeneity_does_not_make_a_chimera(self, tmp_path):
        """Two residue names at one position must not be merged into one."""
        pdb = """
            ATOM      1  N   SER A  10       0.000   0.000   0.000  1.00  0.00           N
            ATOM      2  CA ASER A  10       1.000   0.000   0.000  0.60  0.00           C
            ATOM      3  OG ASER A  10       2.000   0.000   0.000  0.60  0.00           O
            ATOM      4  CA BALA A  10       1.100   0.000   0.000  0.40  0.00           C
            ATOM      5  CB BALA A  10       2.100   0.000   0.000  0.40  0.00           C
            ATOM      6  CA  GLY A  11       5.000   0.000   0.000  1.00  0.00           C
            END
        """
        s = read_pdb(_write(tmp_path, "micro.pdb", pdb))
        assert [(r.seq_id, r.resname) for r in s["A"]] == [(10, "SER"), (11, "GLY")]
        assert s["A"].sequence == "SG"
        # the alanine-only atom must not have been spliced into the serine
        assert s["A"][0].atom("CB") is None


class TestModels:
    PDB = """
        MODEL        1
        ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
        ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00  0.00           C
        ENDMDL
        MODEL        2
        ATOM      1  CA  ALA A   1      10.000   0.000   0.000  1.00  0.00           C
        ATOM      2  CA  GLY A   2      13.800   0.000   0.000  1.00  0.00           C
        ENDMDL
        MODEL        3
        ATOM      1  CA  ALA A   1      20.000   0.000   0.000  1.00  0.00           C
        ATOM      2  CA  GLY A   2      23.800   0.000   0.000  1.00  0.00           C
        ENDMDL
        END
    """

    def test_first_model_by_default(self, tmp_path):
        s = read_pdb(_write(tmp_path, "multi.pdb", self.PDB))
        assert len(s["A"]) == 2
        assert np.allclose(s["A"][0].atom("CA").xyz, [0.0, 0.0, 0.0])

    def test_explicit_model(self, tmp_path):
        p = _write(tmp_path, "multi2.pdb", self.PDB)
        s = read_pdb(p, model=3)
        assert np.allclose(s["A"][0].atom("CA").xyz, [20.0, 0.0, 0.0])
        assert len(s["A"]) == 2

    def test_missing_model_raises(self, tmp_path):
        p = _write(tmp_path, "multi3.pdb", self.PDB)
        with pytest.raises(StructureParseError):
            read_pdb(p, model=9)

    def test_single_model_file_ignores_model_arg(self, tmp_path):
        pdb = """
            ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
            END
        """
        s = read_pdb(_write(tmp_path, "one.pdb", pdb), model=1)
        assert len(s["A"]) == 1


class TestHetatmOrdering:
    def test_hetatm_written_last_is_repaired(self, tmp_path):
        # MSE is deposited as HETATM; some writers emit every HETATM after
        # every ATOM, which would append it to the end of the chain.
        pdb = """
            ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
            ATOM      2  CA  GLY A   3       7.600   0.000   0.000  1.00  0.00           C
            ATOM      3  CA  SER A   4      11.400   0.000   0.000  1.00  0.00           C
            HETATM    4  CA  MSE A   2       3.800   0.000   0.000  1.00  0.00           C
            END
        """
        s = read_pdb(_write(tmp_path, "hetlast.pdb", pdb))
        assert s["A"].sequence == "AMGS"
        assert [r.seq_id for r in s["A"]] == [1, 2, 3, 4]
        assert s.warnings and "numbering order" in s.warnings[0]

    def test_in_order_file_gets_no_warning(self, tmp_path):
        pdb = """
            ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
            HETATM    2  CA  MSE A   2       3.800   0.000   0.000  1.00  0.00           C
            ATOM      3  CA  GLY A   3       7.600   0.000   0.000  1.00  0.00           C
            END
        """
        s = read_pdb(_write(tmp_path, "inorder.pdb", pdb))
        assert s["A"].sequence == "AMG"
        assert not s.warnings


class TestPlddtDetection:
    """The B-factor column holds pLDDT in predicted structures. Detect it only
    from an explicit declaration, never from magnitude."""

    USER = ("idr_flanks/data/structures/"
            "test_binder_chain_A_target_chain_B.cif")

    def test_predicted_cif_is_detected(self):
        import os
        if not os.path.isfile(self.USER):
            import pytest
            pytest.skip("user structure not present")
        assert read_structure(self.USER).plddt_from_bfactor is True

    def test_crystal_structures_are_not(self):
        assert read_structure(structure_path("1ycr.pdb")).plddt_from_bfactor is False
        assert read_structure(structure_path("1ycr.cif")).plddt_from_bfactor is False

    def test_alphafold_pdb_remark_is_detected(self, tmp_path):
        pdb = """
            REMARK   1  pLDDT per-residue confidence in the B-factor column
            ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 88.00           C
            ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00 91.00           C
            END
        """
        s = read_pdb(_write(tmp_path, "af.pdb", pdb))
        assert s.plddt_from_bfactor is True

    def test_not_guessed_from_magnitude(self, tmp_path):
        """High B-factors in a plain PDB must not be read as pLDDT."""
        pdb = """
            ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 88.00           C
            ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00 91.00           C
            END
        """
        s = read_pdb(_write(tmp_path, "plain.pdb", pdb))
        assert s.plddt_from_bfactor is False


class TestFullSequence:
    """SEQRES / _entity_poly give the chain as deposited. Without it, missing
    *termini* are invisible -- and that is where a flank gets attached."""

    def test_seqres_parsed_from_pdb(self, pdb):
        assert len(pdb["A"].full_sequence) == 109
        assert len(pdb["B"].full_sequence) == 15
        assert pdb["B"].full_sequence == "SQ" + P53

    def test_entity_poly_parsed_from_cif(self, cif):
        assert len(cif["A"].full_sequence) == 109
        assert cif["B"].full_sequence == "SQ" + P53

    def test_pdb_and_cif_agree(self, pdb, cif):
        for cid in pdb.chain_ids:
            assert pdb[cid].full_sequence == cif[cid].full_sequence

    def test_resolved_sequence_is_a_subsequence(self, pdb):
        for chain in pdb:
            assert chain.sequence in chain.full_sequence

    def test_terminal_truncation_is_measured(self, pdb):
        # Both chains of 1YCR are truncated, and chain_breaks() cannot see it.
        assert pdb["A"].unresolved_termini() == (8, 16)
        assert pdb["B"].unresolved_termini() == (2, 0)
        assert pdb["A"].chain_breaks() == []
        assert pdb["B"].chain_breaks() == []

    def test_no_full_sequence_means_no_claim(self, tmp_path):
        pdb_text = """
            ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
            ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00  0.00           C
            END
        """
        s = read_pdb(_write(tmp_path, "noseqres.pdb", pdb_text))
        assert s["A"].full_sequence == ""
        assert s["A"].unresolved_termini() == (0, 0)


class TestNumberingCollisions:
    def test_duplicate_identity_is_reported(self, tmp_path):
        """A chain whose numbering restarts must not silently lose atoms."""
        pdb_text = """
            ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
            ATOM      2  CA  GLY A   1       1.400   0.000   0.000  1.00  0.00           C
            ATOM      3  N   ALA A   2       3.800   0.000   0.000  1.00  0.00           N
            ATOM      4  CA  ALA A   2       5.200   0.000   0.000  1.00  0.00           C
            ATOM      5  N   TRP A   1      20.000   0.000   0.000  1.00  0.00           N
            ATOM      6  CA  TRP A   1      21.400   0.000   0.000  1.00  0.00           C
            END
        """
        s = read_pdb(_write(tmp_path, "dup.pdb", pdb_text))
        assert s["A"].sequence == "GA"
        assert s.warnings, "collapsing residues must not be silent"

    def test_clean_chain_has_no_such_warning(self, pdb):
        assert not any("more than one" in w or "collides" in w
                       for w in pdb.warnings)


class TestCifAltlocFallback:
    def test_sidechain_altlocs_do_not_split_a_residue(self, tmp_path):
        """The standard pattern -- backbone unlabelled, sidechain A/B -- must
        stay one residue when the residue number is unusable."""
        cif = """
            data_T
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_alt_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_seq_id
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            _atom_site.occupancy
            ATOM 1 N N  . SER A . 0.0 0.0 0.0 1.00
            ATOM 2 C CA . SER A . 1.4 0.0 0.0 1.00
            ATOM 3 C C  . SER A . 2.4 0.0 0.0 1.00
            ATOM 4 C CB A SER A . 3.0 1.0 0.0 0.60
            ATOM 5 O OG A SER A . 3.5 2.0 0.0 0.60
            ATOM 6 C CB B SER A . 3.1 1.1 0.0 0.40
            ATOM 7 O OG B SER A . 3.6 2.1 0.0 0.40
            ATOM 8 N N  . GLY A . 5.0 0.0 0.0 1.00
            ATOM 9 C CA . GLY A . 6.4 0.0 0.0 1.00
            #
        """
        s = read_cif(_write(tmp_path, "altfb.cif", cif))
        assert s["A"].sequence == "SG"
        assert len(s["A"]) == 2


class TestHeteroatomRetention:
    """Skipped residues are dropped from the chains but their coordinates are
    kept, because they still occlude solvent."""

    def test_ligand_coordinates_are_retained(self, tmp_path):
        pdb = """
            ATOM      1  CA  ALA A   1      11.639   6.399  -5.147  1.00 20.00           C
            HETATM    2 FE   HEM A 200      30.000  30.000  30.000  1.00 20.00          FE
            HETATM    3  NA  HEM A 200      31.000  30.000  30.000  1.00 20.00           N
            HETATM    4  O   HOH A 300      50.000  50.000  50.000  1.00 20.00           O
            END
        """
        s = read_pdb(_write(tmp_path, "het.pdb", pdb))
        coords, elements = s.heteroatoms()
        # the heme atoms are kept; water is not
        assert coords.shape == (2, 3)
        assert sorted(elements) == ["Fe", "N"]
        assert s["A"].sequence == "A"

    def test_skipped_residues_counts_residues_not_atoms(self, tmp_path):
        """The field is named and displayed as a residue count."""
        pdb = """
            ATOM      1  CA  ALA A   1      11.639   6.399  -5.147  1.00 20.00           C
            HETATM    2 FE   HEM A 200      30.000  30.000  30.000  1.00 20.00          FE
            HETATM    3  NA  HEM A 200      31.000  30.000  30.000  1.00 20.00           N
            HETATM    4  NB  HEM A 200      32.000  30.000  30.000  1.00 20.00           N
            HETATM    5 FE   HEM A 201      40.000  30.000  30.000  1.00 20.00          FE
            END
        """
        s = read_pdb(_write(tmp_path, "count.pdb", pdb))
        assert s.skipped_residues == {"HEM": 2}
        # coordinates are still kept per atom, for occlusion
        assert s.heteroatoms()[0].shape == (4, 3)

    def test_md_style_ion_names_are_treated_as_solvent(self, tmp_path):
        """Simulation tools write NA+/CL-/MG2+ where the wwPDB writes NA/CL/MG."""
        pdb = """
            ATOM      1  CA  ALA A   1      11.639   6.399  -5.147  1.00 20.00           C
            HETATM    2 NA   NA+ A 100      20.000  20.000  20.000  1.00 20.00          NA
            HETATM    3 CL   CL- A 101      30.000  30.000  30.000  1.00 20.00          CL
            HETATM    4 MG   MG2+A 102      40.000  40.000  40.000  1.00 20.00          MG
            HETATM    5 FE   HEM A 200      50.000  50.000  50.000  1.00 20.00          FE
            END
        """
        s = read_pdb(_write(tmp_path, "ions.pdb", pdb))
        assert s.skipped_residues == {"HEM": 1}
        assert s.heteroatoms()[1] == ["Fe"]

    def test_none_retained_for_a_clean_structure(self):
        s = read_structure(structure_path("1ycr.pdb"))
        coords, elements = s.heteroatoms()
        assert coords.shape == (0, 3)
        assert elements == []

    def test_hydrogens_are_not_retained(self, tmp_path):
        pdb = """
            ATOM      1  CA  ALA A   1      11.639   6.399  -5.147  1.00 20.00           C
            HETATM    2  C1  LIG A 200      30.000  30.000  30.000  1.00 20.00           C
            HETATM    3  H1  LIG A 200      31.000  30.000  30.000  1.00 20.00           H
            END
        """
        s = read_pdb(_write(tmp_path, "heth.pdb", pdb))
        coords, elements = s.heteroatoms()
        assert elements == ["C"]
        assert coords.shape == (1, 3)

    def test_cif_ligand_hydrogens_are_not_retained(self, tmp_path):
        """A ligand hydrogen must not become a carbon-sized occluder when the
        mmCIF omits type_symbol."""
        cif = """
            data_T
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_seq_id
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            ATOM   1 C  CA  ALA A 1 0.0 0.0 0.0
            HETATM 2 ?  C1  LIG A 2 5.0 0.0 0.0
            HETATM 3 ?  H1  LIG A 2 6.0 0.0 0.0
            #
        """
        s = read_cif(_write(tmp_path, "hetel.cif", cif))
        coords, elements = s.heteroatoms()
        assert elements == ["C"]
        assert coords.shape == (1, 3)

    def test_retained_from_cif_too(self, tmp_path):
        cif = """
            data_T
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_seq_id
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            ATOM   1 C CA ALA A 1 0.0 0.0 0.0
            HETATM 2 P P   DA  B 1 9.0 0.0 0.0
            HETATM 3 O O   HOH C 1 20.0 0.0 0.0
            #
        """
        s = read_cif(_write(tmp_path, "het.cif", cif))
        coords, elements = s.heteroatoms()
        assert elements == ["P"]
        assert np.allclose(coords[0], [9.0, 0.0, 0.0])


class TestResidueMapping:
    def test_standard_twenty_present(self):
        for one in "ACDEFGHIKLMNPQRSTVWY":
            assert one in THREE_TO_ONE.values()

    def test_mse_maps_to_methionine(self):
        assert THREE_TO_ONE["MSE"] == "M"

    def test_common_modified_residues(self):
        assert THREE_TO_ONE["SEP"] == "S"   # phosphoserine
        assert THREE_TO_ONE["TPO"] == "T"   # phosphothreonine
        assert THREE_TO_ONE["PTR"] == "Y"   # phosphotyrosine
        assert THREE_TO_ONE["HYP"] == "P"   # hydroxyproline
        assert THREE_TO_ONE["MLY"] == "K"   # methyllysine

    def test_unknown_residue_is_not_mapped(self):
        assert "HEM" not in THREE_TO_ONE
        assert "HOH" not in THREE_TO_ONE


class TestCifQuoting:
    HEADER = """
        data_T
        loop_
        _atom_site.group_PDB
        _atom_site.id
        _atom_site.type_symbol
        _atom_site.label_atom_id
        _atom_site.label_comp_id
        _atom_site.label_asym_id
        _atom_site.label_seq_id
        _atom_site.Cartn_x
        _atom_site.Cartn_y
        _atom_site.Cartn_z
        _atom_site.pdbx_formal_charge
    """

    def _read(self, tmp_path, name, rows):
        return read_cif(_write(tmp_path, name, self.HEADER + rows + "        #\n"))

    def test_quoted_value_with_spaces(self, tmp_path):
        rows = """
        ATOM 1 C CA ALA A 1 0.0 0.0 0.0 'a b c'
        ATOM 2 C CA GLY A 2 3.8 0.0 0.0 'd e f'
        """
        assert self._read(tmp_path, "sp.cif", rows)["A"].sequence == "AG"

    def test_quoted_value_starting_with_underscore(self, tmp_path):
        """A quoted value is data, not a tag, so it must not end the loop."""
        rows = """
        ATOM 1 C CA ALA A 1 0.0 0.0 0.0 '_not_a_tag'
        ATOM 2 C CA GLY A 2 3.8 0.0 0.0 '_not_a_tag'
        ATOM 3 C CA SER A 3 7.6 0.0 0.0 '_not_a_tag'
        """
        assert self._read(tmp_path, "us.cif", rows)["A"].sequence == "AGS"

    def test_quoted_value_containing_loop_keyword(self, tmp_path):
        rows = """
        ATOM 1 C CA ALA A 1 0.0 0.0 0.0 'loop_'
        ATOM 2 C CA GLY A 2 3.8 0.0 0.0 'loop_'
        """
        assert self._read(tmp_path, "lk.cif", rows)["A"].sequence == "AG"

    def test_double_quoted_value(self, tmp_path):
        rows = """
        ATOM 1 C CA ALA A 1 0.0 0.0 0.0 "x y"
        ATOM 2 C CA GLY A 2 3.8 0.0 0.0 "x y"
        """
        assert self._read(tmp_path, "dq.cif", rows)["A"].sequence == "AG"

    def test_apostrophe_inside_a_token(self, tmp_path):
        """A quote only opens a string at a token boundary, so O5' parses."""
        rows = """
        ATOM 1 C CA ALA A 1 0.0 0.0 0.0 O5'
        ATOM 2 C CA GLY A 2 3.8 0.0 0.0 O5'
        """
        assert self._read(tmp_path, "ap.cif", rows)["A"].sequence == "AG"

    def test_comment_between_rows(self, tmp_path):
        rows = """
        ATOM 1 C CA ALA A 1 0.0 0.0 0.0 ?
        # a comment in the middle of the loop
        ATOM 2 C CA GLY A 2 3.8 0.0 0.0 ?
        """
        assert self._read(tmp_path, "cm.cif", rows)["A"].sequence == "AG"

    def test_residue_number_fallback_groups_by_residue(self, tmp_path):
        """With no usable residue number, atoms must still group into residues."""
        cif = """
            data_T
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_seq_id
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            ATOM 1 N N  ALA A . 0.0 0.0 0.0
            ATOM 2 C CA ALA A . 1.5 0.0 0.0
            ATOM 3 C C  ALA A . 3.0 0.0 0.0
            ATOM 4 N N  GLY A . 3.8 0.0 0.0
            ATOM 5 C CA GLY A . 5.3 0.0 0.0
            ATOM 6 N N  ALA A . 7.0 0.0 0.0
            ATOM 7 C CA ALA A . 8.5 0.0 0.0
            #
        """
        s = read_cif(_write(tmp_path, "fb.cif", cif))
        assert s["A"].sequence == "AGA"
        assert [len(r.atoms) for r in s["A"]] == [3, 2, 2]


class TestCifTokenizing:
    def test_quoted_and_placeholder_values(self, tmp_path):
        cif = """
            data_TEST
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_alt_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_seq_id
            _atom_site.pdbx_PDB_ins_code
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            _atom_site.occupancy
            _atom_site.B_iso_or_equiv
            _atom_site.auth_seq_id
            _atom_site.auth_comp_id
            _atom_site.auth_asym_id
            _atom_site.pdbx_PDB_model_num
            ATOM 1 C CA . ALA A 1 ? 0.000 0.000 0.000 1.00 10.0 5 ALA X 1
            ATOM 2 C CA . GLY A 2 ? 3.800 0.000 0.000 1.00 10.0 6 GLY X 1
            #
        """
        s = read_cif(_write(tmp_path, "t.cif", cif))
        # author identifiers win: chain X, numbering from 5
        assert s.chain_ids == ["X"]
        assert s["X"].sequence == "AG"
        assert [r.seq_id for r in s["X"]] == [5, 6]

    def test_falls_back_to_label_ids(self, tmp_path):
        cif = """
            data_TEST
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_seq_id
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            ATOM 1 C CA ALA Q 1 0.000 0.000 0.000
            ATOM 2 C CA GLY Q 2 3.800 0.000 0.000
            #
        """
        s = read_cif(_write(tmp_path, "lbl.cif", cif))
        assert s.chain_ids == ["Q"]
        assert s["Q"].sequence == "AG"
        assert [r.seq_id for r in s["Q"]] == [1, 2]

    def test_skips_unrelated_loops_before_and_after(self, tmp_path):
        cif = """
            data_TEST
            loop_
            _entity.id
            _entity.type
            1 polymer
            2 water
            #
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_seq_id
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            ATOM 1 C CA ALA A 1 0.000 0.000 0.000
            ATOM 2 C CA GLY A 2 3.800 0.000 0.000
            #
            loop_
            _struct_conf.id
            _struct_conf.name
            HELX1 helix
            #
        """
        s = read_cif(_write(tmp_path, "loops.cif", cif))
        assert s["A"].sequence == "AG"

    def test_multiline_text_block_before_atom_site(self, tmp_path):
        cif = """
            data_TEST
            _struct.title
            ;This is a long title that spans
            several lines and contains loop_ and _atom_site. text
            that must not be parsed as tags
            ;
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_seq_id
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            ATOM 1 C CA ALA A 1 0.000 0.000 0.000
            #
        """
        s = read_cif(_write(tmp_path, "ml.cif", cif))
        # The block's contents must not have been parsed as tags or rows: if
        # semicolon handling were dropped, the "_atom_site." text inside it
        # would be read as a tag and the real loop would be misaligned.
        assert s["A"].sequence == "A"
        assert len(s["A"]) == 1
        assert s["A"][0].resname == "ALA"

    def test_model_selection(self, tmp_path):
        cif = """
            data_TEST
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_seq_id
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            _atom_site.pdbx_PDB_model_num
            ATOM 1 C CA ALA A 1 0.000 0.000 0.000 1
            ATOM 2 C CA ALA A 1 9.000 0.000 0.000 2
            #
        """
        p = _write(tmp_path, "models.cif", cif)
        s1 = read_cif(p)
        assert np.allclose(s1["A"][0].atom("CA").xyz, [0.0, 0.0, 0.0])
        assert s1.model == 1
        s2 = read_cif(p, model=2)
        assert np.allclose(s2["A"][0].atom("CA").xyz, [9.0, 0.0, 0.0])

    def test_hetatm_mse_included(self, tmp_path):
        cif = """
            data_TEST
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_seq_id
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            ATOM   1 C CA ALA A 1 0.000 0.000 0.000
            HETATM 2 C CA MSE A 2 3.800 0.000 0.000
            HETATM 3 O O  HOH A 3 9.000 0.000 0.000
            #
        """
        s = read_cif(_write(tmp_path, "mse.cif", cif))
        assert s["A"].sequence == "AM"
        assert s["A"][1].atoms[0].is_hetatm

    def test_altloc_occupancy_in_cif(self, tmp_path):
        cif = """
            data_TEST
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_alt_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_seq_id
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            _atom_site.occupancy
            ATOM 1 C CA A ALA A 1 0.000 0.000 0.000 0.30
            ATOM 2 C CA B ALA A 1 1.000 0.000 0.000 0.70
            #
        """
        s = read_cif(_write(tmp_path, "alt.cif", cif))
        res = s["A"][0]
        assert len(res.atoms) == 1
        assert res.atom("CA").altloc == "B"
        assert np.allclose(res.atom("CA").xyz, [1.0, 0.0, 0.0])


class TestStructureHelpers:
    def test_summary_mentions_chains_and_sequences(self, pdb):
        text = pdb.summary()
        assert "chain 'A'" in text and "chain 'B'" in text
        assert P53 in text

    def test_contains_and_iteration(self, pdb):
        assert "A" in pdb and "Z" not in pdb
        assert [c.chain_id for c in pdb] == ["A", "B"]
        assert len(pdb) == 2

    def test_get_chain_alias(self, pdb):
        assert pdb.get_chain("B") is pdb["B"]

    def test_seq_ids_array(self, pdb):
        ids = pdb["B"].seq_ids
        assert ids.tolist() == list(range(17, 30))
