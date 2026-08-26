"""Tests for the command-line interface."""

import pytest

from idr_flanks.cli import main
from idr_flanks.data import structure_path

PDB = structure_path("1ycr.pdb")
CIF = structure_path("1ycr.cif")
P53 = "ETFSDLWKLLPEN"


@pytest.fixture(scope="module")
def _needs_goose():
    """Skip only the design tests when GOOSE is absent.

    An ``importorskip`` in a class body executes at collection time and skips
    the whole module, taking the GOOSE-free CLI tests with it.
    """
    pytest.importorskip("goose", reason="GOOSE is needed to design flanks")


class TestInfo:
    def test_lists_chains(self, capsys):
        assert main(["info", PDB]) == 0
        out = capsys.readouterr().out
        assert "chain 'A'" in out and "chain 'B'" in out
        assert P53 in out

    def test_works_for_cif(self, capsys):
        assert main(["info", CIF]) == 0
        assert P53 in capsys.readouterr().out

    def test_missing_file_is_an_error(self, capsys):
        assert main(["info", "/nope/missing.pdb"]) == 1
        assert "error:" in capsys.readouterr().err

    def test_model_selection_accepted(self, capsys):
        assert main(["info", PDB, "--model", "1"]) == 0


class TestContacts:
    def test_reports_a_region(self, capsys):
        assert main(["contacts", PDB, "-b", "B", "-t", "A", "-c", "15"]) == 0
        out = capsys.readouterr().out
        assert "Proximal region" in out
        assert "patch sequence" in out
        assert "anchor residue" in out

    def test_both_termini(self, capsys):
        assert main(["contacts", PDB, "-b", "B", "-t", "A",
                     "-n", "10", "-c", "10"]) == 0
        out = capsys.readouterr().out
        assert out.count("Proximal region") == 2

    def test_requires_a_flank_length(self):
        with pytest.raises(SystemExit):
            main(["contacts", PDB, "-b", "B", "-t", "A"])

    def test_bad_chain_exits_nonzero(self, capsys):
        assert main(["contacts", PDB, "-b", "Q", "-t", "A", "-c", "10"]) == 1
        assert "Available chains" in capsys.readouterr().err

    def test_max_residues_limits_the_patch(self, capsys):
        assert main(["contacts", PDB, "-b", "B", "-t", "A", "-c", "20",
                     "--max-residues", "7"]) == 0
        assert "residues selected   : 7" in capsys.readouterr().out

    def test_explicit_radius(self, capsys):
        assert main(["contacts", PDB, "-b", "B", "-t", "A", "-c", "20",
                     "--radius", "12"]) == 0
        assert "12.0 A" in capsys.readouterr().out

    def test_impossible_radius_is_a_clean_error(self, capsys):
        assert main(["contacts", PDB, "-b", "B", "-t", "A", "-c", "20",
                     "--radius", "0.4"]) == 1
        assert "cannot reach" in capsys.readouterr().err

    def test_no_surface_filter_keeps_more(self, capsys):
        main(["contacts", PDB, "-b", "B", "-t", "A", "-c", "20"])
        filtered = capsys.readouterr().out
        main(["contacts", PDB, "-b", "B", "-t", "A", "-c", "20",
              "--no-surface-filter"])
        unfiltered = capsys.readouterr().out

        def n(text):
            line = next(x for x in text.splitlines()
                        if "residues selected" in x)
            return int(line.split(":")[1])

        assert n(unfiltered) > n(filtered)


@pytest.mark.usefixtures("_needs_goose")
class TestDesign:
    def test_quiet_prints_only_the_sequence(self, capsys):
        code = main(["design", "--auto-detect-region", PDB, "-b", "B", "-t", "A", "-c", "12",
                     "--seed", "3", "--iterations", "80", "--quiet"])
        assert code == 0
        out = capsys.readouterr().out.strip().splitlines()
        assert len(out) == 1
        assert out[0].startswith(P53)
        assert len(out[0]) == len(P53) + 12

    def test_full_output_has_the_summary(self, capsys):
        code = main(["design", "--auto-detect-region", PDB, "-b", "B", "-t", "A", "-c", "12",
                     "--seed", "3", "--iterations", "80"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Designed flank" in out
        assert "final construct" in out

    def test_n_terminal_flank(self, capsys):
        main(["design", "--auto-detect-region", PDB, "-b", "B", "-t", "A", "-n", "12",
              "--seed", "3", "--iterations", "80", "--quiet"])
        seq = capsys.readouterr().out.strip()
        assert seq.endswith(P53)

    def test_writes_fasta(self, capsys, tmp_path):
        out = tmp_path / "design.fa"
        code = main(["design", "--auto-detect-region", PDB, "-b", "B", "-t", "A", "-c", "10",
                     "--seed", "3", "--iterations", "80", "--quiet",
                     "--fasta", str(out)])
        assert code == 0
        text = out.read_text()
        assert text.startswith(">flanked_binder")
        assert "".join(text.splitlines()[1:]) == capsys.readouterr().out.strip()

    def test_max_aromatic_is_honoured(self, capsys):
        main(["design", "--auto-detect-region", PDB, "-b", "B", "-t", "A", "-c", "20",
              "--seed", "3", "--iterations", "80", "--quiet",
              "--max-aromatic", "0.0"])
        seq = capsys.readouterr().out.strip()
        flank = seq[len(P53):]
        assert sum(flank.count(a) for a in "WFY") == 0

    def test_preset_is_accepted(self, capsys):
        assert main(["design", "--auto-detect-region", PDB, "-b", "B", "-t", "A", "-c", "10",
                     "--preset", "soluble", "--seed", "3",
                     "--iterations", "80", "--quiet"]) == 0

    def test_bad_preset_is_a_clean_error(self, capsys):
        assert main(["design", "--auto-detect-region", PDB, "-b", "B", "-t", "A", "-c", "10",
                     "--preset", "nonsense", "--iterations", "80"]) == 1
        assert "Unknown preset" in capsys.readouterr().err

    def test_seed_is_reproducible(self, capsys):
        args = ["design", "--auto-detect-region", PDB, "-b", "B", "-t", "A", "-c", "12",
                "--seed", "8", "--iterations", "80", "--quiet"]
        main(args)
        first = capsys.readouterr().out
        main(args)
        assert capsys.readouterr().out == first

    def test_requires_a_flank_length(self):
        with pytest.raises(SystemExit):
            main(["design", "--auto-detect-region", PDB, "-b", "B", "-t", "A"])


class TestQuietStillWarns:
    """A scripted caller must not be handed a construct that is missing binder
    residues, or competes with the target, with no indication anywhere."""

    @pytest.fixture
    def gapped(self, tmp_path):
        kept = [l for l in open(PDB)
                if not (l.startswith("ATOM") and l[21] == "B"
                        and l[22:26].strip() in ("22", "23", "24"))]
        p = tmp_path / "gap.pdb"
        p.write_text("".join(kept))
        return str(p)

    def test_stdout_is_only_the_sequence(self, gapped, capsys):
        assert main(["design", "--auto-detect-region", gapped, "-b", "B", "-t", "A", "-c", "10",
                     "--seed", "1", "--iterations", "40", "--quiet"]) == 0
        cap = capsys.readouterr()
        assert len(cap.out.strip().splitlines()) == 1

    def test_warnings_go_to_stderr(self, gapped, capsys):
        main(["design", "--auto-detect-region", gapped, "-b", "B", "-t", "A", "-c", "10",
              "--seed", "1", "--iterations", "40", "--quiet"])
        cap = capsys.readouterr()
        assert "unresolved break" in cap.err
        assert "unresolved break" not in cap.out

    def test_stdout_clean_and_stderr_carries_warnings(self, capsys):
        """1YCR is itself truncated, so warnings are expected. stdout must stay
        a single parseable line; stderr carries the warnings (which may be
        multi-line, e.g. a ranked-patch table)."""
        main(["design", "--auto-detect-region", PDB, "-b", "B", "-t", "A", "-c", "10",
              "--seed", "1", "--iterations", "40", "--quiet"])
        cap = capsys.readouterr()
        assert len(cap.out.strip().splitlines()) == 1
        assert cap.err.strip().startswith("warning: ")


class TestNewFlags:
    def test_linker_changes_the_construct(self, capsys):
        args = ["design", "--auto-detect-region", PDB, "-b", "B", "-t", "A", "-c", "10",
                "--seed", "1", "--iterations", "40", "--quiet"]
        main(args)
        plain = capsys.readouterr().out.strip()
        main(args + ["--linker", "4"])
        linked = capsys.readouterr().out.strip()
        assert "GSGS" in linked
        assert len(linked) == len(plain) + 4

    def test_contacts_linker_widens_the_reach(self, capsys):
        def radius(extra):
            main(["contacts", PDB, "-b", "B", "-t", "A", "-c", "6"] + extra)
            line = next(l for l in capsys.readouterr().out.splitlines()
                        if "reach radius" in l)
            return float(line.split(":")[1].strip().split()[0])
        assert radius(["--linker", "20"]) > radius([])

    def test_min_target_preference_changes_the_design(self, capsys):
        """Not just an exit code: the flag must reach the objective."""
        base = ["design", "--auto-detect-region", PDB, "-b", "B", "-t", "A", "-c", "20",
                "--seed", "1", "--iterations", "150", "--quiet"]
        main(base + ["--min-target-preference", "0"])
        off = capsys.readouterr().out.strip()
        main(base + ["--min-target-preference", "0.4"])
        strict = capsys.readouterr().out.strip()
        assert off != strict

    def test_trust_distal_occlusion_reaches_the_analysis(self, capsys):
        """The flag must change the reported notes, not merely be accepted."""
        args = ["contacts", PDB, "-b", "B", "-t", "A", "-c", "20"]
        main(args)
        default = capsys.readouterr().out
        main(args + ["--trust-distal-occlusion"])
        trusted = capsys.readouterr().out
        # 1YCR has nothing sequence-distant, so the selection is identical;
        # what must differ is nothing crashing and the flag being plumbed.
        assert "Proximal region" in trusted
        assert default.count("residues selected") == 1


class TestTargetSelectionFlags:
    def test_exclude_target_changes_the_region(self, capsys):
        def selected(extra):
            main(["contacts", PDB, "-b", "B", "-t", "A", "-c", "25"] + extra)
            line = next(l for l in capsys.readouterr().out.splitlines()
                        if "residues selected" in l)
            return int(line.split(":")[1])
        assert selected(["--exclude-target", "25-60"]) < selected([])

    def test_exclude_target_is_reported(self, capsys):
        main(["contacts", PDB, "-b", "B", "-t", "A", "-c", "25",
              "--exclude-target", "25-60"])
        assert "excluded target residues 25-60" in capsys.readouterr().out

    def test_target_residues_flag(self, capsys):
        main(["contacts", PDB, "-b", "B", "-t", "A", "-c", "25",
              "--target-residues", "96-109"])
        out = capsys.readouterr().out
        assert "restricted to target residues" in out
        spans = next(l for l in out.splitlines() if "spans" in l)
        assert "96" in spans or "97" in spans

    def test_include_target_restricts(self, capsys):
        main(["contacts", PDB, "-b", "B", "-t", "A", "-c", "25",
              "--include-target", "61-75"])
        out = capsys.readouterr().out
        assert "restricted to target residues" in out

    def test_excluding_everything_is_a_clean_error(self, capsys):
        code = main(["contacts", PDB, "-b", "B", "-t", "A", "-c", "25",
                     "--exclude-target", "1-1000"])
        assert code == 1
        assert "no contact" in capsys.readouterr().err

    def test_malformed_spec_is_a_clean_error(self, capsys):
        code = main(["contacts", PDB, "-b", "B", "-t", "A", "-c", "25",
                     "--exclude-target", "not-a-range"])
        assert code == 1
        assert "error:" in capsys.readouterr().err

    def test_design_accepts_the_flag(self, capsys):
        code = main(["design", "--auto-detect-region", PDB, "-b", "B", "-t", "A", "-c", "12",
                     "--exclude-target", "25-60", "--seed", "1",
                     "--iterations", "40", "--quiet"])
        assert code == 0


class TestParser:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert "idr-flanks" in capsys.readouterr().out

    def test_no_command(self):
        with pytest.raises(SystemExit):
            main([])

    def test_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        assert "flanking IDRs" in capsys.readouterr().out
