Developer Guide
===============

Layout
------

======================== =====================================================
Module                   Responsibility
======================== =====================================================
``idr_flanks/io.py``     PDB and mmCIF readers. numpy only.
``idr_flanks/sasa.py``   Shrake-Rupley solvent accessibility. numpy + scipy.
``idr_flanks/interface.py`` Selecting the reachable, exposed, sequence-local
                         target region.
``idr_flanks/design.py`` The GOOSE objective, presets, and scoring.
``idr_flanks/pipeline.py`` End-to-end orchestration.
``idr_flanks/cli.py``    ``idr-flanks`` command line.
======================== =====================================================

Dependencies are layered deliberately. ``io``, ``sasa`` and ``interface`` need
only numpy and scipy, so structure reading and interface analysis work in a
minimal environment. GOOSE, FINCHES, sparrow and metapredict are imported
lazily inside ``design`` and reached through ``idr_flanks.__getattr__``, so
``import idr_flanks`` stays fast and does not require them.

Running the tests
-----------------

.. code-block:: bash

    python -m pytest idr_flanks/tests -q

Tests that need GOOSE are guarded with ``pytest.importorskip("goose")`` and skip
cleanly without it. The SASA cross-check against ``mdtraj`` skips if mdtraj is
missing.

Test data lives in ``idr_flanks/data/structures`` and is reached through
``idr_flanks.data.structure_path``. The suite must not require network access.

Conventions that matter
-----------------------

**Author identifiers everywhere.** Chain ids and residue numbers in the public
API are always the author/PDB numbering a user would read off a viewer, never
internal indices. A residue's identity is ``(chain, seq_id, ins_code)``.

**Never key on a residue number alone.** Insertion codes repeat a number, so a
dict keyed on ``seq_id`` silently collapses residues. This produced a real bug:
a patch of ``52, 52A, 52B, 53`` came out as ``WWWK`` instead of ``GPWK``,
meaning the design was optimised against the wrong chemistry. Group residues by
chain adjacency (``Residue.index``) instead, and there is a regression test for
it in ``TestInsertionCodes``.

**Epsilon sign and scaling.** Negative epsilon is attractive. Epsilon scales
linearly with the length of the *first* argument and is nearly invariant to the
second, so any epsilon target must be expressed per residue and multiplied by
the designed length. ``epsilon(a, b) != epsilon(b, a)``.

**Verify against the dependency's source.** GOOSE and FINCHES have several
surprising behaviours that are easy to get wrong from their docstrings alone:
``add_property`` renames duplicates using a substring test against the class
name, ``aa_fraction_ranges`` converts fractions to counts with ``ceil``,
``enable_error_tolerance`` defaults to stopping the run as soon as every target
is met, and ``seed=`` has no effect on GOOSE's default C-RNG mutation path.
Each of these is handled explicitly in ``design.py`` with a comment saying why.
When touching that file, read the GOOSE source rather than trusting a docstring.

**Claims in docs must be measured.** Every number in the README and user guide
came from a command that was actually run. If you change the objective, re-measure
them.

Adding a design constraint
--------------------------

Constraints are GOOSE properties added in :func:`idr_flanks.design.design_flank`
and exposed as :class:`~idr_flanks.design.DesignConfig` fields. For anything
GOOSE does not provide, subclass its ``CustomProperty`` and implement
``calculate_raw_value(protein)`` returning the *raw value* (GOOSE applies the
constraint and target itself; returning an error instead causes the constraint
to be applied twice). Set ``calculate_in_batch = True`` and implement
``calculate_raw_value_batch`` where the underlying calculation batches, as
metapredict does. Do not assign ``tracking_property_name``.

Building the docs
-----------------

.. code-block:: bash

    cd docs && make html
