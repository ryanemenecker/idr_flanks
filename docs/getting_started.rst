Getting Started
===============

``idr_flanks`` takes a structure of a binder bound to its target and designs a
disordered tail for the binder that is chemically complementary to the target
surface next to the binding site.

Installation
------------

.. code-block:: bash

    pip install -e .

Structure reading and interface analysis need only ``numpy`` and ``scipy``.
Designing a flank additionally needs `GOOSE <https://github.com/idptools/goose>`_,
FINCHES, sparrow and metapredict; those are imported lazily, so the analysis
half of the package works without them.

Step 1: identify the chains
---------------------------

You need to know which chain is the binder (the one to extend) and which is the
target.

.. code-block:: python

    from idr_flanks import describe_chains

    print(describe_chains("complex.pdb"))

.. code-block:: text

    Structure: 1ycr.pdb (pdb, model None)
      chain 'A':   85 residues [A:25 .. A:109]
          ETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV
      chain 'B':   13 residues [B:17 .. B:29]
          ETFSDLWKLLPEN

Here chain B is the p53 transactivation-domain peptide and chain A is MDM2, so
B is the binder and A is the target.

Step 2: look at what a flank could reach
----------------------------------------

Before designing anything, check which target residues a flank of a given
length could actually interact with.

.. code-block:: python

    from idr_flanks import find_proximal_region, read_structure

    structure = read_structure("complex.pdb")
    region = find_proximal_region(structure, binder_chain="B",
                                  target_chain="A", terminus="C",
                                  flank_length=30)
    print(region.summary())

The summary reports the anchor residue, the reach radius, the residues selected,
their contiguous spans, the patch sequence handed to the design step, and every
residue that was excluded and why.

Step 3: design the flank
------------------------

.. code-block:: python

    from idr_flanks import build_flanked_binder

    result = build_flanked_binder("complex.pdb",
                                  binder_chain="B",
                                  target_chain="A",
                                  c_flank_length=30,
                                  seed=1)

    print(result.final_sequence)
    print(result.summary())

``result.summary()`` prints the proximal-region analysis and the design
diagnostics together. ``result.fasta()`` gives a FASTA record, and
``result.annotated_sequence()`` brackets the added flanks so you can see what
changed.

Both termini at once:

.. code-block:: python

    result = build_flanked_binder("complex.pdb", binder_chain="B",
                                  target_chain="A",
                                  n_flank_length=20, c_flank_length=20)

The N-terminal flank is designed first and becomes part of the sequence context
for the C-terminal one, because predicted disorder depends on the whole
construct.

From the command line
---------------------

.. code-block:: bash

    idr-flanks info complex.pdb
    idr-flanks contacts complex.pdb -b B -t A -c 30
    idr-flanks design complex.pdb -b B -t A -c 30 --seed 1

Add ``--quiet`` to print only the final sequence, or ``--fasta out.fa`` to
write a FASTA file.

What to check before trusting a design
--------------------------------------

Read the reported **specificity** and **self-epsilon** numbers, not just the
epsilon against the patch. A flank can score strong attraction simply by being
sticky to everything. See :doc:`user_guide` for what the numbers mean and when
to distrust them.
