API Documentation
=================

Top-level pipeline
------------------

.. autosummary::
   :toctree: autosummary

   idr_flanks.build_flanked_binder
   idr_flanks.describe_chains
   idr_flanks.FlankedBinder

Reading structures
------------------

.. autosummary::
   :toctree: autosummary

   idr_flanks.read_structure
   idr_flanks.read_pdb
   idr_flanks.read_cif
   idr_flanks.Structure
   idr_flanks.Chain
   idr_flanks.Residue
   idr_flanks.Atom
   idr_flanks.StructureParseError

Solvent accessibility
---------------------

.. autosummary::
   :toctree: autosummary

   idr_flanks.residue_sasa
   idr_flanks.relative_residue_sasa
   idr_flanks.solvent_accessible_mask

Finding the proximal region
---------------------------

.. autosummary::
   :toctree: autosummary

   idr_flanks.find_proximal_region
   idr_flanks.ProximalRegion
   idr_flanks.ProximalResidue
   idr_flanks.contact_map
   idr_flanks.reach_radius
   idr_flanks.InterfaceError

Designing the flank
-------------------

.. autosummary::
   :toctree: autosummary

   idr_flanks.design_flank
   idr_flanks.score_flank
   idr_flanks.DesignConfig
   idr_flanks.DesignResult
   idr_flanks.PRESETS
   idr_flanks.epsilon_per_residue
   idr_flanks.load_epsilon_model
   idr_flanks.DesignError
