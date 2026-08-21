# Package data

## `structures/`

Small structure files shipped for tests, examples, and documentation.

| File | Description |
| --- | --- |
| `1ycr.pdb` | PDB 1YCR, trimmed to polymer atoms of chains A and B. Chain A is MDM2 (residues 25-109); chain B is the p53 transactivation-domain peptide (residues 17-29). The canonical binder/target test case: chain B is the binder, chain A the target. |
| `1ycr.cif` | The same entry in mmCIF format, retaining the `_atom_site` loop plus a preceding unrelated loop so the reader's loop handling is exercised. |

Both files parse to identical chains and sequences, which is asserted in the
test suite.

Access them from code with:

```python
from idr_flanks.data import structure_path
path = structure_path("1ycr.pdb")
```
