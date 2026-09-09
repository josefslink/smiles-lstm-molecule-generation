# Third-party components

All code in this repository (`src/`, `scripts/`) is my own work.

**The course-provided evaluation harness is not included.** The original
project imported a challenge-supplied `evaluation` package
(`evaluation/evaluate_submission.py`, `evaluation/utils.py`,
`evaluation/metric_fcd.py`, plus `evaluation/data/test_stats.p`, the
precomputed FCD reference statistics) to score generated SMILES against a
held-out reference set. That harness computed validity, uniqueness,
novelty (all divided by the first 10,000 tokens of the submission), and
FCD via a pretrained ChemNet model shipped with the `fcd` PyPI package.
It is course material, not mine to redistribute. The scores in
`results/leaderboard.md` were obtained by submitting generated SMILES to
the challenge server running that harness, not by running it locally.

One trivial exception: `src/molgen/model.py::canonicalize_smiles` is a
three-line "RDKit-canonicalize-or-return-None" function. The original code
imported this from `evaluation.utils._cansmi`; it is reproduced here
(not copied — same well-known pattern, independently present elsewhere in
the original project's own `train_deprecated.py`) so this repository's
`src/` package has no dependency on the excluded harness. It performs no
scoring and contains no course-authored logic.

This project depends on the following third-party libraries, used
unmodified via their standard PyPI packages (see `requirements.txt`):

| Library | Licence |
|---|---|
| PyTorch | BSD-3-Clause |
| RDKit | BSD-3-Clause |
| fcd | MIT |
| NumPy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| matplotlib | PSF-based (matplotlib licence) |
| tqdm | MPL-2.0 / MIT (dual) |

`fcd` (the PyPI package, imported by the excluded evaluation harness, not
by any code in this repository) is listed here only because it is pinned
in `requirements.txt` for completeness with the original project; nothing
in `src/` or `scripts/` imports it.
