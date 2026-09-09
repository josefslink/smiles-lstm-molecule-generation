# Data

**The training corpus is not included in this repository.** It was
provided by a university course (AI in Life Sciences, summer semester
2025) for a bounded challenge exercise. Course permission to publish this
project covers the code and results, not redistribution of the dataset.

## Format

`smiles_train.txt` — 1,036,643 SMILES strings, one per line, no header, no
other columns. First line:

```
COc1ccc(N2CCN(C(=O)c3cc4ccccc4[nH]3)CC2)cc1
```

Read by `src/molgen/data.py::load_preprocess_data`, which strips each line
and skips blanks. No other preprocessing is applied before building the
character vocabulary.

## Provenance and licence

**Not recorded.** No note anywhere in the surviving project (code,
comments, logs, or my own memory) states where this corpus originally came
from or under what licence it was distributed to the course. The challenge
page was checked for this at Gate 2 (2026-09-08) without resolving it. I am
not guessing a source rather than leave this blank.

## Substitute for reproduction

Any SMILES corpus of a similar size (roughly 10^6 drug-like molecules) is a
structural drop-in for training — for example ChEMBL, MOSES, or GuacaMol.
Format it as one SMILES string per line with no header. However, the
**FCD reference statistics** (`test_stats.p`) used to score submissions on
the challenge server are course-provided and not published here (see
`THIRD_PARTY.md`), so the exact leaderboard FCD score in
`results/leaderboard.md` cannot be reproduced outside the course, whatever
training corpus is used. Validity, uniqueness, and novelty can still be
computed against any training set with `scripts/generate.py`'s output.

## `data/morgan_fingerprints.pkl`

Referenced by `scripts/eda.py` as a fingerprint cache. Not included: 80 MB,
and it is derived entirely from the (also excluded) training corpus. It is
regenerated automatically the first time `scripts/eda.py` is run against a
SMILES file.
