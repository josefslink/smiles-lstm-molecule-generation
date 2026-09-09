# SMILES LSTM molecule generation

Character-level LSTM for de novo SMILES generation, evaluated by FCD,
validity, uniqueness and novelty.

## 1. Problem

De novo molecular design asks a model to propose new, chemically valid
molecules that look like they belong to a known drug-like chemical space.
It matters because the enumerable small-molecule space is on the order of
10^60 and a generative model that reproduces the statistics of a real
compound library gives medicinal chemists a tractable starting set. Here
the model learns a distribution over SMILES strings from ~1.04 million
drug-like molecules and is judged on how closely its samples match a
held-out reference set.

## 2. Approach

Character-level tokenisation over a vocabulary built from the training
corpus's characters plus explicit start (`$`), end (`!`), and pad (`&`)
tokens. A 4-layer LSTM as published — embedding 128, hidden 768, dropout
0.2 — **but see the architecture note in Limitations**: the constants as
published probably do not describe the weights that produced the reported
score, and this section is not asserting otherwise. Trained for 25 epochs
with Adam (lr 1e-3, weight decay 1e-5), batch size 128,
`ReduceLROnPlateau` (factor 0.5, patience 4), cross-entropy loss ignoring
padding, gradient clipped to norm 1.0. Sampling is temperature-1.0
autoregressive decoding with RDKit canonicalisation, 15,000 attempts per
submission.

## 3. Results

**Evaluation setting:** public test set via the university challenge
server. Metrics are computed on the **first 10,000 whitespace-separated
tokens** of the submitted file; validity, uniqueness and novelty all use
that count (10,000) as the denominator, and FCD is Fréchet ChemNet
Distance against course-provided reference statistics. Single run, seed
42, no repeats — no error bars.

| Run | FCD (lower better) | Novelty | Uniqueness | Validity | Date |
|---|---|---|---|---|---|
| **This code** (submission #2) | **0.301** | 0.915 | 0.963 | 0.963 | 2025-05-26 |
| Earlier run (submission #1) | 0.283 | 0.953 | 1.000 | 1.000 | 2025-05-25 |

**This repository publishes the code behind submission #2.** Submission #1
scored marginally better on three of four metrics and came from an earlier
code state (commit `8c45a7c`) with a different checkpoint and, crucially,
an output filter that discarded failed canonicalisations before writing —
that filter is why its validity and uniqueness are exactly 1.000. Removing
that filter in the final cleanup is what cost about 3.7 points of validity
in the published version. The code that produced submission #1 is not
published here, and none of its outputs survive. Full detail, including
why the two submissions' metrics move together the way they do, is in
[`results/leaderboard.md`](results/leaderboard.md).

Also worth noting, and kept clearly separate: an internal evaluation of an
**earlier development checkpoint**
([`results/local_eval_earlier_model.json`](results/local_eval_earlier_model.json) —
validity 0.969, novelty 0.961, FCD 6.85 on 1,000 samples via the
`fcd-torch` library) is **not comparable** to the challenge score above —
different library, different sample size, different model.

## 4. Reproduction

```
pip install -r requirements.txt
```

Obtain a training corpus — the original is not redistributable, see
[`data/README.md`](data/README.md) for the format and a substitute.

```
python scripts/train.py --data <corpus>
python scripts/generate.py --checkpoint <ckpt> --data <corpus> --num-samples 15000 --output submission.txt
```

Recorded wall-clock for the training run that produced the results above:
**16 h 50 min for 25 epochs on one GPU** (2025-05-25 13:01:57 →
2025-05-26 05:51:49, from `results/training_losses.txt`), 8,099 steps per
epoch. GPU model **not recorded**.

**No trained weights are shipped.** The checkpoints from the run that
produced the reported score did not survive the original project's
backups (see the architecture note in Limitations), so reproducing the
result means retraining from scratch — it cannot be done by downloading a
checkpoint from this repository.

## 5. Repo structure

```
README.md
LICENSE                       MIT
THIRD_PARTY.md                 course evaluation harness excluded; library licences
requirements.txt               pinned, reused verbatim from the submitted project
.gitignore
src/molgen/
  __init__.py
  vocab.py                     character vocabulary, start/end/pad tokens
  data.py                      SMILES file loading, encoding, DataLoader
  model.py                     LSTM model, canonicalization/validation helpers
  train.py                     Config dataclass, training loop
  generate.py                  checkpoint loading, sampling loop
scripts/
  train.py                     CLI entry point (was mol_gen/train.py __main__)
  generate.py                  CLI entry point (was inference_gen_mols.py)
  eda.py                       exploratory data analysis (was mol_gen/eda.py)
  plot_training_curves.py      loss-curve plots (was MG2 plots.py)
data/README.md                 schema, provenance (not recorded), substitute corpus
results/
  leaderboard.md               both challenge submissions, full explanation
  local_eval_earlier_model.json  earlier, non-comparable internal evaluation
  training_losses.txt          272 KB step/epoch loss log from the published run
  figures/                     3 loss-curve plots + 3 EDA plots
```

## 6. Limitations and next steps

**Architecture note.** The published `train.py` declares `HIDDEN_DIM = 768`
and `NUM_LSTM_LAYERS = 4`. These values were set in a cleanup commit made
about seven hours after the 16h50m training run finished and roughly two
hours before submission — too late for any retraining. The model that
produced the reported FCD of 0.301 was therefore probably a 1024-unit,
5-layer network, matching the two commits that bracket the training run.
No checkpoint from that run survives, so this cannot be confirmed, and the
code is published exactly as submitted rather than edited to match the
inference. If a checkpoint or a clear memory of the cleanup edit ever
surfaces, this note should be corrected and the basis for the correction
stated plainly.

The rest, each specific and none fixed — this repo is a record of what was
actually submitted, not a cleaned-up version of it:

- **Single seed, single run, no ablation.** The architecture change
  between submissions #1 and #2 was not controlled for; there is no
  experiment isolating its effect from the output-filter change discussed
  in Results.
- **Metrics come from one 10,000-molecule sample**; FCD on that sample
  size is noisy, and there is no repeat to bound that noise.
- **The generator writes unparseable placeholders instead of resampling.**
  A failed canonicalisation becomes the literal string `None` in the
  output file rather than triggering another attempt, which costs
  validity outright. Resampling until 15,000 valid molecules are collected
  is the obvious first fix — deliberately not applied here, since it would
  change the reported metrics.
- **Novelty means only "not in the training set."** It says nothing about
  synthesisability or drug-likeness. QED/SA filtering and a
  property-conditioned objective would be the natural next step.
- **A reinforcement fine-tuning stage was implemented and then removed**
  before the final submission (see `training_metrics/finetuning_metrics.json`
  in the original project, not carried into this repository — 49
  iterations of loss/reward from an abandoned branch). Its effect on the
  final result was never measured cleanly, since it predates the run that
  produced the published checkpoint.

## 7. Provenance

University challenge submission, AI in Life Sciences, summer semester
2025. Solo work. Development 2025-05-11 to 2025-05-26; the two challenge
submissions are dated 2025-05-25 and 2025-05-26 (see `results/leaderboard.md`).
All code here is mine. The evaluation harness that produced the scores was
provided by the course and is not included (see `THIRD_PARTY.md`); the
training corpus and the FCD reference statistics were course-provided and
are not redistributed (see `data/README.md`). Published as a fresh
repository — the original six-commit development history is not carried
over. **No grade is stated anywhere in this repository.**
