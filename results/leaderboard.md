# Leaderboard results

**Evaluation setting:** public test set via the university challenge server.
Metrics are computed on the **first 10,000 whitespace-separated tokens** of
the submitted file; validity, uniqueness and novelty all use that count
(10,000) as the denominator. FCD (Fréchet ChemNet Distance) is computed
against course-provided reference statistics. Single run each, seed 42, no
repeats — no error bars.

| Submission | Date | FCD (lower better) | Novelty | Uniqueness | Validity |
|---|---|---|---|---|---|
| **#2 — this code** | 2025-05-26 | **0.301** | 0.915 | 0.963 | 0.963 |
| #1 — earlier code state | 2025-05-25 | 0.283 | 0.953 | **1.000** | **1.000** |

**This repository publishes the code that produced submission #2.** No
generated-SMILES file survives for either submission — the table above is
transcribed from the challenge server, not recomputed from a stored output.

## Why #1 scored better on three of four metrics

Submission #1 was produced by an earlier commit (`8c45a7c`, 2025-05-25
16:01) whose inference loop filtered its output:

```python
if smile:
    generated_smiles_list.append(smile)
```

Every token written was therefore already valid, which forces validity and
(absent duplicates) uniqueness to exactly 1.000. The version published in
this repository (HEAD `46f2a8a`, submission #2) drops that filter —
`generated_smiles_list.append(smile)` with no condition — so a failed
canonicalization is written as the literal string `None`, which the
evaluator's RDKit canonicalization rejects. Validity and uniqueness fall
together to 0.963 (9,630 of 10,000 tokens valid and, in this run, all of
those distinct). This is inferred from reading both commits' code and the
evaluator's counting logic, not from a preserved output file — see the main
README's Results section.

## Architecture caveat

The constants declared in the published `train.py` (`HIDDEN_DIM=768`,
`NUM_LSTM_LAYERS=4`) were set in a commit made after the 16h50m training
run that produced these scores had already finished. The checkpoint that
actually generated submission #2 was almost certainly a 1024-hidden,
5-layer network. No checkpoint from that run survives to confirm this
either way. Full reasoning in the main README's Limitations section.

## Not included here

`results/local_eval_earlier_model.json` — an internal, pre-submission
diagnostic on a different (1,000-sample) evaluation, using a different FCD
implementation (`fcd-torch`). Not comparable to the table above; kept
separate and clearly labelled rather than mixed into this file.
