"""Character-level LSTM for de novo SMILES generation.

Split out of the original single-file ``train.py`` / ``inference_gen_mols.py``
scripts for legibility. See the top-level README for what was and was not
changed in this refactor.
"""

__all__ = ["vocab", "data", "model", "train", "generate"]
