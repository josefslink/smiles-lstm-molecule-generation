"""Character-level vocabulary for SMILES strings.

Unchanged from the original ``mol_gen/train.py``, moved into its own module.
"""

import numpy as np


class Vocabulary:
    """Maps SMILES characters to integer indices, with explicit
    start/end/pad tokens."""

    def __init__(self, smiles_list, start_token='$', end_token='!', pad_token='&'):
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.vocab = sorted(list(set(start_token + end_token + pad_token + ''.join(smiles_list))))
        self.vocab_size = len(self.vocab)
        self.char_to_index = {char: idx for idx, char in enumerate(self.vocab)}
        self.index_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    def encode(self, smiles_string):
        return [self.char_to_index[char] for char in smiles_string]

    def decode(self, index_list):
        return ''.join([self.index_to_char[idx] for idx in index_list])


def encode_smiles_for_training(smiles, vocab, max_length):
    """Wrap a SMILES string with start/end tokens and pad to a fixed length."""
    smiles = vocab.start_token + smiles + vocab.end_token
    smiles = smiles.ljust(max_length + 2, vocab.pad_token)
    smiles = vocab.encode(smiles)
    return np.array(smiles, dtype=np.int64)
