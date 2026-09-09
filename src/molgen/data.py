"""Loading and encoding the training corpus.

Unchanged from the original ``mol_gen/train.py``, moved into its own module.
The training corpus path is now a function argument instead of a hardcoded
module-level constant (see ``scripts/train.py`` for the CLI flag).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from molgen.vocab import Vocabulary, encode_smiles_for_training


class SmilesDataset(Dataset):
    def __init__(self, encoded_smiles_array):
        self.input_data = torch.tensor(encoded_smiles_array[:, :-1], dtype=torch.long)
        self.target_data = torch.tensor(encoded_smiles_array[:, 1:], dtype=torch.long)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]


def load_preprocess_data(smiles_file_path, batch_size, num_workers=0):
    """Read a one-SMILES-per-line file, build the vocabulary, and return a
    DataLoader plus the vocabulary and the max content length (SMILES length
    without start/end tokens)."""
    smiles_list = []

    with open(smiles_file_path, 'r') as file:
        for line in file:
            smiles = line.strip()
            if len(smiles) > 0:
                smiles_list.append(smiles)

    smiles_vocab = Vocabulary(smiles_list)
    max_length_gen_content = max(len(smiles) for smiles in smiles_list)

    encoded_smiles_list = [
        encode_smiles_for_training(smiles, smiles_vocab, max_length_gen_content)
        for smiles in smiles_list
    ]

    encoded_smiles_array = np.array(encoded_smiles_list)
    smiles_dataset = SmilesDataset(encoded_smiles_array)

    smiles_dataloader = DataLoader(
        smiles_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return smiles_dataloader, smiles_vocab, max_length_gen_content
