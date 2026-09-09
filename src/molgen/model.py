"""LSTM generator model and SMILES canonicalization/validation helpers.

Model architecture and generation logic are unchanged from the original
``mol_gen/train.py``. Two changes from the original file, both legibility-only:

- ``canonicalize_smiles`` replaces the import ``from evaluation.utils import
  _cansmi``. The original evaluation harness is course-provided and is not
  included in this repository (see ``THIRD_PARTY.md``); this is the same
  three-line RDKit "canonicalize or return None" pattern that also appears,
  independently written, in the original ``train_deprecated.py``. It is not
  the harness's scoring logic, just a generic RDKit idiom used to be able to
  import this module without the excluded package.
- ``combined_scorer`` and its ``sigmoid`` helper, present but never called
  in the original ``train.py``, are dropped as dead code left over from the
  reinforcement fine-tuning stage that was implemented and then removed
  before the final submission (see the README's Limitations section).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


def canonicalize_smiles(smiles):
    """Return the RDKit canonical SMILES, or None if it does not parse."""
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def validate_smiles(smiles):
    if not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        lstm_output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(lstm_output)
        return logits, hidden

    def generate(self, vocab, max_content_len, temp=1.0, canonicalize=False, device=None):
        self.eval()
        device = device if device is not None else next(self.parameters()).device
        start_idx = vocab.char_to_index[vocab.start_token]
        end_idx = vocab.char_to_index[vocab.end_token]
        current_char_idx = torch.tensor([[start_idx]], dtype=torch.long).to(device)
        generated_indices = [start_idx]
        hidden = None
        with torch.no_grad():
            for _ in range(max_content_len + 1):
                logits, hidden = self.forward(current_char_idx, hidden)
                last_tkn_logits = logits[:, -1, :] / temp
                probs = F.softmax(last_tkn_logits, dim=-1)
                next_char_idx = torch.multinomial(probs, 1)
                token_id = next_char_idx.item()
                generated_indices.append(token_id)
                if token_id == end_idx:
                    break
                current_char_idx = next_char_idx

            smiles_idx_to_decode = generated_indices[1:]
            if end_idx in smiles_idx_to_decode:
                end_pos = smiles_idx_to_decode.index(end_idx)
                smiles_idx_to_decode = smiles_idx_to_decode[:end_pos]

            generated_smiles = vocab.decode(smiles_idx_to_decode)

            if canonicalize:
                generated_smiles = canonicalize_smiles(generated_smiles)
                if generated_smiles is None:
                    return None

            return generated_smiles

    def sample(self, vocab, n_samples, max_content_len, temp=1.0, canonicalize=False, device=None):
        gen_smiles = []
        for _ in range(n_samples):
            generated_smile = self.generate(vocab, max_content_len, temp, canonicalize, device)
            if generated_smile is not None:
                gen_smiles.append(generated_smile)
        return gen_smiles
