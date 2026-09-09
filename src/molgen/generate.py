"""Sampling SMILES from a trained checkpoint.

Behaviourally unchanged from the submitted ``mol_gen/inference_gen_mols.py``
(git HEAD ``46f2a8a``, functionally identical to the script inside the
submission zip — see the README's Results section for the diff). In
particular, this **keeps the unfiltered write**: every attempt is written to
the output file, including the literal string ``None`` for a failed
canonicalization. That is the behaviour that produced the reported
FCD/validity/uniqueness/novelty (submission #2); resampling on failure would
change the metrics and is listed as a future improvement in the README
instead of being applied here.
"""

import torch
from tqdm import tqdm

from molgen.model import LSTM_Model


def load_model(model_path, vocab, embedding_dim, hidden_dim, num_lstm_layers, dropout_rate, device):
    pad_idx = vocab.char_to_index[vocab.pad_token]
    model = LSTM_Model(
        vocab_size=vocab.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_lstm_layers,
        dropout=dropout_rate,
        pad_idx=pad_idx,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def generate_smiles_from_model(model, vocab, max_content_len, num_samples_to_attempt, temperature, output_file, device):
    """Generate ``num_samples_to_attempt`` samples and write every one of
    them to ``output_file``, one per line. A failed canonicalization is
    written as the literal string ``None`` — unchanged from the submitted
    script; see the module docstring."""
    generated_smiles_list = []
    for _ in tqdm(range(num_samples_to_attempt), desc="Generating SMILES"):
        smile = model.generate(vocab, max_content_len, temp=temperature, canonicalize=True, device=device)
        generated_smiles_list.append(smile)

    with open(output_file, 'w') as f:
        for smiles_string in generated_smiles_list:
            f.write(f"{smiles_string}\n")

    return generated_smiles_list
