#!/usr/bin/env python
"""Sample SMILES from a trained checkpoint.

Entry point for ``src/molgen/generate.py`` (ported from the submitted
``mol_gen/inference_gen_mols.py``). Adds an explicit ``--seed`` flag (default
42, matching the seed used for training) — the original inference script set
no seed at all, so generation was not reproducible. Keeps the unfiltered
write behaviour: see the README's Results section for why.

Example:
    python scripts/generate.py --checkpoint models/lstm_dl/model_dl_final.pth \\
        --data data/smiles_train.txt --num-samples 15000 --output submission.txt
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from molgen.data import load_preprocess_data  # noqa: E402
from molgen.generate import generate_smiles_from_model, load_model  # noqa: E402
from molgen.train import Config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SMILES from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pth model checkpoint.")
    parser.add_argument("--data", type=str, required=True,
                         help="Path to the training SMILES file, used to rebuild the vocabulary "
                              "and the max content length (must match the file the checkpoint was trained on).")
    parser.add_argument("--num-samples", type=int, default=15000,
                         help="Number of generation attempts (submission used 15000).")
    parser.add_argument("--output", type=str, default="generated_smiles.txt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=Config.embedding_dim)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--num-lstm-layers", type=int, default=Config.num_lstm_layers)
    parser.add_argument("--dropout", type=float, default=Config.dropout_rate)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    set_seed(args.seed, device)

    print(f"Loading vocabulary from {args.data}...")
    _, vocab, max_content_len = load_preprocess_data(args.data, batch_size=1, num_workers=0)
    print(f"Vocab size: {vocab.vocab_size}, max SMILES content length: {max_content_len}")

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(
        args.checkpoint, vocab,
        args.embedding_dim, args.hidden_dim, args.num_lstm_layers, args.dropout,
        device,
    )

    print(f"Generating {args.num_samples} SMILES strings with temperature {args.temperature}...")
    generate_smiles_from_model(
        model, vocab, max_content_len, args.num_samples, args.temperature, args.output, device,
    )
    print(f"Wrote {args.num_samples} lines to {args.output}.")


if __name__ == "__main__":
    main()
