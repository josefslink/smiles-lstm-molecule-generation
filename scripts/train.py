#!/usr/bin/env python
"""Train the SMILES LSTM.

Entry point for ``src/molgen/train.py``. Replaces the original interactive
"Found existing model at <path>. Skip training? (y/n)" prompt with explicit
``--skip-training`` / ``--checkpoint`` flags so the script can run
non-interactively.

Example:
    python scripts/train.py --data data/smiles_train.txt --output-dir models/lstm_dl
"""

import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from molgen.data import load_preprocess_data  # noqa: E402
from molgen.model import LSTM_Model  # noqa: E402
from molgen.train import Config, set_seed, train_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train the character-level SMILES LSTM.")
    parser.add_argument("--data", type=str, required=True,
                         help="Path to a one-SMILES-per-line training file (see data/README.md).")
    parser.add_argument("--output-dir", type=str, default="models/lstm_dl",
                         help="Directory to save per-epoch and final checkpoints.")
    parser.add_argument("--log-file", type=str, default="logs/training.log")
    parser.add_argument("--epochs", type=int, default=Config.num_epochs)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--embedding-dim", type=int, default=Config.embedding_dim)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--num-lstm-layers", type=int, default=Config.num_lstm_layers)
    parser.add_argument("--dropout", type=float, default=Config.dropout_rate)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--skip-training", action="store_true",
                         help="Skip training and load --checkpoint instead.")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Checkpoint to load when --skip-training is set.")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    config = Config(
        seed=args.seed,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_lstm_layers=args.num_lstm_layers,
        dropout_rate=args.dropout,
    )
    set_seed(config.seed, device)

    dataloader, vocab, max_content_len = load_preprocess_data(args.data, config.batch_size)
    logger.info(f"Data loaded. Vocabulary size: {vocab.vocab_size}, max SMILES length: {max_content_len}")

    pad_idx = vocab.char_to_index[vocab.pad_token]
    model = LSTM_Model(
        vocab_size=vocab.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_lstm_layers,
        dropout=config.dropout_rate,
        pad_idx=pad_idx,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"LSTM model initialized with {num_params:,} trainable parameters.")

    if args.skip_training:
        if not args.checkpoint:
            parser_error = "--skip-training requires --checkpoint"
            logger.error(parser_error)
            raise SystemExit(parser_error)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        logger.info(f"Loaded checkpoint from {args.checkpoint}, skipping training.")
    else:
        logger.info("Starting training...")
        train_model(model, vocab, dataloader, config, args.output_dir, device)
        logger.info("Training finished.")


if __name__ == "__main__":
    main()
