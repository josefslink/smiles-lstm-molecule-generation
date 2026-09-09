#!/usr/bin/env python
"""Plot step-wise and epoch-average training loss from a training log.

Ported from the MG2 salvage copy of ``plots.py``. Changes from the
original, both legibility-only:

- The absolute-path comments in the original file's ``__main__`` block
  (referring to a local development machine's home directory) are removed.
- The original's "create a dummy log file for testing if none exists"
  fallback — explicitly labelled by its own comments as demonstration
  scaffolding, not part of the analysis — is replaced with a plain error
  if the log file is missing, and paths are CLI arguments instead of
  hardcoded.

The log-parsing regexes and the two plots themselves are unchanged.

Example:
    python scripts/plot_training_curves.py --log-file results/training_losses.txt --output-dir results/figures
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


def extract_losses_from_log(log_file_path):
    """Parse a training log file for step-wise and epoch-average losses.

    Returns a DataFrame with columns 'Epoch', 'Step' (NA for epoch
    averages), 'Loss', 'Type' ('step' or 'epoch_avg').
    """
    step_loss_pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)\s+Step\s+(?P<step>\d+)/\d+\s+Loss:\s+(?P<loss>\d+\.\d+)"
    )
    epoch_avg_loss_pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)\s+Loss:\s+(?P<loss>\d+\.\d+)(?!\s+Sample:)"
    )

    data = []
    with open(log_file_path, 'r') as f:
        for line in f:
            step_match = step_loss_pattern.search(line)
            if step_match:
                data.append({
                    "Epoch": int(step_match.group("epoch")),
                    "Step": int(step_match.group("step")),
                    "Loss": float(step_match.group("loss")),
                    "Type": "step",
                })
            else:
                epoch_avg_match = epoch_avg_loss_pattern.search(line)
                if epoch_avg_match:
                    data.append({
                        "Epoch": int(epoch_avg_match.group("epoch")),
                        "Step": pd.NA,
                        "Loss": float(epoch_avg_match.group("loss")),
                        "Type": "epoch_avg",
                    })

    if not data:
        print("No loss data extracted from the log file.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df['Epoch'] = df['Epoch'].astype(int)
    df['Step'] = df['Step'].astype(pd.Int64Dtype())
    df['Loss'] = df['Loss'].astype(float)
    df['Type'] = df['Type'].astype(str)
    return df


def plot_losses(losses_df, output_dir):
    """Plot step-wise and epoch-average losses into two separate PNGs in
    ``output_dir``."""
    if losses_df.empty:
        print("Cannot plot losses: DataFrame is empty.")
        return

    os.makedirs(output_dir, exist_ok=True)
    step_losses = losses_df[losses_df['Type'] == 'step'].copy()
    epoch_avg_losses = losses_df[losses_df['Type'] == 'epoch_avg']

    if not step_losses.empty:
        plt.figure(figsize=(12, 6))
        step_losses.loc[:, 'GlobalStep'] = np.arange(len(step_losses))
        plt.plot(step_losses['GlobalStep'], step_losses['Loss'], label='Step-wise Loss', alpha=0.7)

        epoch_changes = step_losses.drop_duplicates(subset=['Epoch'], keep='first')
        current_ylim = plt.ylim()
        for _, row in epoch_changes.iterrows():
            plt.axvline(x=row['GlobalStep'], color='r', linestyle='--', alpha=0.5)
            plt.text(row['GlobalStep'] + 5, current_ylim[1] * 0.95, f"Epoch {row['Epoch']}", color='r')

        plt.xlabel("Global Training Step")
        plt.ylabel("Loss")
        plt.title("Step-wise Training Loss")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_step_losses_plot.png"), dpi=300)
        plt.close()
    else:
        print("No step-wise loss data to plot.")

    if not epoch_avg_losses.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_avg_losses['Epoch'], epoch_avg_losses['Loss'], marker='o', linestyle='-', label='Epoch Average Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("Epoch Average Training Loss")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xticks(epoch_avg_losses['Epoch'].unique())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_epoch_losses_plot.png"), dpi=300)
        plt.close()
    else:
        print("No epoch-average loss data to plot.")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training loss curves from a training log.")
    parser.add_argument("--log-file", type=str, default="results/training_losses.txt")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.log_file):
        raise SystemExit(f"Log file not found: {args.log_file}")

    losses_df = extract_losses_from_log(args.log_file)
    if losses_df.empty:
        raise SystemExit(f"No data extracted from {args.log_file}.")

    print("--- All Extracted Losses ---")
    print(losses_df)
    plot_losses(losses_df, args.output_dir)
    print(f"Plots written to {args.output_dir}")


if __name__ == "__main__":
    main()
