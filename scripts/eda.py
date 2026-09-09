#!/usr/bin/env python
"""Exploratory data analysis on the training SMILES / Morgan fingerprints.

Ported from ``mol_gen/eda.py``. Hardcoded paths (9 sites in the original)
are now CLI arguments with the same default relative layout; no other logic
is changed. Note: ``load_data`` calls ``pandas.read_csv`` without
``header=None``, so the first line of the input file is silently treated as
a header and dropped from the analysis. This is unchanged from the
original — flagged here rather than fixed, per the rule against altering
behaviour during this refactor.

Example:
    python scripts/eda.py --smiles-file data/smiles_train.txt --output-dir results/figures
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from tqdm import tqdm


def load_data(file_path):
    """Load SMILES data. Note: uses the pandas default header inference, so
    the first line of a headerless SMILES file is dropped (unchanged from
    the original ``mol_gen/eda.py``)."""
    try:
        smiles_csv = pd.read_csv(file_path)
        smiles_csv = smiles_csv.dropna()
        n_smiles = len(smiles_csv)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None, 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, 0
    return smiles_csv, n_smiles


def get_MorganFingerprints(smiles_df, radius=2, fpSize=2048):
    morgan_fingerprints = []
    mg = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
    for smiles in tqdm(smiles_df.iloc[:, 0], desc="Generating Morgan fingerprints", unit="SMILES"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            morgan_fp = mg.GetFingerprint(mol)
            morgan_fingerprints.append(morgan_fp)
        else:
            print(f"Invalid SMILES: {smiles}")
    return morgan_fingerprints


def calc_sparsity(fingerprints: list, fpSize: int = 2048):
    if not fingerprints:
        return [], 0.0

    individual_sparsity = []
    total_zeros = 0
    total_bits_in_data_set = 0

    for fp in fingerprints:
        num_on_bits = fp.GetNumOnBits()
        num_zero_bits = fpSize - num_on_bits
        individual_sparsity.append(num_zero_bits / fpSize)
        total_zeros += num_zero_bits
        total_bits_in_data_set += fpSize

    overall_sparsity = total_zeros / total_bits_in_data_set if total_bits_in_data_set > 0 else 0.0
    return individual_sparsity, overall_sparsity


def cacl_bit_frequencies(fingerprints: list, fpSize: int = 2048):
    num_fingerprints = len(fingerprints)
    if num_fingerprints == 0:
        print("No fingerprints to analyze.")
        return np.array([])

    bit_counts = np.zeros(fpSize, dtype=np.int32)
    for fp in tqdm(fingerprints, desc='Calculating bit frequencies', unit='fingerprint'):
        if fp is not None:
            on_bits = list(fp.GetOnBits())
            if on_bits:
                bit_counts[on_bits] += 1

    bit_frequencies = bit_counts / num_fingerprints if num_fingerprints > 0 else np.zeros(fpSize, dtype=float)
    return bit_frequencies


def calc_pairwise_tanimoto_similarity(fingerprints: list, sample_size: int = 2000):
    n_total = len(fingerprints)
    if n_total < 2:
        print("Not enough fingerprints to calculate pairwise Tanimoto similarity.")
        return None

    if sample_size is not None and sample_size < n_total:
        actual_sample_size = max(2, sample_size)
        sampled_indices = np.random.choice(n_total, actual_sample_size, replace=False)
        fps_to_compare = [fingerprints[i] for i in sampled_indices]
        print(f"Calculating pairwise Tanimoto similarity for random sample of {len(fps_to_compare)} fingerprints.")
    else:
        fps_to_compare = fingerprints
        if n_total > 5000:
            print(f"Warning: Calculating pairwise Tanimoto similarity for {n_total} fingerprints. This may take a while.")
        else:
            print(f"Calculating all pairwise Tanimoto similarities for {len(fps_to_compare)} fingerprints.")

    similarities = []
    n_compare = len(fps_to_compare)
    for i in tqdm(range(n_compare), desc="Calculating Tanimoto similarity", unit="pair"):
        for j in range(i + 1, n_compare):
            sim = DataStructs.TanimotoSimilarity(fps_to_compare[i], fps_to_compare[j])
            similarities.append(sim)

    return np.array(similarities)


def plot_tanimoto_similarity(similarities, output_path):
    if not isinstance(similarities, np.ndarray) or similarities.size == 0:
        print("No similarity scores to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=50, color='coral', edgecolor='black', density=True)
    plt.title('Distribution of Pairwise Tanimoto Similarities')
    plt.xlabel('Tanimoto Similarity Score')
    plt.ylabel('Density')
    plt.xlim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Tanimoto similarity distribution plot saved to {output_path}")


def plot_bit_frequencies(bit_frequencies, fpSize: int, output_path):
    plt.figure(figsize=(10, 6))
    plt.bar(range(fpSize), bit_frequencies, color='blue', alpha=0.7)
    plt.title('Frequency of each bit in Morgan fingerprints being "on"')
    plt.xlabel('Bit index')
    plt.ylabel('Frequency (Proportion of fingerprints)')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Bit frequency plot saved to {output_path}")


def plot_sparsity(individual_sparsity, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(individual_sparsity, bins=50, color='blue', alpha=0.7, density=True)
    plt.title('Sparsity Distribution of Morgan Fingerprints')
    plt.xlabel('Sparsity')
    plt.ylabel('Density')
    plt.grid(True, linestyle='--', alpha=0.6)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Sparsity distribution plot saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="EDA on the training SMILES / Morgan fingerprints.")
    parser.add_argument("--smiles-file", type=str, default="data/smiles_train.txt")
    parser.add_argument("--fingerprints-cache", type=str, default="data/morgan_fingerprints.pkl")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    parser.add_argument("--fp-size", type=int, default=2048)
    parser.add_argument("--diversity-sample-size", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    fp_size = args.fp_size
    diversity_sample_size = args.diversity_sample_size

    smiles_df, n_smiles = load_data(args.smiles_file)
    if smiles_df is None:
        return

    if not os.path.exists(args.fingerprints_cache):
        print("Morgan fingerprints file not found. Generating fingerprints...")
        morgan_fingerprints = get_MorganFingerprints(smiles_df, fpSize=fp_size)
        print(f"Number of Morgan fingerprints: {len(morgan_fingerprints)}")
        os.makedirs(os.path.dirname(args.fingerprints_cache) or ".", exist_ok=True)
        with open(args.fingerprints_cache, 'wb') as f:
            pickle.dump(morgan_fingerprints, f)
        print(f"Morgan fingerprints saved to {args.fingerprints_cache}")
    else:
        print("Morgan fingerprints file already exists. Loading fingerprints...")
        with open(args.fingerprints_cache, 'rb') as f:
            morgan_fingerprints = pickle.load(f)
        print(f"Number of loaded Morgan fingerprints: {len(morgan_fingerprints)}")

    if not morgan_fingerprints:
        print("No valid fingerprints found. Please check the input data.")
        return

    individual_sparsity, overall_sparsity = calc_sparsity(morgan_fingerprints, fp_size)
    print(f"Overall fingerprint sparsity: {overall_sparsity:.4f}")
    print(f"Average individual fingerprint sparsity: {np.mean(individual_sparsity):.4f}")
    print(f"Median individual fingerprint sparsity: {np.median(individual_sparsity):.4f}")
    print(f"Standard deviation of individual fingerprint sparsity: {np.std(individual_sparsity):.4f}")
    print(f"Minimum individual fingerprint sparsity: {np.min(individual_sparsity):.4f}")
    print(f"Maximum individual fingerprint sparsity: {np.max(individual_sparsity):.4f}")
    plot_sparsity(individual_sparsity, os.path.join(args.output_dir, 'sparsity_distribution.png'))

    print("\nCalculating bit frequencies...")
    bit_frequencies = cacl_bit_frequencies(morgan_fingerprints, fp_size)
    if bit_frequencies.size > 0:
        plot_bit_frequencies(bit_frequencies, fp_size, os.path.join(args.output_dir, 'bit_frequency_distribution.png'))

        num_bits_never_on = np.sum(bit_frequencies == 0)
        num_bits_always_on = np.sum(bit_frequencies == 1)
        num_bits_sometimes_on = fp_size - num_bits_never_on - num_bits_always_on

        print(f"\n--- Bit Frequency Stats (Total Bits: {fp_size}) ---")
        print(f"Number of bits that are NEVER 'on': {num_bits_never_on} ({num_bits_never_on/fp_size*100:.2f}%)")
        print(f"Number of bits that are ALWAYS 'on': {num_bits_always_on} ({num_bits_always_on/fp_size*100:.2f}%)")
        print(f"Number of bits SOMETIMES 'on': {num_bits_sometimes_on} ({num_bits_sometimes_on/fp_size*100:.2f}%)")
        if num_bits_sometimes_on > 0:
            active_bit_frequencies = bit_frequencies[bit_frequencies > 0]
            if active_bit_frequencies.size > 0:
                print(f"Lowest frequency among 'on' bits: {np.min(active_bit_frequencies):.4f}")
                print(f"Highest frequency among 'on' bits: {np.max(active_bit_frequencies):.4f}")
                print(f"Average frequency among 'on' bits: {np.mean(active_bit_frequencies):.4f}")

    print("\nCalculating dataset diversity (Tanimoto similarities)...")
    pairwise_similarities = calc_pairwise_tanimoto_similarity(morgan_fingerprints, sample_size=diversity_sample_size)
    if pairwise_similarities is not None and isinstance(pairwise_similarities, np.ndarray) and pairwise_similarities.size > 0:
        plot_tanimoto_similarity(pairwise_similarities, os.path.join(args.output_dir, 'similarity_distribution.png'))
        print(f"\n--- Tanimoto Similarity Stats (sample of {diversity_sample_size} fingerprints, {len(pairwise_similarities)} pairs) ---")
        print(f"Average Tanimoto similarity: {np.mean(pairwise_similarities):.4f}")
        print(f"Median Tanimoto similarity: {np.median(pairwise_similarities):.4f}")
        print(f"Standard deviation of Tanimoto similarity: {np.std(pairwise_similarities):.4f}")
        print(f"Minimum Tanimoto similarity: {np.min(pairwise_similarities):.4f}")
        print(f"Maximum Tanimoto similarity: {np.max(pairwise_similarities):.4f}")
        print(f"25th percentile Tanimoto similarity: {np.percentile(pairwise_similarities, 25):.4f}")
        print(f"75th percentile Tanimoto similarity: {np.percentile(pairwise_similarities, 75):.4f}")
    else:
        print("Similarity calculation was skipped or failed (e.g. not enough fingerprints).")


if __name__ == "__main__":
    main()
