"""Training loop for the SMILES LSTM.

Every hyperparameter value is unchanged from the submitted ``mol_gen/train.py``
(git HEAD ``46f2a8a``). What changed for legibility:

- The scattered module-level constants are hoisted into a single ``Config``
  dataclass so ``scripts/train.py`` can override them from the CLI instead of
  editing this file.
- The post-training call to an internal ``evaluate_generated_smiles``
  function (FCD via the ``fcd_torch`` package) is dropped. That function is
  not part of the training loop that produced the published checkpoint, its
  only surviving output is already published as
  ``results/local_eval_earlier_model.json`` (explicitly labelled as not
  comparable to the challenge score), and ``fcd_torch`` is not in
  ``requirements.txt`` with a recorded version — adding one would mean
  guessing a version that was never recorded. See the README's Limitations
  and the repository build notes for the alternatives considered.
- Logging goes to a caller-supplied path instead of a hardcoded
  ``mol_gen/logs/training.log``.
"""

import logging
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from molgen.model import validate_smiles

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Hyperparameters as submitted (HEAD 46f2a8a). See the README's
    architecture note: these constants (HIDDEN_DIM=768, NUM_LSTM_LAYERS=4)
    were set in a cleanup commit made after the run that produced the
    reported FCD score finished, so they probably do not describe the
    weights that produced that score. Published unchanged regardless."""

    seed: int = 42
    num_epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    embedding_dim: int = 128
    hidden_dim: int = 768
    num_lstm_layers: int = 4
    dropout_rate: float = 0.2


def set_seed(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def train_model(model, vocab, dataloader, config: Config, save_dir: str, device: torch.device):
    """Train for ``config.num_epochs`` epochs, saving a checkpoint after every
    epoch plus a final ``model_dl_final.pth``. Unchanged from the original
    ``train_model`` in ``mol_gen/train.py`` other than taking its
    hyperparameters from ``config`` and its device explicitly."""
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, threshold=0.01, threshold_mode='abs')
    logger.info(f'Training model for {config.num_epochs} epochs...')

    all_step_losses = []
    avg_epoch_losses = []
    learning_rates_per_epoch = []
    max_content_len = dataloader.dataset.input_data.shape[1] - 1

    for epoch in range(1, config.num_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates_per_epoch.append(current_lr)
        logger.info(f'Starting epoch {epoch}/{config.num_epochs} with learning rate: {current_lr}')

        epoch_loss_sum = 0
        num_batches_in_epoch = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}", leave=False)
        for batch_idx, (input_data, target_data) in enumerate(progress_bar):
            model.train()

            input_data, target_data = input_data.to(device), target_data.to(device)
            optimizer.zero_grad()
            logits, _ = model(input_data)
            logits = logits.view(-1, logits.size(2))
            target_data = target_data.view(-1)
            loss = criterion(logits, target_data)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            all_step_losses.append(loss.item())
            epoch_loss_sum += loss.item()
            num_batches_in_epoch += 1

            progress_bar.set_postfix(loss=loss.item())

            if batch_idx % 100 == 0:
                model.eval()
                with torch.no_grad():
                    sampled_list = model.sample(vocab, 1, max_content_len, temp=1.0, canonicalize=True, device=device)
                    sample_smiles = sampled_list[0] if sampled_list else "N/A"
                logger.info(
                    f'Epoch {epoch} Step {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f} '
                    f'Sample: {sample_smiles} Valid: {validate_smiles(sample_smiles)}'
                )

        avg_epoch_loss = epoch_loss_sum / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
        avg_epoch_losses.append(avg_epoch_loss)
        logger.info(f'Epoch {epoch} Loss: {avg_epoch_loss:.4f}')
        scheduler.step(avg_epoch_loss)
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch}.pth'))

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_dl_final.pth'))
    logger.info(f'Model saved to {save_dir}')

    return {
        'all_step_losses': all_step_losses,
        'avg_epoch_losses': avg_epoch_losses,
        'learning_rates_per_epoch': learning_rates_per_epoch,
    }
