"""
Pretrain the DCG (Diffusion-driven Classifier Guidance) auxiliary model
on the Chest X-ray dataset before running the full DiffMIC training.

Usage:
    python pretraining/pretrain_dcg.py
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from easydict import EasyDict

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretraining.dcg import DCG
from dataloader.loading import ChestXrayDataSet


def pretrain_dcg(config_path, num_epochs=20, batch_size=32, lr=1e-4, save_path=None):
    """
    Pretrain the DCG model as a multi-label classifier on Chest X-ray data.
    """
    # Load config
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = ChestXrayDataSet(
        csv_file=config.data.traindata,
        data_dir=config.data.data_dir,
        train=True
    )
    test_dataset = ChestXrayDataSet(
        csv_file=config.data.testdata,
        data_dir=config.data.data_dir,
        train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Initialize DCG model
    model = DCG(config).to(device)
    print("DCG model initialized.")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Default save path
    if save_path is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ckpt')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'chest_aux_model.pth')

    best_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # DCG forward returns: y_fusion, y_global, y_local, patches, patch_attns, saliency_map
            y_fusion, y_global, y_local, _, _, _ = model(images)

            # Compute losses for all three outputs
            loss_fusion = criterion(y_fusion, labels)
            loss_global = criterion(y_global, labels)
            loss_local = criterion(y_local, labels)
            loss = loss_fusion + 0.5 * loss_global + 0.5 * loss_local

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        scheduler.step()
        avg_train_loss = running_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                y_fusion, y_global, y_local, _, _, _ = model(images)
                loss = criterion(y_fusion, labels)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Save as tuple — diffuser_trainer.py loads with torch.load(...)[0]
            torch.save((model.state_dict(),), save_path)
            print(f"  -> Best model saved to {save_path}")

    # Also save final model
    final_path = save_path.replace('.pth', '_final.pth')
    torch.save((model.state_dict(),), final_path)
    print(f"\nFinal model saved to {final_path}")
    print(f"Best model (val_loss={best_loss:.4f}) saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain DCG auxiliary model")
    parser.add_argument("--config", type=str,
                        default="/kaggle/working/DiffMICv2/configs/placental.yml",
                        help="Path to config YAML file")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save the checkpoint")
    args = parser.parse_args()

    pretrain_dcg(
        config_path=args.config,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path
    )
