import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import time
import json
from sklearn.model_selection import train_test_split
from supervised_contrastive_utils import MultiClassSkinDataset, prepare_df, recalculate_weights, run_experiment_torch, save_predictions_torch
from torchvision import models

class SimpleMobileNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.base = models.mobilenet_v2(weights='DEFAULT')
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        return self.base(x)

class CrossEntropyOnlyWrapper(nn.Module):
    """Wrapper to make CrossEntropyLoss work with existing training loop"""
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, y_pred, z1=None, z2=None, y_true=None):
        # Handle both cases where y_true might be 1D or 2D
        if y_true is not None:
            if y_true.dim() > 1:  # If it's [batch, 2] tensor
                labels = y_true[:, 0].long()
            else:  # If it's just [batch] tensor
                labels = y_true.long()
            loss = self.ce_loss(y_pred, labels)
        else:
            loss = self.ce_loss(y_pred, torch.zeros(y_pred.size(0), dtype=torch.long, device=y_pred.device))
        
        return loss, {'ce': loss.item()}

def dict_collate(batch):
    """Collate function that maintains dictionary format but ensures proper shapes"""
    return {
        'x': torch.stack([item['x'] for item in batch]),
        'label': torch.tensor([item['label'] for item in batch], dtype=torch.long),
        'A': torch.tensor([item['A'] for item in batch], dtype=torch.float),
        'weight': torch.tensor([item['weight'] for item in batch], dtype=torch.float)
    }

def main(seed):
    batch_size = 32
    lr = 1e-4
    ita_threshold = 41
    exp_name = f"ce_baseline_{seed}"
    lesion_types = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    TRAINING_PATH = "ISIC2018_Task3_Training_Input"
    df = prepare_df(TRAINING_PATH).dropna()
    print(f"{len(df)} samples are used")
    df['A'] = (df['estimated_ita'] > ita_threshold).astype(int)
    df_train_temp = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Train-test split
    idx_train, idx_test = train_test_split(df_train_temp.index, 
                                         stratify=df_train_temp["lesion"],
                                         test_size=0.2,
                                         random_state=seed)
    df_train = df_train_temp.loc[idx_train].copy()
    df_test = df_train_temp.loc[idx_test].copy()

    # Train-val split
    idx_train, idx_valid = train_test_split(df_train.index,
                                          stratify=df_train["lesion"],
                                          test_size=0.2,
                                          random_state=seed)
    df_valid = df_train.loc[idx_valid].copy()
    df_train = df_train.loc[idx_train].copy()

    # Recalculate weights
    df_train = recalculate_weights(df_train, label_col='lesion')
    df_valid = recalculate_weights(df_valid, label_col='lesion')
    df_test = recalculate_weights(df_test, label_col='lesion')

    # Data loaders using dict_collate
    train_loader = DataLoader(
        MultiClassSkinDataset(df_train, TRAINING_PATH, contrastive=False),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dict_collate,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        MultiClassSkinDataset(df_valid, TRAINING_PATH, contrastive=False),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dict_collate,
        num_workers=4,
        pin_memory=True
    )

    # Model and training setup
    model = SimpleMobileNet(len(lesion_types)).to(device)
    criterion = CrossEntropyOnlyWrapper()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    trained_model = run_experiment_torch(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        exp_name=exp_name,
        epochs=100,
        patience=15,
        device=device,
        contrastive=False
    )

    # Evaluation
    fairness_metrics = save_predictions_torch(
        model=trained_model,
        df_test=df_test,
        data_path=TRAINING_PATH,
        exp_name=exp_name,
        class_names=lesion_types,
        batch_size=batch_size,
        device=device
    )

if __name__ == "__main__":
    seeds = [10, 20, 30, 40, 50]
    for seed in seeds:
        print(f"\nRunning experiment with seed {seed}")
        main(seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()