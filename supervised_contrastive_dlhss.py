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
from supervised_contrastive_loss import TPRLoss, CombinedLoss, DPLoss
from supervised_contrastive_loss import SupConLoss
from torchvision import models
import torch.nn.functional as F
from models.simclr import SimCLR_MobileNetV2
from loss.nt_xent import nt_xent_loss

def pretrain_simclr(df_train, data_path, batch_size=128, epochs=100, device='cpu'):
    train_data = MultiClassSkinDataset(df_train, data_path, contrastive=True)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=0, pin_memory=False, collate_fn=contrastive_collate)
    
    model = ContrastiveMobileNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    supcon_loss = SupConLoss(temperature=0.07, base_temperature=0.07)
    
    print(f"Starting SimCLR pretraining with {len(train_data)} samples...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for batch in train_loader:
            x1, x2 = batch['x1'].to(device), batch['x2'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            z1 = model(x1, return_projection=True)
            z2 = model(x2, return_projection=True)
            
            features = torch.stack([z1, z2], dim=1)
            
            loss = supcon_loss(features, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            del x1, x2, z1, z2, features, loss
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs} | "f"Avg Loss: {avg_loss:.4f} | "f"Time: {epoch_time:.1f}s")
        
        if not torch.isfinite(torch.tensor(avg_loss)) or avg_loss > 1e6:
            print(f"Stopping early due to unstable loss: {avg_loss}")
            break
    
    print("SimCLR pretraining completed!")
    
    return model

class ContrastiveMobileNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.encoder = models.mobilenet_v2(weights='DEFAULT').features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.feature_projection = nn.Sequential(
            nn.Linear(1280, 2048),
            nn.ReLU(inplace=True)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128) 
        )
        
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x, return_embedding=False, return_projection=False):
        features = self.encoder(x)
        features = self.pool(features)
        features = torch.flatten(features, 1)
        
        r = self.feature_projection(features) 
        r = F.normalize(r, dim=1)  
        
        if return_embedding:
            return r
        elif return_projection:
            z = self.projection(r)  
            z = F.normalize(z, dim=1)  
            return z
        else:
            return self.classifier(r)

def contrastive_collate(batch):
    return {
        'x1': torch.stack([item['x1'] for item in batch]),
        'x2': torch.stack([item['x2'] for item in batch]),
        'label': torch.tensor([item['label'] for item in batch], dtype=torch.long),
        'A': torch.tensor([item['A'] for item in batch], dtype=torch.float),
        'weight': torch.tensor([item['weight'] for item in batch], dtype=torch.float)
    }    

def standard_collate(batch):
    return {
        'x': torch.stack([item['x'] for item in batch]),
        'label': torch.tensor([item['label'] for item in batch], dtype=torch.long),
        'A': torch.tensor([item['A'] for item in batch], dtype=torch.float),
        'weight': torch.tensor([item['weight'] for item in batch], dtype=torch.float)
    }

def main(seed):
    batch_size = 32
    lr = 1e-5
    ita_threshold = 41
    exp_name = f"dlhss_tpr_only_{seed}"
    lesion_types = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    TRAINING_PATH = r"D:\ISI_task_3\RevisitingSkinToneFairness-main\ISIC2018_Task3_Training_Input"
    df = prepare_df(TRAINING_PATH).dropna()
    print(f"{len(df)} samples are used")
    df['A'] = (df['estimated_ita'] > ita_threshold).astype(int)
    df_train_temp = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df_train, df_test = train_test_split(df_train_temp, test_size=0.2, random_state=seed, stratify=df_train_temp[["lesion", "A"]])
    idx_train, idx_valid = train_test_split(df_train.index, stratify=df_train["lesion"], test_size=0.20, random_state=seed, shuffle=True)

    df_valid = df_train.loc[idx_valid,:]
    df_train = df_train.loc[idx_train,:]

    df_train = recalculate_weights(df_train)
    df_valid = recalculate_weights(df_valid)
    df_test = recalculate_weights(df_test)

    train_data = MultiClassSkinDataset(df_train, TRAINING_PATH, contrastive=True, shuffle=True)
    valid_data = MultiClassSkinDataset(df_valid, TRAINING_PATH, shuffle=False)
    test_data = MultiClassSkinDataset(df_test, TRAINING_PATH, shuffle=False)

    train_loader = DataLoader(MultiClassSkinDataset(df_train, TRAINING_PATH, contrastive=True),batch_size=batch_size, shuffle=True, collate_fn=contrastive_collate,num_workers=0, pin_memory=False)
    valid_loader = DataLoader(MultiClassSkinDataset(df_valid, TRAINING_PATH, contrastive=False),batch_size=batch_size,shuffle=False, collate_fn=standard_collate,num_workers=0, pin_memory=False)
    test_loader = DataLoader(MultiClassSkinDataset(df_test, TRAINING_PATH, contrastive=False), 
                           batch_size=batch_size, shuffle=False, collate_fn=standard_collate,
                           num_workers=0, pin_memory=False)
    
    base_loss = nn.CrossEntropyLoss()
    tpr_loss = TPRLoss(name='TPRLoss', weight_vector=None, threshold=0.5, attribute_index=1, reg_lambda=0.1, reg_type='tanh', 
                      num_classes=len(lesion_types))    
    dp_loss = DPLoss(name='DPLoss', weight_vector=None, threshold=0.5, attribute_index=1, reg_lambda=0.1, reg_type='tanh', 
                    reg_beta=0.0, good_value=1, num_classes=len(lesion_types))
    
    loss_fn = CombinedLoss(base_loss=base_loss, tpr_loss=tpr_loss, dp_loss=dp_loss,
                     ntxent_weight=0.3, ce_weight=0.4, tpr_weight=0.15,
                     dp_weight=0.15)

    print("Starting SimCLR pretraining...")

    pretrained_model = pretrain_simclr(df_train, TRAINING_PATH, batch_size=64, epochs=100, device=device)
    model = ContrastiveMobileNet(len(lesion_types))
    model.encoder.load_state_dict(pretrained_model.encoder.state_dict())
    model.feature_projection.load_state_dict(pretrained_model.feature_projection.state_dict())
    
    model = model.to(device)  
    optimizer = optim.Adam(model.parameters(), lr=lr)  

    if hasattr(base_loss, 'weight') and base_loss.weight is not None:
        base_loss.weight = base_loss.weight.to(device)

    print("Model created with TPRLoss only. Start training now...")
    trained_model = run_experiment_torch(model=model, train_loader=train_loader, valid_loader=valid_loader,
                                       criterion=loss_fn, optimizer=optimizer, exp_name=exp_name,
                                       epochs=100, patience=15, device=device, contrastive=True)

    print("Training completed. Saving predictions...")
    fairness_metrics = save_predictions_torch(model=trained_model, df_test=df_test, data_path=TRAINING_PATH,
                                            exp_name=exp_name, class_names=lesion_types,
                                            batch_size=batch_size, device=device)

    print("Fairness metrics:")
    print(json.dumps(fairness_metrics, indent=4))
    
if __name__ == "__main__":
    seeds = [10,20,30,40,50]
    for seed in seeds:
        print(f"\nRunning experiment with seed {seed}")
        main(seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()