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
from supervised_contrastive_loss import TPRLoss, CombinedLoss, DPLoss, SupConLoss
from torchvision import models
import torch.nn.functional as F

def pretrain_supcon(df_train, data_path, batch_size=128, epochs=100, device='cpu', temperature=0.07):
    train_data = MultiClassSkinDataset(df_train, data_path, contrastive=True)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, collate_fn=contrastive_collate)
    
    model = ContrastiveMobileNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = SupConLoss(temperature=temperature)

    print(f"Starting supervised contrastive pretraining with {len(train_data)} samples...")
    print(f"Using temperature: {temperature}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch in train_loader:
            x1, x2 = batch['x1'].to(device), batch['x2'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            z1 = model(x1, return_projection=True)
            z2 = model(x2, return_projection=True)
            
            features = torch.stack([z1, z2], dim=1)
            loss = criterion(features, labels)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | Time: {time.time()-start_time:.1f}s")
    
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

def safe_train_test_split(df, test_size=0.2, random_state=None, stratify_col=None):
    if stratify_col is not None:
        class_counts = df[stratify_col].value_counts()
        if (class_counts < 2).any():
            print(f"Warning: Some classes have <2 samples. Using random split instead of stratified.")
            return train_test_split(df, test_size=test_size, random_state=random_state)
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[stratify_col])

def main(seed, label_type, label_name, number, temperature=0.07):
    batch_size = 8
    lr = 1e-5
    Fitzpatrick_threshold = 2
    exp_name = f"supcon_rp_{label_type}_{seed}_v{number}_temp{temperature}"  # Include temperature in exp_name
    lesion_types = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    TRAINING_PATH = r"D:\ISI_task_3\RevisitingSkinToneFairness-main\ISIC2018_Task3_Training_Input"
    df_train_temp = prepare_df(TRAINING_PATH)
    df_Bevan = pd.read_csv("Bevan_corrected.csv")
    df_train_temp = pd.merge(df_train_temp, df_Bevan, left_on='image', right_on='image')
    df_train_temp = df_train_temp.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    df_train_temp[f'A_{label_type}'] = (df_train_temp[label_name] > Fitzpatrick_threshold).astype(int)
    df_train_temp['A'] = df_train_temp[f'A_{label_type}']
    
    df_A0 = df_train_temp[df_train_temp[f'A_{label_type}'] == 0]
    df_A1 = df_train_temp[df_train_temp[f'A_{label_type}'] == 1]
    
    A0_train, A0_test = safe_train_test_split(df_A0, test_size=0.2, random_state=seed, stratify_col="lesion")
    A1_train, A1_test = safe_train_test_split(df_A1, test_size=0.2, random_state=seed, stratify_col="lesion")
    
    df_train = pd.concat([A0_train, A1_train])
    df_test = pd.concat([A0_test, A1_test])
    
    idx_train, idx_valid = train_test_split(df_train.index, test_size=0.20, random_state=seed, stratify=df_train["lesion"])
    df_valid = df_train.loc[idx_valid]
    df_train = df_train.loc[idx_train]
    
    df_train = recalculate_weights(df_train)
    df_valid = recalculate_weights(df_valid)
    df_test = recalculate_weights(df_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader = DataLoader(
        MultiClassSkinDataset(df_train, TRAINING_PATH, contrastive=True),
        batch_size=batch_size, shuffle=True, collate_fn=contrastive_collate,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    valid_loader = DataLoader(
        MultiClassSkinDataset(df_valid, TRAINING_PATH, contrastive=False),
        batch_size=batch_size, shuffle=False, collate_fn=standard_collate
    )

    base_loss = nn.CrossEntropyLoss()
    tpr_loss = TPRLoss(name='TPRLoss', weight_vector=None, threshold=0.5, 
                      attribute_index=1, reg_lambda=0.1, reg_type='tanh',
                      num_classes=len(lesion_types))
    dp_loss = DPLoss(name='DPLoss', weight_vector=None, threshold=0.5,
                    attribute_index=1, reg_lambda=0.1, reg_type='tanh',
                    reg_beta=0.0, good_value=1, num_classes=len(lesion_types))
    
    loss_fn = CombinedLoss(base_loss=base_loss, tpr_loss=tpr_loss, dp_loss=dp_loss,
                         ntxent_weight=0.3, ce_weight=0.4, 
                         tpr_weight=0.15, dp_weight=0.15)

    print(f"Starting supervised contrastive pretraining with temperature={temperature}...")
    pretrained_model = pretrain_supcon(df_train, TRAINING_PATH, batch_size=64, epochs=100, device=device, temperature=temperature)
    
    model = ContrastiveMobileNet(len(lesion_types)).to(device)
    model.encoder.load_state_dict(pretrained_model.encoder.state_dict())
    model.feature_projection.load_state_dict(pretrained_model.feature_projection.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Model created with {label_type}. Start training now...")
    trained_model = run_experiment_torch(model=model,train_loader=train_loader,valid_loader=valid_loader,criterion=loss_fn,
                                        optimizer=optimizer,exp_name=exp_name,epochs=100,patience=60,device=device,contrastive=True)

    print("Training completed. Saving predictions...")
    fairness_metrics = save_predictions_torch(model=trained_model,df_test=df_test,data_path=TRAINING_PATH,
                                            exp_name=exp_name,class_names=lesion_types,batch_size=batch_size,device=device)
    print(json.dumps(fairness_metrics, indent=4))

if __name__ == "__main__":
    label_types = ["RP","RP2"]
    label_names = ["fitzpatrick_Bevan", "fitzpatrick_corrected"]
    seeds = [10, 20, 30, 40, 50]
    temperatures = [0.2, 0.3]  # List of temperatures to try
    
    for temp in temperatures:
        for i in range(len(label_types)):
            if label_types[i] != "RP2":
                continue
            label_name = label_names[i]
            label_type = label_types[i]

            for j in range(len(seeds)):
                seed = seeds[j]
                number = i * 3 + j + 1
                print(f"\nRunning experiment with {label_type}, seed {seed}, temperature {temp}")
                main(seed, label_type, label_name, number, temperature=temp)
                torch.cuda.empty_cache()