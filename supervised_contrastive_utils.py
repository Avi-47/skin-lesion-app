import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import json
from sklearn.model_selection import train_test_split
from metric.metrics_new import TPRDiff, DPDiff  
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

class SimCLRAugmentation:
    def __init__(self, image_size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            AutoAugment(AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x)

class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class MultiClassSkinDataset(Dataset):
    def __init__(self, df, data_path, contrastive=False, shuffle=True):
        self.df = df
        self.data_path = data_path
        self.contrastive = contrastive
        
        valid_labels = {'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'}
        assert set(df['lesion'].unique()).issubset(valid_labels), "Invalid labels in dataset"
        
        self.label_mapping = {
            'MEL': 0, 'NV': 1, 'BCC': 2, 
            'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6
        }
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.contrastive_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            AutoAugment(AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_path, row['image'])
        img = Image.open(img_path).convert('RGB')
        label_str = row['lesion']
        label = self.label_mapping[label_str]
        
        if self.contrastive:
            x1 = self.contrastive_transform(img)
            x2 = self.contrastive_transform(img)
            
            return {
                'x1': x1,
                'x2': x2,
                'label': torch.tensor(label, dtype=torch.long),
                'A': torch.tensor(row['A'], dtype=torch.float),
                'weight': torch.tensor(row['weight'], dtype=torch.float)
            }
        else:
            x = self.transform(img)
            return {
                'x': x,
                'label': torch.tensor(label, dtype=torch.long),
                'A': torch.tensor(row['A'], dtype=torch.float),
                'weight': torch.tensor(row['weight'], dtype=torch.float)
            }

def prepare_df(data_path, metadata_filename="metadata.csv"):
    metadata_path = os.path.join(data_path, metadata_filename)
    df_meta = pd.read_csv(metadata_path)
    df_meta['image'] = df_meta['image'].apply(lambda x: x + '.jpg' if not x.lower().endswith('.jpg') else x)
    return df_meta

def recalculate_weights(df, label_col='lesion'):
    df = df.copy()  # Ensure we're working with a copy
    class_counts = df[label_col].value_counts()
    total_samples = len(df)
    num_classes = len(class_counts)
    weights = total_samples / (num_classes * class_counts)
    df.loc[:, 'weight'] = df[label_col].map(weights)  # Use .loc
    return df


def run_experiment_torch(model, train_loader, valid_loader, criterion, optimizer, exp_name="experiment", epochs=10, device="cuda", patience=5, contrastive=False):
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for batch in train_loader:
            if contrastive:
                # Handle contrastive training
                x1, x2 = batch['x1'].to(device), batch['x2'].to(device)
                y_true = batch['label'].to(device)
                A = batch['A'].to(device)
                
                # Stack labels and attributes for loss computation
                y_true_combined = torch.stack([batch['label'].to(device), batch['A'].to(device)], dim=1)
                
                optimizer.zero_grad()
                
                # Get predictions and projections
                outputs = model(x1)  # Classification outputs
                z1 = model(x1, return_projection=True)  # Contrastive projections
                z2 = model(x2, return_projection=True)
                
                # Use the combined loss
                if hasattr(criterion, 'forward'):
                    loss, loss_dict = criterion(outputs, z1, z2, y_true_combined)
                else:
                    loss = criterion(outputs, y_true_combined)
                
                loss.backward()
                optimizer.step()
                
                # Update metrics
                _, predicted = torch.max(outputs, 1)
                total += y_true.size(0)
                correct += (predicted == y_true).sum().item()
                running_loss += loss.item() * x1.size(0)
                
            else:
                # Handle standard training
                images, y_true, weights = batch
                images, y_true, weights = images.to(device), y_true.to(device), weights.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                # Use consistent loss calculation
                if hasattr(criterion, 'compute_loss'):
                    loss = criterion.compute_loss(y_true, outputs)
                else:
                    loss = criterion(outputs, y_true[:, 0].long())
                
                loss.backward()
                optimizer.step()
                
                # Update metrics
                _, predicted = torch.max(outputs, 1)
                total += y_true.size(0)
                correct += (predicted == y_true[:, 0].long()).sum().item()
                running_loss += loss.item() * images.size(0)
        
        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation - use consistent approach
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in valid_loader:
                if isinstance(batch, dict):
                    images = batch['x'].to(device)
                    y_true = batch['label'].to(device)
                    A = batch['A'].to(device)
                    y_true_combined = torch.stack([y_true, A], dim=1)
                    outputs = model(images)
                    if hasattr(criterion, 'forward'):
                        z_dummy = torch.zeros_like(outputs).to(device)
                        loss, _ = criterion(outputs, z_dummy, z_dummy, y_true_combined)
                    else:
                        loss = criterion(outputs, y_true_combined)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs, 1)
                val_total += y_true_combined.size(0)
                if len(y_true_combined.shape) > 1:
                    correct_labels = y_true_combined[:, 0].long()
                else:
                    correct_labels = y_true_combined.long()
                val_correct += (predicted == correct_labels).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Early stopping with better logging
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
            os.makedirs("results", exist_ok=True)
            torch.save(model.state_dict(), f"results/{exp_name}_best_model.pth")
            print(f"New best model saved with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (best: {best_val_loss:.4f})")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} - no improvement for {patience} epochs")
                break
        
        print(f"Epoch {epoch+1}/{epochs} - {time.time()-start_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Add loss stability check
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            print(f"Stopping due to unstable loss: train={train_loss}, val={val_loss}")
            break
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }
    
    os.makedirs("results", exist_ok=True)
    with open(f"results/{exp_name}_history.json", "w") as f:
        json.dump(history, f)
        
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss over epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy over epochs')
    plt.legend()
    
    plt.savefig(f"results/{exp_name}_history.png")
    plt.close()
    
    return model

def save_predictions_torch(model, df_test, data_path, exp_name, class_names, batch_size=16, device="cuda"):
    model.eval()
    test_dataset = MultiClassSkinDataset(df_test, data_path, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_probs = []
    all_true = []
    all_A = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['x'].to(device)
            y_true = batch['label'].to(device)
            A = batch['A'].to(device)
            outputs = model(images)
            all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
            all_true.append(y_true.cpu().numpy())
            all_A.append(A.cpu().numpy())
    
    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_true)
    y_pred = np.argmax(probs, axis=1)
    A_values = np.concatenate(all_A)
    metrics = calculate_fairness_metrics(y_true, y_pred, A_values, class_names)
    os.makedirs("results", exist_ok=True)
    with open(f"results/{exp_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    
    results = pd.DataFrame({
        'image': df_test['image'].values,
        'true_lesion': df_test['lesion'].values,
        'predicted_lesion': [class_names[p] for p in y_pred],
        'A': A_values
    })
    
    for i, class_name in enumerate(class_names):
        results[class_name] = probs[:, i]
    
    os.makedirs("predictions", exist_ok=True)
    output_path = f"predictions/{exp_name}_predictions.csv"
    results.to_csv(output_path, index=False, float_format='%.8f')
    
    print(f"Predictions saved to {output_path}")
    return metrics

def calculate_tpr_diff(y_true, y_pred, a_values):
    """Calculate TPR difference between groups (max - min)"""
    tpr = {}
    for group in [0, 1]:
        group_mask = (a_values == group)
        pos_mask = group_mask & (y_true == 1)  # For binary case, modify for multi-class
        if pos_mask.sum() > 0:
            tpr[group] = (y_pred[pos_mask] == y_true[pos_mask]).mean()
    return max(tpr.values()) - min(tpr.values()) if len(tpr) == 2 else 0

def calculate_dp_diff(y_true, y_pred, a_values):
    dp = {}
    for group in [0, 1]:
        group_mask = (a_values == group)
        if group_mask.sum() > 0:
            dp[group] = (y_pred[group_mask] == 1).mean()  # For binary case, modify for multi-class
    return max(dp.values()) - min(dp.values()) if len(dp) == 2 else 0


def calculate_fairness_metrics(y_true, y_pred, a_values, class_names=None, weights=None):
    metrics = {}
    
    # Combine y_true and a_values into the format expected by your Metric classes
    y_true_combined = np.column_stack([y_true, a_values])
    
    # Initialize metric calculators
    tpr_diff_calculator = TPRDiff(attribute_index=1, num_classes=len(np.unique(y_true)))
    dp_diff_calculator = DPDiff(attribute_index=1, num_classes=len(np.unique(y_true)))
    
    # Calculate metrics
    metrics['Accuracy'] = np.mean(y_true == y_pred)
    metrics['TPRDiff'] = 1 - tpr_diff_calculator.evaluate(y_true_combined, y_pred, None, None)
    metrics['DPDiff'] = 1 - dp_diff_calculator.evaluate(y_true_combined, y_pred, None, None)
    
    return metrics