import torch
import torch.nn as nn
from loss.loss_class import Loss
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        
        device = features.device
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
            
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        device = z_i.device
        N = z_i.size(0)
        
        z = torch.cat([z_i, z_j], dim=0)  # [2N, 128]
        
        sim = self.cossim(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature  # [2N, 2N]
        
        mask = torch.eye(2*N, dtype=torch.bool, device=device)
        pos_sim = sim[mask].view(2*N, 1)  # [2N, 1]
        neg_mask = mask.logical_not()
        neg_sim = sim[neg_mask].view(2*N, -1)  # [2N, 2N-2]
        
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [2N, 2N-1]
        
        labels = torch.zeros(2*N, dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)

class CombinedLoss(nn.Module):
    def __init__(self, base_loss, tpr_loss, dp_loss, ntxent_weight=0.3, ce_weight=0.4, 
                 tpr_weight=0.15, dp_weight=0.15):
        super().__init__()
        self.base_loss = base_loss
        self.tpr_loss = tpr_loss
        self.dp_loss = dp_loss
        self.ntxent_loss = SupConLoss(temperature=0.07)  # Fixed temperature as in paper
        self.weights = {
            'ntxent': ntxent_weight,
            'ce': ce_weight,
            'tpr': tpr_weight,
            'dp': dp_weight
        }

    def forward(self, y_pred, z_i, z_j, y_true):
        device = y_pred.device
        labels = y_true[:, 0].long()
        A = y_true[:, 1]
        
        # Classification loss
        ce_loss = self.base_loss(y_pred, labels)
        
        # Fairness losses
        tpr_loss = self.tpr_loss.compute_loss(y_true, y_pred)
        dp_loss = self.dp_loss.compute_loss(y_true, y_pred)
        
        # Supervised contrastive loss
        if z_i is not None and z_j is not None:
            # Stack the two views along the second dimension [bsz, n_views, ...]
            features = torch.stack([z_i, z_j], dim=1)
            ntxent_loss = self.ntxent_loss(features, labels)
        else:
            ntxent_loss = torch.tensor(0.0, device=device)
        
        # Combine losses
        total_loss = (
            self.weights['ce'] * ce_loss +
            self.weights['ntxent'] * ntxent_loss +
            self.weights['tpr'] * tpr_loss +
            self.weights['dp'] * dp_loss
        )
        
        return total_loss, {
            'ce': ce_loss.item(),
            'ntxent': ntxent_loss.item(),
            'tpr': tpr_loss.item(),
            'dp': dp_loss.item()
        }
                
class DPLoss(Loss):
    def __init__(self, name='DPLoss', weight_vector=None,
                 threshold=0.5, attribute_index=1, reg_lambda=0.1,
                 reg_type='tanh', reg_beta=0.0, good_value=1, num_classes=3):
        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index
        self.reg_lambda = reg_lambda
        self.reg_beta = reg_beta
        self.reg_type = reg_type
        self.good_value = good_value
        self.num_classes = num_classes
       
    def _differentiable_round(self, x):
        x = x.float()
        return torch.tanh(3*(x - self.threshold))/2 + 0.5
   
    def _DP_torch_multiclass(self, y_true, y_pred, target_class, reg):
        """
        Compute Demographic Parity for a specific class in multi-class setting
        DP = P(Y_hat = target_class | A = a) for each protected group
        """
        # Get predictions for the target class
        class_probs = y_pred[:, target_class]
        total_samples = y_pred.shape[0] + 1e-7
       
        if reg == 'tanh':
            # Apply differentiable rounding to probabilities
            rounded_probs = self._differentiable_round(class_probs)
            if self.good_value:
                # Count predictions above threshold (positive predictions)
                positive_predictions = torch.sum(rounded_probs > self.threshold)
            else:
                # Count predictions below threshold (negative predictions)
                positive_predictions = torch.sum(rounded_probs <= self.threshold)
        elif reg == 'ccr':
            # Direct thresholding without differentiable rounding
            if self.good_value:
                positive_predictions = torch.sum(class_probs > self.threshold)
            else:
                positive_predictions = torch.sum(class_probs <= self.threshold)
        elif reg == 'linear':
            # Use raw probabilities
            if self.good_value:
                positive_predictions = torch.sum(class_probs)
            else:
                positive_predictions = torch.sum(1.0 - class_probs)
       
        # Return the rate: proportion of samples predicted as positive for this class
        return positive_predictions / total_samples
   
    def _DP_torch_multiclass_alternative(self, y_true, y_pred, target_class, reg):
        """
        Alternative DP computation: P(Y_hat = target_class) regardless of true labels
        This is the standard demographic parity definition
        """
        # Get the maximum probability class predictions
        if reg == 'tanh':
            rounded_probs = self._differentiable_round(y_pred)
            if self.good_value:
                predicted_class = torch.argmax(rounded_probs, dim=1)
            else:
                # For negative case, might want to use minimum instead
                predicted_class = torch.argmin(rounded_probs, dim=1)
        elif reg == 'ccr':
            predicted_class = torch.argmax(y_pred, dim=1)
        elif reg == 'linear':
            # Use soft assignment with probabilities
            class_assignment = y_pred[:, target_class]
            total_samples = y_pred.shape[0] + 1e-7
            return torch.sum(class_assignment) / total_samples
       
        # Count how many samples are predicted as the target class
        predicted_as_target = torch.sum(predicted_class == target_class).float()
        total_samples = y_pred.shape[0] + 1e-7
       
        return predicted_as_target / total_samples
   
    def compute_loss(self, y_true, y_pred):
        # Extract protected attribute
        a = y_true[:, self.idx]  # Protected attribute (0 or 1)
        y_true_labels = y_true[:, 0].long()  # True class labels
       
        # Apply softmax to get probabilities if not already applied
        if y_pred.dim() > 1 and y_pred.size(1) > 1:
            y_pred = F.softmax(y_pred, dim=1)
       
        # Separate by protected attribute
        mask_0 = (a == 0)
        mask_1 = (a == 1)
       
        y_pred_0 = y_pred[mask_0]
        y_true_0 = y_true_labels[mask_0]
        y_pred_1 = y_pred[mask_1]
        y_true_1 = y_true_labels[mask_1]
       
        # Compute Demographic Parity disparity across all classes
        total_disparity = 0.0
        for class_idx in range(self.num_classes):
            if len(y_pred_0) > 0:
                DP_0 = self._DP_torch_multiclass(y_true_0, y_pred_0, class_idx, self.reg_type)
            else:
                DP_0 = torch.tensor(0.0, device=y_pred.device)
               
            if len(y_pred_1) > 0:
                DP_1 = self._DP_torch_multiclass(y_true_1, y_pred_1, class_idx, self.reg_type)
            else:
                DP_1 = torch.tensor(0.0, device=y_pred.device)
           
            total_disparity += torch.abs(DP_0 - DP_1)
       
        # Compute classification loss
        if self.weight_vector is not None:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(self.weight_vector, device=y_pred.device))
        else:
            loss_fn = nn.CrossEntropyLoss()
       
        # Convert back to logits for CrossEntropyLoss
        y_pred_logits = torch.log(y_pred + 1e-7)
        classification_loss = loss_fn(y_pred_logits, y_true_labels)
       
        # L2 regularization term on predictions
        l2_reg = torch.mean(y_pred**2)
       
        return (total_disparity +
                self.reg_lambda * classification_loss +
                self.reg_beta * l2_reg)
class TPRLoss(Loss):
    def __init__(self, name='TPRLoss', weight_vector=None, 
                 threshold=0.5, attribute_index=1, reg_lambda=0.1,
                 reg_type='tanh', num_classes=3):
        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index
        self.reg_lambda = reg_lambda
        self.reg_type = reg_type
        self.num_classes = num_classes
        
    def _differentiable_round(self, x):
        x = x.float()
        return torch.tanh(5*(x - self.threshold))/2 + 0.5
    
    def _TPR_torch_multiclass(self, y_true, y_pred, target_class, reg):
        
        class_probs = y_pred[:, target_class]
        
        true_positives_mask = (y_true == target_class)
        
        if torch.sum(true_positives_mask) == 0:
            return torch.tensor(0.0, device=y_pred.device)
        
        class_probs_for_target = class_probs[true_positives_mask]
        
        if reg == 'tanh':
            rounded_probs = self._differentiable_round(class_probs_for_target)
            predicted_positives = torch.sum(rounded_probs > self.threshold)
        elif reg == 'ccr':
            predicted_positives = torch.sum(class_probs_for_target > self.threshold)
        elif reg == 'linear':
            predicted_positives = torch.sum(class_probs_for_target)
        
        total_positives = torch.sum(true_positives_mask).float() + 1e-7
        return predicted_positives / total_positives
    
    def compute_loss(self, y_true, y_pred):
        a = y_true[:, self.idx]  # Protected attribute (0 or 1)
        y_true_labels = y_true[:, 0].long()  # True class labels
        
        if y_pred.dim() > 1 and y_pred.size(1) > 1:
            y_pred = F.softmax(y_pred, dim=1)
        
        mask_0 = (a == 0)
        mask_1 = (a == 1)
        
        y_pred_0 = y_pred[mask_0]
        y_true_0 = y_true_labels[mask_0]
        y_pred_1 = y_pred[mask_1]
        y_true_1 = y_true_labels[mask_1]
        
        total_disparity = 0.0
        for class_idx in range(self.num_classes):
            if len(y_true_0) > 0:
                TPR_0 = self._TPR_torch_multiclass(y_true_0, y_pred_0, class_idx, self.reg_type)
            else:
                TPR_0 = torch.tensor(0.0, device=y_pred.device)
                
            if len(y_true_1) > 0:
                TPR_1 = self._TPR_torch_multiclass(y_true_1, y_pred_1, class_idx, self.reg_type)
            else:
                TPR_1 = torch.tensor(0.0, device=y_pred.device)
            
            total_disparity += torch.abs(TPR_0 - TPR_1)
        
        if self.weight_vector is not None:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(self.weight_vector, device=y_pred.device))
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        y_pred_logits = torch.log(y_pred + 1e-7)
        classification_loss = loss_fn(y_pred_logits, y_true_labels)
        
        return total_disparity + self.reg_lambda * classification_loss

class FNRLoss(Loss):
    def __init__(self, name='FNRLoss', weight_vector=None, 
                 threshold=0.5, attribute_index=1, reg_lambda=0.1,
                 reg_type='tanh', num_classes=3):
        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index
        self.reg_lambda = reg_lambda
        self.reg_type = reg_type
        self.num_classes = num_classes
        
    def _differentiable_round(self, x):
        x = x.float()
        return torch.tanh(5*(x - self.threshold))/2 + 0.5
    
    def _FNR_torch_multiclass(self, y_true, y_pred, target_class, reg):
        class_probs = y_pred[:, target_class]
        
        true_positives_mask = (y_true == target_class)
        
        if torch.sum(true_positives_mask) == 0:
            return torch.tensor(0.0, device=y_pred.device)  # No positives -> perfect FNR
        
        class_probs_for_positives = class_probs[true_positives_mask]
        
        if reg == 'tanh':
            rounded_probs = self._differentiable_round(class_probs_for_positives)
            false_negatives = torch.sum(rounded_probs <= self.threshold)
        elif reg == 'ccr':
            false_negatives = torch.sum(class_probs_for_positives <= self.threshold)
        elif reg == 'linear':
            false_negatives = torch.sum(1.0 - class_probs_for_positives)
        
        total_positives = torch.sum(true_positives_mask).float() + 1e-7
        return false_negatives / total_positives
    
    def compute_loss(self, y_true, y_pred):
        a = y_true[:, self.idx]
        y_true_labels = y_true[:, 0].long()
        
        if y_pred.dim() > 1 and y_pred.size(1) > 1:
            y_pred = F.softmax(y_pred, dim=1)
        
        mask_0 = (a == 0)
        mask_1 = (a == 1)
        
        y_pred_0 = y_pred[mask_0]
        y_true_0 = y_true_labels[mask_0]
        y_pred_1 = y_pred[mask_1]
        y_true_1 = y_true_labels[mask_1]
        
        total_disparity = 0.0
        for class_idx in range(self.num_classes):
            if len(y_true_0) > 0:
                FNR_0 = self._FNR_torch_multiclass(y_true_0, y_pred_0, class_idx, self.reg_type)
            else:
                FNR_0 = torch.tensor(0.0, device=y_pred.device)
                
            if len(y_true_1) > 0:
                FNR_1 = self._FNR_torch_multiclass(y_true_1, y_pred_1, class_idx, self.reg_type)
            else:
                FNR_1 = torch.tensor(0.0, device=y_pred.device)
            
            total_disparity += torch.abs(FNR_0 - FNR_1)
        
        if self.weight_vector is not None:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(self.weight_vector, device=y_pred.device))
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        y_pred_logits = torch.log(y_pred + 1e-7)
        classification_loss = loss_fn(y_pred_logits, y_true_labels)
        
        return total_disparity + self.reg_lambda * classification_loss

class TNRLoss(Loss):
    def __init__(self, name='TNRLoss', weight_vector=None, 
                 threshold=0.5, attribute_index=1, reg_lambda=0.1,
                 reg_type='tanh', num_classes=3):
        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index
        self.reg_lambda = reg_lambda
        self.reg_type = reg_type
        self.num_classes = num_classes
        
    def _differentiable_round(self, x):
        x = x.float()
        return torch.tanh(5*(x - self.threshold))/2 + 0.5
    
    def _TNR_torch_multiclass(self, y_true, y_pred, target_class, reg):
        """
        Compute TNR for a specific class in multi-class setting
        TNR = True Negative Rate = TN / (TN + FP)
        """
        class_probs = y_pred[:, target_class]
        
        true_negatives_mask = (y_true != target_class)
        
        if torch.sum(true_negatives_mask) == 0:
            return torch.tensor(1.0, device=y_pred.device)  # Perfect TNR when no negatives
        
        class_probs_for_negatives = class_probs[true_negatives_mask]
        
        if reg == 'tanh':
            rounded_probs = self._differentiable_round(class_probs_for_negatives)
            correct_negatives = torch.sum(rounded_probs <= self.threshold)
        elif reg == 'ccr':
            correct_negatives = torch.sum(class_probs_for_negatives <= self.threshold)
        elif reg == 'linear':
            correct_negatives = torch.sum(1.0 - class_probs_for_negatives)
        
        total_negatives = torch.sum(true_negatives_mask).float() + 1e-7
        return correct_negatives / total_negatives
    
    def compute_loss(self, y_true, y_pred):
        a = y_true[:, self.idx]  # Protected attribute (0 or 1)
        y_true_labels = y_true[:, 0].long()  # True class labels
        
        if y_pred.dim() > 1 and y_pred.size(1) > 1:
            y_pred = F.softmax(y_pred, dim=1)
        
        mask_0 = (a == 0)
        mask_1 = (a == 1)
        
        y_pred_0 = y_pred[mask_0]
        y_true_0 = y_true_labels[mask_0]
        y_pred_1 = y_pred[mask_1]
        y_true_1 = y_true_labels[mask_1]
        
        total_disparity = 0.0
        for class_idx in range(self.num_classes):
            if len(y_true_0) > 0:
                TNR_0 = self._TNR_torch_multiclass(y_true_0, y_pred_0, class_idx, self.reg_type)
            else:
                TNR_0 = torch.tensor(1.0, device=y_pred.device)
                
            if len(y_true_1) > 0:
                TNR_1 = self._TNR_torch_multiclass(y_true_1, y_pred_1, class_idx, self.reg_type)
            else:
                TNR_1 = torch.tensor(1.0, device=y_pred.device)
            
            total_disparity += torch.abs(TNR_0 - TNR_1)
        
        if self.weight_vector is not None:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(self.weight_vector, device=y_pred.device))
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        y_pred_logits = torch.log(y_pred + 1e-7)
        classification_loss = loss_fn(y_pred_logits, y_true_labels)
        
        return total_disparity + self.reg_lambda * classification_loss

class FPRLoss(Loss):
    def __init__(self, name='FPRLoss', weight_vector=None, 
                 threshold=0.5, attribute_index=1, reg_lambda=0.1,
                 reg_type='tanh', num_classes=3):
        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index
        self.reg_lambda = reg_lambda
        self.reg_type = reg_type
        self.num_classes = num_classes
        
    def _differentiable_round(self, x):
        x = x.float()
        return torch.tanh(5*(x - self.threshold))/2 + 0.5
    
    def _FPR_torch_multiclass(self, y_true, y_pred, target_class, reg):
        class_probs = y_pred[:, target_class]
        
        true_negatives_mask = (y_true != target_class)
        
        if torch.sum(true_negatives_mask) == 0:
            return torch.tensor(0.0, device=y_pred.device)  # No negatives -> perfect FPR
        
        class_probs_for_negatives = class_probs[true_negatives_mask]
        
        if reg == 'tanh':
            rounded_probs = self._differentiable_round(class_probs_for_negatives)
            false_positives = torch.sum(rounded_probs > self.threshold)
        elif reg == 'ccr':
            false_positives = torch.sum(class_probs_for_negatives > self.threshold)
        elif reg == 'linear':
            false_positives = torch.sum(class_probs_for_negatives)
        
        total_negatives = torch.sum(true_negatives_mask).float() + 1e-7
        return false_positives / total_negatives
    
    def compute_loss(self, y_true, y_pred):
        a = y_true[:, self.idx]
        y_true_labels = y_true[:, 0].long()
        
        if y_pred.dim() > 1 and y_pred.size(1) > 1:
            y_pred = F.softmax(y_pred, dim=1)
        
        mask_0 = (a == 0)
        mask_1 = (a == 1)
        
        y_pred_0 = y_pred[mask_0]
        y_true_0 = y_true_labels[mask_0]
        y_pred_1 = y_pred[mask_1]
        y_true_1 = y_true_labels[mask_1]
        
        total_disparity = 0.0
        for class_idx in range(self.num_classes):
            if len(y_true_0) > 0:
                FPR_0 = self._FPR_torch_multiclass(y_true_0, y_pred_0, class_idx, self.reg_type)
            else:
                FPR_0 = torch.tensor(0.0, device=y_pred.device)
                
            if len(y_true_1) > 0:
                FPR_1 = self._FPR_torch_multiclass(y_true_1, y_pred_1, class_idx, self.reg_type)
            else:
                FPR_1 = torch.tensor(0.0, device=y_pred.device)
            
            total_disparity += torch.abs(FPR_0 - FPR_1)
        
        if self.weight_vector is not None:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(self.weight_vector, device=y_pred.device))
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        y_pred_logits = torch.log(y_pred + 1e-7)
        classification_loss = loss_fn(y_pred_logits, y_true_labels)
        
        return total_disparity + self.reg_lambda * classification_loss

