from loss.loss_class import Loss

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BCELoss(Loss):
    def __init__(self, name='BCELoss', weight_vector=None):
        super().__init__(name)
        self.weight_vector = weight_vector

    def compute_loss(self, y_true, y_pred):
        y_true = y_true[:, [0]]
        if self.weight_vector is not None:
            batch_weights = torch.zeros_like(y_true, device=device)
            batch_weights[y_true==0] = self.weight_vector[0]
            batch_weights[y_true==1] = self.weight_vector[1]
            loss_fn = nn.BCELoss(weight=batch_weights)
        else:
            loss_fn = nn.BCELoss()
        return(loss_fn(y_pred, y_true))


class MSELoss(Loss):
    def __init__(self, name='MSELoss'):
        super().__init__(name)

    def compute_loss(self, y_true, y_pred):
        y_true = y_true[:, [0]]
        loss_fn = nn.MSELoss()
        return(loss_fn(y_pred, y_true))

class MultiClassBaseFairnessLoss(Loss):
    def __init__(self, target_class=0, name='MultiClassBaseFairnessLoss', 
                 weight_vector=None, threshold=0.5, attribute_index=1, 
                 reg_lambda=0.1, reg_type='tanh'):
        super().__init__(name)
        self.target_class = target_class
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index
        self.reg_lambda = reg_lambda
        self.reg_type = reg_type
        
    def _differentiable_round(self, x):
        x = x.float()
        return torch.tanh(3*(x - self.threshold))/2 + 0.5
        
    def _to_binary(self, y_true_labels):
        return (y_true_labels == self.target_class).float()
    
class MultiClassDPLoss(MultiClassBaseFairnessLoss):
    def _DP_torch(self, y_true_binary, y_pred_prob, reg):
        if reg == 'tanh':
            y_pred = self._differentiable_round(y_pred_prob)
            y_pred = y_pred[y_pred > self.threshold]
        elif reg == 'ccr':
            y_pred = y_pred_prob[y_pred_prob > self.threshold]
        elif reg == 'linear':
            y_pred = y_pred_prob
        total = y_true_binary.shape[0] + 1e-7
        return torch.sum(y_pred)/total
    
    def compute_loss(self, y_true, y_pred, weights=None):
        a = y_true[:, self.idx]
        y_true_labels = y_true[:, 0].long()
        y_true_binary = self._to_binary(y_true_labels)
        y_pred_prob = y_pred[:, self.target_class]  # Probability of target class
        
        y_pred_0 = y_pred_prob[a==0]
        y_true_0 = y_true_binary[a==0]
        y_pred_1 = y_pred_prob[a==1]
        y_true_1 = y_true_binary[a==1]
        
        DP_0 = self._DP_torch(y_true_0, y_pred_0, self.reg_type)
        DP_1 = self._DP_torch(y_true_1, y_pred_1, self.reg_type)
        
        # Cross entropy loss for the multi-class problem
        ce_loss = nn.CrossEntropyLoss(reduction='none')(y_pred, y_true_labels)
        if weights is not None:
            ce_loss = (ce_loss * weights).mean()
        else:
            ce_loss = ce_loss.mean()
        
        fairness_loss = torch.abs(DP_0 - DP_1)
        total_loss = fairness_loss + self.reg_lambda * ce_loss
        
        # Ensure we return a scalar
        return total_loss if isinstance(total_loss, torch.Tensor) else torch.tensor(total_loss, device=y_pred.device)

class MultiClassTPRLoss(MultiClassBaseFairnessLoss):
    def _TPR_torch(self, y_true_binary, y_pred_prob, reg):
        if reg == 'tanh':
            y_pred = self._differentiable_round(y_pred_prob)
            y_pred = y_pred[(y_true_binary == 1) & (y_pred > self.threshold)]  # TP: Predicted positive when true is positive
        elif reg == 'ccr':
            y_pred = y_pred_prob[(y_true_binary == 1) & (y_pred_prob > self.threshold)]
        elif reg == 'linear':
            y_pred = y_pred_prob[y_true_binary == 1]
        total_positives = torch.sum(y_true_binary == 1) + 1e-7
        return torch.sum(y_pred) / total_positives
    
    def compute_loss(self, y_true, y_pred):
        a = y_true[:, self.idx]
        y_true_labels = y_true[:, 0].long()
        y_true_binary = self._to_binary(y_true_labels)
        y_pred_prob = y_pred[:, self.target_class]
        
        y_pred_0 = y_pred_prob[a==0]
        y_true_0 = y_true_binary[a==0]
        y_pred_1 = y_pred_prob[a==1]
        y_true_1 = y_true_binary[a==1]
        
        TPR_0 = self._TPR_torch(y_true_0, y_pred_0, self.reg_type)
        TPR_1 = self._TPR_torch(y_true_1, y_pred_1, self.reg_type)
        
        ce_loss = nn.CrossEntropyLoss()(y_pred, y_true_labels)
        return torch.abs(TPR_0 - TPR_1) + self.reg_lambda * ce_loss
    
class MultiClassTNRLoss(MultiClassBaseFairnessLoss):
    def _TNR_torch(self, y_true_binary, y_pred_prob, reg):
        if reg == 'tanh':
            y_pred = self._differentiable_round(y_pred_prob)
            y_pred = 1 - y_pred[(y_true_binary == 0) & (y_pred <= self.threshold)]  # TN: Predicted negative when true is negative
        elif reg == 'ccr':
            y_pred = 1 - y_pred_prob[(y_true_binary == 0) & (y_pred_prob <= self.threshold)]
        elif reg == 'linear':
            y_pred = 1 - y_pred_prob[y_true_binary == 0]
        total_negatives = torch.sum(y_true_binary == 0) + 1e-7
        return torch.sum(y_pred) / total_negatives  # TN / (TN + FP)
    
    def compute_loss(self, y_true, y_pred):
        a = y_true[:, self.idx]
        y_true_labels = y_true[:, 0].long()
        y_true_binary = self._to_binary(y_true_labels)
        y_pred_prob = y_pred[:, self.target_class]
        
        y_pred_0 = y_pred_prob[a==0]
        y_true_0 = y_true_binary[a==0]
        y_pred_1 = y_pred_prob[a==1]
        y_true_1 = y_true_binary[a==1]
        
        TNR_0 = self._TNR_torch(y_true_0, y_pred_0, self.reg_type)
        TNR_1 = self._TNR_torch(y_true_1, y_pred_1, self.reg_type)
        
        ce_loss = nn.CrossEntropyLoss()(y_pred, y_true_labels)
        return torch.abs(TNR_0 - TNR_1) + self.reg_lambda * ce_loss

class MultiClassFPRLoss(MultiClassBaseFairnessLoss):
    def _FPR_torch(self, y_true_binary, y_pred_prob, reg):
        if reg == 'tanh':
            y_pred = self._differentiable_round(y_pred_prob)
            y_pred = y_pred[(y_true_binary == 0) & (y_pred > self.threshold)]  # FP: Predicted positive when true is negative
        elif reg == 'ccr':
            y_pred = y_pred_prob[(y_true_binary == 0) & (y_pred_prob > self.threshold)]
        elif reg == 'linear':
            y_pred = y_pred_prob[y_true_binary == 0]
        total_negatives = torch.sum(y_true_binary == 0) + 1e-7
        return torch.sum(y_pred) / total_negatives  # FP / (FP + TN)
    
    def compute_loss(self, y_true, y_pred):
        a = y_true[:, self.idx]
        y_true_labels = y_true[:, 0].long()
        y_true_binary = self._to_binary(y_true_labels)
        y_pred_prob = y_pred[:, self.target_class]
        
        y_pred_0 = y_pred_prob[a==0]
        y_true_0 = y_true_binary[a==0]
        y_pred_1 = y_pred_prob[a==1]
        y_true_1 = y_true_binary[a==1]
        
        FPR_0 = self._FPR_torch(y_true_0, y_pred_0, self.reg_type)
        FPR_1 = self._FPR_torch(y_true_1, y_pred_1, self.reg_type)
        
        ce_loss = nn.CrossEntropyLoss()(y_pred, y_true_labels)
        return torch.abs(FPR_0 - FPR_1) + self.reg_lambda * ce_loss

class MultiClassFNRLoss(MultiClassBaseFairnessLoss):
    def _FNR_torch(self, y_true_binary, y_pred_prob, reg):
        if reg == 'tanh':
            y_pred = self._differentiable_round(y_pred_prob)
            y_pred = 1 - y_pred[(y_true_binary == 1) & (y_pred <= self.threshold)]  # FN: Predicted negative when true is positive
        elif reg == 'ccr':
            y_pred = 1 - y_pred_prob[(y_true_binary == 1) & (y_pred_prob <= self.threshold)]
        elif reg == 'linear':
            y_pred = 1 - y_pred_prob[y_true_binary == 1]
        total_positives = torch.sum(y_true_binary == 1) + 1e-7
        return torch.sum(y_pred) / total_positives  # FN / (FN + TP)
    
    def compute_loss(self, y_true, y_pred):
        a = y_true[:, self.idx]
        y_true_labels = y_true[:, 0].long()
        y_true_binary = self._to_binary(y_true_labels)
        y_pred_prob = y_pred[:, self.target_class]
        
        y_pred_0 = y_pred_prob[a==0]
        y_true_0 = y_true_binary[a==0]
        y_pred_1 = y_pred_prob[a==1]
        y_true_1 = y_true_binary[a==1]
        
        FNR_0 = self._FNR_torch(y_true_0, y_pred_0, self.reg_type)
        FNR_1 = self._FNR_torch(y_true_1, y_pred_1, self.reg_type)
        
        ce_loss = nn.CrossEntropyLoss()(y_pred, y_true_labels)
        return torch.abs(FNR_0 - FNR_1) + self.reg_lambda * ce_loss

class CFLoss(Loss):
    def __init__(self, name='CounterfactualLoss', sen_attributes_idx=[1],reg_lambda=0.1, weight_vector=None, augmentation=False):

        super().__init__(name)
        self.reg_lambda = reg_lambda
        self.needs_model = True
        self.sen_attributes_idx = sen_attributes_idx
        self.weight_vector = weight_vector
        self.augmentation = augmentation

    def _logit(self, x):
        eps = torch.tensor(1e-7)
        x = x.float()
        x = torch.clamp(x, eps, 1-eps)
        return torch.log(x/(1-x)) 

    def _get_subsets(self, s):     
        x = len(s)
        subset_list = []
        for i in range(1, 1 << x):
            subset_list.append([s[j] for j in range(x) if (i & (1 << j))])
        return(subset_list)
        
    def _get_counterfactuals(self, x):
        i = self.sen_attributes_idx[0]
        x1 = x.clone()
        x1[:,[-i]] = 1 - x1[:,[-i]]
        return(x1)

    def compute_loss(self, y_true, y_pred, x, model):
        
        y_pred_x = self._logit(y_pred)
        x_new = self._get_counterfactuals(x)
        y_pred_new = model(x_new)
        y_pred_new_logit = self._logit(y_pred_new)
        pred_diff = torch.abs(y_pred_x - y_pred_new)

        if self.weight_vector is not None:
            batch_weights = torch.zeros_like(y_true, device=device)
            batch_weights[y_true==0] = self.weight_vector[0]
            batch_weights[y_true==1] = self.weight_vector[1]
            loss_fn = nn.BCELoss(weight=batch_weights)
        else:
            loss_fn = nn.MSELoss()

        if(self.augmentation):
        	loss = loss_fn(y_pred, y_true) + loss_fn(y_pred_new, y_true)
        else:
            loss = loss_fn(y_pred, y_true)
        
        return(self.reg_lambda*torch.mean(pred_diff) + loss)
    
class addLosses(Loss):
    def __call__(self, y_pred, y_true, x=None, model=None, weights=None):
        return self.compute_loss(y_true, y_pred, x, model, weights)
    def forward(self, y_pred, y_true, x=None, model=None, weights=None):
        return self.compute_loss(y_true, y_pred, x, model, weights)
    def __init__(self, name='addLosses', loss_list=[], loss_weights=[]):
        super().__init__(name)
        self.loss_list = loss_list
        self.loss_weights = loss_weights
    def compute_loss(self, y_true, y_pred, x=None, model=None, weights=None):
        final_loss = 0.0
        for alpha, loss_fn in zip(self.loss_weights, self.loss_list):
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                # Handle standard CrossEntropyLoss
                if weights is not None:
                    loss_per_sample = loss_fn(y_pred, y_true[:,0].long())
                    loss = alpha * (loss_per_sample * weights).mean()  # Ensure mean reduction
                else:
                    loss = alpha * loss_fn(y_pred, y_true[:,0].long())
            elif hasattr(loss_fn, 'isFairnessLoss') and loss_fn.isFairnessLoss:
                # Handle our fairness losses
                if hasattr(loss_fn, 'needs_model') and loss_fn.needs_model:
                    current_loss = loss_fn.compute_loss(y_true, y_pred, x, model)
                else:
                    current_loss = loss_fn.compute_loss(y_true, y_pred)
                # Ensure the loss is scalar
                if current_loss.dim() > 0:
                    current_loss = current_loss.mean()
                loss = alpha * current_loss
            else:
                # Handle other PyTorch losses
                loss = alpha * loss_fn(y_pred, y_true[:,0].float())
                if loss.dim() > 0:
                    loss = loss.mean()
            final_loss += loss
        return final_loss