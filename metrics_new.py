import numpy as np
from metric.metric import Metric
from sklearn.metrics import confusion_matrix

class TPRDiff(Metric):
    def __init__(self, name='TPR_diff', weight_vector=None, 
                 threshold=0.5, attribute_index=1, num_classes=None):
        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index
        self.num_classes = num_classes
    
    def _TPR_multiclass(self, y_true, y_pred, num_classes):
        """Calculate TPR for each class in multi-class setting"""
        epsilon = 1e-7
        
        # Get unique classes if num_classes not specified
        if num_classes is None:
            classes = np.unique(np.concatenate([y_true.flatten(), y_pred.flatten()]))
            num_classes = len(classes)
        else:
            classes = np.arange(num_classes)
        
        # Handle case where there are no samples
        if len(y_true) == 0:
            return np.zeros(num_classes)
        
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=classes)
        
        # Calculate TPR for each class (diagonal / row sum)
        tpr_per_class = np.zeros(num_classes)
        for i in range(num_classes):
            if i < cm.shape[0]:  # Check if class exists in confusion matrix
                tp = cm[i, i] if i < cm.shape[1] else 0
                fn = cm[i, :].sum() - tp if i < cm.shape[0] else 0
                tpr_per_class[i] = tp / (tp + fn + epsilon)
        
        return tpr_per_class
    
    def evaluate(self, y_true, y_pred, x, model):
        """
        Evaluate TPR difference across sensitive attribute groups for multi-class
        Returns: 1 - max_difference_across_classes
        """
        # Extract sensitive attribute and target
        a = y_true[:, [self.idx]]
        y_true_target = y_true[:, [0]]
        
        # Convert predictions to class predictions (assuming they're probabilities)
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_class = np.argmax(y_pred, axis=1).reshape(-1, 1)
        else:
            y_pred_class = np.rint(y_pred).astype(int)
        
        # Determine number of classes
        if self.num_classes is None:
            num_classes = len(np.unique(y_true_target))
        else:
            num_classes = self.num_classes
        
        # Split by sensitive attribute
        unique_attrs = np.unique(a)
        
        # Calculate TPR for each sensitive attribute group
        tpr_by_group = {}
        for attr_val in unique_attrs:
            mask = (a == attr_val).flatten()
            if np.sum(mask) > 0:  # Only calculate if group has samples
                y_true_group = y_true_target[mask]
                y_pred_group = y_pred_class[mask]
                tpr_by_group[attr_val] = self._TPR_multiclass(y_true_group, y_pred_group, num_classes)
        
        # Calculate maximum difference across all classes and groups
        max_diff = 0
        for class_idx in range(num_classes):
            class_tprs = [tpr_by_group[attr][class_idx] for attr in tpr_by_group.keys()]
            if len(class_tprs) > 1:
                class_diff = np.max(class_tprs) - np.min(class_tprs)
                max_diff = max(max_diff, class_diff)
        
        return 1 - max_diff


class DPDiff(Metric):
    def __init__(self, name='DP_diff', weight_vector=None, 
                 threshold=0.5, attribute_index=1, num_classes=None):
        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index
        self.num_classes = num_classes
    
    def _DP_multiclass(self, y_true, y_pred, num_classes):
        """Calculate Demographic Parity (positive prediction rate) for each class"""
        epsilon = 1e-7
        
        # Get unique classes if num_classes not specified
        if num_classes is None:
            classes = np.unique(np.concatenate([y_true.flatten(), y_pred.flatten()]))
            num_classes = len(classes)
        else:
            classes = np.arange(num_classes)
        
        # Handle case where there are no samples
        if len(y_pred) == 0:
            return np.zeros(num_classes)
        
        total_samples = len(y_pred)
        dp_per_class = np.zeros(num_classes)
        
        # Calculate positive prediction rate for each class
        for class_idx in range(num_classes):
            positive_predictions = np.sum(y_pred.flatten() == class_idx)
            dp_per_class[class_idx] = (positive_predictions + epsilon) / (total_samples + epsilon)
        
        return dp_per_class
    
    def evaluate(self, y_true, y_pred, x, model):
        """
        Evaluate Demographic Parity difference across sensitive attribute groups for multi-class
        Returns: 1 - max_difference_across_classes
        """
        # Extract sensitive attribute and target
        a = y_true[:, [self.idx]]
        y_true_target = y_true[:, [0]]
        
        # Convert predictions to class predictions (assuming they're probabilities)
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_class = np.argmax(y_pred, axis=1).reshape(-1, 1)
        else:
            y_pred_class = np.rint(y_pred).astype(int)
        
        # Determine number of classes
        if self.num_classes is None:
            num_classes = len(np.unique(y_true_target))
        else:
            num_classes = self.num_classes
        
        # Split by sensitive attribute
        unique_attrs = np.unique(a)
        
        # Calculate DP for each sensitive attribute group
        dp_by_group = {}
        for attr_val in unique_attrs:
            mask = (a == attr_val).flatten()
            if np.sum(mask) > 0:  # Only calculate if group has samples
                y_true_group = y_true_target[mask]
                y_pred_group = y_pred_class[mask]
                dp_by_group[attr_val] = self._DP_multiclass(y_true_group, y_pred_group, num_classes)
        
        # Calculate maximum difference across all classes and groups
        max_diff = 0
        for class_idx in range(num_classes):
            class_dps = [dp_by_group[attr][class_idx] for attr in dp_by_group.keys()]
            if len(class_dps) > 1:
                class_diff = np.max(class_dps) - np.min(class_dps)
                max_diff = max(max_diff, class_diff)
        
        return 1 - max_diff


# Alternative implementation that averages across classes
class TPRDiffAvg(Metric):
    def __init__(self, name='TPR_diff_avg', weight_vector=None, 
                 threshold=0.5, attribute_index=1, num_classes=None):
        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index
        self.num_classes = num_classes
    
    def _TPR_multiclass(self, y_true, y_pred, num_classes):
        """Calculate TPR for each class in multi-class setting"""
        epsilon = 1e-7
        
        if num_classes is None:
            classes = np.unique(np.concatenate([y_true.flatten(), y_pred.flatten()]))
            num_classes = len(classes)
        else:
            classes = np.arange(num_classes)
        
        if len(y_true) == 0:
            return np.zeros(num_classes)
        
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=classes)
        
        tpr_per_class = np.zeros(num_classes)
        for i in range(num_classes):
            if i < cm.shape[0]:
                tp = cm[i, i] if i < cm.shape[1] else 0
                fn = cm[i, :].sum() - tp if i < cm.shape[0] else 0
                tpr_per_class[i] = tp / (tp + fn + epsilon)
        
        return tpr_per_class
    
    def evaluate(self, y_true, y_pred, x, model):
        """
        Evaluate average TPR difference across sensitive attribute groups for multi-class
        Returns: 1 - average_difference_across_classes
        """
        # Extract sensitive attribute and target
        a = y_true[:, [self.idx]]
        y_true_target = y_true[:, [0]]
        
        # Convert predictions to class predictions
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_class = np.argmax(y_pred, axis=1).reshape(-1, 1)
        else:
            y_pred_class = np.rint(y_pred).astype(int)
        
        # Determine number of classes
        if self.num_classes is None:
            num_classes = len(np.unique(y_true_target))
        else:
            num_classes = self.num_classes
        
        # Split by sensitive attribute
        unique_attrs = np.unique(a)
        
        # Calculate TPR for each sensitive attribute group
        tpr_by_group = {}
        for attr_val in unique_attrs:
            mask = (a == attr_val).flatten()
            if np.sum(mask) > 0:
                y_true_group = y_true_target[mask]
                y_pred_group = y_pred_class[mask]
                tpr_by_group[attr_val] = self._TPR_multiclass(y_true_group, y_pred_group, num_classes)
        
        # Calculate average difference across all classes
        class_diffs = []
        for class_idx in range(num_classes):
            class_tprs = [tpr_by_group[attr][class_idx] for attr in tpr_by_group.keys()]
            if len(class_tprs) > 1:
                class_diff = np.max(class_tprs) - np.min(class_tprs)
                class_diffs.append(class_diff)
        
        avg_diff = np.mean(class_diffs) if class_diffs else 0
        return 1 - avg_diff
