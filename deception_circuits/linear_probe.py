"""
Linear probe implementation for deception circuit detection.

This module implements linear classifiers (probes) that learn to detect deception
signals in model activations. The main idea is that if deception is encoded in
the model's internal representations, a simple linear classifier should be able
to distinguish between truthful and deceptive activations.

The module contains two main classes:
1. DeceptionLinearProbe: A simple linear classifier with sigmoid output
2. LinearProbeTrainer: Handles training probes across multiple layers

Linear probes are a key tool in mechanistic interpretability because they:
- Provide a simple way to test if information is linearly decodable
- Can be trained quickly and efficiently
- Give interpretable weights showing which features matter most
- Can be applied to any layer of the model

Based on the LLMProbe architecture, adapted specifically for deception research.

=== FOR FIRST-TIME READERS ===

This module implements "deception detectors" - simple AI models that scan the neural network's
internal states to find deception signals. Think of them as lie detectors for AI!

=== WHAT ARE LINEAR PROBES? ===

Linear probes are simple classifiers that try to detect deception by looking at the neural
network's internal activations (its "brain states"). They work like this:

1. **INPUT**: Activations from a specific layer of the neural network
2. **PROCESSING**: Simple linear transformation (like a weighted sum)
3. **OUTPUT**: Probability that the response is deceptive (0-1)

The key insight: If deception is encoded in the neural network's activations, a simple
linear classifier should be able to detect it.

=== WHY LINEAR PROBES MATTER ===

- **SIMPLE BUT POWERFUL**: If a simple linear classifier can detect deception, 
  then deception signals are clearly encoded in the activations
- **FAST TO TRAIN**: Can train hundreds of probes quickly
- **INTERPRETABLE**: The weights tell you which neurons matter most
- **LAYER-BY-LAYER**: Test every layer to find where deception signals emerge

=== HOW IT WORKS IN PRACTICE ===

1. **Train probes on each layer**: See which layers can best detect deception
2. **Find the best layer**: Layer with highest accuracy/AUC
3. **Analyze weights**: See which specific neurons are most important
4. **Target for causal testing**: Use the best layer for activation patching

=== EXAMPLE USAGE ===

# Create a probe for a specific layer:
probe = DeceptionLinearProbe(input_dim=768)

# Train it to detect deception:
trainer = LinearProbeTrainer(device="cpu")
results = trainer.train_probe(probe, activations, labels)

# Check how well it works:
accuracy = results['final_metrics']['accuracy']
print(f"Probe accuracy: {accuracy:.3f}")

This is the foundation of deception circuit discovery!
"""

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.linear_model import LogisticRegression
import json
from pathlib import Path


def binary_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error for binary labels (paper-style metric)."""
    probs = np.asarray(probs).flatten().astype(np.float64)
    labels = np.asarray(labels).flatten().astype(np.float64)
    if len(probs) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


class DeceptionLinearProbe(nn.Module):
    """
    Linear probe for detecting deception in model activations.
    
    This is a simple linear classifier that takes model activations as input
    and outputs a probability of deception. The probe consists of:
    - A single linear layer that maps from activation space to a single score
    - A sigmoid activation to convert the score to a probability [0,1]
    - Optional dropout for regularization
    
    The probe learns weights that capture which dimensions of the activation
    vector are most indicative of deception. High probe accuracy suggests
    that deception is linearly separable in the activation space.
    
    Attributes:
        input_dim (int): Dimension of input activations
        dropout (float): Dropout rate for regularization
        linear (nn.Linear): Main linear transformation layer
        dropout_layer (nn.Dropout): Optional dropout layer
    """
    
    def __init__(self, input_dim: int, dropout: float = 0.0):
        """
        Initialize linear probe.
        
        Args:
            input_dim: Dimension of input activations (typically 768, 1024, etc.)
            dropout: Dropout rate for regularization (0.0 = no dropout)
        """
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        
        # Main linear layer: maps from activation space to single score
        # The weights of this layer will show which activation dimensions
        # are most important for detecting deception
        self.linear = nn.Linear(input_dim, 1)
        
        # Optional dropout layer for regularization
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the probe.
        
        This method takes model activations and returns deception probabilities.
        The process is:
        1. Apply dropout (if enabled) for regularization
        2. Apply linear transformation to get raw scores
        3. Apply sigmoid to convert scores to probabilities [0,1]
        
        Args:
            x: Input activations of shape [batch_size, input_dim]
               These are the internal representations from a specific layer
               of the language model
               
        Returns:
            Probabilities of shape [batch_size] where each value represents
            the probability that the corresponding activation represents
            deceptive behavior (1.0 = definitely deceptive, 0.0 = definitely truthful)
        """
        # Apply dropout for regularization during training
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
            
        # Apply linear transformation: activations -> single score
        # This learns which activation dimensions are most important
        logits = self.linear(x)
        
        # Convert raw scores to probabilities using sigmoid
        # sigmoid(score) gives a value between 0 and 1
        probabilities = torch.sigmoid(logits).squeeze(-1)
        
        return probabilities
        
    def get_weights(self) -> torch.Tensor:
        """Get the learned weight vector."""
        return self.linear.weight.squeeze(0)
        
    def get_important_features(self, top_k: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the most important features for deception detection.
        
        Args:
            top_k: Number of top features to return
            
        Returns:
            Tuple of (indices, weights) for top-k features
        """
        weights = self.get_weights().abs()
        top_indices = torch.topk(weights, top_k).indices
        top_weights = weights[top_indices]
        
        return top_indices, top_weights


class LinearProbeTrainer:
    """
    Trainer for linear probes on deception detection.
    
    Handles training, evaluation, and cross-layer analysis.
    """
    
    def __init__(self, device: str = "cpu", lr: float = 0.01, 
                 weight_decay: float = 0.0):
        """
        Initialize trainer.
        
        Args:
            device: Device for training
            lr: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        
    def train_probe(self, activations: torch.Tensor, 
                   labels: torch.Tensor,
                   epochs: int = 100,
                   batch_size: Optional[int] = None,
                   validation_split: float = 0.2,
                   early_stopping_patience: int = 10,
                   val_activations: Optional[torch.Tensor] = None,
                   val_labels: Optional[torch.Tensor] = None) -> Dict:
        """
        Train a linear probe on deception detection.
        
        Args:
            activations: Activation tensor [batch_size, input_dim]
            labels: Binary labels [batch_size]
            epochs: Number of training epochs
            batch_size: Batch size (None for full batch)
            validation_split: Fraction for validation (ignored if val_activations given)
            early_stopping_patience: Epochs to wait for improvement
            val_activations: Optional held-out activations (e.g. disjoint base items)
            val_labels: Optional held-out labels
            
        Returns:
            Dictionary with training results and metrics
        """
        # Move to device
        activations = activations.to(self.device)
        labels = labels.to(self.device)
        
        if val_activations is not None and val_labels is not None:
            if len(val_activations) == 0:
                raise ValueError("val_activations is empty; disable val split or add samples.")
            train_activations = activations
            train_labels = labels
            val_activations = val_activations.to(self.device)
            val_labels = val_labels.to(self.device)
        else:
            n_train = int(len(activations) * (1 - validation_split))
            train_activations = activations[:n_train]
            train_labels = labels[:n_train]
            val_activations = activations[n_train:]
            val_labels = labels[n_train:]
        
        # Initialize probe
        probe = DeceptionLinearProbe(activations.shape[1]).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.lr, 
                                   weight_decay=self.weight_decay)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_auc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            probe.train()
            train_loss = 0.0
            train_correct = 0
            
            # Batch training
            if batch_size:
                num_batches = len(train_activations) // batch_size
                for i in range(0, len(train_activations), batch_size):
                    batch_activations = train_activations[i:i+batch_size]
                    batch_labels = train_labels[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = probe(batch_activations)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_correct += ((outputs > 0.5) == batch_labels).sum().item()
            else:
                optimizer.zero_grad()
                outputs = probe(train_activations)
                loss = criterion(outputs, train_labels)
                loss.backward()
                optimizer.step()
                
                train_loss = loss.item()
                train_correct = ((outputs > 0.5) == train_labels).sum().item()
            
            # Validation
            probe.eval()
            with torch.no_grad():
                val_outputs = probe(val_activations)
                val_loss = criterion(val_outputs, val_labels).item()
                val_correct = ((val_outputs > 0.5) == val_labels).sum().item()
                
                # Calculate AUC
                try:
                    val_auc = roc_auc_score(val_labels.cpu().numpy(), 
                                          val_outputs.cpu().numpy())
                except ValueError as exc:
                    raise ValueError("Validation split must contain both classes for AUROC") from exc
                    
            # Record metrics
            train_acc = train_correct / len(train_labels)
            val_acc = val_correct / len(val_labels)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['val_auc'].append(val_auc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # state_dict().copy() is shallow: tensor values keep mutating during
                # later optimizer steps.  A real early-stop checkpoint needs a clone.
                best_probe_state = copy.deepcopy(probe.state_dict())
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        # Load best model
        probe.load_state_dict(best_probe_state)
        
        # Final evaluation
        probe.eval()
        with torch.no_grad():
            final_outputs = probe(val_activations)
            final_preds = (final_outputs > 0.5).long()
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_labels.cpu().numpy(), final_preds.cpu().numpy(), 
                average='binary', zero_division=0
            )
            
            final_auc = roc_auc_score(val_labels.cpu().numpy(), 
                                    final_outputs.cpu().numpy())
            final_ece = binary_ece(
                final_outputs.cpu().numpy(), val_labels.cpu().numpy()
            )
            final_accuracy = (final_preds == val_labels.long()).float().mean().item()
        
        results = {
            'probe': probe,
            'history': history,
            'final_metrics': {
                'accuracy': final_accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': final_auc,
                'ece': final_ece,
                'val_loss': best_val_loss
            },
            'best_epoch': len(history['val_loss']) - patience_counter - 1
        }
        
        return results
        
    def train_multi_layer_probes(self, activations: torch.Tensor,
                                labels: torch.Tensor,
                                layer_names: Optional[List[str]] = None,
                                val_activations: Optional[torch.Tensor] = None,
                                val_labels: Optional[torch.Tensor] = None,
                                **train_kwargs) -> Dict:
        """
        Train probes for multiple layers.
        
        Args:
            activations: Activation tensor [batch_size, num_layers, hidden_dim]
            labels: Binary labels [batch_size]
            layer_names: Names for each layer
            val_activations: Optional [batch_val, num_layers, hidden_dim]
            val_labels: Optional [batch_val]
            **train_kwargs: Additional training arguments
            
        Returns:
            Dictionary with results for each layer
        """
        num_layers = activations.shape[1]
        if layer_names is None:
            layer_names = [f"layer_{i}" for i in range(num_layers)]
            
        results = {}
        
        for layer_idx in range(num_layers):
            layer_name = layer_names[layer_idx]
            print(f"Training probe for {layer_name}...")
            
            # Extract activations for this layer
            layer_activations = activations[:, layer_idx, :]
            layer_val_act = None
            layer_val_lab = None
            if val_activations is not None and val_labels is not None:
                layer_val_act = val_activations[:, layer_idx, :]
                layer_val_lab = val_labels
            
            # Train probe
            layer_results = self.train_probe(
                layer_activations, labels,
                val_activations=layer_val_act, val_labels=layer_val_lab,
                **train_kwargs
            )
            
            results[layer_name] = {
                'layer_idx': layer_idx,
                'results': layer_results,
                'final_accuracy': layer_results['final_metrics']['accuracy'],
                'final_auc': layer_results['final_metrics']['auc']
            }
            
        return results
        
    def evaluate_probe(self, probe: DeceptionLinearProbe,
                      activations: torch.Tensor,
                      labels: torch.Tensor) -> Dict:
        """
        Evaluate a trained probe.
        
        Args:
            probe: Trained probe
            activations: Test activations
            labels: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        probe.eval()
        activations = activations.to(self.device)
        labels = labels.to(self.device)
        
        with torch.no_grad():
            outputs = probe(activations)
            predictions = (outputs > 0.5).long()
            
            # Calculate metrics
            accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels.cpu().numpy(), predictions.cpu().numpy(),
                average='binary', zero_division=0
            )
            
            try:
                auc = roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy())
            except Exception:
                auc = 0.5
            ece = binary_ece(outputs.cpu().numpy(), labels.cpu().numpy())
                
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'ece': ece,
            'predictions': predictions.cpu().numpy(),
            'probabilities': outputs.cpu().numpy()
        }
        
    def save_probe(self, probe: DeceptionLinearProbe, 
                   filepath: Union[str, Path]) -> None:
        """Save trained probe to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': probe.state_dict(),
            'input_dim': probe.input_dim,
            'dropout': probe.dropout
        }, filepath)
        
    def load_probe(self, filepath: Union[str, Path]) -> DeceptionLinearProbe:
        """Load trained probe from file."""
        filepath = Path(filepath)
        
        checkpoint = torch.load(filepath, map_location=self.device)
        probe = DeceptionLinearProbe(
            input_dim=checkpoint['input_dim'],
            dropout=checkpoint['dropout']
        )
        probe.load_state_dict(checkpoint['model_state_dict'])
        probe.to(self.device)
        
        return probe
        
    def analyze_feature_importance(self, probe: DeceptionLinearProbe,
                                 top_k: int = 100) -> Dict:
        """
        Analyze which features are most important for deception detection.
        
        Args:
            probe: Trained probe
            top_k: Number of top features to analyze
            
        Returns:
            Dictionary with feature importance analysis
        """
        weights = probe.get_weights()
        abs_weights = weights.abs()
        
        # Get top-k features
        top_indices, top_weights = probe.get_important_features(top_k)
        
        # Calculate statistics
        weight_stats = {
            'mean': abs_weights.mean().item(),
            'std': abs_weights.std().item(),
            'max': abs_weights.max().item(),
            'min': abs_weights.min().item(),
            'sparsity': (abs_weights < 1e-6).float().mean().item()
        }
        
        return {
            'top_features': {
                'indices': top_indices.detach().cpu().numpy(),
                'weights': top_weights.detach().cpu().numpy()
            },
            'weight_statistics': weight_stats,
            'total_features': len(weights),
            'top_k': top_k
        }


def train_sklearn_probes_all_layers(
    train_activations: torch.Tensor,
    train_labels: torch.Tensor,
    test_activations: torch.Tensor,
    test_labels: torch.Tensor,
    C: float = 1.0,
    max_iter: int = 2000,
) -> Dict:
    """
    Paper-facing L2-regularized logistic regression per layer (sklearn).
    Returns metrics and a unit-norm steering vector per layer (coef direction).
    """
    y_tr = train_labels.detach().cpu().numpy().astype(int)
    y_te = test_labels.detach().cpu().numpy().astype(int)
    n_layers = train_activations.shape[1]
    out: Dict = {}
    for li in range(n_layers):
        X_tr = train_activations[:, li, :].detach().cpu().numpy()
        X_te = test_activations[:, li, :].detach().cpu().numpy()
        clf = LogisticRegression(
            C=C, max_iter=max_iter, solver="lbfgs", class_weight="balanced"
        )
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_te, preds)
        try:
            auc = roc_auc_score(y_te, probs)
        except Exception:
            auc = 0.5
        ece = binary_ece(probs, y_te.astype(float))
        w = torch.tensor(clf.coef_.astype(np.float32).squeeze(0))
        norm = w.norm().clamp(min=1e-12)
        steering = (w / norm).numpy()
        out[f"layer_{li}"] = {
            "layer_idx": li,
            "accuracy": float(acc),
            "auc": float(auc),
            "ece": float(ece),
            "steering_vector": steering,
        }
    best = max(
        ((k, v) for k, v in out.items() if k != "best_layer"),
        key=lambda x: x[1]["auc"],
    )
    out["best_layer"] = {"name": best[0], **best[1]}
    return out


def normalized_probe_steering_vector(probe: DeceptionLinearProbe) -> torch.Tensor:
    """Unit direction from PyTorch linear probe weights (deception-positive)."""
    w = probe.get_weights().detach()
    return w / w.norm().clamp(min=1e-12)
