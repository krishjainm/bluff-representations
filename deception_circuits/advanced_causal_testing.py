"""
Advanced causal testing methods for deception circuit research.

This module implements sophisticated causal intervention techniques beyond basic
activation patching, including:

1. Attention Patching: Intervene on attention weights to test causal relationships
2. Steering Vectors: Learn and apply steering vectors to control model behavior
3. Gradient-Based Interventions: Use gradients to identify and manipulate causal pathways
4. Multi-Layer Interventions: Coordinate interventions across multiple layers
5. Interpretable Interventions: Make interventions more interpretable and controllable

These methods provide deeper insights into the causal mechanisms underlying
deception in language models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class InterventionConfig:
    """Configuration for advanced causal interventions."""
    intervention_type: str  # 'attention', 'steering', 'gradient', 'multi_layer'
    layer_indices: List[int]
    strength: float = 1.0
    direction: str = 'positive'  # 'positive', 'negative', 'both'
    interpretable: bool = True
    save_intermediate: bool = False


class AttentionPatcher:
    """
    Attention patching for causal testing.
    
    This class implements attention-based interventions to test whether
    specific attention patterns are causally important for deception.
    """
    
    def __init__(self, device: str = "cpu"):
        """Initialize attention patcher."""
        self.device = device
        self.attention_hooks = []
        self.attention_cache = {}
        
    def patch_attention_weights(self,
                              model: nn.Module,
                              truthful_attention: Dict[int, torch.Tensor],
                              deceptive_attention: Dict[int, torch.Tensor],
                              layer_indices: List[int],
                              patch_strength: float = 1.0) -> Callable:
        """
        Create a function that patches attention weights during forward pass.
        
        Args:
            model: The model to patch
            truthful_attention: Attention weights from truthful examples
            deceptive_attention: Attention weights from deceptive examples
            layer_indices: Layers to patch
            patch_strength: Strength of the patch (0.0 = no patch, 1.0 = full patch)
            
        Returns:
            Function that applies the attention patch
        """
        def attention_patch_fn(module, input, output):
            # Store original attention
            if hasattr(output, 'attentions') and output.attentions is not None:
                original_attention = output.attentions
                
                # Apply patch
                layer_idx = self._get_layer_index(module)
                if layer_idx in layer_indices:
                    if patch_strength > 0:
                        # Replace with truthful attention patterns
                        if layer_idx in truthful_attention:
                            patched_attention = (
                                patch_strength * truthful_attention[layer_idx] + 
                                (1 - patch_strength) * original_attention
                            )
                            output.attentions = patched_attention
                            
        return attention_patch_fn
    
    def _get_layer_index(self, module: nn.Module) -> int:
        """Extract layer index from module."""
        # This is model-specific and may need customization
        for name, child in module.named_modules():
            if child is module:
                # Try to extract layer index from name
                parts = name.split('.')
                for part in parts:
                    if part.startswith('layer_') or part.isdigit():
                        try:
                            return int(part.split('_')[-1])
                        except:
                            continue
        return -1
    
    def test_attention_causality(self,
                               model: nn.Module,
                               truthful_inputs: torch.Tensor,
                               deceptive_inputs: torch.Tensor,
                               truthful_attention: Dict[int, torch.Tensor],
                               deceptive_attention: Dict[int, torch.Tensor],
                               layer_indices: List[int],
                               probe: Optional[nn.Module] = None) -> Dict:
        """
        Test whether attention patterns are causally important.
        
        Args:
            model: Model to test
            truthful_inputs: Inputs from truthful examples
            deceptive_inputs: Inputs from deceptive examples
            truthful_attention: Attention weights from truthful examples
            deceptive_attention: Attention weights from deceptive examples
            layer_indices: Layers to test
            probe: Optional probe for evaluation
            
        Returns:
            Dictionary with causality test results
        """
        results = {}
        
        for layer_idx in layer_indices:
            logger.info(f"Testing attention causality for layer {layer_idx}")
            
            # Test 1: Replace deceptive attention with truthful attention
            patcher_fn = self.patch_attention_weights(
                model, truthful_attention, deceptive_attention, [layer_idx], 1.0
            )
            
            # Apply patch and evaluate
            with torch.no_grad():
                # Get original deceptive predictions
                original_outputs = model(deceptive_inputs)
                original_predictions = self._extract_predictions(original_outputs)
                
                # Apply attention patch
                hook = self._register_attention_hook(model, layer_idx, patcher_fn)
                
                try:
                    patched_outputs = model(deceptive_inputs)
                    patched_predictions = self._extract_predictions(patched_outputs)
                    
                    # Calculate change in predictions
                    prediction_change = patched_predictions - original_predictions
                    
                    results[f'layer_{layer_idx}'] = {
                        'original_predictions': original_predictions.cpu().numpy(),
                        'patched_predictions': patched_predictions.cpu().numpy(),
                        'prediction_change': prediction_change.cpu().numpy(),
                        'mean_change': prediction_change.mean().item(),
                        'std_change': prediction_change.std().item(),
                        'significant_change': (prediction_change.abs() > 0.1).sum().item()
                    }
                    
                finally:
                    hook.remove()
        
        return results
    
    def _extract_predictions(self, outputs) -> torch.Tensor:
        """Extract predictions from model outputs."""
        if hasattr(outputs, 'logits'):
            return torch.softmax(outputs.logits, dim=-1)
        elif hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        else:
            return outputs
    
    def _register_attention_hook(self, model: nn.Module, layer_idx: int, patch_fn: Callable):
        """Register attention hook on specific layer."""
        # This is model-specific and may need customization
        for name, module in model.named_modules():
            if f'layer_{layer_idx}' in name and hasattr(module, 'attention'):
                return module.attention.register_forward_hook(patch_fn)
        raise ValueError(f"Could not find attention module for layer {layer_idx}")


class SteeringVectorLearner:
    """
    Learn steering vectors to control model behavior.
    
    This class implements methods to learn vectors that can steer model
    behavior toward or away from deception.
    """
    
    def __init__(self, device: str = "cpu"):
        """Initialize steering vector learner."""
        self.device = device
        self.steering_vectors = {}
        
    def learn_deception_steering_vector(self,
                                      truthful_activations: torch.Tensor,
                                      deceptive_activations: torch.Tensor,
                                      method: str = 'pca',
                                      layer_idx: int = -1) -> torch.Tensor:
        """
        Learn a steering vector that distinguishes truthful from deceptive activations.
        
        Args:
            truthful_activations: Activations from truthful examples
            deceptive_activations: Activations from deceptive examples
            method: Method to learn steering vector ('pca', 'logistic', 'mean_diff')
            layer_idx: Layer index for the steering vector
            
        Returns:
            Learned steering vector
        """
        if layer_idx >= 0:
            truthful_layer = truthful_activations[:, layer_idx, :]
            deceptive_layer = deceptive_activations[:, layer_idx, :]
        else:
            truthful_layer = truthful_activations.mean(dim=1)  # Average across layers
            deceptive_layer = deceptive_activations.mean(dim=1)
        
        if method == 'mean_diff':
            # Simple mean difference
            steering_vector = deceptive_layer.mean(dim=0) - truthful_layer.mean(dim=0)
            steering_vector = F.normalize(steering_vector, p=2, dim=0)
            
        elif method == 'pca':
            # PCA-based steering vector
            combined_activations = torch.cat([truthful_layer, deceptive_layer], dim=0)
            labels = torch.cat([
                torch.zeros(len(truthful_layer)),
                torch.ones(len(deceptive_layer))
            ])
            
            # Apply PCA
            pca = PCA(n_components=min(10, combined_activations.shape[1]))
            pca_activations = pca.fit_transform(combined_activations.cpu().numpy())
            
            # Learn linear classifier on PCA space
            clf = LogisticRegression()
            clf.fit(pca_activations, labels.cpu().numpy())
            
            # Convert back to original space
            steering_vector = torch.zeros_like(truthful_layer[0])
            for i, coef in enumerate(clf.coef_[0]):
                steering_vector += coef * torch.tensor(pca.components_[i]).to(self.device)
            
            steering_vector = F.normalize(steering_vector, p=2, dim=0)
            
        elif method == 'logistic':
            # Direct logistic regression on activations
            combined_activations = torch.cat([truthful_layer, deceptive_layer], dim=0)
            labels = torch.cat([
                torch.zeros(len(truthful_layer)),
                torch.ones(len(deceptive_layer))
            ])
            
            clf = LogisticRegression(max_iter=1000)
            clf.fit(combined_activations.cpu().numpy(), labels.cpu().numpy())
            
            steering_vector = torch.tensor(clf.coef_[0]).to(self.device)
            steering_vector = F.normalize(steering_vector, p=2, dim=0)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.steering_vectors[layer_idx] = steering_vector
        return steering_vector
    
    def apply_steering_intervention(self,
                                  activations: torch.Tensor,
                                  steering_vector: torch.Tensor,
                                  layer_idx: int,
                                  strength: float = 1.0,
                                  direction: str = 'positive') -> torch.Tensor:
        """
        Apply steering intervention to activations.
        
        Args:
            activations: Input activations
            steering_vector: Learned steering vector
            layer_idx: Layer to apply intervention to
            strength: Strength of intervention
            direction: Direction of intervention ('positive', 'negative', 'both')
            
        Returns:
            Steered activations
        """
        steered_activations = activations.clone()
        
        # Apply steering to specified layer
        if direction == 'positive':
            steered_activations[:, layer_idx, :] += strength * steering_vector
        elif direction == 'negative':
            steered_activations[:, layer_idx, :] -= strength * steering_vector
        elif direction == 'both':
            # Apply both directions and return both results
            pos_steered = steered_activations.clone()
            neg_steered = steered_activations.clone()
            pos_steered[:, layer_idx, :] += strength * steering_vector
            neg_steered[:, layer_idx, :] -= strength * steering_vector
            return torch.stack([pos_steered, neg_steered], dim=0)
        
        return steered_activations
    
    def test_steering_effectiveness(self,
                                  activations: torch.Tensor,
                                  labels: torch.Tensor,
                                  steering_vector: torch.Tensor,
                                  layer_idx: int,
                                  probe: Optional[nn.Module] = None) -> Dict:
        """
        Test the effectiveness of steering intervention.
        
        Args:
            activations: Input activations
            labels: Ground truth labels
            steering_vector: Learned steering vector
            layer_idx: Layer to test
            probe: Optional probe for evaluation
            
        Returns:
            Dictionary with steering effectiveness results
        """
        results = {}
        
        # Test different steering strengths
        strengths = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        for strength in strengths:
            # Apply positive steering
            pos_steered = self.apply_steering_intervention(
                activations, steering_vector, layer_idx, strength, 'positive'
            )
            
            # Apply negative steering
            neg_steered = self.apply_steering_intervention(
                activations, steering_vector, layer_idx, strength, 'negative'
            )
            
            # Evaluate with probe if available
            if probe is not None:
                probe.eval()
                with torch.no_grad():
                    # Original predictions
                    original_preds = probe(activations[:, layer_idx, :])
                    
                    # Positive steering predictions
                    pos_preds = probe(pos_steered[:, layer_idx, :])
                    
                    # Negative steering predictions
                    neg_preds = probe(neg_steered[:, layer_idx, :])
                    
                    results[f'strength_{strength}'] = {
                        'original_deception_prob': original_preds.mean().item(),
                        'positive_steering_prob': pos_preds.mean().item(),
                        'negative_steering_prob': neg_preds.mean().item(),
                        'positive_change': (pos_preds - original_preds).mean().item(),
                        'negative_change': (neg_preds - original_preds).mean().item(),
                        'steering_magnitude': strength
                    }
        
        return results


class GradientBasedInterventions:
    """
    Gradient-based causal interventions.
    
    This class implements interventions based on gradients to identify
    and manipulate causal pathways in the model.
    """
    
    def __init__(self, device: str = "cpu"):
        """Initialize gradient-based interventions."""
        self.device = device
        
    def compute_deception_gradients(self,
                                  model: nn.Module,
                                  inputs: torch.Tensor,
                                  labels: torch.Tensor,
                                  layer_idx: int) -> torch.Tensor:
        """
        Compute gradients of deception prediction with respect to activations.
        
        Args:
            model: Model to analyze
            inputs: Input tensors
            labels: Deception labels
            layer_idx: Layer to compute gradients for
            
        Returns:
            Gradients of deception with respect to activations
        """
        model.train()
        inputs.requires_grad_(True)
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss (assuming binary classification)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
            
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            loss, inputs, retain_graph=True, create_graph=True
        )[0]
        
        return gradients
    
    def identify_causal_features(self,
                               gradients: torch.Tensor,
                               activations: torch.Tensor,
                               top_k: int = 100) -> Dict:
        """
        Identify the most causally important features based on gradients.
        
        Args:
            gradients: Gradients with respect to activations
            activations: Original activations
            top_k: Number of top features to identify
            
        Returns:
            Dictionary with causal feature analysis
        """
        # Compute gradient magnitudes
        gradient_magnitudes = gradients.abs().mean(dim=0)
        
        # Get top-k features
        top_indices = torch.topk(gradient_magnitudes, top_k).indices
        top_magnitudes = gradient_magnitudes[top_indices]
        
        # Analyze feature importance
        causal_analysis = {
            'top_causal_features': {
                'indices': top_indices.cpu().numpy(),
                'magnitudes': top_magnitudes.cpu().numpy()
            },
            'gradient_statistics': {
                'mean_magnitude': gradient_magnitudes.mean().item(),
                'std_magnitude': gradient_magnitudes.std().item(),
                'max_magnitude': gradient_magnitudes.max().item(),
                'sparsity': (gradient_magnitudes < 1e-6).float().mean().item()
            }
        }
        
        return causal_analysis
    
    def apply_gradient_intervention(self,
                                  activations: torch.Tensor,
                                  gradients: torch.Tensor,
                                  intervention_strength: float = 1.0,
                                  intervention_type: str = 'additive') -> torch.Tensor:
        """
        Apply intervention based on gradients.
        
        Args:
            activations: Input activations
            gradients: Gradients to use for intervention
            intervention_strength: Strength of intervention
            intervention_type: Type of intervention ('additive', 'multiplicative', 'threshold')
            
        Returns:
            Intervened activations
        """
        if intervention_type == 'additive':
            # Add gradients to activations
            intervened = activations + intervention_strength * gradients
            
        elif intervention_type == 'multiplicative':
            # Multiply activations by gradient-based weights
            weights = 1 + intervention_strength * torch.sigmoid(gradients)
            intervened = activations * weights
            
        elif intervention_type == 'threshold':
            # Apply threshold-based intervention
            mask = gradients.abs() > gradients.abs().quantile(0.9)
            intervened = activations.clone()
            intervened[mask] += intervention_strength * gradients[mask]
            
        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")
        
        return intervened


class MultiLayerIntervention:
    """
    Coordinate interventions across multiple layers.
    
    This class implements sophisticated multi-layer interventions that
    can test complex causal relationships across the entire model.
    """
    
    def __init__(self, device: str = "cpu"):
        """Initialize multi-layer intervention."""
        self.device = device
        
    def coordinate_intervention(self,
                              activations: torch.Tensor,
                              intervention_configs: List[InterventionConfig],
                              truthful_activations: Optional[torch.Tensor] = None,
                              deceptive_activations: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Coordinate interventions across multiple layers.
        
        Args:
            activations: Input activations
            intervention_configs: List of intervention configurations
            truthful_activations: Optional truthful activations for patching
            deceptive_activations: Optional deceptive activations for patching
            
        Returns:
            Intervened activations
        """
        intervened_activations = activations.clone()
        
        for config in intervention_configs:
            if config.intervention_type == 'attention':
                # Apply attention patching
                if truthful_activations is not None and deceptive_activations is not None:
                    patcher = AttentionPatcher(self.device)
                    # Apply attention patching by replacing deceptive patterns with truthful ones
                    # For each specified layer, blend activations based on patch strength
                    for layer_idx in config.layer_indices:
                        if layer_idx < intervened_activations.shape[1]:
                            # Extract truthful and deceptive patterns for this layer
                            truth_layer = truthful_activations[:, layer_idx, :].mean(dim=0, keepdim=True)
                            deceptive_layer = deceptive_activations[:, layer_idx, :].mean(dim=0, keepdim=True)
                            
                            # Apply patch: replace deceptive pattern with truthful pattern
                            original_layer = intervened_activations[:, layer_idx, :]
                            # Blend based on patch strength
                            patched_layer = (
                                config.strength * truth_layer + 
                                (1 - config.strength) * original_layer
                            )
                            intervened_activations[:, layer_idx, :] = patched_layer.squeeze(0)
                    
            elif config.intervention_type == 'steering':
                # Apply steering intervention
                if hasattr(self, 'steering_vectors'):
                    for layer_idx in config.layer_indices:
                        if layer_idx in self.steering_vectors:
                            steering_vector = self.steering_vectors[layer_idx]
                            steered = self.apply_steering_intervention(
                                intervened_activations, steering_vector, layer_idx,
                                config.strength, config.direction
                            )
                            intervened_activations = steered
                            
            elif config.intervention_type == 'gradient':
                # Apply gradient-based intervention
                # Compute gradients using activation differences as proxy
                if truthful_activations is not None and deceptive_activations is not None:
                    # Compute gradient-like signal as difference between truthful and deceptive
                    for layer_idx in config.layer_indices:
                        if layer_idx < intervened_activations.shape[1]:
                            truth_mean = truthful_activations[:, layer_idx, :].mean(dim=0, keepdim=True)
                            deceptive_mean = deceptive_activations[:, layer_idx, :].mean(dim=0, keepdim=True)
                            # Gradient-like direction pointing from deceptive to truthful
                            gradient_direction = truth_mean - deceptive_mean
                            # Normalize
                            grad_norm = gradient_direction.norm()
                            if grad_norm > 1e-8:
                                gradient_direction = gradient_direction / grad_norm
                            # Apply intervention
                            intervened_activations[:, layer_idx, :] += (
                                config.strength * gradient_direction.squeeze(0)
                            )
                
            elif config.intervention_type == 'multi_layer':
                # Apply coordinated multi-layer intervention
                intervened_activations = self._apply_multi_layer_intervention(
                    intervened_activations, config
                )
        
        return intervened_activations
    
    def _apply_multi_layer_intervention(self,
                                      activations: torch.Tensor,
                                      config: InterventionConfig) -> torch.Tensor:
        """Apply multi-layer intervention."""
        intervened = activations.clone()
        
        # Example: Apply progressive intervention across layers
        # Progressive strength based on layer position (stronger in later layers)
        for i, layer_idx in enumerate(config.layer_indices):
            if layer_idx < intervened.shape[1]:
                # Progressive strength based on layer position
                progressive_strength = config.strength * (i + 1) / len(config.layer_indices)
                
                # Apply intervention to this layer based on intervention type
                # Default: additive intervention (can be customized)
                if hasattr(config, 'intervention_method') and config.intervention_method == 'multiplicative':
                    # Multiplicative scaling
                    intervened[:, layer_idx, :] *= (1 + progressive_strength)
                else:
                    # Additive intervention (default)
                    # Use layer mean as reference for intervention direction
                    layer_mean = intervened[:, layer_idx, :].mean(dim=0, keepdim=True)
                    intervened[:, layer_idx, :] += progressive_strength * layer_mean.squeeze(0)
        
        return intervened


class AdvancedCausalTester:
    """
    Comprehensive advanced causal testing framework.
    
    This class orchestrates all advanced causal testing methods and provides
    a unified interface for sophisticated causal interventions.
    """
    
    def __init__(self, device: str = "cpu"):
        """Initialize advanced causal tester."""
        self.device = device
        self.attention_patcher = AttentionPatcher(device)
        self.steering_learner = SteeringVectorLearner(device)
        self.gradient_interventions = GradientBasedInterventions(device)
        self.multi_layer_intervention = MultiLayerIntervention(device)
        
    def run_comprehensive_advanced_test(self,
                                      model: Optional[nn.Module],
                                      truthful_activations: torch.Tensor,
                                      deceptive_activations: torch.Tensor,
                                      truthful_labels: torch.Tensor,
                                      deceptive_labels: torch.Tensor,
                                      probe: Optional[nn.Module] = None,
                                      layer_indices: Optional[List[int]] = None) -> Dict:
        """
        Run comprehensive advanced causal testing suite.
        
        Args:
            model: Optional model for attention/gradient interventions
            truthful_activations: Activations from truthful examples
            deceptive_activations: Activations from deceptive examples
            truthful_labels: Labels for truthful examples
            deceptive_labels: Labels for deceptive examples
            probe: Optional probe for evaluation
            layer_indices: Layers to test
            
        Returns:
            Comprehensive causal test results
        """
        if layer_indices is None:
            layer_indices = list(range(truthful_activations.shape[1]))
        
        results = {
            'steering_vector_analysis': {},
            'gradient_analysis': {},
            'attention_analysis': {},
            'multi_layer_analysis': {},
            'summary': {}
        }
        
        logger.info("Starting comprehensive advanced causal testing...")
        
        # 1. Learn steering vectors
        logger.info("Learning steering vectors...")
        for layer_idx in layer_indices:
            steering_vector = self.steering_learner.learn_deception_steering_vector(
                truthful_activations, deceptive_activations, 'pca', layer_idx
            )
            
            # Test steering effectiveness
            all_activations = torch.cat([truthful_activations, deceptive_activations])
            all_labels = torch.cat([truthful_labels, deceptive_labels])
            
            steering_results = self.steering_learner.test_steering_effectiveness(
                all_activations, all_labels, steering_vector, layer_idx, probe
            )
            
            results['steering_vector_analysis'][f'layer_{layer_idx}'] = {
                'steering_vector': steering_vector.cpu().numpy(),
                'effectiveness': steering_results
            }
        
        # 2. Gradient-based analysis (if model provided)
        if model is not None:
            logger.info("Performing gradient-based analysis...")
            for layer_idx in layer_indices:
                layer_activations = torch.cat([
                    truthful_activations[:, layer_idx, :],
                    deceptive_activations[:, layer_idx, :]
                ])
                layer_labels = torch.cat([truthful_labels, deceptive_labels])
                
                gradients = self.gradient_interventions.compute_deception_gradients(
                    model, layer_activations, layer_labels, layer_idx
                )
                
                causal_features = self.gradient_interventions.identify_causal_features(
                    gradients, layer_activations
                )
                
                results['gradient_analysis'][f'layer_{layer_idx}'] = {
                    'gradients': gradients.cpu().numpy(),
                    'causal_features': causal_features
                }
        
        # 3. Multi-layer coordinated interventions
        logger.info("Testing multi-layer interventions...")
        intervention_configs = [
            InterventionConfig('steering', layer_indices[:3], strength=1.0, direction='positive'),
            InterventionConfig('steering', layer_indices[3:6], strength=0.5, direction='negative'),
            InterventionConfig('multi_layer', layer_indices, strength=1.0)
        ]
        
        all_activations = torch.cat([truthful_activations, deceptive_activations])
        coordinated_results = self.multi_layer_intervention.coordinate_intervention(
            all_activations, intervention_configs, truthful_activations, deceptive_activations
        )
        
        results['multi_layer_analysis'] = {
            'intervention_configs': [config.__dict__ for config in intervention_configs],
            'coordinated_activations_shape': coordinated_results.shape
        }
        
        # 4. Generate summary
        results['summary'] = self._generate_advanced_summary(results)
        
        logger.info("Advanced causal testing completed!")
        return results
    
    def _generate_advanced_summary(self, results: Dict) -> Dict:
        """Generate summary of advanced causal test results."""
        summary = {
            'steering_effectiveness': {},
            'gradient_insights': {},
            'multi_layer_coordination': {},
            'recommendations': []
        }
        
        # Analyze steering effectiveness
        for layer, data in results['steering_vector_analysis'].items():
            effectiveness = data['effectiveness']
            if effectiveness:
                changes = [
                    abs(eff['positive_change']) + abs(eff['negative_change'])
                    for eff in effectiveness.values()
                ]
                max_change = max(changes) if changes else 0.0
            else:
                max_change = 0.0
            summary['steering_effectiveness'][layer] = {
                'max_effect': max_change,
                'effective': max_change > 0.1
            }
        
        # Analyze gradient insights
        if results['gradient_analysis']:
            for layer, data in results['gradient_analysis'].items():
                causal_features = data['causal_features']
                summary['gradient_insights'][layer] = {
                    'num_causal_features': len(causal_features['top_causal_features']['indices']),
                    'gradient_sparsity': causal_features['gradient_statistics']['sparsity']
                }
        
        # Generate recommendations
        effective_layers = [
            layer for layer, data in summary['steering_effectiveness'].items()
            if data['effective']
        ]
        
        if effective_layers:
            summary['recommendations'].append(
                f"Layers {effective_layers} show strong steering effectiveness"
            )
        
        return summary


# Example usage function
def run_advanced_deception_analysis(truthful_activations: torch.Tensor,
                                  deceptive_activations: torch.Tensor,
                                  truthful_labels: torch.Tensor,
                                  deceptive_labels: torch.Tensor,
                                  probe: Optional[nn.Module] = None,
                                  model: Optional[nn.Module] = None) -> Dict:
    """
    Run advanced deception circuit analysis.
    
    Args:
        truthful_activations: Activations from truthful examples
        deceptive_activations: Activations from deceptive examples
        truthful_labels: Labels for truthful examples
        deceptive_labels: Labels for deceptive examples
        probe: Optional probe for evaluation
        model: Optional model for advanced interventions
        
    Returns:
        Advanced causal analysis results
    """
    tester = AdvancedCausalTester()
    
    return tester.run_comprehensive_advanced_test(
        model=model,
        truthful_activations=truthful_activations,
        deceptive_activations=deceptive_activations,
        truthful_labels=truthful_labels,
        deceptive_labels=deceptive_labels,
        probe=probe
    )
