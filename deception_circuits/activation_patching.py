"""
Activation patching implementation for causal testing of deception circuits.

This module implements the core causal intervention techniques from the BLKY project
for testing whether discovered circuits actually cause deceptive behavior. The key
idea is to manipulate model activations and observe if this changes the model's
behavior in predictable ways.

The module contains two main classes:
1. ActivationPatcher: Low-level patching operations (replace, steer, ablate)
2. CausalTester: High-level causal experiments

Core causal experiments implemented:
1. **Deception Suppression**: Replace deceptive activations with truthful ones
   - If successful, should reduce deception in the output
2. **Deception Injection**: Replace truthful activations with deceptive ones  
   - If successful, should increase deception in the output
3. **Cross-Context Patching**: Apply circuits from one scenario to another
   - Tests if deception circuits generalize across contexts
4. **Steering Interventions**: Add learned vectors to steer behavior
   - Tests if we can control deception with learned steering vectors
5. **Neuron Ablation**: Zero out or mean-ablate specific neurons
   - Tests which specific neurons are causally important

These experiments are crucial for proving that discovered circuits actually
drive deceptive behavior, rather than just correlating with it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ActivationPatcher:
    """
    Implements activation patching for causal testing of deception circuits.
    
    This class provides low-level operations for manipulating model activations
    to test causal relationships. It implements the core patching techniques
    used in mechanistic interpretability:
    
    1. **Activation Replacement**: Replace activations from one run with another
    2. **Activation Steering**: Add vectors to steer behavior in specific directions
    3. **Neuron Ablation**: Zero out or replace specific neurons
    4. **Selective Patching**: Patch only specific layers, samples, or neurons
    
    These operations are the building blocks for the higher-level causal experiments
    in the CausalTester class. The key insight is that if a circuit is causally
    important, manipulating it should change the model's behavior predictably.
    
    Based on the BLKY project methodology for testing whether identified
    circuits causally drive deceptive behavior.
    
    Attributes:
        device (str): Device for computation ("cpu" or "cuda")
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize activation patcher.
        
        Args:
            device: Device for computation ("cpu" or "cuda")
        """
        self.device = device
        
    def patch_activations(self, original_activations: torch.Tensor,
                         patch_activations: torch.Tensor,
                         patch_indices: Optional[Union[int, List[int], slice]] = None,
                         layer_indices: Optional[Union[int, List[int], slice]] = None) -> torch.Tensor:
        """
        Patch activations by replacing specified indices.
        
        Args:
            original_activations: Original activation tensor [batch_size, num_layers, hidden_dim]
            patch_activations: Activations to patch in [batch_size, num_layers, hidden_dim]
            patch_indices: Which samples to patch (None = all)
            layer_indices: Which layers to patch (None = all)
            
        Returns:
            Patched activation tensor
        """
        patched = original_activations.clone()
        
        # Handle patch indices
        if patch_indices is None:
            patch_indices = slice(None)
        elif isinstance(patch_indices, int):
            patch_indices = [patch_indices]
            
        # Handle layer indices
        if layer_indices is None:
            layer_indices = slice(None)
        elif isinstance(layer_indices, int):
            layer_indices = [layer_indices]
            
        # Apply patches
        patched[patch_indices, layer_indices, :] = patch_activations[patch_indices, layer_indices, :]
        
        return patched
        
    def patch_specific_neurons(self, original_activations: torch.Tensor,
                             patch_activations: torch.Tensor,
                             neuron_indices: List[int],
                             layer_idx: int,
                             sample_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Patch specific neurons in a specific layer.
        
        Args:
            original_activations: Original activation tensor
            patch_activations: Activations to patch in
            neuron_indices: List of neuron indices to patch
            layer_idx: Layer index to patch
            sample_indices: Which samples to patch (None = all)
            
        Returns:
            Patched activation tensor
        """
        patched = original_activations.clone()
        
        if sample_indices is None:
            sample_indices = slice(None)
            
        # Patch specific neurons
        patched[sample_indices, layer_idx, neuron_indices] = patch_activations[sample_indices, layer_idx, neuron_indices]
        
        return patched
        
    def steering_intervention(self, activations: torch.Tensor,
                            steering_vector: torch.Tensor,
                            steering_strength: float = 1.0,
                            layer_indices: Optional[Union[int, List[int]]] = None,
                            sample_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Apply steering intervention by adding a vector to activations.
        
        Args:
            activations: Original activation tensor
            steering_vector: Vector to add [hidden_dim] or [num_layers, hidden_dim]
            steering_strength: Multiplier for steering vector
            layer_indices: Which layers to apply steering to
            sample_indices: Which samples to apply steering to
            
        Returns:
            Steered activation tensor
        """
        steered = activations.clone()
        
        # Handle layer indices
        if layer_indices is None:
            layer_indices = list(range(activations.shape[1]))
        elif isinstance(layer_indices, int):
            layer_indices = [layer_indices]
            
        # Handle sample indices
        if sample_indices is None:
            sample_indices = slice(None)
            
        # Ensure steering vector has correct shape
        if steering_vector.dim() == 1:
            # Broadcast to all layers
            steering_vector = steering_vector.unsqueeze(0).expand(len(layer_indices), -1)
        elif steering_vector.dim() == 2:
            # Should be [num_layers, hidden_dim]
            steering_vector = steering_vector[layer_indices]
        else:
            raise ValueError("Steering vector must be 1D or 2D")
            
        # Apply steering
        for i, layer_idx in enumerate(layer_indices):
            steered[sample_indices, layer_idx, :] += steering_strength * steering_vector[i]
            
        return steered
        
    def zero_ablation(self, activations: torch.Tensor,
                     neuron_indices: List[int],
                     layer_idx: int,
                     sample_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Zero-ablate specific neurons (set to zero).
        
        Args:
            activations: Original activation tensor
            neuron_indices: List of neuron indices to zero out
            layer_idx: Layer index to ablate
            sample_indices: Which samples to ablate (None = all)
            
        Returns:
            Ablated activation tensor
        """
        ablated = activations.clone()
        
        if sample_indices is None:
            sample_indices = slice(None)
            
        ablated[sample_indices, layer_idx, neuron_indices] = 0.0
        
        return ablated
        
    def mean_ablation(self, activations: torch.Tensor,
                     neuron_indices: List[int],
                     layer_idx: int,
                     sample_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Mean-ablate specific neurons (set to mean across dataset).
        
        Args:
            activations: Original activation tensor
            neuron_indices: List of neuron indices to mean-ablate
            layer_idx: Layer index to ablate
            sample_indices: Which samples to ablate (None = all)
            
        Returns:
            Ablated activation tensor
        """
        ablated = activations.clone()
        
        if sample_indices is None:
            sample_indices = slice(None)
            
        # Calculate mean across all samples
        mean_values = activations[:, layer_idx, neuron_indices].mean(dim=0)
        ablated[sample_indices, layer_idx, neuron_indices] = mean_values
        
        return ablated


class CausalTester:
    """
    Comprehensive causal testing framework for deception circuits.
    
    Implements the core causal experiments from the BLKY project:
    1. Replace deceptive activations with truthful ones
    2. Inject deception into truthful runs
    3. Cross-context patching
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize causal tester.
        
        Args:
            device: Device for computation
        """
        self.device = device
        self.patcher = ActivationPatcher(device)
        
    def test_deception_suppression(self, truthful_activations: torch.Tensor,
                                 deceptive_activations: torch.Tensor,
                                 truthful_labels: torch.Tensor,
                                 deceptive_labels: torch.Tensor,
                                 layer_idx: int,
                                 probe: Optional[nn.Module] = None,
                                 autoencoder: Optional[nn.Module] = None) -> Dict:
        """
        Test 1: Replace deceptive activations with truthful ones.
        
        This tests whether removing the "deception signal" causes
        the model to behave more truthfully.
        
        Args:
            truthful_activations: Activations from truthful runs
            deceptive_activations: Activations from deceptive runs
            truthful_labels: Labels for truthful runs
            deceptive_labels: Labels for deceptive runs
            layer_idx: Layer to patch
            probe: Trained probe for evaluation
            autoencoder: Trained autoencoder for evaluation
            
        Returns:
            Dictionary with suppression test results
        """
        results = {}
        
        # Patch deceptive activations with truthful ones at specified layer
        patched_activations = self.patcher.patch_activations(
            deceptive_activations, truthful_activations, 
            layer_indices=layer_idx
        )
        
        # Evaluate with probe if provided
        if probe is not None:
            probe.eval()
            with torch.no_grad():
                # Original deceptive predictions
                original_predictions = probe(deceptive_activations[:, layer_idx, :])
                
                # Patched predictions
                patched_predictions = probe(patched_activations[:, layer_idx, :])
                
                # Calculate change in deception probability
                deception_reduction = original_predictions - patched_predictions
                
                results['probe_evaluation'] = {
                    'original_deception_prob': original_predictions.mean().item(),
                    'patched_deception_prob': patched_predictions.mean().item(),
                    'deception_reduction': deception_reduction.mean().item(),
                    'samples_with_reduction': (deception_reduction > 0).sum().item(),
                    'total_samples': len(deception_reduction)
                }
                
        # Evaluate with autoencoder if provided
        if autoencoder is not None:
            autoencoder.eval()
            with torch.no_grad():
                # Original reconstructions
                original_reconstructed, _, _ = autoencoder(deceptive_activations[:, layer_idx, :])
                
                # Patched reconstructions
                patched_reconstructed, _, _ = autoencoder(patched_activations[:, layer_idx, :])
                
                # Calculate reconstruction changes
                reconstruction_change = F.mse_loss(original_reconstructed, patched_reconstructed)
                
                results['autoencoder_evaluation'] = {
                    'reconstruction_change': reconstruction_change.item(),
                    'original_reconstruction_error': F.mse_loss(deceptive_activations[:, layer_idx, :], original_reconstructed).item(),
                    'patched_reconstruction_error': F.mse_loss(patched_activations[:, layer_idx, :], patched_reconstructed).item()
                }
                
        return results
        
    def test_deception_injection(self, truthful_activations: torch.Tensor,
                               deceptive_activations: torch.Tensor,
                               truthful_labels: torch.Tensor,
                               deceptive_labels: torch.Tensor,
                               layer_idx: int,
                               probe: Optional[nn.Module] = None,
                               autoencoder: Optional[nn.Module] = None) -> Dict:
        """
        Test 2: Inject deception into truthful runs.
        
        This tests whether adding the "deception signal" causes
        the model to behave more deceptively.
        
        Args:
            truthful_activations: Activations from truthful runs
            deceptive_activations: Activations from deceptive runs
            truthful_labels: Labels for truthful runs
            deceptive_labels: Labels for deceptive runs
            layer_idx: Layer to patch
            probe: Trained probe for evaluation
            autoencoder: Trained autoencoder for evaluation
            
        Returns:
            Dictionary with injection test results
        """
        results = {}
        
        # Patch truthful activations with deceptive ones at specified layer
        patched_activations = self.patcher.patch_activations(
            truthful_activations, deceptive_activations,
            layer_indices=layer_idx
        )
        
        # Evaluate with probe if provided
        if probe is not None:
            probe.eval()
            with torch.no_grad():
                # Original truthful predictions
                original_predictions = probe(truthful_activations[:, layer_idx, :])
                
                # Patched predictions
                patched_predictions = probe(patched_activations[:, layer_idx, :])
                
                # Calculate change in deception probability
                deception_increase = patched_predictions - original_predictions
                
                results['probe_evaluation'] = {
                    'original_deception_prob': original_predictions.mean().item(),
                    'patched_deception_prob': patched_predictions.mean().item(),
                    'deception_increase': deception_increase.mean().item(),
                    'samples_with_increase': (deception_increase > 0).sum().item(),
                    'total_samples': len(deception_increase)
                }
                
        # Evaluate with autoencoder if provided
        if autoencoder is not None:
            autoencoder.eval()
            with torch.no_grad():
                # Original reconstructions
                original_reconstructed, _, _ = autoencoder(truthful_activations[:, layer_idx, :])
                
                # Patched reconstructions
                patched_reconstructed, _, _ = autoencoder(patched_activations[:, layer_idx, :])
                
                # Calculate reconstruction changes
                reconstruction_change = F.mse_loss(original_reconstructed, patched_reconstructed)
                
                results['autoencoder_evaluation'] = {
                    'reconstruction_change': reconstruction_change.item(),
                    'original_reconstruction_error': F.mse_loss(truthful_activations[:, layer_idx, :], original_reconstructed).item(),
                    'patched_reconstruction_error': F.mse_loss(patched_activations[:, layer_idx, :], patched_reconstructed).item()
                }
                
        return results
        
    def test_cross_context_patching(self, scenario1_activations: torch.Tensor,
                                  scenario2_activations: torch.Tensor,
                                  scenario1_labels: torch.Tensor,
                                  scenario2_labels: torch.Tensor,
                                  layer_idx: int,
                                  probe: Optional[nn.Module] = None,
                                  autoencoder: Optional[nn.Module] = None) -> Dict:
        """
        Test 3: Cross-context patching.
        
        This tests whether deception circuits generalize across
        different contexts (e.g., poker vs sandbagging).
        
        Args:
            scenario1_activations: Activations from scenario 1
            scenario2_activations: Activations from scenario 2
            scenario1_labels: Labels from scenario 1
            scenario2_labels: Labels from scenario 2
            layer_idx: Layer to patch
            probe: Trained probe for evaluation
            autoencoder: Trained autoencoder for evaluation
            
        Returns:
            Dictionary with cross-context test results
        """
        results = {}
        
        # Patch scenario 2 with scenario 1 activations
        patched_activations = self.patcher.patch_activations(
            scenario2_activations, scenario1_activations,
            layer_indices=layer_idx
        )
        
        # Evaluate with probe if provided
        if probe is not None:
            probe.eval()
            with torch.no_grad():
                # Original scenario 2 predictions
                original_predictions = probe(scenario2_activations[:, layer_idx, :])
                
                # Patched predictions (scenario 2 patched with scenario 1)
                patched_predictions = probe(patched_activations[:, layer_idx, :])
                
                # Calculate change in predictions
                prediction_change = patched_predictions - original_predictions
                
                results['probe_evaluation'] = {
                    'original_scenario2_prob': original_predictions.mean().item(),
                    'patched_prob': patched_predictions.mean().item(),
                    'prediction_change': prediction_change.mean().item(),
                    'samples_with_change': (prediction_change.abs() > 0.1).sum().item(),
                    'total_samples': len(prediction_change)
                }
                
        # Evaluate with autoencoder if provided
        if autoencoder is not None:
            autoencoder.eval()
            with torch.no_grad():
                # Original reconstructions
                original_reconstructed, _, _ = autoencoder(scenario2_activations[:, layer_idx, :])
                
                # Patched reconstructions
                patched_reconstructed, _, _ = autoencoder(patched_activations[:, layer_idx, :])
                
                # Calculate reconstruction changes
                reconstruction_change = F.mse_loss(original_reconstructed, patched_reconstructed)
                
                results['autoencoder_evaluation'] = {
                    'reconstruction_change': reconstruction_change.item(),
                    'original_reconstruction_error': F.mse_loss(scenario2_activations[:, layer_idx, :], original_reconstructed).item(),
                    'patched_reconstruction_error': F.mse_loss(patched_activations[:, layer_idx, :], patched_reconstructed).item()
                }
                
        return results
    
    def run_comprehensive_cross_context_test(self,
                                           scenario_activations: Dict[str, torch.Tensor],
                                           scenario_labels: Dict[str, torch.Tensor],
                                           layer_indices: Optional[List[int]] = None,
                                           probe: Optional[nn.Module] = None) -> Dict:
        """
        Run comprehensive cross-context patching tests.
        
        Tests patching between all pairs of scenarios to see if deception
        circuits generalize. This implements the proposal requirement:
        "Patch deception circuits from poker into sandbagging → does the lie transfer?"
        
        Args:
            scenario_activations: Dictionary mapping scenario names to activations
            scenario_labels: Dictionary mapping scenario names to labels
            layer_indices: Which layers to test (None = all)
            probe: Trained probe for evaluation
            
        Returns:
            Comprehensive cross-context test results
        """
        scenario_names = list(scenario_activations.keys())
        results = {
            'cross_context_patches': {},
            'generalization_summary': {}
        }
        
        if layer_indices is None:
            # Use all layers
            num_layers = scenario_activations[scenario_names[0]].shape[1]
            layer_indices = list(range(num_layers))
        
        # Test all pairs of scenarios
        for i, scenario1 in enumerate(scenario_names):
            for scenario2 in scenario_names[i+1:]:
                patch_key = f"{scenario1}_to_{scenario2}"
                
                scenario1_acts = scenario_activations[scenario1]
                scenario2_acts = scenario_activations[scenario2]
                scenario1_labs = scenario_labels[scenario1]
                scenario2_labs = scenario_labels[scenario2]
                
                patch_results = {}
                
                # Test patching at each layer
                for layer_idx in layer_indices:
                    layer_results = self.test_cross_context_patching(
                        scenario1_acts, scenario2_acts,
                        scenario1_labs, scenario2_labs,
                        layer_idx, probe
                    )
                    patch_results[f'layer_{layer_idx}'] = layer_results
                
                # Compute generalization score
                generalization_scores = []
                for layer_key, layer_data in patch_results.items():
                    if 'probe_evaluation' in layer_data:
                        # Higher prediction change = better generalization
                        pred_change = abs(layer_data['probe_evaluation'].get('prediction_change', 0.0))
                        generalization_scores.append(pred_change)
                
                avg_generalization = np.mean(generalization_scores) if generalization_scores else 0.0
                
                results['cross_context_patches'][patch_key] = {
                    'layer_results': patch_results,
                    'generalization_score': avg_generalization,
                    'generalizes': avg_generalization > 0.1  # Threshold for generalization
                }
        
        # Summary statistics
        all_scores = [data['generalization_score'] 
                     for data in results['cross_context_patches'].values()]
        results['generalization_summary'] = {
            'avg_generalization_score': np.mean(all_scores) if all_scores else 0.0,
            'num_generalizing_pairs': sum(1 for data in results['cross_context_patches'].values() 
                                         if data['generalizes']),
            'total_pairs': len(results['cross_context_patches']),
            'generalization_rate': sum(1 for data in results['cross_context_patches'].values() 
                                      if data['generalizes']) / len(results['cross_context_patches']) 
                                      if results['cross_context_patches'] else 0.0
        }
        
        return results
        
    def test_steering_intervention(self, activations: torch.Tensor,
                                 labels: torch.Tensor,
                                 steering_vector: torch.Tensor,
                                 layer_idx: int,
                                 steering_strength: float = 1.0,
                                 probe: Optional[nn.Module] = None) -> Dict:
        """
        Test 4: Steering intervention.
        
        This tests whether we can steer the model toward or away from
        deception by adding a learned steering vector.
        
        Args:
            activations: Original activations
            labels: Original labels
            steering_vector: Vector to steer with
            layer_idx: Layer to apply steering to
            steering_strength: Strength of steering intervention
            probe: Trained probe for evaluation
            
        Returns:
            Dictionary with steering test results
        """
        results = {}
        
        # Apply steering intervention
        steered_activations = self.patcher.steering_intervention(
            activations, steering_vector, steering_strength, layer_idx
        )
        
        # Evaluate with probe if provided
        if probe is not None:
            probe.eval()
            with torch.no_grad():
                # Original predictions
                original_predictions = probe(activations[:, layer_idx, :])
                
                # Steered predictions
                steered_predictions = probe(steered_activations[:, layer_idx, :])
                
                # Calculate change in predictions
                prediction_change = steered_predictions - original_predictions
                
                # Analyze by label
                truthful_mask = labels == 0
                deceptive_mask = labels == 1
                
                results['probe_evaluation'] = {
                    'original_deception_prob': original_predictions.mean().item(),
                    'steered_deception_prob': steered_predictions.mean().item(),
                    'overall_change': prediction_change.mean().item(),
                    'truthful_change': prediction_change[truthful_mask].mean().item() if truthful_mask.sum() > 0 else 0,
                    'deceptive_change': prediction_change[deceptive_mask].mean().item() if deceptive_mask.sum() > 0 else 0,
                    'steering_strength': steering_strength
                }
                
        return results
        
    def test_neuron_ablation(self, activations: torch.Tensor,
                           labels: torch.Tensor,
                           neuron_indices: List[int],
                           layer_idx: int,
                           ablation_type: str = 'zero',
                           probe: Optional[nn.Module] = None) -> Dict:
        """
        Test 5: Neuron ablation.
        
        This tests whether specific neurons are causally important
        for deception by ablating them.
        
        Args:
            activations: Original activations
            labels: Original labels
            neuron_indices: Neurons to ablate
            layer_idx: Layer to ablate in
            ablation_type: Type of ablation ('zero' or 'mean')
            probe: Trained probe for evaluation
            
        Returns:
            Dictionary with ablation test results
        """
        results = {}
        
        # Apply ablation
        if ablation_type == 'zero':
            ablated_activations = self.patcher.zero_ablation(
                activations, neuron_indices, layer_idx
            )
        elif ablation_type == 'mean':
            ablated_activations = self.patcher.mean_ablation(
                activations, neuron_indices, layer_idx
            )
        else:
            raise ValueError("Ablation type must be 'zero' or 'mean'")
            
        # Evaluate with probe if provided
        if probe is not None:
            probe.eval()
            with torch.no_grad():
                # Original predictions
                original_predictions = probe(activations[:, layer_idx, :])
                
                # Ablated predictions
                ablated_predictions = probe(ablated_activations[:, layer_idx, :])
                
                # Calculate change in predictions
                prediction_change = ablated_predictions - original_predictions
                
                # Analyze by label
                truthful_mask = labels == 0
                deceptive_mask = labels == 1
                
                results['probe_evaluation'] = {
                    'original_deception_prob': original_predictions.mean().item(),
                    'ablated_deception_prob': ablated_predictions.mean().item(),
                    'overall_change': prediction_change.mean().item(),
                    'truthful_change': prediction_change[truthful_mask].mean().item() if truthful_mask.sum() > 0 else 0,
                    'deceptive_change': prediction_change[deceptive_mask].mean().item() if deceptive_mask.sum() > 0 else 0,
                    'ablated_neurons': len(neuron_indices),
                    'ablation_type': ablation_type
                }
                
        return results
        
    def run_comprehensive_causal_test(self, truthful_activations: torch.Tensor,
                                    deceptive_activations: torch.Tensor,
                                    truthful_labels: torch.Tensor,
                                    deceptive_labels: torch.Tensor,
                                    layer_idx: int,
                                    probe: Optional[nn.Module] = None,
                                    autoencoder: Optional[nn.Module] = None,
                                    steering_vector: Optional[torch.Tensor] = None,
                                    important_neurons: Optional[List[int]] = None) -> Dict:
        """
        Run comprehensive causal testing suite.
        
        Args:
            truthful_activations: Activations from truthful runs
            deceptive_activations: Activations from deceptive runs
            truthful_labels: Labels for truthful runs
            deceptive_labels: Labels for deceptive runs
            layer_idx: Layer to test
            probe: Trained probe for evaluation
            autoencoder: Trained autoencoder for evaluation
            steering_vector: Optional steering vector
            important_neurons: Optional list of important neurons to ablate
            
        Returns:
            Dictionary with comprehensive causal test results
        """
        results = {
            'layer_idx': layer_idx,
            'test_summary': {}
        }
        
        # Test 1: Deception suppression
        print(f"Running deception suppression test on layer {layer_idx}...")
        suppression_results = self.test_deception_suppression(
            truthful_activations, deceptive_activations,
            truthful_labels, deceptive_labels, layer_idx,
            probe, autoencoder
        )
        results['deception_suppression'] = suppression_results
        
        # Test 2: Deception injection
        print(f"Running deception injection test on layer {layer_idx}...")
        injection_results = self.test_deception_injection(
            truthful_activations, deceptive_activations,
            truthful_labels, deceptive_labels, layer_idx,
            probe, autoencoder
        )
        results['deception_injection'] = injection_results
        
        # Test 3: Steering intervention (if vector provided)
        if steering_vector is not None:
            print(f"Running steering intervention test on layer {layer_idx}...")
            steering_results = self.test_steering_intervention(
                torch.cat([truthful_activations, deceptive_activations]),
                torch.cat([truthful_labels, deceptive_labels]),
                steering_vector, layer_idx, probe=probe
            )
            results['steering_intervention'] = steering_results
            
        # Test 4: Neuron ablation (if neurons provided)
        if important_neurons is not None:
            print(f"Running neuron ablation test on layer {layer_idx}...")
            ablation_results = self.test_neuron_ablation(
                torch.cat([truthful_activations, deceptive_activations]),
                torch.cat([truthful_labels, deceptive_labels]),
                important_neurons, layer_idx, probe=probe
            )
            results['neuron_ablation'] = ablation_results
            
        # Compile summary
        if probe is not None:
            results['test_summary'] = {
                'suppression_effective': suppression_results.get('probe_evaluation', {}).get('deception_reduction', 0) > 0.1,
                'injection_effective': injection_results.get('probe_evaluation', {}).get('deception_increase', 0) > 0.1,
                'layer_causally_important': (
                    suppression_results.get('probe_evaluation', {}).get('deception_reduction', 0) > 0.05 or
                    injection_results.get('probe_evaluation', {}).get('deception_increase', 0) > 0.05
                )
            }
            
        return results
