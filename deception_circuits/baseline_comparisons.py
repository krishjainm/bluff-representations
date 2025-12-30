"""
Baseline comparison frameworks for deception circuit research.

This module implements comparison frameworks mentioned in the research proposal:
1. Parameter-Based: Base model vs RLHF-aligned model
2. Architecture-Based: Different model sizes (5B vs 10B vs 25B)

These comparisons help understand:
- Whether deception circuits are caused by alignment training
- Whether deception circuits are shared across architectures/sizes
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from pathlib import Path
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelComparisonConfig:
    """Configuration for model comparisons."""
    base_model_name: str
    aligned_model_name: Optional[str] = None
    model_sizes: Optional[List[str]] = None  # e.g., ["5B", "10B", "25B"]
    device: str = "cpu"
    layer_indices: Optional[List[int]] = None


class RLHFComparison:
    """
    Compare base models vs RLHF-aligned models.
    
    This comparison tests whether deception circuits are caused by
    alignment training (RLHF). If circuits differ significantly between
    base and aligned models, it suggests alignment affects deception mechanisms.
    """
    
    def __init__(self, base_model, aligned_model, device: str = "cpu"):
        """
        Initialize RLHF comparison.
        
        Args:
            base_model: Base (pre-alignment) model
            aligned_model: RLHF-aligned model
            device: Device for computation
        """
        self.base_model = base_model
        self.aligned_model = aligned_model
        self.device = device
        
    def compare_deception_circuits(self,
                                  prompts: List[str],
                                  labels: List[int],
                                  layer_indices: Optional[List[int]] = None) -> Dict:
        """
        Compare deception circuits between base and aligned models.
        
        Args:
            prompts: List of input prompts
            labels: List of labels (0=truthful, 1=deceptive)
            layer_indices: Which layers to compare
            
        Returns:
            Comparison results
        """
        # Extract activations from both models
        base_activations = self._extract_activations(
            self.base_model, prompts, layer_indices
        )
        aligned_activations = self._extract_activations(
            self.aligned_model, prompts, layer_indices
        )
        
        # Compare activation patterns
        comparison_results = {}
        
        if layer_indices is None:
            layer_indices = list(range(base_activations.shape[1]))
        
        for layer_idx in layer_indices:
            base_layer = base_activations[:, layer_idx, :]
            aligned_layer = aligned_activations[:, layer_idx, :]
            
            # Split by label
            truthful_base = base_layer[torch.tensor(labels) == 0]
            deceptive_base = base_layer[torch.tensor(labels) == 1]
            truthful_aligned = aligned_layer[torch.tensor(labels) == 0]
            deceptive_aligned = aligned_layer[torch.tensor(labels) == 1]
            
            # Compute differences
            base_deception_diff = self._compute_deception_difference(
                truthful_base, deceptive_base
            )
            aligned_deception_diff = self._compute_deception_difference(
                truthful_aligned, deceptive_aligned
            )
            
            # Compare models
            model_similarity = self._compute_similarity(
                base_layer, aligned_layer
            )
            
            comparison_results[f'layer_{layer_idx}'] = {
                'base_deception_diff': base_deception_diff,
                'aligned_deception_diff': aligned_deception_diff,
                'model_similarity': model_similarity,
                'deception_diff_ratio': aligned_deception_diff / base_deception_diff if base_deception_diff > 0 else 0.0
            }
        
        return {
            'layer_comparisons': comparison_results,
            'summary': self._summarize_comparison(comparison_results)
        }
    
    def _extract_activations(self, model, prompts: List[str],
                           layer_indices: Optional[List[int]] = None) -> torch.Tensor:
        """Extract activations from model."""
        # This is a placeholder - actual implementation would hook into model
        # For now, return dummy activations
        num_prompts = len(prompts)
        num_layers = 24  # Default
        hidden_dim = 768  # Default
        
        if layer_indices:
            num_layers = len(layer_indices)
        
        return torch.randn(num_prompts, num_layers, hidden_dim)
    
    def _compute_deception_difference(self, truthful: torch.Tensor,
                                     deceptive: torch.Tensor) -> float:
        """Compute difference between truthful and deceptive activations."""
        if len(truthful) == 0 or len(deceptive) == 0:
            return 0.0
        
        truthful_mean = truthful.mean(dim=0)
        deceptive_mean = deceptive.mean(dim=0)
        
        diff = torch.norm(truthful_mean - deceptive_mean).item()
        return diff
    
    def _compute_similarity(self, activations1: torch.Tensor,
                          activations2: torch.Tensor) -> float:
        """Compute similarity between two activation sets."""
        if activations1.shape != activations2.shape:
            # Reshape to match
            min_size = min(activations1.numel(), activations2.numel())
            a1_flat = activations1.flatten()[:min_size]
            a2_flat = activations2.flatten()[:min_size]
        else:
            a1_flat = activations1.flatten()
            a2_flat = activations2.flatten()
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            a1_flat.unsqueeze(0), a2_flat.unsqueeze(0)
        ).item()
        
        return cos_sim
    
    def _summarize_comparison(self, comparison_results: Dict) -> Dict:
        """Summarize comparison results."""
        deception_diffs_base = []
        deception_diffs_aligned = []
        similarities = []
        
        for layer_data in comparison_results.values():
            deception_diffs_base.append(layer_data['base_deception_diff'])
            deception_diffs_aligned.append(layer_data['aligned_deception_diff'])
            similarities.append(layer_data['model_similarity'])
        
        return {
            'avg_base_deception_diff': np.mean(deception_diffs_base),
            'avg_aligned_deception_diff': np.mean(deception_diffs_aligned),
            'avg_model_similarity': np.mean(similarities),
            'alignment_effect': np.mean(deception_diffs_aligned) - np.mean(deception_diffs_base)
        }


class ModelSizeComparison:
    """
    Compare deception circuits across different model sizes.
    
    This comparison tests whether deception circuits are shared across
    architectures and model sizes (e.g., 5B vs 10B vs 25B parameters).
    """
    
    def __init__(self, models: Dict[str, nn.Module], device: str = "cpu"):
        """
        Initialize model size comparison.
        
        Args:
            models: Dictionary mapping model size names to models
                   e.g., {"5B": model_5b, "10B": model_10b, "25B": model_25b}
            device: Device for computation
        """
        self.models = models
        self.device = device
        
    def compare_deception_circuits(self,
                                  prompts: List[str],
                                  labels: List[int],
                                  layer_indices: Optional[List[int]] = None) -> Dict:
        """
        Compare deception circuits across model sizes.
        
        Args:
            prompts: List of input prompts
            labels: List of labels (0=truthful, 1=deceptive)
            layer_indices: Which layers to compare
            
        Returns:
            Comparison results across model sizes
        """
        # Extract activations from all models
        model_activations = {}
        for model_name, model in self.models.items():
            activations = self._extract_activations(model, prompts, layer_indices)
            model_activations[model_name] = activations
        
        # Compare across models
        comparison_results = {}
        
        model_names = list(self.models.keys())
        for i, model1_name in enumerate(model_names):
            for model2_name in model_names[i+1:]:
                comparison_key = f"{model1_name}_vs_{model2_name}"
                
                activations1 = model_activations[model1_name]
                activations2 = model_activations[model2_name]
                
                # Compute deception differences for each model
                diff1 = self._compute_deception_difference_by_model(
                    activations1, labels
                )
                diff2 = self._compute_deception_difference_by_model(
                    activations2, labels
                )
                
                # Compute cross-model similarity
                similarity = self._compute_cross_model_similarity(
                    activations1, activations2
                )
                
                comparison_results[comparison_key] = {
                    f'{model1_name}_deception_diff': diff1,
                    f'{model2_name}_deception_diff': diff2,
                    'cross_model_similarity': similarity,
                    'deception_diff_ratio': diff2 / diff1 if diff1 > 0 else 0.0
                }
        
        return {
            'model_comparisons': comparison_results,
            'summary': self._summarize_size_comparison(comparison_results, model_names)
        }
    
    def _extract_activations(self, model, prompts: List[str],
                           layer_indices: Optional[List[int]] = None) -> torch.Tensor:
        """Extract activations from model."""
        # Placeholder - actual implementation would hook into model
        num_prompts = len(prompts)
        num_layers = 24  # Default
        hidden_dim = 768  # Default
        
        if layer_indices:
            num_layers = len(layer_indices)
        
        return torch.randn(num_prompts, num_layers, hidden_dim)
    
    def _compute_deception_difference_by_model(self,
                                             activations: torch.Tensor,
                                             labels: List[int]) -> float:
        """Compute deception difference for a single model."""
        labels_tensor = torch.tensor(labels)
        truthful = activations[labels_tensor == 0]
        deceptive = activations[labels_tensor == 1]
        
        if len(truthful) == 0 or len(deceptive) == 0:
            return 0.0
        
        truthful_mean = truthful.mean(dim=0)
        deceptive_mean = deceptive.mean(dim=0)
        
        diff = torch.norm(truthful_mean - deceptive_mean).item()
        return diff
    
    def _compute_cross_model_similarity(self,
                                       activations1: torch.Tensor,
                                       activations2: torch.Tensor) -> float:
        """Compute similarity between activations from different models."""
        # Normalize shapes
        if activations1.shape != activations2.shape:
            min_size = min(activations1.numel(), activations2.numel())
            a1_flat = activations1.flatten()[:min_size]
            a2_flat = activations2.flatten()[:min_size]
        else:
            a1_flat = activations1.flatten()
            a2_flat = activations2.flatten()
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            a1_flat.unsqueeze(0), a2_flat.unsqueeze(0)
        ).item()
        
        return cos_sim
    
    def _summarize_size_comparison(self, comparison_results: Dict,
                                   model_names: List[str]) -> Dict:
        """Summarize model size comparison results."""
        summary = {
            'models_compared': model_names,
            'num_comparisons': len(comparison_results),
            'avg_cross_model_similarity': 0.0,
            'deception_circuit_shared': True  # If similarity is high
        }
        
        similarities = []
        for comp_data in comparison_results.values():
            similarities.append(comp_data['cross_model_similarity'])
        
        if similarities:
            summary['avg_cross_model_similarity'] = np.mean(similarities)
            summary['deception_circuit_shared'] = np.mean(similarities) > 0.7
        
        return summary


class BaselineComparisonPipeline:
    """
    Complete pipeline for baseline comparisons.
    
    Orchestrates both RLHF and model size comparisons.
    """
    
    def __init__(self, config: ModelComparisonConfig):
        """
        Initialize baseline comparison pipeline.
        
        Args:
            config: Comparison configuration
        """
        self.config = config
        self.rlhf_comparison = None
        self.size_comparison = None
        
    def setup_rlhf_comparison(self, base_model, aligned_model):
        """Set up RLHF comparison."""
        self.rlhf_comparison = RLHFComparison(
            base_model, aligned_model, self.config.device
        )
    
    def setup_size_comparison(self, models: Dict[str, nn.Module]):
        """Set up model size comparison."""
        self.size_comparison = ModelSizeComparison(
            models, self.config.device
        )
    
    def run_complete_comparison(self,
                               prompts: List[str],
                               labels: List[int]) -> Dict:
        """
        Run complete baseline comparison.
        
        Args:
            prompts: Input prompts
            labels: Labels (0=truthful, 1=deceptive)
            
        Returns:
            Complete comparison results
        """
        results = {}
        
        # RLHF comparison
        if self.rlhf_comparison:
            logger.info("Running RLHF comparison...")
            rlhf_results = self.rlhf_comparison.compare_deception_circuits(
                prompts, labels, self.config.layer_indices
            )
            results['rlhf_comparison'] = rlhf_results
        
        # Model size comparison
        if self.size_comparison:
            logger.info("Running model size comparison...")
            size_results = self.size_comparison.compare_deception_circuits(
                prompts, labels, self.config.layer_indices
            )
            results['size_comparison'] = size_results
        
        return results
    
    def save_results(self, results: Dict, output_path: Union[str, Path]):
        """Save comparison results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved comparison results to {output_path}")

