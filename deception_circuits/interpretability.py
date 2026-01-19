"""
Advanced interpretability features for deception circuit research.

This module provides comprehensive interpretability tools for understanding
deception circuits in language models, including:

1. Attention Visualization: Visualize attention patterns in deceptive vs truthful responses
2. Feature Attribution: Understand which input features contribute to deception
3. Neuron Analysis: Analyze individual neurons and their role in deception
4. Circuit Visualization: Visualize the flow of information through deception circuits
5. Interactive Exploration: Tools for interactive exploration of model behavior

These tools help researchers understand not just that deception circuits exist,
but how they work and what they represent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
import networkx as nx
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for interpretability visualizations."""
    figure_size: Tuple[int, int] = (12, 8)
    color_scheme: str = 'viridis'
    save_format: str = 'png'
    dpi: int = 300
    interactive: bool = True


class AttentionVisualizer:
    """
    Visualize attention patterns in deception circuits.
    
    This class provides tools to visualize and analyze attention patterns
    in transformer models, particularly focusing on differences between
    truthful and deceptive responses.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize attention visualizer."""
        self.config = config or VisualizationConfig()
        
    def visualize_attention_patterns(self,
                                  attention_weights: Dict[str, torch.Tensor],
                                  tokens: List[str],
                                  layer_indices: List[int],
                                  save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Visualize attention patterns across layers.
        
        Args:
            attention_weights: Dictionary mapping layer names to attention weights
            tokens: List of input tokens
            layer_indices: Layers to visualize
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        num_layers = len(layer_indices)
        fig, axes = plt.subplots(2, (num_layers + 1) // 2, 
                               figsize=(self.config.figure_size[0], self.config.figure_size[1] * 2))
        axes = axes.flatten() if num_layers > 1 else [axes]
        
        for i, layer_idx in enumerate(layer_indices):
            layer_key = f"layer_{layer_idx}"
            if layer_key in attention_weights:
                attention = attention_weights[layer_key].cpu().numpy()
                
                # Average across heads and samples
                if attention.ndim == 4:  # [batch, heads, seq, seq]
                    attention = attention.mean(axis=(0, 1))
                elif attention.ndim == 3:  # [heads, seq, seq]
                    attention = attention.mean(axis=0)
                
                # Create heatmap
                im = axes[i].imshow(attention, cmap=self.config.color_scheme, aspect='auto')
                axes[i].set_title(f'Layer {layer_idx}')
                axes[i].set_xlabel('Key Position')
                axes[i].set_ylabel('Query Position')
                
                # Set token labels
                if len(tokens) <= 20:  # Only show labels if not too many tokens
                    axes[i].set_xticks(range(len(tokens)))
                    axes[i].set_yticks(range(len(tokens)))
                    axes[i].set_xticklabels(tokens, rotation=45, ha='right')
                    axes[i].set_yticklabels(tokens)
                
                plt.colorbar(im, ax=axes[i])
        
        # Hide unused subplots
        for i in range(num_layers, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def compare_attention_patterns(self,
                                 truthful_attention: Dict[str, torch.Tensor],
                                 deceptive_attention: Dict[str, torch.Tensor],
                                 tokens: List[str],
                                 layer_idx: int,
                                 save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Compare attention patterns between truthful and deceptive responses.
        
        Args:
            truthful_attention: Attention weights for truthful responses
            deceptive_attention: Attention weights for deceptive responses
            tokens: List of input tokens
            layer_idx: Layer to compare
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        layer_key = f"layer_{layer_idx}"
        
        # Truthful attention
        if layer_key in truthful_attention:
            truth_attn = truthful_attention[layer_key].cpu().numpy()
            if truth_attn.ndim > 2:
                truth_attn = truth_attn.mean(axis=tuple(range(truth_attn.ndim - 2)))
            
            im1 = axes[0].imshow(truth_attn, cmap='Blues', aspect='auto')
            axes[0].set_title('Truthful Attention')
            axes[0].set_xlabel('Key Position')
            axes[0].set_ylabel('Query Position')
            plt.colorbar(im1, ax=axes[0])
        
        # Deceptive attention
        if layer_key in deceptive_attention:
            deceit_attn = deceptive_attention[layer_key].cpu().numpy()
            if deceit_attn.ndim > 2:
                deceit_attn = deceit_attn.mean(axis=tuple(range(deceit_attn.ndim - 2)))
            
            im2 = axes[1].imshow(deceit_attn, cmap='Reds', aspect='auto')
            axes[1].set_title('Deceptive Attention')
            axes[1].set_xlabel('Key Position')
            axes[1].set_ylabel('Query Position')
            plt.colorbar(im2, ax=axes[1])
        
        # Difference
        if layer_key in truthful_attention and layer_key in deceptive_attention:
            diff = deceit_attn - truth_attn
            im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto')
            axes[2].set_title('Difference (Deceptive - Truthful)')
            axes[2].set_xlabel('Key Position')
            axes[2].set_ylabel('Query Position')
            plt.colorbar(im3, ax=axes[2])
        
        # Set token labels
        if len(tokens) <= 15:
            for ax in axes:
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha='right')
                ax.set_yticklabels(tokens)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def create_interactive_attention_plot(self,
                                        attention_weights: Dict[str, torch.Tensor],
                                        tokens: List[str],
                                        layer_idx: int,
                                        save_path: Optional[Union[str, Path]] = None) -> go.Figure:
        """
        Create interactive attention visualization.
        
        Args:
            attention_weights: Attention weights
            tokens: Input tokens
            layer_idx: Layer to visualize
            save_path: Optional path to save HTML file
            
        Returns:
            Plotly figure
        """
        layer_key = f"layer_{layer_idx}"
        if layer_key not in attention_weights:
            raise ValueError(f"Layer {layer_idx} not found in attention weights")
        
        attention = attention_weights[layer_key].cpu().numpy()
        if attention.ndim > 2:
            attention = attention.mean(axis=tuple(range(attention.ndim - 2)))
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attention,
            x=tokens,
            y=tokens,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='Query: %{y}<br>Key: %{x}<br>Attention: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Attention Patterns - Layer {layer_idx}',
            xaxis_title='Key Position',
            yaxis_title='Query Position',
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(str(save_path))
        
        return fig


class FeatureAttributor:
    """
    Analyze feature attribution in deception circuits.
    
    This class provides tools to understand which input features
    contribute most to deception predictions.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize feature attributor."""
        self.config = config or VisualizationConfig()
        
    def compute_integrated_gradients(self,
                                   model: nn.Module,
                                   inputs: torch.Tensor,
                                   target_class: int = 1,
                                   steps: int = 50) -> torch.Tensor:
        """
        Compute integrated gradients for feature attribution.
        
        Args:
            model: Model to analyze
            inputs: Input tensors
            target_class: Target class for attribution
            steps: Number of integration steps
            
        Returns:
            Integrated gradients
        """
        model.eval()
        inputs.requires_grad_(True)
        
        # Baseline (zeros)
        baseline = torch.zeros_like(inputs)
        
        # Compute gradients along integration path
        integrated_gradients = torch.zeros_like(inputs)
        
        for i in range(steps):
            # Interpolate between baseline and inputs
            alpha = i / steps
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            outputs = model(interpolated)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Compute loss for target class
            if logits.dim() > 1:
                target_logit = logits[:, target_class]
            else:
                target_logit = logits.squeeze()
            
            # Compute gradients
            gradients = torch.autograd.grad(
                target_logit.sum(), interpolated, retain_graph=True
            )[0]
            
            integrated_gradients += gradients / steps
        
        # Multiply by input difference
        integrated_gradients *= (inputs - baseline)
        
        return integrated_gradients
    
    def compute_shap_values(self,
                          model: nn.Module,
                          inputs: torch.Tensor,
                          baseline: Optional[torch.Tensor] = None,
                          num_samples: int = 100) -> torch.Tensor:
        """
        Compute SHAP values for feature attribution.
        
        Args:
            model: Model to analyze
            inputs: Input tensors
            baseline: Baseline values (if None, use zeros)
            num_samples: Number of samples for SHAP estimation
            
        Returns:
            SHAP values
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        
        model.eval()
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = model(baseline)
            if hasattr(baseline_output, 'logits'):
                baseline_pred = torch.sigmoid(baseline_output.logits)
            else:
                baseline_pred = torch.sigmoid(baseline_output)
        
        # Compute SHAP values
        shap_values = torch.zeros_like(inputs)
        
        for i in range(inputs.shape[1]):  # For each feature
            feature_contributions = []
            
            for _ in range(num_samples):
                # Create random mask
                mask = torch.rand_like(inputs) < 0.5
                
                # Create input with current feature included
                input_with_feature = torch.where(
                    mask, inputs, baseline
                )
                input_with_feature[:, i] = inputs[:, i]  # Always include current feature
                
                # Create input without current feature
                input_without_feature = torch.where(
                    mask, inputs, baseline
                )
                input_without_feature[:, i] = baseline[:, i]  # Exclude current feature
                
                # Get predictions
                with torch.no_grad():
                    pred_with = model(input_with_feature)
                    pred_without = model(input_without_feature)
                    
                    if hasattr(pred_with, 'logits'):
                        pred_with = torch.sigmoid(pred_with.logits)
                        pred_without = torch.sigmoid(pred_without.logits)
                    else:
                        pred_with = torch.sigmoid(pred_with)
                        pred_without = torch.sigmoid(pred_without)
                
                # Compute contribution
                contribution = pred_with - pred_without
                feature_contributions.append(contribution)
            
            # Average contribution
            shap_values[:, i] = torch.stack(feature_contributions).mean(dim=0)
        
        return shap_values
    
    def visualize_feature_attribution(self,
                                    attribution_scores: torch.Tensor,
                                    feature_names: List[str],
                                    title: str = "Feature Attribution",
                                    save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Visualize feature attribution scores.
        
        Args:
            attribution_scores: Attribution scores for each feature
            feature_names: Names of features
            title: Title for the plot
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Average across samples
        avg_scores = attribution_scores.mean(dim=0).cpu().numpy()
        
        # Sort by importance
        sorted_indices = np.argsort(np.abs(avg_scores))[-20:]  # Top 20 features
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(sorted_indices))
        colors = ['red' if score < 0 else 'blue' for score in avg_scores[sorted_indices]]
        
        bars = ax.barh(y_pos, avg_scores[sorted_indices], color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in sorted_indices])
        ax.set_xlabel('Attribution Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, avg_scores[sorted_indices])):
            ax.text(score + (0.01 if score >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', va='center', ha='left' if score >= 0 else 'right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig


class NeuronAnalyzer:
    """
    Analyze individual neurons and their role in deception.
    
    This class provides tools to understand which specific neurons
    are most important for deception detection.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize neuron analyzer."""
        self.config = config or VisualizationConfig()
        
    def analyze_neuron_importance(self,
                                activations: torch.Tensor,
                                labels: torch.Tensor,
                                method: str = 'correlation') -> Dict:
        """
        Analyze the importance of individual neurons for deception detection.
        
        Args:
            activations: Neuron activations [batch_size, num_layers, num_neurons]
            labels: Binary labels (0=truthful, 1=deceptive)
            method: Method for importance calculation ('correlation', 'mutual_info', 'gradient')
            
        Returns:
            Dictionary with neuron importance analysis
        """
        num_layers = activations.shape[1]
        num_neurons = activations.shape[2]
        
        importance_scores = {}
        
        for layer_idx in range(num_layers):
            layer_activations = activations[:, layer_idx, :].cpu().numpy()
            layer_labels = labels.cpu().numpy()
            
            if method == 'correlation':
                # Compute correlation between neuron activations and labels
                correlations = np.array([
                    np.corrcoef(layer_activations[:, i], layer_labels)[0, 1]
                    for i in range(num_neurons)
                ])
                importance = np.abs(correlations)
                
            elif method == 'mutual_info':
                from sklearn.feature_selection import mutual_info_classif
                importance = mutual_info_classif(layer_activations, layer_labels)
                
            elif method == 'gradient':
                # Compute gradient-based importance (placeholder)
                importance = np.random.rand(num_neurons)  # Placeholder
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            importance_scores[f'layer_{layer_idx}'] = {
                'importance': importance,
                'top_neurons': np.argsort(importance)[-50:],  # Top 50 neurons
                'mean_importance': np.mean(importance),
                'std_importance': np.std(importance)
            }
        
        return importance_scores
    
    def visualize_neuron_importance(self,
                                  importance_scores: Dict,
                                  layer_indices: List[int],
                                  save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Visualize neuron importance across layers.
        
        Args:
            importance_scores: Neuron importance scores
            layer_indices: Layers to visualize
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        num_layers = len(layer_indices)
        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 6))
        if num_layers == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(layer_indices):
            layer_key = f'layer_{layer_idx}'
            if layer_key in importance_scores:
                importance = importance_scores[layer_key]['importance']
                
                # Create histogram
                axes[i].hist(importance, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Neuron Importance - Layer {layer_idx}')
                axes[i].set_xlabel('Importance Score')
                axes[i].set_ylabel('Number of Neurons')
                axes[i].grid(True, alpha=0.3)
                
                # Add statistics
                mean_imp = importance_scores[layer_key]['mean_importance']
                std_imp = importance_scores[layer_key]['std_importance']
                axes[i].axvline(mean_imp, color='red', linestyle='--', 
                               label=f'Mean: {mean_imp:.3f}')
                axes[i].axvline(mean_imp + 2*std_imp, color='orange', linestyle='--',
                               label=f'+2σ: {mean_imp + 2*std_imp:.3f}')
                axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def find_deception_neurons(self,
                             activations: torch.Tensor,
                             labels: torch.Tensor,
                             threshold: float = 0.7) -> Dict:
        """
        Find neurons that are specifically active during deception.
        
        Args:
            activations: Neuron activations
            labels: Binary labels
            threshold: Threshold for considering a neuron as 'deception-specific'
            
        Returns:
            Dictionary with deception-specific neurons
        """
        num_layers = activations.shape[1]
        deception_neurons = {}
        
        for layer_idx in range(num_layers):
            layer_activations = activations[:, layer_idx, :].cpu().numpy()
            
            # Separate truthful and deceptive activations
            truthful_mask = labels.cpu().numpy() == 0
            deceptive_mask = labels.cpu().numpy() == 1
            
            truthful_activations = layer_activations[truthful_mask]
            deceptive_activations = layer_activations[deceptive_mask]
            
            # Find neurons that are significantly more active during deception
            truthful_means = np.mean(truthful_activations, axis=0)
            deceptive_means = np.mean(deceptive_activations, axis=0)
            
            # Compute activation ratio
            activation_ratio = deceptive_means / (truthful_means + 1e-8)
            
            # Find neurons with high activation ratio
            deception_specific = np.where(activation_ratio > threshold)[0]
            
            deception_neurons[f'layer_{layer_idx}'] = {
                'deception_neurons': deception_specific.tolist(),
                'activation_ratios': activation_ratio.tolist(),
                'num_deception_neurons': len(deception_specific),
                'total_neurons': len(activation_ratio)
            }
        
        return deception_neurons


class CircuitVisualizer:
    """
    Visualize the flow of information through deception circuits.
    
    This class provides tools to create network visualizations
    showing how information flows through the model during deception.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize circuit visualizer."""
        self.config = config or VisualizationConfig()
        
    def create_circuit_diagram(self,
                             layer_connections: Dict[str, List[str]],
                             node_importance: Dict[str, float],
                             save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create a network diagram of the deception circuit.
        
        Args:
            layer_connections: Dictionary mapping layers to connected layers
            node_importance: Dictionary mapping nodes to importance scores
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes
        for node, importance in node_importance.items():
            G.add_node(node, importance=importance)
        
        # Add edges
        for source, targets in layer_connections.items():
            for target in targets:
                G.add_edge(source, target)
        
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Draw nodes
        node_sizes = [node_importance[node] * 1000 for node in G.nodes()]
        node_colors = [node_importance[node] for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                              cmap='viridis', alpha=0.7, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True, arrowsize=20, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        ax.set_title("Deception Circuit Flow")
        ax.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                  norm=plt.Normalize(vmin=min(node_importance.values()),
                                                   vmax=max(node_importance.values())))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Node Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def visualize_activation_flow(self,
                                activations: torch.Tensor,
                                layer_names: List[str],
                                method: str = 'pca',
                                save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Visualize how activations flow through layers.
        
        Args:
            activations: Layer activations [batch_size, num_layers, hidden_dim]
            layer_names: Names of layers
            method: Dimensionality reduction method ('pca', 'tsne', 'mds')
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Reduce dimensionality for each layer
        reduced_activations = {}
        
        for layer_idx, layer_name in enumerate(layer_names):
            layer_acts = activations[:, layer_idx, :].cpu().numpy()
            
            if method == 'pca':
                reducer = PCA(n_components=2)
            elif method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42)
            elif method == 'mds':
                reducer = MDS(n_components=2, random_state=42)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            reduced_acts = reducer.fit_transform(layer_acts)
            reduced_activations[layer_name] = reduced_acts
        
        # Create subplots
        num_layers = len(layer_names)
        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
        if num_layers == 1:
            axes = [axes]
        
        for i, layer_name in enumerate(layer_names):
            reduced_acts = reduced_activations[layer_name]
            
            # Create scatter plot
            scatter = axes[i].scatter(reduced_acts[:, 0], reduced_acts[:, 1], 
                                    c=range(len(reduced_acts)), cmap='viridis', alpha=0.6)
            axes[i].set_title(f'{layer_name} - {method.upper()}')
            axes[i].set_xlabel(f'{method.upper()} Component 1')
            axes[i].set_ylabel(f'{method.upper()} Component 2')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig


class InterpretabilitySuite:
    """
    Comprehensive interpretability suite for deception circuits.
    
    This class provides a unified interface for all interpretability
    tools and generates comprehensive analysis reports.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize interpretability suite."""
        self.config = config or VisualizationConfig()
        self.attention_visualizer = AttentionVisualizer(config)
        self.feature_attributor = FeatureAttributor(config)
        self.neuron_analyzer = NeuronAnalyzer(config)
        self.circuit_visualizer = CircuitVisualizer(config)
        
    def generate_comprehensive_report(self,
                                    model: Optional[nn.Module],
                                    activations: torch.Tensor,
                                    labels: torch.Tensor,
                                    output_dir: Union[str, Path],
                                    tokens: Optional[List[str]] = None,
                                    attention_weights: Optional[Dict] = None,
                                    probe: Optional[nn.Module] = None) -> Dict:
        """
        Generate comprehensive interpretability report.
        
        Args:
            model: Optional model for gradient-based analysis
            activations: Model activations
            labels: Binary labels
            tokens: Optional input tokens
            attention_weights: Optional attention weights
            output_dir: Directory to save report
            probe: Optional probe for evaluation
            
        Returns:
            Dictionary with report summary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating comprehensive interpretability report...")
        
        report = {
            'neuron_analysis': {},
            'feature_attribution': {},
            'attention_analysis': {},
            'circuit_analysis': {},
            'files_created': []
        }
        
        # 1. Neuron Analysis
        logger.info("Analyzing neuron importance...")
        neuron_importance = self.neuron_analyzer.analyze_neuron_importance(
            activations, labels, method='correlation'
        )
        report['neuron_analysis'] = neuron_importance
        
        # Visualize neuron importance
        layer_indices = list(range(min(6, activations.shape[1])))  # Top 6 layers
        neuron_fig = self.neuron_analyzer.visualize_neuron_importance(
            neuron_importance, layer_indices,
            save_path=output_dir / 'neuron_importance.png'
        )
        report['files_created'].append('neuron_importance.png')
        
        # Find deception-specific neurons
        deception_neurons = self.neuron_analyzer.find_deception_neurons(
            activations, labels, threshold=1.5
        )
        report['neuron_analysis']['deception_neurons'] = deception_neurons
        
        # 2. Feature Attribution (if model provided)
        if model is not None:
            logger.info("Computing feature attribution...")
            # Use a subset of activations for efficiency
            subset_size = min(50, activations.shape[0])
            subset_activations = activations[:subset_size]
            subset_labels = labels[:subset_size]
            
            # Compute integrated gradients
            integrated_grads = self.feature_attributor.compute_integrated_gradients(
                model, subset_activations, target_class=1
            )
            
            # Create feature names
            feature_names = [f'Neuron_{i}' for i in range(activations.shape[2])]
            
            # Visualize attribution
            attribution_fig = self.feature_attributor.visualize_feature_attribution(
                integrated_grads, feature_names,
                title="Feature Attribution for Deception",
                save_path=output_dir / 'feature_attribution.png'
            )
            report['feature_attribution'] = {
                'integrated_gradients': integrated_grads.cpu().numpy().tolist(),
                'top_features': np.argsort(integrated_grads.abs().mean(dim=0))[-20:].tolist()
            }
            report['files_created'].append('feature_attribution.png')
        
        # 3. Attention Analysis (if attention weights provided)
        if attention_weights is not None and tokens is not None:
            logger.info("Analyzing attention patterns...")
            
            # Visualize attention patterns
            attention_fig = self.attention_visualizer.visualize_attention_patterns(
                attention_weights, tokens, layer_indices,
                save_path=output_dir / 'attention_patterns.png'
            )
            report['attention_analysis'] = {
                'layers_analyzed': layer_indices,
                'num_tokens': len(tokens)
            }
            report['files_created'].append('attention_patterns.png')
            
            # Create interactive attention plot
            if layer_indices:
                interactive_fig = self.attention_visualizer.create_interactive_attention_plot(
                    attention_weights, tokens, layer_indices[0],
                    save_path=output_dir / 'interactive_attention.html'
                )
                report['files_created'].append('interactive_attention.html')
        
        # 4. Circuit Analysis
        logger.info("Analyzing circuit flow...")
        
        # Create layer names
        layer_names = [f'Layer_{i}' for i in range(activations.shape[1])]
        
        # Visualize activation flow
        circuit_fig = self.circuit_visualizer.visualize_activation_flow(
            activations, layer_names, method='pca',
            save_path=output_dir / 'activation_flow.png'
        )
        report['circuit_analysis'] = {
            'num_layers': activations.shape[1],
            'hidden_dim': activations.shape[2],
            'method': 'pca'
        }
        report['files_created'].append('activation_flow.png')
        
        # 5. Save report summary
        report_path = output_dir / 'interpretability_report.json'
        with open(report_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_report = self._make_serializable(report)
            json.dump(serializable_report, f, indent=2)
        
        report['files_created'].append('interpretability_report.json')
        
        logger.info(f"Interpretability report generated: {output_dir}")
        logger.info(f"Files created: {report['files_created']}")
        
        return report
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert non-serializable objects to JSON-serializable format."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


# Example usage function
def run_comprehensive_interpretability_analysis(model: nn.Module,
                                              activations: torch.Tensor,
                                              labels: torch.Tensor,
                                              output_dir: str = "interpretability_report",
                                              tokens: Optional[List[str]] = None,
                                              attention_weights: Optional[Dict] = None) -> Dict:
    """
    Run comprehensive interpretability analysis.
    
    Args:
        model: Model to analyze
        activations: Model activations
        labels: Binary labels
        output_dir: Directory to save report
        tokens: Optional input tokens
        attention_weights: Optional attention weights
        
    Returns:
        Interpretability analysis results
    """
    suite = InterpretabilitySuite()
    
    return suite.generate_comprehensive_report(
        model=model,
        activations=activations,
        labels=labels,
        tokens=tokens,
        attention_weights=attention_weights,
        output_dir=output_dir
    )
