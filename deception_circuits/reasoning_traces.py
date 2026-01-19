"""
Multi-step reasoning trace collection and analysis.

This module implements collection and analysis of chain-of-thought reasoning traces
for deception circuit research. It tracks activations at each step of multi-step
reasoning to understand how deception emerges over the course of reasoning.

Key Features:
- Step-by-step activation tracking during reasoning
- Chain-of-thought trace collection
- Analysis of deception emergence over reasoning steps
- Comparison of truthful vs deceptive reasoning trajectories
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class ReasoningStep:
    """Represents a single step in a reasoning trace."""
    step_index: int
    prompt: str
    response: str
    activations: torch.Tensor  # [num_layers, hidden_dim]
    tokens: List[str]
    attention_weights: Optional[torch.Tensor] = None
    timestamp: float = 0.0


@dataclass
class ReasoningTrace:
    """Complete reasoning trace with all steps."""
    trace_id: str
    scenario: str
    label: int  # 0 = truthful, 1 = deceptive
    steps: List[ReasoningStep]
    final_response: str
    metadata: Dict[str, Any] = None


class ReasoningTraceCollector:
    """
    Collects multi-step reasoning traces from language models.
    
    This class hooks into model forward passes to capture activations
    at each step of chain-of-thought reasoning, allowing analysis of
    how deception emerges over multiple reasoning steps.
    """
    
    def __init__(self, model, tokenizer, device: str = "cpu"):
        """
        Initialize reasoning trace collector.
        
        Args:
            model: Language model to collect traces from
            tokenizer: Tokenizer for the model
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.traces: List[ReasoningTrace] = []
        self.hooks = []
        self.current_trace: Optional[ReasoningTrace] = None
        self.current_steps: List[ReasoningStep] = []
        self.step_counter = 0
        
    def start_trace(self, trace_id: str, scenario: str, label: int, 
                   initial_prompt: str, metadata: Optional[Dict] = None):
        """
        Start collecting a new reasoning trace.
        
        Args:
            trace_id: Unique identifier for this trace
            scenario: Deception scenario type
            label: 0 for truthful, 1 for deceptive
            initial_prompt: Initial prompt/question
            metadata: Optional metadata dictionary
        """
        self.current_trace = ReasoningTrace(
            trace_id=trace_id,
            scenario=scenario,
            label=label,
            steps=[],
            final_response="",
            metadata=metadata or {}
        )
        self.current_steps = []
        self.step_counter = 0
        
    def collect_step(self, prompt: str, response: str, 
                    activations: torch.Tensor,
                    tokens: Optional[List[str]] = None,
                    attention_weights: Optional[torch.Tensor] = None):
        """
        Collect a single reasoning step.
        
        Args:
            prompt: Input prompt for this step
            response: Model's response for this step
            activations: Activations tensor [num_layers, hidden_dim]
            tokens: List of tokens (optional)
            attention_weights: Attention weights (optional)
        """
        if self.current_trace is None:
            raise ValueError("Must call start_trace() before collect_step()")
            
        step = ReasoningStep(
            step_index=self.step_counter,
            prompt=prompt,
            response=response,
            activations=activations,
            tokens=tokens or [],
            attention_weights=attention_weights,
            timestamp=time.time()
        )
        
        self.current_steps.append(step)
        self.current_trace.steps.append(step)
        self.step_counter += 1
        
    def finish_trace(self, final_response: str):
        """
        Finish collecting the current trace.
        
        Args:
            final_response: Final response from the model
        """
        if self.current_trace is None:
            raise ValueError("No trace in progress")
            
        self.current_trace.final_response = final_response
        self.traces.append(self.current_trace)
        self.current_trace = None
        self.current_steps = []
        self.step_counter = 0
        
    def collect_chain_of_thought(self, 
                                 trace_id: str,
                                 scenario: str,
                                 label: int,
                                 prompt: str,
                                 model,
                                 max_steps: int = 10,
                                 metadata: Optional[Dict] = None) -> ReasoningTrace:
        """
        Collect a complete chain-of-thought reasoning trace.
        
        Args:
            trace_id: Unique identifier
            scenario: Deception scenario
            label: 0 for truthful, 1 for deceptive
            prompt: Initial prompt
            model: Model to run inference on
            max_steps: Maximum number of reasoning steps
            metadata: Optional metadata
            
        Returns:
            Complete reasoning trace
        """
        self.start_trace(trace_id, scenario, label, prompt, metadata)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set up hooks to capture activations
        activations_dict = {}
        hooks = []
        
        def create_hook(layer_idx):
            def hook(module, input, output):
                if hasattr(output, 'last_hidden_state'):
                    activations_dict[f"layer_{layer_idx}"] = output.last_hidden_state
                elif isinstance(output, tuple):
                    activations_dict[f"layer_{layer_idx}"] = output[0]
            return hook
        
        # Hook into model layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            layers = model.encoder.layer
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            raise ValueError("Cannot identify model layers")
        
        for layer_idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(create_hook(layer_idx))
            hooks.append(hook)
        
        # Run inference with chain-of-thought
        current_prompt = prompt
        step_count = 0
        
        with torch.no_grad():
            while step_count < max_steps:
                # Forward pass
                activations_dict.clear()
                outputs = model(**inputs)
                
                # Extract activations
                num_layers = len(activations_dict)
                if num_layers == 0:
                    break
                    
                # Stack activations from all layers
                layer_activations = []
                for layer_idx in range(num_layers):
                    if f"layer_{layer_idx}" in activations_dict:
                        # Use [CLS] token or mean pooling
                        layer_act = activations_dict[f"layer_{layer_idx}"]
                        if len(layer_act.shape) == 3:
                            layer_act = layer_act[:, 0, :]  # [CLS] token
                        layer_activations.append(layer_act.squeeze(0))
                
                if layer_activations:
                    stacked_activations = torch.stack(layer_activations)  # [num_layers, hidden_dim]
                else:
                    break
                
                # Get response
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    next_token_id = logits[0, -1, :].argmax().item()
                    next_token = self.tokenizer.decode([next_token_id])
                    response = next_token
                else:
                    response = ""
                
                # Collect step
                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                self.collect_step(
                    prompt=current_prompt,
                    response=response,
                    activations=stacked_activations,
                    tokens=tokens
                )
                
                # Check if reasoning is complete
                if response.strip() in [".", "!", "?", "\n"] or step_count >= max_steps - 1:
                    break
                
                # Update prompt for next step
                current_prompt = current_prompt + response
                inputs = self.tokenizer(current_prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                step_count += 1
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Finish trace
        final_response = current_prompt[len(prompt):].strip()
        self.finish_trace(final_response)
        
        return self.traces[-1]
    
    def analyze_trace_evolution(self, trace: ReasoningTrace) -> Dict:
        """
        Analyze how deception signals evolve over reasoning steps.
        
        Args:
            trace: Reasoning trace to analyze
            
        Returns:
            Dictionary with evolution analysis
        """
        if len(trace.steps) == 0:
            return {}
        
        # Extract activations for each step
        step_activations = [step.activations for step in trace.steps]
        
        # Compute similarity between steps
        similarities = []
        for i in range(len(step_activations) - 1):
            # Flatten activations
            act1 = step_activations[i].flatten()
            act2 = step_activations[i + 1].flatten()
            
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                act1.unsqueeze(0), act2.unsqueeze(0)
            ).item()
            similarities.append(cos_sim)
        
        # Compute activation magnitude over steps
        magnitudes = [torch.norm(step.activations).item() for step in trace.steps]
        
        # Compute layer-wise changes
        layer_changes = []
        if len(step_activations) > 1:
            for layer_idx in range(step_activations[0].shape[0]):
                layer_acts = [act[layer_idx] for act in step_activations]
                layer_diffs = []
                for i in range(len(layer_acts) - 1):
                    diff = torch.norm(layer_acts[i + 1] - layer_acts[i]).item()
                    layer_diffs.append(diff)
                layer_changes.append(layer_diffs)
        
        return {
            'num_steps': len(trace.steps),
            'step_similarities': similarities,
            'activation_magnitudes': magnitudes,
            'layer_changes': layer_changes,
            'avg_similarity': np.mean(similarities) if similarities else 0.0,
            'total_change': magnitudes[-1] - magnitudes[0] if magnitudes else 0.0
        }
    
    def compare_traces(self, truthful_trace: ReasoningTrace, 
                     deceptive_trace: ReasoningTrace) -> Dict:
        """
        Compare truthful and deceptive reasoning traces.
        
        Args:
            truthful_trace: Truthful reasoning trace
            deceptive_trace: Deceptive reasoning trace
            
        Returns:
            Comparison analysis
        """
        truthful_analysis = self.analyze_trace_evolution(truthful_trace)
        deceptive_analysis = self.analyze_trace_evolution(deceptive_trace)
        
        # Compare activation patterns
        if len(truthful_trace.steps) > 0 and len(deceptive_trace.steps) > 0:
            # Compare final activations
            truthful_final = truthful_trace.steps[-1].activations
            deceptive_final = deceptive_trace.steps[-1].activations
            
            # Flatten for comparison
            t_flat = truthful_final.flatten()
            d_flat = deceptive_final.flatten()
            
            # Ensure same size
            min_size = min(len(t_flat), len(d_flat))
            t_flat = t_flat[:min_size]
            d_flat = d_flat[:min_size]
            
            final_similarity = torch.nn.functional.cosine_similarity(
                t_flat.unsqueeze(0), d_flat.unsqueeze(0)
            ).item()
            
            final_diff = torch.norm(t_flat - d_flat).item()
        else:
            final_similarity = 0.0
            final_diff = 0.0
        
        return {
            'truthful_analysis': truthful_analysis,
            'deceptive_analysis': deceptive_analysis,
            'final_activation_similarity': final_similarity,
            'final_activation_difference': final_diff,
            'num_steps_truthful': len(truthful_trace.steps),
            'num_steps_deceptive': len(deceptive_trace.steps)
        }
    
    def save_traces(self, output_path: Union[str, Path]):
        """Save all collected traces to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert traces to serializable format
        traces_data = []
        for trace in self.traces:
            trace_dict = {
                'trace_id': trace.trace_id,
                'scenario': trace.scenario,
                'label': trace.label,
                'final_response': trace.final_response,
                'metadata': trace.metadata or {},
                'steps': []
            }
            
            for step in trace.steps:
                step_dict = {
                    'step_index': step.step_index,
                    'prompt': step.prompt,
                    'response': step.response,
                    'tokens': step.tokens,
                    'timestamp': step.timestamp,
                    'activations_shape': list(step.activations.shape)
                }
                trace_dict['steps'].append(step_dict)
            
            traces_data.append(trace_dict)
        
        # Save as JSON (activations saved separately)
        with open(output_path, 'w') as f:
            json.dump(traces_data, f, indent=2)
        
        # Save activations separately
        activations_dir = output_path.parent / f"{output_path.stem}_activations"
        activations_dir.mkdir(exist_ok=True)
        
        for trace_idx, trace in enumerate(self.traces):
            for step_idx, step in enumerate(trace.steps):
                act_path = activations_dir / f"trace_{trace_idx}_step_{step_idx}.pt"
                torch.save(step.activations, act_path)
        
        print(f"Saved {len(self.traces)} traces to {output_path}")
        print(f"Saved activations to {activations_dir}")
    
    def load_traces(self, traces_path: Union[str, Path], 
                   activations_dir: Optional[Union[str, Path]] = None):
        """Load traces from disk."""
        traces_path = Path(traces_path)
        activations_dir = activations_dir or (traces_path.parent / f"{traces_path.stem}_activations")
        activations_dir = Path(activations_dir)
        
        with open(traces_path, 'r') as f:
            traces_data = json.load(f)
        
        self.traces = []
        for trace_dict in traces_data:
            steps = []
            for step_idx, step_dict in enumerate(trace_dict['steps']):
                # Load activations
                act_path = activations_dir / f"trace_{len(self.traces)}_step_{step_idx}.pt"
                if act_path.exists():
                    activations = torch.load(act_path)
                else:
                    # Create dummy activations if not found
                    activations = torch.randn(*step_dict['activations_shape'])
                
                step = ReasoningStep(
                    step_index=step_dict['step_index'],
                    prompt=step_dict['prompt'],
                    response=step_dict['response'],
                    activations=activations,
                    tokens=step_dict.get('tokens', []),
                    timestamp=step_dict.get('timestamp', 0.0)
                )
                steps.append(step)
            
            trace = ReasoningTrace(
                trace_id=trace_dict['trace_id'],
                scenario=trace_dict['scenario'],
                label=trace_dict['label'],
                steps=steps,
                final_response=trace_dict['final_response'],
                metadata=trace_dict.get('metadata', {})
            )
            self.traces.append(trace)
        
        print(f"Loaded {len(self.traces)} traces from {traces_path}")
    
    def convert_traces_to_activations(self, 
                                     traces: Optional[List[ReasoningTrace]] = None,
                                     use_final_step: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert reasoning traces to activation tensors for training.
        
        Args:
            traces: List of traces to convert (None to use all collected traces)
            use_final_step: If True, use final step activations; if False, average all steps
            
        Returns:
            Tuple of (activations, labels) where activations is [num_samples, num_layers, hidden_dim]
        """
        if traces is None:
            traces = self.traces
        
        if len(traces) == 0:
            raise ValueError("No traces available to convert")
        
        activations_list = []
        labels_list = []
        
        for trace in traces:
            if len(trace.steps) == 0:
                continue
            
            if use_final_step:
                # Use final step activations
                step_activations = trace.steps[-1].activations  # [num_layers, hidden_dim]
            else:
                # Average across all steps
                step_activations_list = [step.activations for step in trace.steps]
                step_activations = torch.stack(step_activations_list).mean(dim=0)  # [num_layers, hidden_dim]
            
            activations_list.append(step_activations)
            labels_list.append(trace.label)
        
        if len(activations_list) == 0:
            raise ValueError("No valid traces with steps found")
        
        # Stack activations: [num_samples, num_layers, hidden_dim]
        activations = torch.stack(activations_list)
        labels = torch.tensor(labels_list, dtype=torch.float32)
        
        return activations, labels
    
    def convert_traces_to_dataframe(self, 
                                   traces: Optional[List[ReasoningTrace]] = None) -> 'pd.DataFrame':
        """
        Convert reasoning traces to DataFrame format compatible with DeceptionDataLoader.
        
        Args:
            traces: List of traces to convert (None to use all collected traces)
            
        Returns:
            DataFrame with columns: statement, response, label, scenario
        """
        import pandas as pd
        
        if traces is None:
            traces = self.traces
        
        rows = []
        for trace in traces:
            # Use first step prompt as statement, final response as response
            statement = trace.steps[0].prompt if trace.steps else ""
            response = trace.final_response
            
            rows.append({
                'statement': statement,
                'response': response,
                'label': trace.label,
                'scenario': trace.scenario,
                'trace_id': trace.trace_id,
                'num_steps': len(trace.steps)
            })
        
        return pd.DataFrame(rows)

