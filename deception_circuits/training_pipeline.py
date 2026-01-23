"""
Training pipeline for deception circuit research.

This module contains the main orchestration class for running complete deception
circuit experiments. It coordinates all components of the research pipeline:

1. Data loading and preprocessing
2. Linear probe training across all layers
3. Sparse autoencoder training (both unsupervised and supervised)
4. Cross-scenario generalization testing
5. Result analysis and saving

The DeceptionTrainingPipeline class provides a high-level interface that:
- Loads CSV data with paired truthful/deceptive examples
- Trains probes and autoencoders across all model layers
- Tests generalization across different deception scenarios
- Saves comprehensive results and model checkpoints
- Provides analysis of which layers are most informative

This implements the core methodology from the BLKY project for discovering
and analyzing deception circuits in LLM reasoning traces.

=== FOR FIRST-TIME READERS ===

This is the "MAIN CONTROL CENTER" of your deception circuit research! The DeceptionTrainingPipeline
is like your research assistant that runs the entire experiment for you.

=== WHAT THIS PIPELINE DOES ===

Think of it as an automated research workflow that:

1. **LOADS YOUR DATA**: Reads CSV files with truthful vs deceptive examples
2. **TRAINS DETECTORS**: Creates "deception detectors" (linear probes) for each layer
3. **FINDS FEATURES**: Discovers important features using sparse autoencoders
4. **TESTS GENERALIZATION**: Checks if circuits work across different scenarios
5. **SAVES EVERYTHING**: Stores all results, models, and plots

=== THE COMPLETE RESEARCH WORKFLOW ===

```
Input: CSV file with deceptive vs truthful examples
  ↓
Step 1: Load and validate data
  ↓
Step 2: Train linear probes on each layer
  ↓
Step 3: Train sparse autoencoders on each layer
  ↓
Step 4: Test cross-scenario generalization
  ↓
Step 5: Analyze results and identify circuits
  ↓
Output: Complete analysis of deception circuits
```

=== HOW TO USE (SIMPLE) ===

```python
# Create the pipeline:
pipeline = DeceptionTrainingPipeline(device="cpu", output_dir="my_results")

# Run the complete experiment:
results = pipeline.run_full_experiment("my_data.csv")

# That's it! Everything is automated.
print(f"Best layer: {results['analysis_results']['best_probe_layer']}")
```

=== WHAT YOU GET ===

After running the pipeline, you'll have:
- **Best performing layers** for deception detection
- **Feature importance** showing which neurons matter
- **Generalization results** across different scenarios
- **Trained models** ready for causal testing
- **Visualizations** showing your results
- **Complete analysis** of deception circuits

This is the heart of the BLKY research methodology!
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import time
from collections import defaultdict

from .data_loader import DeceptionDataLoader
from .linear_probe import LinearProbeTrainer, DeceptionLinearProbe
from .sparse_autoencoder import AutoencoderTrainer, DeceptionSparseAutoencoder, SupervisedDeceptionAutoencoder


class DeceptionTrainingPipeline:
    """
    Main training pipeline for deception circuit research.
    
    This is the primary interface for running complete deception circuit experiments.
    It orchestrates all components of the research pipeline:
    
    1. **Data Loading**: Loads CSV files with paired truthful/deceptive examples
    2. **Probe Training**: Trains linear probes across all model layers
    3. **Autoencoder Training**: Trains sparse autoencoders for feature discovery
    4. **Cross-Scenario Analysis**: Tests generalization across deception contexts
    5. **Result Analysis**: Identifies most informative layers and circuits
    6. **Saving**: Saves models, results, and visualizations
    
    The pipeline implements the core methodology from the BLKY project:
    - Multi-layer analysis to find where deception signals emerge
    - Cross-context testing to see if circuits generalize
    - Comprehensive evaluation metrics and visualizations
    
    Attributes:
        device (str): Device for training ("cpu" or "cuda")
        output_dir (Path): Directory to save all results
        data_loader (DeceptionDataLoader): Handles CSV data loading
        probe_trainer (LinearProbeTrainer): Trains linear probes
        autoencoder_trainer (AutoencoderTrainer): Trains autoencoders
        results (Dict): Stores experiment results
    """
    
    def __init__(self, device: str = "cpu", 
                 output_dir: Union[str, Path] = "deception_results"):
        """
        Initialize training pipeline.
        
        Args:
            device: Device for training ("cpu" or "cuda")
            output_dir: Directory to save all results (models, plots, data)
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all component classes
        self.data_loader = DeceptionDataLoader(device=device)
        self.probe_trainer = LinearProbeTrainer(device=device, lr=0.01)  # Default learning rate
        self.autoencoder_trainer = AutoencoderTrainer(device=device, lr=0.001)  # Default learning rate
        
        # Store experiment results
        self.results = {}
        
    def run_full_experiment(self, csv_path: Union[str, Path],
                          activation_dir: Optional[Union[str, Path]] = None,
                          scenarios: Optional[List[str]] = None,
                          max_samples: Optional[int] = None,
                          probe_config: Optional[Dict] = None,
                          autoencoder_config: Optional[Dict] = None,
                          save_results: bool = True) -> Dict:
        """
        Run complete deception circuit experiment.
        
        Args:
            csv_path: Path to CSV data file
            activation_dir: Directory containing activation files
            scenarios: List of scenarios to analyze (None for all)
            max_samples: Maximum number of samples to use
            probe_config: Configuration for linear probes
            autoencoder_config: Configuration for autoencoders
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with complete experiment results
        """
        print("Starting Deception Circuit Research Experiment")
        print("=" * 50)
        
        # Default configurations
        if probe_config is None:
            probe_config = {
                'epochs': 100,
                'validation_split': 0.2,
                'early_stopping_patience': 10
            }
            
        if autoencoder_config is None:
            autoencoder_config = {
                'epochs': 100,
                'lr': 0.001,
                'l1_coeff': 0.01,
                'bottleneck_dim': 0,  # Same as input
                'tied_weights': True,
                'activation_type': 'ReLU',
                'topk_percent': 10,
                'lambda_classify': 1.0,
                'validation_split': 0.2,
                'early_stopping_patience': 10
            }
        
        # Step 1: Load data
        print("\n1. Loading Data...")
        data_results = self.data_loader.load_csv(
            csv_path=csv_path,
            activation_dir=activation_dir,
            max_samples=max_samples
        )
        
        df = data_results['dataframe']
        print(f"   Loaded {len(df)} samples")
        print(f"   Truthful: {data_results['num_truthful']}, Deceptive: {data_results['num_deceptive']}")
        
        # Filter by scenarios if specified
        if scenarios is not None:
            df = self.data_loader.filter_by_scenario(df, scenarios)
            print(f"   Filtered to scenarios {scenarios}: {len(df)} samples")
        
        # Step 2: Split data
        print("\n2. Splitting Data...")
        train_data, test_data = self.data_loader.split_data(
            test_size=0.2, stratify=True, random_state=42
        )
        print(f"   Train: {train_data['num_samples']} samples")
        print(f"   Test: {test_data['num_samples']} samples")
        
        # Step 3: Extract activations
        print("\n3. Extracting Activations...")
        train_activations = self.data_loader.get_activations_tensor(train_data['dataframe'])
        train_labels = self.data_loader.get_labels_tensor(train_data['dataframe'])
        test_activations = self.data_loader.get_activations_tensor(test_data['dataframe'])
        test_labels = self.data_loader.get_labels_tensor(test_data['dataframe'])
        
        print(f"   Activations shape: {train_activations.shape}")
        print(f"   Number of layers: {train_activations.shape[1]}")
        
        # Step 4: Train linear probes
        print("\n4. Training Linear Probes...")
        probe_results = self._train_linear_probes(
            train_activations, train_labels,
            test_activations, test_labels,
            probe_config
        )
        
        # Step 5: Train sparse autoencoders
        print("\n5. Training Sparse Autoencoders...")
        autoencoder_results = self._train_sparse_autoencoders(
            train_activations, train_labels,
            test_activations, test_labels,
            autoencoder_config
        )
        
        # Step 6: Analyze results
        print("\n6. Analyzing Results...")
        analysis_results = self._analyze_results(
            probe_results, autoencoder_results,
            train_activations, train_labels,
            test_activations, test_labels
        )
        
        # Step 7: Cross-scenario analysis (if multiple scenarios)
        cross_scenario_results = {}
        if 'scenario' in df.columns and len(df['scenario'].unique()) > 1:
            print("\n7. Cross-Scenario Analysis...")
            cross_scenario_results = self._cross_scenario_analysis(
                df, train_activations, train_labels, test_activations, test_labels,
                probe_config, autoencoder_config
            )
        
        # Compile final results
        # Prepare data_info without DataFrame (to avoid serialization issues)
        data_info_serializable = {k: v for k, v in data_results.items() if k != 'dataframe'}
        data_info_serializable['dataframe_info'] = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'num_samples': len(df)
        }
        
        final_results = {
            'data_info': data_info_serializable,
            'train_test_split': {
                'train_samples': train_data['num_samples'],
                'test_samples': test_data['num_samples']
            },
            'probe_results': probe_results,
            'autoencoder_results': autoencoder_results,
            'analysis_results': analysis_results,
            'cross_scenario_results': cross_scenario_results,
            'configurations': {
                'probe_config': probe_config,
                'autoencoder_config': autoencoder_config
            },
            'experiment_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': self.device,
                'total_samples': len(df)
            }
        }
        
        # Save results
        if save_results:
            self._save_results(final_results)
            
        # Re-inject dataframe into results for in-memory use
        # We do this after saving to avoid serializing the potentially large dataframe into the JSON
        if 'data_info' in final_results:
             final_results['data_info']['dataframe'] = df
            
        self.results = final_results
        
        print("\n" + "=" * 50)
        print("Experiment Complete!")
        print(f"Results saved to: {self.output_dir}")
        
        return final_results
        
    def _train_linear_probes(self, train_activations: torch.Tensor,
                           train_labels: torch.Tensor,
                           test_activations: torch.Tensor,
                           test_labels: torch.Tensor,
                           config: Dict) -> Dict:
        """Train linear probes across all layers."""
        num_layers = train_activations.shape[1]
        layer_names = [f"layer_{i}" for i in range(num_layers)]
        
        # Update probe trainer learning rate if specified in config
        if 'lr' in config:
            self.probe_trainer.lr = config['lr']
            # Remove lr from config to avoid passing it to train_multi_layer_probes
            config = {k: v for k, v in config.items() if k != 'lr'}
        
        # Train probes for each layer
        probe_results = self.probe_trainer.train_multi_layer_probes(
            train_activations, train_labels,
            layer_names=layer_names,
            **config
        )
        
        # Evaluate on test set
        test_results = {}
        for layer_name, layer_results in probe_results.items():
            probe = layer_results['results']['probe']
            test_metrics = self.probe_trainer.evaluate_probe(
                probe, test_activations[:, layer_results['layer_idx'], :], test_labels
            )
            test_results[layer_name] = test_metrics
            
        # Find best performing layer
        best_layer = max(test_results.items(), key=lambda x: x[1]['auc'])
        
        return {
            'layer_results': probe_results,
            'test_results': test_results,
            'best_layer': {
                'layer_name': best_layer[0],
                'metrics': best_layer[1]
            },
            'summary': {
                'num_layers': num_layers,
                'best_auc': best_layer[1]['auc'],
                'best_accuracy': best_layer[1]['accuracy']
            }
        }
        
    def _train_sparse_autoencoders(self, train_activations: torch.Tensor,
                                 train_labels: torch.Tensor,
                                 test_activations: torch.Tensor,
                                 test_labels: torch.Tensor,
                                 config: Dict) -> Dict:
        """Train sparse autoencoders across all layers."""
        num_layers = train_activations.shape[1]
        
        # Update autoencoder trainer learning rate if specified in config
        if 'lr' in config:
            self.autoencoder_trainer.lr = config['lr']
            # Remove lr from config to avoid passing it to training methods
            config = {k: v for k, v in config.items() if k != 'lr'}
        
        # Filter config for unsupervised autoencoder (remove supervised-specific parameters)
        unsupervised_config = {k: v for k, v in config.items() 
                             if k not in ['lambda_classify']}
        
        # Train unsupervised autoencoders
        print("   Training unsupervised autoencoders...")
        unsupervised_results = {}
        for layer_idx in range(num_layers):
            print(f"   Layer {layer_idx + 1}/{num_layers}")
            layer_activations = train_activations[:, layer_idx, :]
            
            result = self.autoencoder_trainer.train_unsupervised_autoencoder(
                layer_activations, **unsupervised_config
            )
            unsupervised_results[f"layer_{layer_idx}"] = result
            
        # Train supervised autoencoders
        print("   Training supervised autoencoders...")
        supervised_results = {}
        for layer_idx in range(num_layers):
            print(f"   Layer {layer_idx + 1}/{num_layers}")
            layer_activations = train_activations[:, layer_idx, :]
            
            result = self.autoencoder_trainer.train_supervised_autoencoder(
                layer_activations, train_labels, **config
            )
            supervised_results[f"layer_{layer_idx}"] = result
            
        # Evaluate on test set
        unsupervised_test_results = {}
        supervised_test_results = {}
        
        for layer_idx in range(num_layers):
            layer_name = f"layer_{layer_idx}"
            
            # Unsupervised evaluation
            autoencoder = unsupervised_results[layer_name]['autoencoder']
            test_layer_activations = test_activations[:, layer_idx, :]
            unsupervised_analysis = self.autoencoder_trainer.analyze_features(
                autoencoder, test_layer_activations
            )
            unsupervised_test_results[layer_name] = unsupervised_analysis
            
            # Supervised evaluation
            autoencoder = supervised_results[layer_name]['autoencoder']
            supervised_analysis = self.autoencoder_trainer.analyze_features(
                autoencoder, test_layer_activations, test_labels
            )
            supervised_test_results[layer_name] = supervised_analysis
            
        return {
            'unsupervised_results': unsupervised_results,
            'supervised_results': supervised_results,
            'unsupervised_test_results': unsupervised_test_results,
            'supervised_test_results': supervised_test_results,
            'summary': {
                'num_layers': num_layers,
                'unsupervised_best_reconstruction': min(
                    result['final_metrics']['reconstruction_loss'] 
                    for result in unsupervised_results.values()
                ),
                'supervised_best_accuracy': max(
                    result['final_metrics']['accuracy']
                    for result in supervised_results.values()
                )
            }
        }
        
    def _analyze_results(self, probe_results: Dict, autoencoder_results: Dict,
                        train_activations: torch.Tensor, train_labels: torch.Tensor,
                        test_activations: torch.Tensor, test_labels: torch.Tensor) -> Dict:
        """Analyze and compare results across methods."""
        
        # Compare layer-wise performance
        layer_comparison = {}
        num_layers = train_activations.shape[1]
        
        for layer_idx in range(num_layers):
            layer_name = f"layer_{layer_idx}"
            
            # Probe performance
            probe_auc = probe_results['test_results'][layer_name]['auc']
            probe_acc = probe_results['test_results'][layer_name]['accuracy']
            
            # Autoencoder performance
            unsupervised_recon = autoencoder_results['unsupervised_results'][layer_name]['final_metrics']['reconstruction_loss']
            supervised_acc = autoencoder_results['supervised_results'][layer_name]['final_metrics']['accuracy']
            
            layer_comparison[layer_name] = {
                'probe_auc': probe_auc,
                'probe_accuracy': probe_acc,
                'unsupervised_reconstruction_loss': unsupervised_recon,
                'supervised_accuracy': supervised_acc,
                'combined_score': probe_auc + supervised_acc  # Simple combination
            }
            
        # Find best layers for each method
        best_probe_layer = max(
            layer_comparison.items(),
            key=lambda x: x[1]['probe_auc']
        )
        
        best_supervised_layer = max(
            layer_comparison.items(),
            key=lambda x: x[1]['supervised_accuracy']
        )
        
        # Feature importance analysis
        feature_analysis = {}
        if probe_results['best_layer']['layer_name'] in probe_results['layer_results']:
            best_probe = probe_results['layer_results'][probe_results['best_layer']['layer_name']]['results']['probe']
            feature_analysis['probe_importance'] = self.probe_trainer.analyze_feature_importance(
                best_probe, top_k=100
            )
            
        return {
            'layer_comparison': layer_comparison,
            'best_layers': {
                'probe': best_probe_layer,
                'supervised_autoencoder': best_supervised_layer
            },
            'feature_analysis': feature_analysis,
            'summary': {
                'best_probe_auc': best_probe_layer[1]['probe_auc'],
                'best_supervised_acc': best_supervised_layer[1]['supervised_accuracy'],
                'most_informative_layer': max(
                    layer_comparison.items(),
                    key=lambda x: x[1]['combined_score']
                )[0]
            }
        }
        
    def _cross_scenario_analysis(self, df, train_activations: torch.Tensor,
                               train_labels: torch.Tensor,
                               test_activations: torch.Tensor,
                               test_labels: torch.Tensor,
                               probe_config: Dict,
                               autoencoder_config: Dict) -> Dict:
        """Analyze generalization across different deception scenarios."""
        
        scenarios = df['scenario'].unique()
        cross_results = {}
        
        # Train on one scenario, test on others
        for train_scenario in scenarios:
            print(f"   Training on {train_scenario}...")
            
            # Filter training data for this scenario
            train_scenario_df = self.data_loader.filter_by_scenario(
                df.iloc[:len(train_activations)], [train_scenario]
            )
            if len(train_scenario_df) < 10:  # Skip if too few samples
                continue
                
            train_scenario_activations = self.data_loader.get_activations_tensor(train_scenario_df)
            train_scenario_labels = self.data_loader.get_labels_tensor(train_scenario_df)
            
            # Train probe on this scenario
            scenario_probe_results = self.probe_trainer.train_multi_layer_probes(
                train_scenario_activations, train_scenario_labels,
                **probe_config
            )
            
            # Test on all scenarios
            scenario_test_results = {}
            for test_scenario in scenarios:
                test_scenario_df = self.data_loader.filter_by_scenario(
                    df.iloc[len(train_activations):], [test_scenario]
                )
                if len(test_scenario_df) < 5:  # Skip if too few samples
                    continue
                    
                test_scenario_activations = self.data_loader.get_activations_tensor(test_scenario_df)
                test_scenario_labels = self.data_loader.get_labels_tensor(test_scenario_df)
                
                # Evaluate best probe on this test scenario
                best_layer_name = max(
                    scenario_probe_results.items(),
                    key=lambda x: x[1]['final_auc']
                )[0]
                
                best_probe = scenario_probe_results[best_layer_name]['results']['probe']
                test_metrics = self.probe_trainer.evaluate_probe(
                    best_probe,
                    test_scenario_activations[:, scenario_probe_results[best_layer_name]['layer_idx'], :],
                    test_scenario_labels
                )
                
                scenario_test_results[test_scenario] = test_metrics
                
            cross_results[train_scenario] = {
                'train_samples': len(train_scenario_df),
                'test_results': scenario_test_results
            }
            
        return cross_results
        
    def _save_results(self, results: Dict) -> None:
        """Save results to disk."""
        
        # Save main results
        results_file = self.output_dir / "experiment_results.json"
        
        # Convert torch tensors to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        # Save model checkpoints
        models_dir = self.output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Save best probe
        if 'probe_results' in results and 'best_layer' in results['probe_results']:
            best_layer_name = results['probe_results']['best_layer']['layer_name']
            best_probe = results['probe_results']['layer_results'][best_layer_name]['results']['probe']
            probe_path = models_dir / f"best_probe_{best_layer_name}.pt"
            self.probe_trainer.save_probe(best_probe, probe_path)
            
        # Save best autoencoder
        if 'autoencoder_results' in results and 'supervised_results' in results['autoencoder_results']:
            supervised_results = results['autoencoder_results']['supervised_results']
            best_layer_name = max(
                supervised_results.items(),
                key=lambda x: x[1]['final_metrics']['accuracy']
            )[0]
            
            best_autoencoder = supervised_results[best_layer_name]['autoencoder']
            autoencoder_path = models_dir / f"best_autoencoder_{best_layer_name}.pt"
            torch.save(best_autoencoder.state_dict(), autoencoder_path)
            
        print(f"Results saved to {self.output_dir}")
        
    def _make_serializable(self, obj: Any) -> Any:
        """Convert torch tensors and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, torch.nn.Module):
            # Skip nn.Module objects - they're saved separately as checkpoints
            return f"<nn.Module: {type(obj).__name__}>"
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            # Check for pandas DataFrame
            try:
                import pandas as pd
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
            except ImportError:
                pass
            # For other unknown types, try to convert to string representation
            try:
                return str(obj)
            except:
                return f"<non-serializable: {type(obj).__name__}>"
            
    def load_results(self, results_dir: Union[str, Path]) -> Dict:
        """Load previously saved results."""
        results_dir = Path(results_dir)
        results_file = results_dir / "experiment_results.json"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
            
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        self.results = results
        return results
        
    def create_sample_data(self, output_path: Union[str, Path], 
                          num_samples: int = 100) -> None:
        """Create sample CSV data for testing."""
        self.data_loader.create_sample_csv(output_path, num_samples)
        
    def quick_test(self, csv_path: Union[str, Path], 
                  num_samples: int = 50) -> Dict:
        """Run a quick test with limited samples."""
        return self.run_full_experiment(
            csv_path=csv_path,
            max_samples=num_samples,
            probe_config={'epochs': 10},
            autoencoder_config={'epochs': 10}
        )
    
    def run_experiment_with_reasoning_traces(self,
                                            traces_path: Union[str, Path],
                                            activations_dir: Optional[Union[str, Path]] = None,
                                            use_final_step: bool = True,
                                            probe_config: Optional[Dict] = None,
                                            autoencoder_config: Optional[Dict] = None,
                                            save_results: bool = True) -> Dict:
        """
        Run deception circuit experiment using reasoning traces.
        
        This method integrates reasoning traces into the training pipeline by:
        1. Loading reasoning traces from disk
        2. Converting traces to activation tensors and DataFrame format
        3. Running the standard training pipeline
        
        Args:
            traces_path: Path to saved reasoning traces JSON file
            activations_dir: Directory containing activation files (if separate)
            use_final_step: If True, use final step activations; if False, average all steps
            probe_config: Configuration for linear probes
            autoencoder_config: Configuration for autoencoders
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with complete experiment results
        """
        from .reasoning_traces import ReasoningTraceCollector
        
        print("Loading reasoning traces...")
        collector = ReasoningTraceCollector(model=None, tokenizer=None, device=self.device)
        collector.load_traces(traces_path, activations_dir)
        
        if len(collector.traces) == 0:
            raise ValueError(f"No traces found in {traces_path}")
        
        print(f"Loaded {len(collector.traces)} reasoning traces")
        
        # Convert traces to DataFrame
        print("Converting traces to DataFrame format...")
        df = collector.convert_traces_to_dataframe()
        
        # Convert traces to activations
        print("Converting traces to activation tensors...")
        activations, labels = collector.convert_traces_to_activations(use_final_step=use_final_step)
        
        # Save temporary CSV for compatibility with existing pipeline
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_csv = f.name
            df.to_csv(temp_csv, index=False)
        
        # Save activations temporarily
        import tempfile as tf
        temp_act_dir = tf.mkdtemp()
        from pathlib import Path
        act_dir_path = Path(temp_act_dir)
        act_dir_path.mkdir(exist_ok=True)
        
        # Save activations in expected format
        for i in range(len(activations)):
            torch.save(activations[i], act_dir_path / f"activations_{i}.pt")
        
        try:
            # Run standard experiment with the converted data
            results = self.run_full_experiment(
                csv_path=temp_csv,
                activation_dir=temp_act_dir,
                probe_config=probe_config,
                autoencoder_config=autoencoder_config,
                save_results=save_results
            )
            
            # Add reasoning trace metadata
            results['reasoning_trace_metadata'] = {
                'num_traces': len(collector.traces),
                'traces_path': str(traces_path),
                'use_final_step': use_final_step,
                'average_steps_per_trace': sum(len(trace.steps) for trace in collector.traces) / len(collector.traces)
            }
            
            return results
            
        finally:
            # Cleanup temporary files
            import os
            try:
                os.unlink(temp_csv)
                import shutil
                shutil.rmtree(temp_act_dir)
            except:
                pass