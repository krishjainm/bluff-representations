# Deception LLMs Research Infrastructure

> **Research status:** The strict, validated V2 probe workflow is
> `deception_circuits.paper`; historical modules below are legacy/experimental.
> This repository does not establish a general “deception circuit,” and no
> research command substitutes random activations for missing data.

## Paper V2 workflow

Copy `configs/paper_v2.yaml`, set it to a real canonical dataset and activation
directory, and run:

```bash
uv sync
deception-paper validate-data --config my_run.yaml
deception-paper make-splits --config my_run.yaml
# Loads subject-model weights and runs real forward passes; requires the flag.
deception-paper collect-activations --config my_run.yaml --confirm-model-load
deception-paper train-probes --config my_run.yaml
deception-paper audit --config my_run.yaml
```

The V2 workflow requires one finite `[layers, hidden_size]` tensor per sample,
uses group-safe frozen splits, chooses layers on validation only, and records
config/environment/commit metadata. It does not download a model or make API
calls. See [the protocol](docs/EXPERIMENT_PROTOCOL.md),
[reproducibility instructions](docs/REPRODUCIBILITY.md), and the
[review-response matrix](docs/REVIEW_RESPONSE_MATRIX.md).

A comprehensive framework for discovering and manipulating deception circuits in LLM reasoning traces using linear probes and sparse autoencoders.

## Overview

This project implements the research methodology described in **"Can deception circuits be discovered and causally manipulated inside LLM reasoning traces?"** The framework provides tools for:

- **Data Loading**: CSV-based loading of truthful vs deceptive reasoning pairs
- **Linear Probes**: Training probes to detect deception signals in model activations
- **Sparse Autoencoders**: Discovering interpretable features related to deception
- **Causal Testing**: Activation patching, steering vectors, and advanced interventions
- **Dataset Integrations**: Built-in support for GSM8K, TruthfulQA, MMLU, and PokerBench
- **Reasoning Traces**: Multi-step reasoning trace collection and analysis
- **Real Model Integration**: GPT-4o integration and activation extraction from transformer models
- **Analysis & Visualization**: Comprehensive analysis and visualization tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/krishjainm/Deception-LLMs.git
cd Deception-LLMs
```

2. Install dependencies with uv:
```bash
# If you don't have uv installed, install it first:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and sync environment
uv sync
```

## Quick Start

### Basic Usage

```python
from deception_circuits import DeceptionTrainingPipeline

# Initialize pipeline
pipeline = DeceptionTrainingPipeline(device="cpu", output_dir="results")

# Run experiment
results = pipeline.run_full_experiment(
    csv_path="your_data.csv",
    max_samples=1000,
    probe_config={'epochs': 100, 'lr': 0.01},
    autoencoder_config={'epochs': 100, 'lr': 0.001}
)

# See results
print(f"Best detection layer: {results['analysis_results']['best_probe_layer']}")
```

### Using Integrated Datasets

```python
from deception_circuits import DatasetIntegrationPipeline

# Create dataset from multiple sources
pipeline = DatasetIntegrationPipeline()
stats = pipeline.create_complete_dataset(
    output_path="integrated_dataset.csv",
    gsm8k_examples=100,      # Sandbagging examples
    truthfulqa_examples=100,  # Truthful vs deceptive Q&A
    mmlu_subsets=["high_school_mathematics", "history"],
    mmlu_examples_per_subset=50,
    pokerbench_examples=100   # Poker bluffing scenarios
)
```

### Reasoning Traces

```python
from deception_circuits import ReasoningTraceCollector, ReasoningTrace, ReasoningStep

# Collect multi-step reasoning traces
collector = ReasoningTraceCollector()
trace = ReasoningTrace(
    statement="What is 2+2?",
    response="4",
    label=0,
    scenario="general",
    steps=[...]  # Multi-step reasoning with activations
)

# Convert to activations or DataFrame
activations, labels = collector.convert_traces_to_activations()
df = collector.convert_traces_to_dataframe()
```

### Advanced Causal Testing

```python
from deception_circuits import AdvancedCausalTester, SteeringVectorLearner

# Learn steering vectors
learner = SteeringVectorLearner(device="cpu")
steering_vector = learner.learn_deception_steering_vector(
    truthful_activations, deceptive_activations, method='pca'
)

# Run advanced causal tests
advanced_tester = AdvancedCausalTester(device="cpu")
results = advanced_tester.run_comprehensive_advanced_test(
    model=your_model,
    truthful_activations=truthful_acts,
    deceptive_activations=deceptive_acts,
    truthful_labels=truthful_labels,
    deceptive_labels=deceptive_labels,
    probe=trained_probe,
    layer_indices=[15, 16, 17, 18, 19]
)
```

## CSV Data Format

Your CSV should contain the following columns:
- `statement`: The input prompt/statement
- `response`: The model's response
- `label`: 0 for truthful, 1 for deceptive
- `scenario`: Type of deception (poker, roleplay, sandbagging, etc.)

Example:
```csv
statement,response,label,scenario
"What is 2+2?","4",0,general
"What is 2+2?","I'm not sure",1,sandbagging
"Do you have a strong hand?","Yes, very strong",0,poker
"Do you have a strong hand?","Yes, very strong",1,poker
```

## Core Components

### 1. Data Loading (`data_loader.py`)

```python
from deception_circuits import DeceptionDataLoader

loader = DeceptionDataLoader(device="cpu")
data = loader.load_csv("data.csv", activation_dir="activations/")
train_data, test_data = loader.split_data(test_size=0.2)
```

### 2. Dataset Integrations (`dataset_integrations.py`)

```python
from deception_circuits import (
    DatasetIntegrationPipeline,
    PokerBenchIntegration,
    GSM8KIntegration,
    TruthfulQAIntegration,
    MMLUIntegration
)

# Complete pipeline
pipeline = DatasetIntegrationPipeline()
stats = pipeline.create_complete_dataset("output.csv", ...)

# Individual integrations
pokerbench = PokerBenchIntegration()
pokerbench.load_dataset("train")
scenarios = pokerbench.create_bluffing_scenarios(num_examples=100)
```

### 3. Linear Probes (`linear_probe.py`)

```python
from deception_circuits import LinearProbeTrainer

trainer = LinearProbeTrainer(device="cpu")
results = trainer.train_probe(activations, labels, epochs=100)
```

### 4. Sparse Autoencoders (`sparse_autoencoder.py`)

```python
from deception_circuits import AutoencoderTrainer

trainer = AutoencoderTrainer(device="cpu")
results = trainer.train_unsupervised_autoencoder(
    activations, epochs=100, l1_coeff=0.01
)
```

### 5. Causal Testing (`activation_patching.py`, `advanced_causal_testing.py`)

**Basic causal testing:**
```python
from deception_circuits import CausalTester

tester = CausalTester(device="cpu")
results = tester.test_deception_suppression(
    truthful_activations, deceptive_activations, 
    truthful_labels, deceptive_labels, layer_idx=20
)
```

**Advanced causal testing:**
```python
from deception_circuits import AdvancedCausalTester

advanced_tester = AdvancedCausalTester(device="cpu")
results = advanced_tester.run_comprehensive_advanced_test(
    model=model, activations=acts, labels=labels, ...
)
```

### 6. Reasoning Traces (`reasoning_traces.py`)

```python
from deception_circuits import ReasoningTraceCollector

collector = ReasoningTraceCollector()
# Add traces with multi-step reasoning
collector.add_trace(trace)

# Convert to activations/DataFrame for training
activations, labels = collector.convert_traces_to_activations()
df = collector.convert_traces_to_dataframe()

# Use with training pipeline
pipeline = DeceptionTrainingPipeline(device="cpu")
results = pipeline.run_experiment_with_reasoning_traces(collector)
```

### 7. Real Model Integration (`model_integration.py`)

```python
from deception_circuits import GPT4oIntegration, ActivationExtractor

# GPT-4o integration for data generation
gpt4o = GPT4oIntegration(api_key="your-key")
dataset = gpt4o.create_deception_dataset(num_examples=100)

# Extract activations from transformer models
extractor = ActivationExtractor(model_name="gpt2", device="cpu")
extractor.load_model()
activations = extractor.extract_activations(texts=["prompt1", "prompt2"])
```

### 8. Analysis & Visualization (`analysis.py`)

```python
from deception_circuits import CircuitAnalyzer, VisualizationTools

analyzer = CircuitAnalyzer(results)
probe_analysis = analyzer.analyze_probe_performance()

viz_tools = VisualizationTools()
fig = viz_tools.plot_layer_performance(results)
dashboard = viz_tools.create_interactive_dashboard(results)
```

## Example Experiment

Run the complete example:

```bash
uv run example_usage.py
```

This will:
1. Create sample deception data
2. Demonstrate dataset integrations (PokerBench, GSM8K, etc.)
3. Show reasoning trace collection
4. Train linear probes and sparse autoencoders
5. Analyze results across layers
6. Run causal intervention tests (basic and advanced)
7. Generate visualizations

See `example_usage.py` for complete examples of:
- Basic training pipeline
- Dataset integrations
- Reasoning trace collection
- Advanced causal testing

## Research Applications

### Deception Scenarios

The framework supports multiple deception types:

- **Poker Bluffing**: Strategic deception in game contexts
- **Sandbagging**: Deliberate underperformance
- **Roleplay**: Character-based deception
- **Pressure/Blackmail**: Deception under threat
- **Password Gating**: Conditional knowledge hiding

### Cross-Context Generalization

Test whether deception circuits generalize across scenarios:

```python
# Train on poker scenarios, test on sandbagging
cross_results = pipeline._cross_scenario_analysis(
    df, train_activations, train_labels, test_activations, test_labels
)
```

### Causal Interventions

**Basic interventions:**
```python
# Test 1: Replace deceptive activations with truthful ones
suppression_results = tester.test_deception_suppression(...)

# Test 2: Inject deception into truthful runs
injection_results = tester.test_deception_injection(...)

# Test 3: Cross-context patching
cross_context_results = tester.test_cross_context_patching(...)
```

**Advanced interventions:**
```python
# Attention patching
# Steering vectors
# Gradient-based interventions
# Multi-layer coordinated interventions
```

## Architecture

```
deception_circuits/
├── __init__.py              # Main package interface
├── data_loader.py           # CSV data loading and processing
├── linear_probe.py          # Linear probe implementation
├── sparse_autoencoder.py    # Sparse autoencoder implementation
├── training_pipeline.py     # Main training orchestration
├── analysis.py              # Analysis and visualization tools
├── activation_patching.py   # Basic causal testing framework
├── advanced_causal_testing.py  # Advanced causal testing
├── dataset_integrations.py  # Dataset integrations (GSM8K, TruthfulQA, MMLU, PokerBench)
├── reasoning_traces.py      # Multi-step reasoning trace collection
├── model_integration.py     # Real model integration (GPT-4o, activation extraction)
├── interpretability.py      # Interpretability tools
├── production.py            # Production features (logging, monitoring)
├── baseline_comparisons.py  # Baseline comparison tools
├── advanced_metrics.py      # Advanced metrics (win rate, latency, etc.)
├── enhanced_scenarios.py    # Enhanced deception scenarios
└── game_data_loaders.py     # Game data loaders (Mafia, Bullshit, Poker)
```

## Key Features

- **Multi-Layer Analysis**: Train probes and autoencoders across all model layers
- **Cross-Scenario Testing**: Test generalization across different deception contexts
- **Causal Validation**: Prove causal relationships through activation patching and steering
- **Dataset Integrations**: Built-in support for standard datasets (GSM8K, TruthfulQA, MMLU, PokerBench)
- **Reasoning Traces**: Collect and analyze multi-step reasoning with layer-wise activations
- **Real Model Support**: GPT-4o integration and activation extraction from transformer models
- **Advanced Causal Testing**: Attention patching, steering vectors, gradient-based interventions
- **Comprehensive Visualization**: Interactive dashboards and static plots
- **Production-Ready**: Logging, monitoring, error handling, Docker support
- **Flexible Data Format**: Easy CSV-based data loading
- **Extensible Design**: Easy to add new deception scenarios or analysis methods

## Testing

Run the test suites:

```bash
# Core framework tests
uv run test_framework.py

# Integration tests
uv run test_integration.py

# Quick validation
uv run quick_validation.py
```

## Documentation

- **README.md** (this file): Overview and quick start
- **SETUP_GUIDE.md**: Detailed setup instructions
- **DATA_COLLECTION_GUIDE.md**: Comprehensive guide to data collection and dataset integrations
- **PRODUCTION_README.md**: Production deployment guide
- **example_usage.py**: Complete working examples

## Citation

If you use this infrastructure in your research, please cite:

```bibtex
@misc{deception_circuits_2025,
  title={Deception Circuits Research Infrastructure},
  author={Krish Jain},
  year={2025},
  url={https://github.com/krishjainm/Deception-LLMs}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
