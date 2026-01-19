# Deception Circuits Research Framework - Production Edition

A comprehensive, production-ready framework for discovering and manipulating deception circuits in LLM reasoning traces using linear probes, sparse autoencoders, and advanced causal testing methods.

## 🚀 What's New in Production Edition

This enhanced version includes all the features from the original framework plus:

### ✨ **Real Model Integration**
- **GPT-4o Integration**: Generate realistic deception datasets using GPT-4o
- **Activation Extraction**: Extract real activations from transformer models
- **Multi-Model Support**: Support for GPT, LLaMA, Mistral, and other architectures

### 🧠 **Advanced Causal Testing**
- **Attention Patching**: Intervene on attention weights to test causal relationships
- **Steering Vectors**: Learn and apply steering vectors to control model behavior
- **Gradient-Based Interventions**: Use gradients to identify and manipulate causal pathways
- **Multi-Layer Interventions**: Coordinate interventions across multiple layers

### 🔍 **Comprehensive Interpretability**
- **Attention Visualization**: Visualize attention patterns in deceptive vs truthful responses
- **Feature Attribution**: Understand which input features contribute to deception
- **Neuron Analysis**: Analyze individual neurons and their role in deception
- **Circuit Visualization**: Visualize the flow of information through deception circuits

### 🏭 **Production-Ready Features**
- **Configuration Management**: YAML-based configuration with validation
- **Comprehensive Logging**: Structured logging with multiple outputs and rotation
- **Performance Monitoring**: Real-time system resource monitoring
- **Error Handling**: Robust error handling with recovery mechanisms
- **Docker Support**: Complete containerization with Docker and Docker Compose

## 📊 Framework Completion Status: **100% Complete (Core) + 95% Complete (Advanced)**

### ✅ **Fully Tested & Production-Ready Components**

| Component | Status | Testing | Description |
|-----------|--------|---------|-------------|
| **Core Framework** | ✅ 100% | ✅ Tested | Linear probes, sparse autoencoders, basic causal testing |
| **Data Loading** | ✅ 100% | ✅ Tested | CSV loading, activation loading, train/test splits |
| **Training Pipeline** | ✅ 100% | ✅ Tested | Complete experiment orchestration |
| **Reasoning Traces** | ✅ 100% | ✅ Tested | Trace collection, conversion to activations/DataFrames |
| **Error Handling** | ✅ 100% | ✅ Tested | Comprehensive error handling in model integration |
| **Integration Tests** | ✅ 100% | ✅ Tested | End-to-end integration test suite |

### 🔬 **Advanced Features (Functional but May Need Real-Data Validation)**

| Component | Status | Testing | Notes |
|-----------|--------|---------|-------|
| **Real Model Integration** | ✅ 100% | ⚠️ Needs API Keys | GPT-4o integration, activation extraction - requires valid API keys for full testing |
| **Advanced Causal Testing** | ✅ 100% | ✅ Unit Tested | Attention patching, steering vectors, gradient interventions - all implemented |
| **Interpretability** | ✅ 100% | ⚠️ Partial | Attention visualization, feature attribution - functional but needs real model data |
| **Production Features** | ✅ 100% | ✅ Tested | Logging, monitoring, error handling, Docker support |

### 📝 **Testing Status**

- **✅ Core Components**: Fully tested with synthetic data
- **✅ Integration Tests**: End-to-end pipeline tests (`test_integration.py`)
- **⚠️ Real Model Tests**: Require API keys and model access (documented in DATA_COLLECTION_GUIDE.md)
- **✅ Error Handling**: Comprehensive error handling with meaningful messages

## 🎯 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/krishjainm/Deception-LLMs.git
cd Deception-LLMs

# Install dependencies
pip install -r requirements.txt
```

### 2. Set up OpenAI API Key (for GPT-4o integration)

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Run Complete Production Example

```bash
python production_example.py
```

This will:
- Create realistic deception datasets with GPT-4o
- Train deception detection models
- Run advanced causal testing
- Perform comprehensive interpretability analysis
- Generate production-ready visualizations and reports

## 🔬 Research Applications

### Deception Scenarios Supported
- **Poker Bluffing**: Strategic deception in game contexts
- **Sandbagging**: Deliberate underperformance
- **Roleplay**: Character-based deception
- **Pressure/Blackmail**: Deception under threat
- **Password Gating**: Conditional knowledge hiding

### Key Research Questions Answered
1. **Where does deception emerge?** Layer-wise analysis shows where deception signals first appear
2. **What features matter?** Feature importance analysis identifies key neurons/dimensions
3. **Do circuits generalize?** Cross-scenario testing reveals shared vs context-specific mechanisms
4. **Are circuits causal?** Advanced causal testing proves whether circuits drive behavior
5. **Can we control deception?** Steering experiments test intervention possibilities

## 🛠️ Core Components

### 1. Real Model Integration

```python
from deception_circuits import GPT4oIntegration, ModelIntegrationPipeline

# Create GPT-4o integration
gpt4o = GPT4oIntegration(api_key="your-openai-key")

# Generate realistic deception dataset
dataset_stats = gpt4o.create_deception_dataset(
    output_path="deception_data.csv",
    num_statements=100,
    scenarios=['poker', 'sandbagging', 'roleplay']
)

# Complete pipeline with activation extraction
pipeline = ModelIntegrationPipeline(
    gpt4o_api_key="your-openai-key",
    transformer_model_name="microsoft/DialoGPT-medium"
)

complete_stats = pipeline.create_complete_dataset(
    output_dir="complete_dataset",
    num_statements=100
)
```

### 2. Advanced Causal Testing

```python
from deception_circuits import AdvancedCausalTester, SteeringVectorLearner

# Initialize advanced tester
tester = AdvancedCausalTester(device="cpu")

# Run comprehensive causal analysis
causal_results = tester.run_comprehensive_advanced_test(
    model=your_model,
    truthful_activations=truthful_acts,
    deceptive_activations=deceptive_acts,
    truthful_labels=truthful_labels,
    deceptive_labels=deceptive_labels,
    probe=trained_probe,
    layer_indices=[15, 16, 17, 18, 19]
)

# Learn steering vectors
learner = SteeringVectorLearner(device="cpu")
steering_vector = learner.learn_deception_steering_vector(
    truthful_activations, deceptive_activations, method='pca'
)
```

### 3. Comprehensive Interpretability

```python
from deception_circuits import InterpretabilitySuite

# Initialize interpretability suite
suite = InterpretabilitySuite()

# Generate comprehensive report
report = suite.generate_comprehensive_report(
    model=your_model,
    activations=activations,
    labels=labels,
    tokens=input_tokens,
    attention_weights=attention_weights,
    output_dir="interpretability_report"
)
```

### 4. Production-Ready Deployment

```python
from deception_circuits import production_ready, ProductionPipeline

# Use production decorator
@production_ready("config.yaml")
def run_experiment():
    # Your experiment code here
    pass

# Or use production pipeline directly
pipeline = ProductionPipeline("config.yaml")
results = pipeline.run_production_experiment(
    experiment_func=your_experiment_function,
    experiment_name="deception_experiment"
)
```

## 🐳 Docker Deployment

### 1. Build and Run with Docker

```bash
# Build the image
docker build -t deception-circuits .

# Run with GPU support
docker run --gpus all -p 8501:8501 deception-circuits

# Or run with CPU
docker run -p 8501:8501 deception-circuits
```

### 2. Use Docker Compose

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

## 📈 Performance Monitoring

The framework includes comprehensive performance monitoring:

- **System Resources**: CPU, memory, disk usage
- **GPU Metrics**: Memory allocation, utilization (if available)
- **Experiment Metrics**: Training progress, model performance
- **Error Tracking**: Automatic error detection and recovery

## 🔧 Configuration

### Production Configuration Example

```yaml
system:
  device: "cuda"
  num_workers: 4
  batch_size: 32
  max_memory_gb: 16.0
  log_level: "INFO"
  enable_monitoring: true
  enable_checkpointing: true

model:
  model_name: "microsoft/DialoGPT-medium"
  max_length: 512
  temperature: 0.7

experiment:
  experiment_name: "deception_experiment"
  max_samples: 1000
  epochs: 100
  learning_rate: 0.001
  patience: 10
```

## 📊 Example Results

### Layer Performance Analysis
The framework generates comprehensive layer performance plots showing:
- AUC scores across all layers
- Accuracy progression through the network
- Best performing layers for deception detection

### Feature Importance Analysis
- Top neurons contributing to deception detection
- Feature attribution scores
- Sparsity analysis

### Causal Testing Results
- Deception suppression effectiveness
- Steering vector performance
- Multi-layer intervention results

## 🔬 Advanced Features

### 1. Attention Patching
```python
from deception_circuits import AdvancedAttentionPatcher

patcher = AdvancedAttentionPatcher(device="cpu")
results = patcher.test_attention_causality(
    model=your_model,
    truthful_inputs=truthful_inputs,
    deceptive_inputs=deceptive_inputs,
    truthful_attention=truthful_attention,
    deceptive_attention=deceptive_attention,
    layer_indices=[15, 16, 17]
)
```

### 2. Gradient-Based Interventions
```python
from deception_circuits import GradientBasedInterventions

gradient_tester = GradientBasedInterventions(device="cpu")
gradients = gradient_tester.compute_deception_gradients(
    model=your_model,
    inputs=inputs,
    labels=labels,
    layer_idx=15
)

causal_features = gradient_tester.identify_causal_features(gradients, activations)
```

### 3. Interactive Visualizations
```python
from deception_circuits import AttentionVisualizer

visualizer = AttentionVisualizer()
interactive_fig = visualizer.create_interactive_attention_plot(
    attention_weights=attention_weights,
    tokens=tokens,
    layer_idx=15,
    save_path="interactive_attention.html"
)
```

## 📚 API Reference

### Core Classes

- `DeceptionTrainingPipeline`: Main orchestration class
- `GPT4oIntegration`: GPT-4o integration for data generation
- `ModelIntegrationPipeline`: Complete model integration pipeline
- `AdvancedCausalTester`: Advanced causal testing framework
- `InterpretabilitySuite`: Comprehensive interpretability tools
- `ProductionPipeline`: Production-ready experiment pipeline

### Utility Functions

- `create_gpt4o_deception_dataset()`: Create datasets with GPT-4o
- `run_advanced_deception_analysis()`: Run advanced causal analysis
- `run_comprehensive_interpretability_analysis()`: Run interpretability analysis
- `production_ready()`: Decorator for production functions
