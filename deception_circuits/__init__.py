"""
Deception Circuits Research Infrastructure

A comprehensive framework for discovering and manipulating deception circuits 
in LLM reasoning traces using linear probes and sparse autoencoders.

Based on the research project: "Can deception circuits be discovered and 
causally manipulated inside LLM reasoning traces?"

This framework implements the methodology from the BLKY project, which aims to:
1. Discover internal circuits that drive deceptive behavior in LLMs
2. Test whether these circuits can be causally manipulated
3. Determine if deception circuits generalize across different contexts

Key Components:
- CSV data loading for truthful vs deceptive pairs
- Linear probe training and evaluation
- Sparse autoencoder feature discovery
- Activation patching for causal testing
- Cross-context generalization analysis

Usage Example:
    from deception_circuits import DeceptionTrainingPipeline
    
    pipeline = DeceptionTrainingPipeline(device="cpu")
    results = pipeline.run_full_experiment("deception_data.csv")

=== FOR FIRST-TIME READERS ===

Welcome to the Deception Circuits Research Framework! This is a complete toolkit
for understanding how AI models lie and finding ways to detect/control deception.

=== WHAT IS THIS FRAMEWORK? ===

This framework helps you answer a crucial question: "How do AI models lie, and can we detect it?"

When AI models are deceptive, they must encode that deception somewhere in their internal
neural computations. This framework helps you find those "deception circuits" - the specific
patterns of neural activity that correspond to lying vs telling the truth.

=== THE RESEARCH PROCESS ===

1. **COLLECT DATA**: Get examples of truthful vs deceptive responses
2. **TRAIN DETECTORS**: Create AI models that can spot deception signals
3. **FIND CIRCUITS**: Identify which parts of the neural network are responsible
4. **TEST CAUSALITY**: Prove that manipulating those circuits changes behavior
5. **CHECK GENERALIZATION**: See if circuits work across different types of lies

=== QUICK START ===

```python
# Import the main pipeline
from deception_circuits import DeceptionTrainingPipeline

# Create and run a complete experiment
pipeline = DeceptionTrainingPipeline(device="cpu")
results = pipeline.run_full_experiment("your_data.csv")

# See which layers are best at detecting deception
print(f"Best layer: {results['analysis_results']['best_probe_layer']}")
```

=== WHAT YOU CAN DO ===

- **Discover Deception Circuits**: Find where lies are encoded in neural networks
- **Test Causal Interventions**: Prove that circuits actually cause deception
- **Cross-Scenario Analysis**: See if circuits generalize across different types of lies
- **Create Visualizations**: Generate plots and dashboards of your results
- **Save Everything**: Complete experiment logging and model checkpoints

=== WHO IS THIS FOR? ===

- AI Safety Researchers studying model behavior
- Mechanistic Interpretability researchers
- Anyone interested in understanding AI deception
- Researchers working on AI alignment and transparency

This framework puts the power of cutting-edge AI interpretability research in your hands!
"""

__version__ = "0.1.0"
__author__ = "Krish Jain"

# Import all main classes for easy access
from .data_loader import DeceptionDataLoader
from .linear_probe import DeceptionLinearProbe, LinearProbeTrainer
from .sparse_autoencoder import DeceptionSparseAutoencoder, AutoencoderTrainer
from .activation_patching import ActivationPatcher, CausalTester
from .analysis import CircuitAnalyzer, VisualizationTools
from .training_pipeline import DeceptionTrainingPipeline

# Import new production-ready modules
from .model_integration import (
    GPT4oIntegration, ActivationExtractor, ModelIntegrationPipeline,
    create_gpt4o_deception_dataset
)
from .advanced_causal_testing import (
    AttentionPatcher as AdvancedAttentionPatcher, 
    SteeringVectorLearner, GradientBasedInterventions,
    MultiLayerIntervention, AdvancedCausalTester,
    run_advanced_deception_analysis
)
from .interpretability import (
    AttentionVisualizer, FeatureAttributor, NeuronAnalyzer,
    CircuitVisualizer, InterpretabilitySuite,
    run_comprehensive_interpretability_analysis
)
from .production import (
    ConfigManager, ProductionLogger, PerformanceMonitor,
    ErrorHandler, CheckpointManager, ProductionPipeline,
    production_ready, create_production_config,
    create_dockerfile, create_docker_compose
)

# Import new research modules
from .reasoning_traces import (
    ReasoningTraceCollector, ReasoningTrace, ReasoningStep
)
from .dataset_integrations import (
    GSM8KIntegration, TruthfulQAIntegration, MMLUIntegration,
    DatasetIntegrationPipeline
)
from .baseline_comparisons import (
    RLHFComparison, ModelSizeComparison, BaselineComparisonPipeline,
    ModelComparisonConfig
)
from .advanced_metrics import (
    WinRateCalculator, LatencyTracker, UncertaintyAnalyzer,
    ContradictionAnalyzer, AdvancedMetricsCollector, ResponseMetrics
)
from .enhanced_scenarios import (
    PokerBadHandBluffing, PasswordLocking, EnhancedSandbagging, PokerHand
)
from .game_data_loaders import (
    MafiaDataLoader, BullshitDataLoader, PokerGameDataLoader,
    MafiaGameData, BullshitGameData, PokerGameData
)

# Define what gets imported when someone does "from deception_circuits import *"
__all__ = [
    # Core Framework
    "DeceptionDataLoader",      # Loads CSV data with activation support
    "DeceptionLinearProbe",     # Linear classifier for deception detection
    "LinearProbeTrainer",       # Trains probes across multiple layers
    "DeceptionSparseAutoencoder", # Discovers interpretable features
    "AutoencoderTrainer",       # Trains autoencoders with sparsity
    "ActivationPatcher",        # Core patching operations
    "CausalTester",             # Comprehensive causal testing framework
    "CircuitAnalyzer",          # Analyzes results and identifies circuits
    "VisualizationTools",       # Creates plots and dashboards
    "DeceptionTrainingPipeline", # Main orchestration class
    
    # Real Model Integration
    "GPT4oIntegration",         # GPT-4o integration for data generation
    "ActivationExtractor",      # Extract activations from transformer models
    "ModelIntegrationPipeline", # Complete pipeline for real model integration
    "create_gpt4o_deception_dataset", # Create datasets with GPT-4o
    
    # Advanced Causal Testing
    "AdvancedAttentionPatcher", # Advanced attention patching
    "SteeringVectorLearner",    # Learn steering vectors for behavior control
    "GradientBasedInterventions", # Gradient-based causal interventions
    "MultiLayerIntervention",   # Multi-layer coordinated interventions
    "AdvancedCausalTester",     # Advanced causal testing framework
    "run_advanced_deception_analysis", # Run advanced causal analysis
    
    # Interpretability
    "AttentionVisualizer",      # Visualize attention patterns
    "FeatureAttributor",        # Feature attribution analysis
    "NeuronAnalyzer",           # Individual neuron analysis
    "CircuitVisualizer",        # Circuit flow visualization
    "InterpretabilitySuite",    # Comprehensive interpretability tools
    "run_comprehensive_interpretability_analysis", # Run interpretability analysis
    
    # Production Features
    "ConfigManager",            # Configuration management
    "ProductionLogger",         # Production logging system
    "PerformanceMonitor",       # Performance monitoring
    "ErrorHandler",             # Error handling and recovery
    "CheckpointManager",        # Checkpoint management
    "ProductionPipeline",       # Production-ready pipeline
    "production_ready",         # Decorator for production functions
    "create_production_config", # Create production configuration
    "create_dockerfile",        # Create Dockerfile
    "create_docker_compose",    # Create Docker Compose file
    # New Research Modules
    "ReasoningTraceCollector",  # Multi-step reasoning trace collection
    "ReasoningTrace",           # Reasoning trace data structure
    "ReasoningStep",            # Single reasoning step
    "GSM8KIntegration",         # GSM8K dataset integration
    "TruthfulQAIntegration",    # TruthfulQA dataset integration
    "MMLUIntegration",         # MMLU dataset integration
    "DatasetIntegrationPipeline", # Complete dataset integration
    "RLHFComparison",          # RLHF alignment comparison
    "ModelSizeComparison",     # Model size comparison
    "BaselineComparisonPipeline", # Baseline comparison pipeline
    "ModelComparisonConfig",   # Model comparison configuration
    "WinRateCalculator",       # Win rate calculation for games
    "LatencyTracker",          # Response latency tracking
    "UncertaintyAnalyzer",     # Token-level uncertainty analysis
    "ContradictionAnalyzer",   # Contradiction tracking
    "AdvancedMetricsCollector", # Complete metrics collector
    "ResponseMetrics",         # Response metrics data structure
    "PokerBadHandBluffing",    # Poker bad hand bluffing scenario
    "PasswordLocking",         # Password-locking mechanism
    "EnhancedSandbagging",     # Enhanced sandbagging scenarios
    "PokerHand",               # Poker hand data structure
    "MafiaDataLoader",         # Mafia game data loader
    "BullshitDataLoader",      # Bullshit game data loader
    "PokerGameDataLoader",     # Poker game data loader
    "MafiaGameData",           # Mafia game data structure
    "BullshitGameData",        # Bullshit game data structure
    "PokerGameData",           # Poker game data structure
]
