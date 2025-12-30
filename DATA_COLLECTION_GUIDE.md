# Data Collection Guide

This guide explains how to collect all the data needed for the deception circuit research experiments.

## 📋 Overview

The framework is now **~95% complete**. The remaining work involves collecting actual data from games and setting up model access. This guide will walk you through everything you need.

## 🎮 Game Data Collection

### Mafia Games

**What you need:**
- Completed Mafia game data with model participation
- Game logs showing when the model was deceptive
- Win/loss outcomes

**How to collect:**

1. **Set up Mafia game environment:**
   ```python
   from deception_circuits import MafiaDataLoader
   
   # Create sample structure to see format
   loader = MafiaDataLoader()
   loader.create_sample_structure("mafia_sample_structure.json")
   ```

2. **Run Mafia games with your model:**
   - Use the model as a player in Mafia games
   - Record all statements/responses
   - Track when model is mafia vs town
   - Record win/loss outcomes
   - Note deception instances (when mafia model lies about identity)

3. **Format your data:**
   ```json
   {
     "games": [
       {
         "game_id": "mafia_game_001",
         "players": ["player1", "player2", "model", "player4"],
         "roles": {
           "player1": "town",
           "player2": "mafia",
           "model": "mafia",
           "player4": "town"
         },
         "rounds": [...],
         "model_was_mafia": true,
         "model_won": true,
         "deception_instances": [
           {
             "statement": "I'm a town member",
             "response": "I'm a town member, I saw player2 acting suspicious",
             "was_deceptive": true,
             "round": 1,
             "role": "mafia",
             "context": "trying to hide mafia identity"
           }
         ],
         "final_outcome": "mafia_won"
       }
     ]
   }
   ```

4. **Load and convert:**
   ```python
   loader = MafiaDataLoader()
   games = loader.load_mafiascum_data("your_mafia_data.json")
   df = loader.convert_to_deception_dataset(games)
   df.to_csv("mafia_deception_data.csv", index=False)
   ```

### Bullshit (BS) Card Game

**What you need:**
- Bullshit card game data with model participation
- Records of when model bluffed
- Cards claimed vs actual cards played

**How to collect:**

1. **Create sample structure:**
   ```python
   from deception_circuits import BullshitDataLoader
   
   loader = BullshitDataLoader()
   loader.create_sample_structure("bullshit_sample_structure.json")
   ```

2. **Run Bullshit games:**
   - Play Bullshit with your model
   - Record each turn: cards claimed, cards actually played
   - Track when model bluffs
   - Record win/loss outcomes

3. **Format your data:**
   ```json
   {
     "games": [
       {
         "game_id": "bullshit_game_001",
         "players": ["player1", "model", "player3"],
         "cards_played": [
           {
             "turn": 1,
             "player": "model",
             "cards_claimed": "three 7s",
             "cards_actual": "two 7s, one 5",
             "was_bluff": true,
             "was_called": false
           }
         ],
         "model_bluffed": true,
         "model_won": true,
         "deception_instances": [...]
       }
     ]
   }
   ```

4. **Load and convert:**
   ```python
   loader = BullshitDataLoader()
   games = loader.load_bullshit_data("your_bullshit_data.json")
   df = loader.convert_to_deception_dataset(games)
   df.to_csv("bullshit_deception_data.csv", index=False)
   ```

### Poker Games

**What you need:**
- Structured poker game data
- Model hand information
- Betting rounds and actions
- Bluff instances

**How to collect:**

1. **Run poker games:**
   - Play poker with your model
   - Record hands, betting actions
   - Track when model bluffs
   - Record win/loss outcomes

2. **Format and load:**
   ```python
   from deception_circuits import PokerGameDataLoader
   
   loader = PokerGameDataLoader()
   games = loader.load_poker_data("your_poker_data.json")
   df = loader.convert_to_deception_dataset(games)
   df.to_csv("poker_deception_data.csv", index=False)
   ```

## 📊 Standard Dataset Integration

### GSM8K, TruthfulQA, MMLU

These datasets are automatically downloaded via HuggingFace. No manual collection needed!

**Usage:**
```python
from deception_circuits import DatasetIntegrationPipeline

# Create complete dataset from all sources
pipeline = DatasetIntegrationPipeline()
stats = pipeline.create_complete_dataset(
    output_path="complete_dataset.csv",
    gsm8k_examples=100,
    truthfulqa_examples=100,
    mmlu_subsets=["high_school_mathematics", "history", "astronomy"],
    mmlu_examples_per_subset=50
)
```

**Installation:**
```bash
pip install datasets
```

The datasets will be automatically downloaded on first use.

## 🤖 Model Access Setup

### For RLHF Comparison

**What you need:**
- Base model (pre-alignment)
- RLHF-aligned version of the same model

**How to set up:**
```python
from deception_circuits import BaselineComparisonPipeline, ModelComparisonConfig

# Configure comparison
config = ModelComparisonConfig(
    base_model_name="your_base_model",
    aligned_model_name="your_aligned_model",
    device="cuda"  # or "cpu"
)

# Set up comparison
pipeline = BaselineComparisonPipeline(config)
pipeline.setup_rlhf_comparison(base_model, aligned_model)

# Run comparison
results = pipeline.run_complete_comparison(prompts, labels)
```

**Where to get models:**
- HuggingFace Model Hub
- OpenAI API (for GPT models)
- Your own trained models

### For Model Size Comparison

**What you need:**
- Multiple models of different sizes (e.g., 5B, 10B, 25B parameters)

**How to set up:**
```python
from deception_circuits import ModelSizeComparison

# Load models of different sizes
models = {
    "5B": load_model_5b(),
    "10B": load_model_10b(),
    "25B": load_model_25b()
}

# Set up comparison
comparison = ModelSizeComparison(models, device="cuda")
results = comparison.compare_deception_circuits(prompts, labels)
```

## 🔬 Reasoning Trace Collection

**What you need:**
- Access to model with hooking capabilities
- Chain-of-thought prompts

**How to collect:**
```python
from deception_circuits import ReasoningTraceCollector
from transformers import AutoModel, AutoTokenizer

# Load model
model = AutoModel.from_pretrained("your_model")
tokenizer = AutoTokenizer.from_pretrained("your_model")

# Create collector
collector = ReasoningTraceCollector(model, tokenizer, device="cuda")

# Collect trace
trace = collector.collect_chain_of_thought(
    trace_id="trace_001",
    scenario="poker",
    label=1,  # deceptive
    prompt="You have a 2-7 offsuit. Do you bluff?",
    model=model,
    max_steps=10
)

# Save traces
collector.save_traces("reasoning_traces.json")
```

## 📈 Advanced Metrics Collection

**What you need:**
- Response times
- Token-level logits (for uncertainty)
- Ground truth answers

**How to collect:**
```python
from deception_circuits import AdvancedMetricsCollector
import time

collector = AdvancedMetricsCollector()

# Collect metrics for each response
start_time = time.time()
response = model.generate(prompt)
latency = time.time() - start_time

# Get token uncertainties (from model logits)
token_uncertainties = compute_uncertainty_from_logits(model_logits)

# Collect complete metrics
metrics = collector.collect_complete_metrics(
    response=response,
    ground_truth=ground_truth,
    label=1,  # deceptive
    scenario="poker",
    latency=latency,
    token_uncertainties=token_uncertainties,
    tokens=tokenized_response
)

# Save all metrics
collector.save_metrics("advanced_metrics.json")
```

## 🎯 Complete Workflow Example

Here's a complete example of collecting all data:

```python
from deception_circuits import (
    DatasetIntegrationPipeline,
    ReasoningTraceCollector,
    AdvancedMetricsCollector,
    PokerBadHandBluffing,
    PasswordLocking
)

# 1. Create standard dataset
dataset_pipeline = DatasetIntegrationPipeline()
dataset_pipeline.create_complete_dataset("standard_datasets.csv")

# 2. Create poker scenarios
poker_scenarios = PokerBadHandBluffing()
poker_data = poker_scenarios.create_multiple_scenarios(50)

# 3. Create password scenarios
password_scenarios = PasswordLocking()
password_data = password_scenarios.create_cross_domain_scenarios(
    questions=[("What is 2+2?", "4"), ("What is the capital of France?", "Paris")],
    passwords=["secret123", "password456"],
    domains=["math", "geography"]
)

# 4. Collect reasoning traces (for selected prompts)
# ... (see reasoning trace collection above)

# 5. Collect advanced metrics (during model inference)
# ... (see advanced metrics collection above)

# 6. Combine all data
import pandas as pd
all_data = pd.concat([
    pd.read_csv("standard_datasets.csv"),
    pd.DataFrame([p['truthful'] for p in poker_data] + [p['deceptive'] for p in poker_data]),
    # ... other data sources
])

all_data.to_csv("complete_deception_dataset.csv", index=False)
```

## 🔧 Required Dependencies

```bash
# Core dependencies
pip install torch numpy pandas scikit-learn

# Dataset dependencies
pip install datasets

# Model dependencies (choose based on your models)
pip install transformers  # For HuggingFace models
pip install openai       # For OpenAI API models

# Optional but recommended
pip install matplotlib seaborn plotly  # For visualizations
```

## 📝 Data Format Requirements

All deception data should follow this CSV format:

```csv
statement,response,label,scenario,ground_truth,metadata
"What is 2+2?","4",0,gsm8k_truthful,"4","{}"
"What is 2+2?","I'm not sure",1,gsm8k_sandbagging,"4","{}"
```

**Required columns:**
- `statement`: Input prompt/question
- `response`: Model's response
- `label`: 0 for truthful, 1 for deceptive
- `scenario`: Type of deception scenario

**Optional columns:**
- `ground_truth`: Correct answer (for contradiction analysis)
- `metadata`: JSON string with additional info

## 🚀 Quick Start Checklist

- [ ] Install dependencies: `pip install -r requirements.txt datasets`
- [ ] Collect game data (Mafia, Bullshit, Poker) - **YOU NEED TO DO THIS**
- [ ] Set up model access (base/aligned, different sizes) - **YOU NEED TO DO THIS**
- [ ] Run dataset integration pipeline (automatic)
- [ ] Collect reasoning traces (if using chain-of-thought)
- [ ] Collect advanced metrics (during inference)
- [ ] Combine all data into final dataset
- [ ] Run experiments using `DeceptionTrainingPipeline`

## 💡 Tips

1. **Start small**: Begin with a small subset of data to test the pipeline
2. **Use sample structures**: The game data loaders provide sample structures - use them as templates
3. **Automate where possible**: Use the dataset integration pipeline for standard datasets
4. **Save frequently**: Save intermediate results in case something goes wrong
5. **Document your data**: Keep notes on how data was collected for reproducibility

## ❓ Common Questions

**Q: Do I need to collect all game data manually?**  
A: Yes, for Mafia, Bullshit, and Poker games, you need to run actual games with your model and record the data. The framework provides the structure and loaders.

**Q: Can I use pre-existing game datasets?**  
A: If you find Mafiascum or other game datasets, you can adapt them to the required format using the data loaders.

**Q: How much data do I need?**  
A: Start with 50-100 examples per scenario. You can always add more later.

**Q: Do I need all model sizes for comparison?**  
A: No, but having at least 2-3 different sizes helps test generalization.

---

**Remember**: The framework handles all the analysis - you just need to collect the raw data! 🎉

