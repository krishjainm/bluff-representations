# Data Storage Locations Guide

This document describes where all data is stored in the Deception Circuits Research Infrastructure project.

## 📁 **Data Storage Overview**

Data in this project is stored in several locations:

1. **HuggingFace Datasets Cache** (external, automatic)
2. **Project Data Directories** (local, user-created)
3. **Experiment Results** (local, auto-generated)
4. **Activations** (local, PyTorch tensors)

---

## 1. **HuggingFace Datasets Cache** 🌐

**Location**: `~/.cache/huggingface/datasets/` (or `HF_HOME` environment variable)

**What's stored here:**
- **GSM8K** dataset (`gsm8k/`)
- **TruthfulQA** dataset (`truthful_qa/`)
- **MMLU** dataset (`hendrycks_test/`, `hendrycks_train/`)
- **PokerBench** dataset (`RZ412___poker_bench/`)

**When it's downloaded:**
- Automatically on first use via `DatasetIntegrationPipeline`
- When calling `load_dataset()` from HuggingFace's `datasets` library

**How to manage:**
```bash
# View cache size
du -sh ~/.cache/huggingface/datasets/

# Clear cache (if needed)
rm -rf ~/.cache/huggingface/datasets/*
```

**Note**: This is managed by HuggingFace, not directly by this project. The framework just downloads datasets here automatically.

---

## 2. **Project Data Directories** 📂

These are directories **you create** or **the framework creates** for your experiments:

### **`sample_data/`** (Example/Default)
- **Created by**: `example_usage.py`
- **Contains**:
  - `deception_data.csv` - Sample deception dataset
  - `activations.pt` - Sample activation tensors

### **`production_data/`** (Production Example)
- **Created by**: `production_example.py`
- **Contains**:
  - `deception_data.csv` - Generated deception dataset
  - `synthetic_deception_data.csv` - Fallback synthetic data
  - `activations/` - Activation tensors directory
  - `activation_mapping.json` - Mapping file

### **Custom Data Directories**
You can create your own data directories anywhere. Common patterns:
- `data/` - General data directory
- `datasets/` - Multiple datasets
- `experiments/data/` - Per-experiment data

**Usage:**
```python
# Create custom data directory
data_dir = Path("my_data")
data_dir.mkdir(exist_ok=True)

# Save CSV
df.to_csv(data_dir / "my_dataset.csv")

# Save activations
torch.save(activations, data_dir / "activations.pt")
```

---

## 3. **Experiment Results** 📊

**Default location**: `deception_results/` or `output_dir` specified in pipeline

**Created by**: `DeceptionTrainingPipeline`

**Contains**:
```
deception_results/
├── experiment_results.json     # Complete experiment results
├── models/                     # Trained probe and autoencoder models
│   ├── probe_layer_0.pt
│   ├── probe_layer_1.pt
│   ├── autoencoder_layer_0.pt
│   └── ...
├── layer_performance.png       # Visualization plots
├── feature_importance.png
└── interactive_dashboard.html  # Interactive Plotly dashboard
```

**Other result directories:**
- `quick_demo_results/` - From `quick_demo()`
- `production_results/` - From `production_example.py`
- `interpretability_report/` - From `InterpretabilitySuite`
- `gpt4o_deception_dataset/` - From `create_gpt4o_deception_dataset()`

**Usage:**
```python
# Specify custom output directory
pipeline = DeceptionTrainingPipeline(
    device="cpu",
    output_dir="my_experiment_results"  # Custom location
)
results = pipeline.run_full_experiment("data.csv")
```

---

## 4. **Activation Storage** 💾

Activations are stored as **PyTorch tensor files** (`.pt` format).

**Storage patterns:**

### **Single File**
```python
# Save all activations in one file
torch.save(activations, "activations.pt")  # Shape: (num_samples, num_layers, hidden_dim)
```

### **Directory Structure** (recommended for large datasets)
```
activations/
├── sample_0.pt    # Activations for sample 0
├── sample_1.pt    # Activations for sample 1
├── ...
└── activation_mapping.json  # Maps sample IDs to file paths
```

**Activation dimensions:**
- Shape: `(num_samples, num_layers, hidden_dim)`
- Example: `(100, 24, 768)` = 100 samples, 24 layers, 768 hidden dimensions

**Loading activations:**
```python
from deception_circuits import DeceptionDataLoader

loader = DeceptionDataLoader(device="cpu")
data = loader.load_csv("data.csv", activation_dir="activations/")
# Activations are automatically loaded and matched to CSV rows
```

---

## 5. **Configuration and Cache** ⚙️

**Production cache** (if using production features):
- `cache/` - Default cache directory (configurable via `ConfigManager`)

**Log files** (if using production logging):
- `logs/` - Application logs

**Checkpoints** (if using checkpointing):
- `checkpoints/` - Model checkpoints during training

---

## 📋 **Data Flow Summary**

### **Dataset Collection Flow:**

```
HuggingFace Datasets (auto-downloaded)
  ↓
~/.cache/huggingface/datasets/
  ↓
DatasetIntegrationPipeline processes
  ↓
Saves to: your_output_path.csv (e.g., "integrated_dataset.csv")
```

### **Experiment Flow:**

```
Input: CSV file (e.g., "data.csv")
  ↓
Activation Directory: "activations/" (optional)
  ↓
DeceptionTrainingPipeline processes
  ↓
Results saved to: output_dir/ (default: "deception_results/")
```

---

## 🔍 **Finding Your Data**

### **Where is my CSV data?**
- Check your project root directory
- Look for files created by `DatasetIntegrationPipeline.create_complete_dataset()`
- Check `sample_data/`, `production_data/`, or your custom directories

### **Where are activations stored?**
- Check the `activation_dir` you specified when loading data
- Look in `sample_data/activations.pt` for sample data
- Check `production_data/activations/` for production examples

### **Where are results saved?**
- Default: `deception_results/`
- Or check the `output_dir` you specified in `DeceptionTrainingPipeline()`

### **Where are HuggingFace datasets?**
- Default: `~/.cache/huggingface/datasets/`
- Or check `HF_HOME` environment variable

---

## 💡 **Best Practices**

1. **Keep data organized:**
   ```python
   project_root/
   ├── data/              # Input datasets
   ├── activations/       # Activation tensors
   ├── results/           # Experiment results
   └── cache/             # Temporary cache
   ```

2. **Use relative paths for portability:**
   ```python
   from pathlib import Path
   data_dir = Path("data")  # Not "/absolute/path/data"
   ```

3. **Version control considerations:**
   - Add `*.pt`, `*.csv`, `results/`, `cache/` to `.gitignore`
   - Commit only code and configuration
   - Data should be downloaded/generated, not committed

4. **Large datasets:**
   - Use directory structure for activations (not single large file)
   - Consider compression for long-term storage
   - Use `activation_mapping.json` to track files

---

## 🛠️ **Managing Data**

### **View data directory structure:**
```bash
# Linux/Mac
tree -L 3 data/

# Windows (PowerShell)
Get-ChildItem -Recurse -Depth 2 data/
```

### **Check disk usage:**
```bash
# Linux/Mac
du -sh data/ activations/ results/

# Windows
Get-ChildItem -Recurse data/ | Measure-Object -Property Length -Sum
```

### **Clean up (carefully!):**
```bash
# Remove old results (keeps data)
rm -rf results/old_experiment/

# Clear HuggingFace cache (will re-download)
rm -rf ~/.cache/huggingface/datasets/

# Remove generated CSV (recreate if needed)
rm integrated_dataset.csv
```

---

## 📝 **Summary**

| Data Type | Default Location | How It's Created |
|-----------|-----------------|------------------|
| **HuggingFace Datasets** | `~/.cache/huggingface/datasets/` | Auto-downloaded on first use |
| **CSV Datasets** | Project root (you specify) | `DatasetIntegrationPipeline` or manual |
| **Activations (.pt)** | `sample_data/` or custom `activation_dir/` | Manual save or model extraction |
| **Experiment Results** | `deception_results/` or `output_dir/` | `DeceptionTrainingPipeline` |
| **Logs/Cache** | `logs/`, `cache/` | Production features |

**Key takeaway**: Most data storage locations are **configurable** and **project-specific**. The only automatic external storage is the HuggingFace datasets cache.

