# Paper Visualizations Guide

This guide describes the comprehensive visualization system for generating publication-ready figures for your research paper.

## 📊 **Overview**

The `PaperVisualizationSystem` generates all figures needed for your deception circuits research paper, with:

- ✅ **Publication-ready quality** (300+ DPI, LaTeX-compatible)
- ✅ **Multiple export formats** (PNG, PDF, SVG, EPS)
- ✅ **Automatic data tracking** (updates as new data is added)
- ✅ **Statistical analysis** (error bars, confidence intervals)
- ✅ **Summary tables** and metrics export

## 🚀 **Quick Start**

```python
from deception_circuits import DeceptionTrainingPipeline, PaperVisualizationSystem

# Run your experiment
pipeline = DeceptionTrainingPipeline(device="cpu", output_dir="results")
results = pipeline.run_full_experiment("data.csv")

# Generate all paper figures
viz_system = PaperVisualizationSystem(output_dir="paper_figures")
all_figures = viz_system.generate_all_paper_figures(
    results, 
    experiment_name="experiment_1"
)

# Access saved files
print(f"Layer performance: {all_figures['layer_performance']['png']}")
print(f"ROC curves: {all_figures['roc_curves']['pdf']}")
```

## 📈 **Available Visualizations**

### 1. **Layer Performance with Error Bars**

**Purpose**: Show where deception signals are strongest across network layers

**Figure**: Line plot with AUC and Accuracy across layers, with confidence intervals

**Usage**:
```python
figures = viz_system.plot_layer_performance_with_errors(
    results,
    include_ci=True,
    confidence_level=0.95,
    save_formats=['png', 'pdf', 'svg']
)
```

**What it shows**:
- AUC scores per layer (best metric)
- Accuracy per layer
- Confidence intervals (if multiple experiments)
- Best performing layer highlighted
- Chance baseline (AUC=0.5)

---

### 2. **ROC Curves by Layer**

**Purpose**: Show discriminative power of top-performing layers

**Figure**: ROC curves for top N layers, showing true positive vs false positive rates

**Usage**:
```python
figures = viz_system.plot_roc_curves_by_layer(
    results,
    top_n_layers=5,
    save_formats=['png', 'pdf']
)
```

**What it shows**:
- ROC curves for top-performing layers
- AUC values in legend
- Random classifier baseline (diagonal line)
- Area under curve represents detection quality

---

### 3. **Cross-Scenario Generalization Matrix**

**Purpose**: Show whether deception circuits generalize across scenarios

**Figure**: Heatmap showing performance when training on one scenario, testing on another

**Usage**:
```python
figures = viz_system.plot_cross_scenario_generalization(
    results,
    save_formats=['png', 'pdf']
)
```

**What it shows**:
- Training scenario (rows) vs Test scenario (columns)
- Performance (AUC) in each cell
- Color-coded heatmap (green=good, red=poor)
- Indicates if circuits are scenario-specific or general

---

## 📋 **Essential Figures for Your Paper**

### **Figure 1: Layer-wise Performance**
- **Type**: `plot_layer_performance_with_errors`
- **Shows**: Where deception signals emerge in the network
- **Paper section**: Methods/Results
- **Caption**: "Deception detection performance across network layers. (A) AUC scores. (B) Classification accuracy. Error bars show 95% confidence intervals. Best performing layer (Layer X) is highlighted."

### **Figure 2: ROC Curves**
- **Type**: `plot_roc_curves_by_layer`
- **Shows**: Discriminative power of top layers
- **Paper section**: Results
- **Caption**: "ROC curves for top 5 performing layers. AUC values indicate strong deception detection capability above chance (dashed line)."

### **Figure 3: Cross-Scenario Generalization**
- **Type**: `plot_cross_scenario_generalization`
- **Shows**: Whether circuits generalize across scenarios
- **Paper section**: Results/Generalization
- **Caption**: "Cross-scenario generalization matrix. Performance when training on row scenario and testing on column scenario. Diagonal shows within-scenario performance."

### **Figure 4: Feature Importance** *(to be added)*
- **Type**: Bar plot or heatmap
- **Shows**: Which features/neurons are most important
- **Paper section**: Circuit Analysis

### **Figure 5: Activation Distributions** *(to be added)*
- **Type**: Violin plot or histogram
- **Shows**: Activation distributions for truthful vs deceptive
- **Paper section**: Circuit Analysis

### **Figure 6: Training Progression** *(to be added)*
- **Type**: Line plot over epochs
- **Shows**: How performance improves during training
- **Paper section**: Methods

### **Figure 7: Statistical Comparisons** *(to be added)*
- **Type**: Box plots with significance tests
- **Shows**: Statistical differences between conditions
- **Paper section**: Results

### **Figure 8: Model Size Comparison** *(to be added)*
- **Type**: Bar plot or line plot
- **Shows**: Performance across different model sizes
- **Paper section**: Ablation Studies

---

## 🔄 **Incremental Updates**

The system tracks experiments automatically:

```python
# First experiment
results_1 = pipeline.run_full_experiment("data1.csv")
viz_system.track_experiment("exp1", results_1)
figures_1 = viz_system.generate_all_paper_figures(results_1)

# Second experiment (updates automatically)
results_2 = pipeline.run_full_experiment("data2.csv")
viz_system.track_experiment("exp2", results_2)
figures_2 = viz_system.generate_all_paper_figures(results_2)

# Figures now include confidence intervals across experiments!
```

---

## 📐 **Customization**

### Change Figure Size

```python
viz_system = PaperVisualizationSystem(
    output_dir="paper_figures",
    figsize_standard=(8, 6),  # Wider for papers
    figsize_wide=(12, 4),      # Multi-panel
    figsize_tall=(6, 10)       # Vertical layout
)
```

### Export Formats

```python
# Save in multiple formats
figures = viz_system.plot_layer_performance_with_errors(
    results,
    save_formats=['png', 'pdf', 'svg', 'eps']  # All formats
)
```

### Style Customization

The system uses publication-ready defaults:
- **Font**: Times New Roman (serif, LaTeX-compatible)
- **DPI**: 300 (print quality)
- **Style**: Seaborn whitegrid (clean, professional)

---

## 📊 **Summary Tables and Metrics**

Generate summary tables for your paper:

```python
# Export metrics to CSV/LaTeX
from deception_circuits.analysis import CircuitAnalyzer

analyzer = CircuitAnalyzer(results)
probe_analysis = analyzer.analyze_probe_performance()

# Convert to DataFrame for tables
import pandas as pd
df = pd.DataFrame(probe_analysis['layer_metrics'])

# Save as CSV (for Excel/analysis)
df.to_csv("paper_figures/layer_metrics.csv", index=False)

# Save as LaTeX table (for paper)
df.to_latex("paper_figures/layer_metrics.tex", index=False, float_format="%.3f")
```

---

## 🔍 **Statistical Analysis**

The system includes statistical analysis:

```python
# Confidence intervals (when multiple experiments tracked)
figures = viz_system.plot_layer_performance_with_errors(
    results,
    include_ci=True,
    confidence_level=0.95  # 95% confidence intervals
)

# Statistical significance testing (coming soon)
# - t-tests between layers
# - Mann-Whitney U tests
# - Bonferroni corrections
```

---

## 📁 **File Organization**

All figures are saved in organized structure:

```
paper_figures/
├── layer_performance.png
├── layer_performance.pdf
├── roc_curves.png
├── roc_curves.pdf
├── cross_scenario_generalization.png
├── cross_scenario_generalization.pdf
├── visualization_tracking.json  # Experiment history
└── layer_metrics.csv            # Summary tables
```

---

## 🎯 **Complete Example Workflow**

```python
from deception_circuits import (
    DeceptionTrainingPipeline,
    PaperVisualizationSystem
)
from pathlib import Path

# 1. Run experiments
pipeline = DeceptionTrainingPipeline(device="cpu", output_dir="results")

# Experiment 1: Baseline
results_baseline = pipeline.run_full_experiment("baseline_data.csv")

# Experiment 2: Extended dataset
results_extended = pipeline.run_full_experiment("extended_data.csv")

# 2. Initialize visualization system
viz_system = PaperVisualizationSystem(output_dir="paper_figures")

# 3. Track and generate figures
for i, (name, results) in enumerate([
    ("baseline", results_baseline),
    ("extended", results_extended)
], 1):
    viz_system.track_experiment(name, results)
    figures = viz_system.generate_all_paper_figures(results, experiment_name=name)
    print(f"Experiment {i} figures saved: {list(figures.keys())}")

# 4. Final consolidated figures (with all experiments)
final_figures = viz_system.generate_all_paper_figures(
    results_extended,  # Use most complete dataset
    experiment_name="final"
)

print("All paper figures generated!")
print(f"Saved to: {viz_system.output_dir}")
```

---

## 📝 **Paper Figure Checklist**

Use this checklist when preparing figures for your paper:

- [ ] **Layer Performance** - Shows where signals are strongest
- [ ] **ROC Curves** - Shows discriminative power
- [ ] **Cross-Scenario Generalization** - Shows generalization
- [ ] **Feature Importance** - Shows which features matter
- [ ] **Activation Distributions** - Shows activation differences
- [ ] **Training Progression** - Shows learning curves
- [ ] **Statistical Comparisons** - Shows significance tests
- [ ] **Model Size Comparison** - Shows ablation results
- [ ] **Summary Tables** - Key metrics in tabular format
- [ ] **High-resolution exports** (300+ DPI)
- [ ] **Multiple formats** (PNG, PDF)
- [ ] **LaTeX-compatible** fonts and formatting

---

## 🛠️ **Extending the System**

To add new visualizations:

```python
class PaperVisualizationSystem:
    # ... existing methods ...
    
    def plot_custom_visualization(self, results: Dict, **kwargs):
        """Add your custom visualization here."""
        fig, ax = plt.subplots(figsize=self.figsize_standard)
        
        # Your plotting code here
        
        # Save in multiple formats
        saved_files = {}
        for fmt in ['png', 'pdf']:
            path = self.output_dir / f"custom_plot.{fmt}"
            plt.savefig(path, format=fmt, dpi=300, bbox_inches='tight')
            saved_files[fmt] = path
        
        plt.close()
        return saved_files
```

---

## 💡 **Tips for Paper Figures**

1. **Consistency**: Use same color scheme across all figures
2. **Clarity**: Label all axes clearly, include units
3. **Legends**: Always include legends, place outside plot area
4. **Error Bars**: Always show error bars/confidence intervals
5. **Baselines**: Include chance/random baselines for comparison
6. **Font Size**: Ensure text is readable at print size
7. **Resolution**: Use 300+ DPI for print quality
8. **Formats**: Provide both PNG (presentations) and PDF (papers)

---

## 📚 **Integration with Paper Writing**

### LaTeX Integration

```latex
% In your LaTeX paper
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{paper_figures/layer_performance.pdf}
    \caption{Deception detection performance across layers...}
    \label{fig:layer_performance}
\end{figure}
```

### Word Integration

1. Insert PDF figures for best quality
2. Use "Insert > Picture > From File"
3. Set text wrapping to "In Line with Text"
4. Maintain aspect ratio

---

## ✅ **Summary**

The `PaperVisualizationSystem` provides:

✅ All essential figures for your paper
✅ Automatic updates as data is added
✅ Publication-ready quality (300+ DPI)
✅ Multiple export formats
✅ Statistical analysis and error bars
✅ Incremental data tracking

**Next Steps:**
1. Run your experiments
2. Generate figures using `generate_all_paper_figures()`
3. Review figures in `paper_figures/` directory
4. Use PDF versions in your paper
5. Add captions and references

**For questions or extensions**, see the `paper_visualizations.py` source code or extend the `PaperVisualizationSystem` class with your custom methods.

