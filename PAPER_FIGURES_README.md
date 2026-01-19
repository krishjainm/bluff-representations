# Paper Figures Directory

This directory contains all publication-ready figures and tables for your research paper.

## 📁 Directory Structure

```
paper_figures/
├── Figures (PNG, PDF, SVG, EPS)
│   ├── layer_performance.png/pdf
│   ├── roc_curves.png/pdf
│   ├── cross_scenario_generalization.png/pdf
│   └── ...
│
├── Tables (CSV, LaTeX)
│   ├── layer_performance_metrics.csv
│   ├── layer_performance_metrics.tex
│   ├── best_layer_summary.csv
│   └── cross_scenario_generalization.csv
│
├── Index
│   └── FIGURE_INDEX.md  (Complete list of all figures)
│
└── Tracking
    └── visualization_tracking.json  (Experiment history)
```

## 🚀 Quick Start

### Generate All Figures

```bash
# Option 1: Generate from existing results
python generate_paper_figures.py --results results/experiment_results.json

# Option 2: Run new experiment and generate figures
python generate_paper_figures.py --data your_data.csv

# Option 3: Custom output directory
python generate_paper_figures.py --results results.json --output-dir my_figures
```

### In Python

```python
from deception_circuits import PaperVisualizationSystem
import json

# Load results
with open('results/experiment_results.json', 'r') as f:
    results = json.load(f)

# Generate all figures
viz_system = PaperVisualizationSystem(output_dir="paper_figures")
all_figures = viz_system.generate_all_paper_figures(
    results,
    experiment_name="main_experiment"
)

# Figures are now in paper_figures/ directory
```

## 📊 Available Figures

### 1. Layer Performance (`layer_performance`)
- **Formats**: PNG, PDF, SVG, EPS
- **Shows**: AUC and Accuracy across all layers
- **Use in paper**: Main results figure
- **File**: `layer_performance.pdf`

### 2. ROC Curves (`roc_curves`)
- **Formats**: PNG, PDF
- **Shows**: ROC curves for top-performing layers
- **Use in paper**: Discriminative power analysis
- **File**: `roc_curves.pdf`

### 3. Cross-Scenario Generalization (`cross_scenario_generalization`)
- **Formats**: PNG, PDF
- **Shows**: Generalization matrix across scenarios
- **Use in paper**: Generalization analysis
- **File**: `cross_scenario_generalization.pdf`

## 📋 Available Tables

### 1. Layer Performance Metrics
- **CSV**: `layer_performance_metrics.csv` (for analysis)
- **LaTeX**: `layer_performance_metrics.tex` (for paper)
- **Contains**: AUC, Accuracy, Precision, Recall, F1 for each layer

### 2. Best Layer Summary
- **CSV**: `best_layer_summary.csv`
- **Contains**: Summary of best performing layer

### 3. Cross-Scenario Generalization
- **CSV**: `cross_scenario_generalization.csv`
- **Contains**: Generalization matrix data

## 📝 Using in Your Paper

### LaTeX

```latex
% Include figure
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{paper_figures/layer_performance.pdf}
    \caption{Deception detection performance across network layers. 
             (A) AUC scores. (B) Classification accuracy. 
             Error bars show 95\% confidence intervals.}
    \label{fig:layer_performance}
\end{figure}

% Include table
\input{paper_figures/layer_performance_metrics.tex}
```

### Word

1. Insert > Picture > From File
2. Select `paper_figures/layer_performance.pdf`
3. Set text wrapping to "In Line with Text"
4. Add caption via References > Insert Caption

## 🔄 Updating Figures

Figures automatically update when you add new experiments:

```python
# First experiment
results_1 = pipeline.run_full_experiment("data1.csv")
viz_system.track_experiment("exp1", results_1)
figures_1 = viz_system.generate_all_paper_figures(results_1)

# Second experiment (figures update with confidence intervals)
results_2 = pipeline.run_full_experiment("data2.csv")
viz_system.track_experiment("exp2", results_2)
figures_2 = viz_system.generate_all_paper_figures(results_2)
```

## 📐 Figure Specifications

- **Resolution**: 300 DPI (print quality)
- **Formats**: PNG (presentations), PDF (papers), SVG (scalable), EPS (LaTeX)
- **Font**: Times New Roman (LaTeX-compatible)
- **Size**: Standard (6.4" x 4.8"), Wide (10" x 4"), Tall (6.4" x 8")

## 📂 File Locations

All figures are saved in the `paper_figures/` directory by default.

To change the location:
```python
viz_system = PaperVisualizationSystem(output_dir="my_custom_directory")
```

## ✅ Checklist

Before submitting your paper:

- [ ] All figures generated (check `FIGURE_INDEX.md`)
- [ ] Figures are high-resolution (300+ DPI)
- [ ] PDF versions available for paper
- [ ] PNG versions available for presentations
- [ ] Tables exported (CSV and LaTeX)
- [ ] Figures have clear labels and legends
- [ ] Error bars/confidence intervals included
- [ ] Baselines (chance/random) shown where appropriate

## 🆘 Troubleshooting

### Figures not generating?
- Check that results dictionary has required keys
- Ensure `paper_figures/` directory is writable
- Check console for error messages

### Missing confidence intervals?
- Run multiple experiments and track them
- Confidence intervals appear when 2+ experiments tracked

### Low resolution?
- Default is 300 DPI (check `savefig.dpi` in code)
- PDF format preserves vector quality

## 📚 More Information

- See `PAPER_VISUALIZATIONS_GUIDE.md` for detailed usage
- See `deception_circuits/paper_visualizations.py` for source code
- See `generate_paper_figures.py` for script usage

