"""
Simple script to generate and show graphs.
Generates sample graphs immediately so you can see them.
"""

import json
from pathlib import Path
import torch
import pandas as pd
import numpy as np

from deception_circuits import PaperVisualizationSystem

# Create mock results with realistic data
print("Creating sample results data...")

num_layers = 24
results = {
    'probe_results': {
        'layer_results': {
            f'layer_{i}': {
                'layer_idx': i,
                'final_auc': 0.5 + (i / num_layers) * 0.4 + np.random.normal(0, 0.03),
                'final_accuracy': 0.5 + (i / num_layers) * 0.3 + np.random.normal(0, 0.03)
            }
            for i in range(num_layers)
        },
        'test_results': {
            f'layer_{i}': {
                'layer_idx': i,
                'auc': max(0.5, min(1.0, 0.5 + (i / num_layers) * 0.4 + np.random.normal(0, 0.03))),
                'accuracy': max(0.5, min(1.0, 0.5 + (i / num_layers) * 0.3 + np.random.normal(0, 0.03))),
                'precision': max(0.4, min(1.0, 0.5 + np.random.normal(0, 0.1))),
                'recall': max(0.4, min(1.0, 0.5 + np.random.normal(0, 0.1))),
                'f1': max(0.4, min(1.0, 0.5 + np.random.normal(0, 0.1)))
            }
            for i in range(num_layers)
        }
    },
    'analysis_results': {
        'best_probe_layer': f'layer_{num_layers-3}',
        'best_probe_auc': 0.85,
        'best_probe_accuracy': 0.78,
        'layer_metrics': [
            {
                'layer_idx': i,
                'auc': max(0.5, min(1.0, 0.5 + (i / num_layers) * 0.4 + np.random.normal(0, 0.03))),
                'accuracy': max(0.5, min(1.0, 0.5 + (i / num_layers) * 0.3 + np.random.normal(0, 0.03))),
                'precision': max(0.4, min(1.0, 0.5 + np.random.normal(0, 0.1))),
                'recall': max(0.4, min(1.0, 0.5 + np.random.normal(0, 0.1))),
                'f1': max(0.4, min(1.0, 0.5 + np.random.normal(0, 0.1)))
            }
            for i in range(num_layers)
        ]
    },
    'cross_scenario_results': {
        'generalization_matrix': [
            [0.85, 0.72, 0.68, 0.75],
            [0.70, 0.88, 0.65, 0.71],
            [0.68, 0.64, 0.82, 0.69],
            [0.74, 0.70, 0.67, 0.86]
        ],
        'scenarios': ['poker', 'sandbagging', 'roleplay', 'general']
    }
}

print("Generating graphs...")
viz_system = PaperVisualizationSystem(output_dir="paper_figures")
all_figures = viz_system.generate_all_paper_figures(
    results,
    experiment_name="sample"
)

print("\n" + "="*60)
print("GRAPHS GENERATED SUCCESSFULLY!")
print("="*60)

paper_dir = Path("paper_figures")
print(f"\nDirectory: {paper_dir.absolute()}")
print(f"\nGenerated files:")

for fig_name, formats in all_figures.items():
    print(f"\n{fig_name}:")
    for fmt, path in formats.items():
        if path.exists():
            size = path.stat().st_size / 1024
            print(f"  {fmt.upper()}: {path.name} ({size:.1f} KB)")

print("\n" + "="*60)
print("To view graphs:")
print(f"1. Open folder: {paper_dir.absolute()}")
print("2. Double-click any .png or .pdf file")
print("="*60)

