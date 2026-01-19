"""
Generate all paper figures for deception circuit research.

This script generates all publication-ready figures and saves them to
a dedicated directory structure for easy access when writing your paper.

Usage:
    python generate_paper_figures.py

Or with custom results:
    python generate_paper_figures.py --results results/experiment_results.json
"""

import argparse
import json
from pathlib import Path
import sys

from deception_circuits import (
    DeceptionTrainingPipeline,
    PaperVisualizationSystem,
    CircuitAnalyzer
)
import pandas as pd


def load_results(results_path: Path) -> dict:
    """Load experiment results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def generate_all_figures(results: dict, 
                         output_dir: str = "paper_figures",
                         experiment_name: str = "experiment") -> dict:
    """
    Generate all paper figures and save to dedicated directory.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save all figures
        experiment_name: Name for tracking this experiment
        
    Returns:
        Dictionary mapping figure names to file paths
    """
    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)
    
    # Initialize visualization system
    viz_system = PaperVisualizationSystem(output_dir=output_dir)
    
    # Track experiment for incremental updates
    viz_system.track_experiment(experiment_name, results)
    
    print(f"\n📊 Generating figures...")
    print(f"📁 Output directory: {output_dir}")
    
    # Generate all standard figures
    all_figures = viz_system.generate_all_paper_figures(
        results,
        experiment_name=experiment_name
    )
    
    # Print summary
    print("\n✅ Generated Figures:")
    print("-" * 60)
    for fig_name, formats in all_figures.items():
        print(f"\n📈 {fig_name.replace('_', ' ').title()}:")
        for fmt, path in formats.items():
            print(f"   • {fmt.upper()}: {path}")
    
    # Generate summary tables
    print("\n📋 Generating summary tables...")
    summary_tables = generate_summary_tables(results, output_dir)
    
    print("\n✅ Generated Tables:")
    print("-" * 60)
    for table_name, path in summary_tables.items():
        print(f"   • {table_name}: {path}")
    
    # Create index file
    create_figure_index(all_figures, summary_tables, output_dir)
    
    print("\n" + "=" * 60)
    print(f"✅ All figures saved to: {Path(output_dir).absolute()}")
    print("=" * 60)
    
    return all_figures


def generate_summary_tables(results: dict, output_dir: Path) -> dict:
    """
    Generate summary tables in CSV and LaTeX formats.
    
    Args:
        results: Experiment results
        output_dir: Directory to save tables
        
    Returns:
        Dictionary mapping table names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tables = {}
    
    # Analyze results
    analyzer = CircuitAnalyzer(results)
    
    # 1. Layer performance table
    probe_analysis = analyzer.analyze_probe_performance()
    if 'layer_metrics' in probe_analysis:
        layer_df = pd.DataFrame(probe_analysis['layer_metrics'])
        
        # Save as CSV
        csv_path = output_dir / "layer_performance_metrics.csv"
        layer_df.to_csv(csv_path, index=False)
        tables['layer_performance_csv'] = csv_path
        
        # Save as LaTeX
        latex_path = output_dir / "layer_performance_metrics.tex"
        layer_df.to_latex(
            latex_path,
            index=False,
            float_format="%.3f",
            caption="Layer-wise deception detection performance metrics",
            label="tab:layer_performance"
        )
        tables['layer_performance_latex'] = latex_path
    
    # 2. Best layer summary
    if 'best_layer' in probe_analysis:
        best_layer = probe_analysis['best_layer']
        summary_data = {
            'Metric': ['Best Layer', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1'],
            'Value': [
                best_layer.get('layer', 'N/A'),
                f"{best_layer.get('auc', 0):.3f}",
                f"{best_layer.get('accuracy', 0):.3f}",
                f"{best_layer.get('precision', 0):.3f}",
                f"{best_layer.get('recall', 0):.3f}",
                f"{best_layer.get('f1', 0):.3f}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        
        csv_path = output_dir / "best_layer_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        tables['best_layer_summary'] = csv_path
    
    # 3. Cross-scenario results (if available)
    if 'cross_scenario_results' in results:
        cross_results = results['cross_scenario_results']
        if 'generalization_matrix' in cross_results:
            scenarios = cross_results.get('scenarios', [])
            matrix = cross_results['generalization_matrix']
            
            gen_df = pd.DataFrame(
                matrix,
                index=scenarios,
                columns=scenarios
            )
            
            csv_path = output_dir / "cross_scenario_generalization.csv"
            gen_df.to_csv(csv_path)
            tables['cross_scenario_csv'] = csv_path
    
    return tables


def create_figure_index(all_figures: dict, summary_tables: dict, output_dir: Path):
    """Create an index file listing all generated figures and tables."""
    output_dir = Path(output_dir)
    
    index_path = output_dir / "FIGURE_INDEX.md"
    
    with open(index_path, 'w') as f:
        f.write("# Paper Figures Index\n\n")
        f.write("This file lists all generated figures and tables for your paper.\n\n")
        
        f.write("## 📊 Figures\n\n")
        for fig_name, formats in all_figures.items():
            f.write(f"### {fig_name.replace('_', ' ').title()}\n\n")
            for fmt, path in formats.items():
                f.write(f"- **{fmt.upper()}**: `{path.name}`\n")
            f.write("\n")
        
        f.write("## 📋 Tables\n\n")
        for table_name, path in summary_tables.items():
            f.write(f"- **{table_name.replace('_', ' ').title()}**: `{path.name}`\n")
        
        f.write("\n## 📝 Usage in LaTeX\n\n")
        f.write("```latex\n")
        f.write("% Example: Include layer performance figure\n")
        f.write("\\begin{figure}[h]\n")
        f.write("    \\centering\n")
        f.write("    \\includegraphics[width=0.8\\textwidth]{paper_figures/layer_performance.pdf}\n")
        f.write("    \\caption{Deception detection performance across layers.}\n")
        f.write("    \\label{fig:layer_performance}\n")
        f.write("\\end{figure}\n")
        f.write("```\n")
    
    print(f"\n📄 Created index: {index_path}")


def main():
    """Main function to generate all paper figures."""
    parser = argparse.ArgumentParser(
        description="Generate all paper figures for deception circuit research"
    )
    parser.add_argument(
        '--results',
        type=str,
        help='Path to experiment results JSON file (if not provided, will run new experiment)'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to CSV data file (required if --results not provided)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='paper_figures',
        help='Directory to save all figures (default: paper_figures)'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='experiment',
        help='Name for this experiment (default: experiment)'
    )
    
    args = parser.parse_args()
    
    # Load or generate results
    if args.results:
        print(f"📂 Loading results from: {args.results}")
        results = load_results(Path(args.results))
    elif args.data:
        print(f"🔬 Running new experiment with data: {args.data}")
        pipeline = DeceptionTrainingPipeline(device="cpu", output_dir="results")
        results = pipeline.run_full_experiment(
            csv_path=args.data,
            max_samples=1000
        )
    else:
        print("❌ Error: Must provide either --results or --data")
        parser.print_help()
        sys.exit(1)
    
    # Generate all figures
    generate_all_figures(
        results=results,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )


if __name__ == "__main__":
    main()

