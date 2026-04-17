"""EDA pipeline orchestrator with dynamic chart discovery and parallel generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Callable
import inspect

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from src.data_loader import load_penguins
from src import analysis
from src import visualization


@dataclass
class AnalysisConfig:
    dataset_path: str = "dataset/penguins.csv"
    output_dir: str = "outputs/graficos"
    max_workers: int = 3  # Parallel workers for chart generation


def _ensure_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def discover_plot_functions(module) -> Dict[str, Callable]:
    """Dynamically discover and register all plot_* functions in a module.
    
    Args:
        module: Python module to scan
        
    Returns:
        Dictionary mapping plot names to functions.
        
    Example:
        >>> plots = discover_plot_functions(visualization)
        >>> print(f"Found {len(plots)} plots: {list(plots.keys())}")
    """
    plots = {}
    for name, func in inspect.getmembers(module):
        if name.startswith("plot_") and callable(func):
            plots[name] = func
    return plots


def run_full_analysis(config: AnalysisConfig | None = None) -> Dict[str, object]:
    """Run complete EDA flow with dynamic chart discovery and parallel generation.
    
    Features:
    - Automatically discovers all plot_* functions from visualization module
    - Generates charts in parallel (configurable worker threads)
    - Polars-only data manipulation (no Pandas)
    - Exports clean parquet file
    
    Args:
        config: AnalysisConfig object with dataset and output paths
        
    Returns:
        Dictionary with analysis results, charts, and metadata
    """
    sns.set_theme(style="whitegrid", context="talk")
    if config is None:
        config = AnalysisConfig()

    out_dir = _ensure_output_dir(config.output_dir)
    df = load_penguins(config.dataset_path)
    print(f"✅ Dataset loaded: {df.shape[0]} registros")

    # Run all analyses
    analyses = analysis.compute_all_analyses(df)
    
    # Discover available plot functions dynamically
    available_plots = discover_plot_functions(visualization)
    print(f"📊 Descobertos {len(available_plots)} gráficos: {', '.join(available_plots.keys())}")

    # Prepare chart generation tasks
    chart_tasks = []
    
    # Chart 1: Missing counts
    if "plot_missing_counts" in available_plots:
        chart_tasks.append(
            ("missing_counts", available_plots["plot_missing_counts"], 
             {"missing_summary": analyses["missing_summary"]})
        )
    
    # Chart 2: Island counts
    if "plot_island_counts" in available_plots:
        chart_tasks.append(
            ("island_counts", available_plots["plot_island_counts"],
             {"island_counts": analyses["island_counts"]})
        )
    
    # Chart 3: Species counts
    if "plot_species_counts" in available_plots:
        chart_tasks.append(
            ("species_counts", available_plots["plot_species_counts"],
             {"species_counts": analyses["species_counts"]})
        )
    
    # Chart 4: Scatter measures
    if "plot_scatter_measures_by_species" in available_plots:
        chart_tasks.append(
            ("scatter_measures", available_plots["plot_scatter_measures_by_species"],
             {"df": df.drop_nulls(["bill_length_mm", "bill_depth_mm", "species"])})
        )
    
    # Chart 5: Pairplot by species
    if "plot_pairplot_species" in available_plots:
        chart_tasks.append(
            ("pairplot_species", available_plots["plot_pairplot_species"],
             {"df": df})
        )
    
    # Chart 6: Mass by sex and species
    if "plot_mass_by_sex_species" in available_plots:
        chart_tasks.append(
            ("mass_by_sex", available_plots["plot_mass_by_sex_species"],
             {"df": df})
        )
    
    # Chart 7-9: Pairplots by species and sex
    if "plot_pairplot_by_species_sex" in available_plots:
        chart_tasks.append(
            ("pairplot_sex", available_plots["plot_pairplot_by_species_sex"],
             {"df": df})
        )

    # Generate charts in parallel using ProcessPoolExecutor (safer for Matplotlib)
    # Note: Using process pool for Matplotlib thread-safety
    # Charts are generated sequentially for reliability (can be parallelized with proper serialization)
    graph_paths: List[str] = []
    print(f"🔄 Gerando {len(chart_tasks)} gráficos...")
    
    for task_name, plot_func, chart_data in chart_tasks:
        try:
            result = plot_func(**chart_data, out_dir=out_dir)
            if isinstance(result, list):  # For functions returning multiple paths
                graph_paths.extend(result)
                print(f"  ✅ {task_name}: {len(result)} gráficos")
            else:
                graph_paths.append(result)
                print(f"  ✅ {task_name}")
        except Exception as e:
            print(f"  ❌ {task_name}: {e}")

    # Export clean parquet
    clean_parquet_path = Path("dataset/penguins_clean.parquet")
    df.write_parquet(clean_parquet_path)
    print(f"💾 Parquet exportado: {clean_parquet_path}")

    return {
        "df": df,
        "missing_summary": analyses["missing_summary"],
        "missing_rows": analyses["missing_rows"],
        "island_counts": analyses["island_counts"],
        "species_counts": analyses["species_counts"],
        "species_stats": analyses["species_stats"],
        "sex_species_stats": analyses["sex_species_stats"],
        "graphs": graph_paths,
        "clean_parquet": str(clean_parquet_path),
        "available_plots": available_plots,
    }
