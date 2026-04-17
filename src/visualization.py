"""Visualization functions using Matplotlib and Seaborn (Polars-only compatible)."""

from pathlib import Path
from typing import List, Callable

# Set non-interactive backend for thread safety
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import polars as pl
import seaborn as sns


# Cold palette inspired by ocean/ice tones.
PENGUIN_PALETTE = ["#1B4965", "#5FA8D3", "#62B6CB", "#CAE9FF", "#0B132B"]


def _save_fig(path: Path) -> str:
    """Save and close current figure."""
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(path)


def _autopct_with_counts(values: List[int]) -> Callable:
    """Create formatter for pie charts showing both % and absolute count."""
    total = sum(values)

    def _formatter(pct: float) -> str:
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}% ({count})"

    return _formatter


def plot_missing_counts(missing_summary: pl.DataFrame, out_dir: Path) -> str:
    """Create bar chart of missing values per column without Pandas."""
    plt.figure(figsize=(10, 5))
    data = missing_summary.to_dict(as_series=False)
    n_cols = len(data["column"])
    palette = sns.color_palette("Blues", n_colors=n_cols)
    
    sns.barplot(
        x=data["column"], 
        y=data["missing_count"], 
        hue=data["column"],
        palette=palette,
        legend=False
    )
    plt.title("Número de dados faltantes por coluna")
    plt.xlabel("Coluna")
    plt.ylabel("Quantidade de faltantes")
    plt.xticks(rotation=20)
    return _save_fig(out_dir / "grafico_01_numero_dados_faltantes_por_coluna.png")


def plot_island_counts(island_counts: pl.DataFrame, out_dir: Path) -> str:
    """Create pie chart of penguins per island without Pandas."""
    plt.figure(figsize=(8, 6))
    data = island_counts.to_dict(as_series=False)
    counts = data["count"]
    islands = data["island"]
    
    n_islands = len(islands)
    palette = sns.color_palette("Blues", n_colors=n_islands)
    
    wedges, _, _ = plt.pie(
        counts,
        labels=None,
        autopct=_autopct_with_counts(counts),
        startangle=90,
        colors=palette,
        pctdistance=0.7,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    plt.title("Distribuição de pinguins por ilha")
    legend_labels = [f"{name.title()} - {count}" for name, count in zip(islands, counts)]
    plt.legend(
        wedges,
        legend_labels,
        title="Ilhas",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    plt.axis("equal")
    return _save_fig(out_dir / "grafico_02_numero_de_pinguins_por_ilha.png")


def plot_species_counts(species_counts: pl.DataFrame, out_dir: Path) -> str:
    """Create pie chart of penguins per species without Pandas."""
    plt.figure(figsize=(8, 6))
    data = species_counts.to_dict(as_series=False)
    counts = data["count"]
    species = data["species"]
    
    n_species = len(species)
    palette = sns.color_palette("Blues", n_colors=n_species)
    
    wedges, _, _ = plt.pie(
        counts,
        labels=None,
        autopct=_autopct_with_counts(counts),
        startangle=90,
        colors=palette,
        pctdistance=0.7,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    plt.title("Distribuição de pinguins por espécie")
    legend_labels = [f"{name.capitalize()} - {count}" for name, count in zip(species, counts)]
    plt.legend(
        wedges,
        legend_labels,
        title="Espécies",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    plt.axis("equal")
    return _save_fig(out_dir / "grafico_03_numero_de_pinguins_por_especie.png")


def plot_scatter_measures_by_species(df: pl.DataFrame, out_dir: Path) -> str:
    """Create scatter plot of bill measurements by species without Pandas."""
    plt.figure(figsize=(9, 6))
    data = df.to_dict(as_series=False)
    
    sns.scatterplot(
        x=data["bill_length_mm"],
        y=data["bill_depth_mm"],
        hue=data["species"],
        palette=PENGUIN_PALETTE[:3],
        alpha=0.85,
    )
    plt.title("Comprimento vs profundidade do bico por espécie")
    plt.xlabel("Comprimento do bico (mm)")
    plt.ylabel("Profundidade do bico (mm)")
    return _save_fig(out_dir / "grafico_04_relacao_medidas_por_especie.png")


def plot_pairplot_species(df: pl.DataFrame, out_dir: Path) -> str:
    """Create pairplot-style visualization using matplotlib puro."""
    cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    clean_df = df.select(["species"] + cols).drop_nulls()
    data = clean_df.to_dict(as_series=False)
    
    n_cols = len(cols)
    fig, axes = plt.subplots(n_cols, n_cols, figsize=(12, 12))
    
    species_list = list(set(data["species"]))
    species_colors = {sp: PENGUIN_PALETTE[i % len(PENGUIN_PALETTE)] for i, sp in enumerate(species_list)}
    
    for i, col_y in enumerate(cols):
        for j, col_x in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                # Histogram on diagonal
                for species in species_list:
                    mask = [s == species for s in data["species"]]
                    values = [data[col_x][k] for k in range(len(data["species"])) if mask[k]]
                    ax.hist(values, alpha=0.5, label=species, color=species_colors[species], bins=10)
                if i == 0:
                    ax.legend(fontsize=8)
            else:
                # Scatter plot
                for species in species_list:
                    mask = [s == species for s in data["species"]]
                    x_vals = [data[col_x][k] for k in range(len(data["species"])) if mask[k]]
                    y_vals = [data[col_y][k] for k in range(len(data["species"])) if mask[k]]
                    ax.scatter(x_vals, y_vals, alpha=0.6, label=species, color=species_colors[species], s=30)
            
            if i == n_cols - 1:
                ax.set_xlabel(col_x, fontsize=9)
            else:
                ax.set_xticklabels([])
            
            if j == 0:
                ax.set_ylabel(col_y, fontsize=9)
            else:
                ax.set_yticklabels([])
    
    fig.suptitle("Relações entre medidas por espécie", fontsize=14)
    path = out_dir / "grafico_05_pairplot_medidas_por_especie.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_mass_by_sex_species(df: pl.DataFrame, out_dir: Path) -> str:
    """Create boxplot of mass by sex and species using matplotlib."""
    clean_df = df.filter(pl.col("sex").is_not_null()).select(["species", "sex", "body_mass_g"])
    data = clean_df.to_dict(as_series=False)
    
    species_list = ["adelie", "gentoo", "chinstrap"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    palette = sns.color_palette("Blues", n_colors=2)
    
    for idx, species in enumerate(species_list):
        ax = axes[idx]
        species_data = clean_df.filter(pl.col("species") == species)
        
        if species_data.height == 0:
            continue
        
        data_dict = species_data.to_dict(as_series=False)
        males = [data_dict["body_mass_g"][i] for i in range(len(data_dict["sex"])) if data_dict["sex"][i] == "male"]
        females = [data_dict["body_mass_g"][i] for i in range(len(data_dict["sex"])) if data_dict["sex"][i] == "female"]
        
        box_data = []
        labels = []
        if males:
            box_data.append(males)
            labels.append("male")
        if females:
            box_data.append(females)
            labels.append("female")
        
        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], palette[:len(box_data)]):
                patch.set_facecolor(color)
        
        ax.set_title(f"{species.capitalize()}", fontweight="bold")
        ax.set_ylabel("Massa corporal (g)")
        ax.set_xlabel("Sexo")
        ax.grid(True, alpha=0.3, axis="y")
    
    fig.suptitle("Massa corporal por sexo em cada espécie", fontsize=14)
    path = out_dir / "grafico_06_massa_por_sexo_em_cada_especie.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_pairplot_by_species_sex(df: pl.DataFrame, out_dir: Path) -> List[str]:
    """Create species-specific pairplots with sex distinction using matplotlib."""
    files: List[str] = []
    cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    
    species_order = ["adelie", "gentoo", "chinstrap"]
    palette = sns.color_palette("Blues", n_colors=2)
    
    for file_idx, species in enumerate(species_order, start=7):
        species_df = df.filter((pl.col("species") == species) & (pl.col("sex").is_not_null())).select(["sex"] + cols)
        
        if species_df.height == 0 or species_df.select("sex").n_unique() < 2:
            continue
        
        data_dict = species_df.to_dict(as_series=False)
        n_cols = len(cols)
        fig, axes = plt.subplots(n_cols, n_cols, figsize=(11, 11))
        
        sex_colors = {"male": palette[0], "female": palette[1]}
        
        for i, col_y in enumerate(cols):
            for j, col_x in enumerate(cols):
                ax = axes[i, j]
                if i == j:
                    # Histogram on diagonal
                    for sex_val in ["male", "female"]:
                        mask = [s == sex_val for s in data_dict["sex"]]
                        values = [data_dict[col_x][k] for k in range(len(data_dict["sex"])) if mask[k]]
                        ax.hist(values, alpha=0.5, label=sex_val, color=sex_colors[sex_val], bins=8)
                    if i == 0:
                        ax.legend(fontsize=8)
                else:
                    # Scatter plot
                    for sex_val in ["male", "female"]:
                        mask = [s == sex_val for s in data_dict["sex"]]
                        x_vals = [data_dict[col_x][k] for k in range(len(data_dict["sex"])) if mask[k]]
                        y_vals = [data_dict[col_y][k] for k in range(len(data_dict["sex"])) if mask[k]]
                        ax.scatter(x_vals, y_vals, alpha=0.6, label=sex_val, color=sex_colors[sex_val], s=30)
                
                if i == n_cols - 1:
                    ax.set_xlabel(col_x, fontsize=9)
                else:
                    ax.set_xticklabels([])
                
                if j == 0:
                    ax.set_ylabel(col_y, fontsize=9)
                else:
                    ax.set_yticklabels([])
        
        title = f"Relacao entre medidas e sexo - {species.capitalize()}"
        fig.suptitle(title, fontsize=12)
        path = out_dir / f"grafico_{file_idx:02d}_pairplot_{species}_sexo.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        files.append(str(path))
    
    return files

