"""Data analysis functions using Polars only (no Pandas)."""

from typing import Dict, List
import polars as pl


def missing_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Count missing values per column using Polars native operations."""
    missing_counts = [
        {"column": col, "missing_count": df[col].null_count()}
        for col in df.columns
    ]
    return pl.DataFrame(missing_counts).sort("missing_count", descending=True)


def rows_with_missing(df: pl.DataFrame) -> pl.DataFrame:
    """Return rows that contain at least one missing value."""
    exprs = [pl.col(c).is_null() for c in df.columns]
    return df.filter(pl.any_horizontal(exprs))


def island_counts(df: pl.DataFrame) -> pl.DataFrame:
    """Return counts of penguins per island, sorted descending."""
    return df.group_by("island").agg(pl.len().alias("count")).sort("count", descending=True)


def species_counts(df: pl.DataFrame) -> pl.DataFrame:
    """Return counts of penguins per species, sorted descending."""
    return df.group_by("species").agg(pl.len().alias("count")).sort("count", descending=True)


def species_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Compute mean measurements per species using Polars native operations."""
    return (
        df.group_by("species")
        .agg(
            pl.col("bill_length_mm").mean().alias("bill_length_mm_mean"),
            pl.col("bill_depth_mm").mean().alias("bill_depth_mm_mean"),
            pl.col("flipper_length_mm").mean().alias("flipper_length_mm_mean"),
            pl.col("body_mass_g").mean().alias("body_mass_g_mean"),
        )
        .sort("species")
    )


def sex_species_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Compute mean measurements per species and sex using Polars native operations."""
    return (
        df.filter(pl.col("sex").is_not_null())
        .group_by("species", "sex")
        .agg(
            pl.len().alias("n"),
            pl.col("bill_length_mm").mean().alias("bill_length_mm_mean"),
            pl.col("bill_depth_mm").mean().alias("bill_depth_mm_mean"),
            pl.col("flipper_length_mm").mean().alias("flipper_length_mm_mean"),
            pl.col("body_mass_g").mean().alias("body_mass_g_mean"),
        )
        .sort("species", "sex")
    )


def numeric_columns(df: pl.DataFrame) -> List[str]:
    """Return a list of numeric column names detected in the DataFrame."""
    return [c for c, dt in zip(df.columns, df.dtypes) if ("Float" in str(dt)) or ("Int" in str(dt))]


def compute_all_analyses(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """Compute all analysis summaries at once."""
    return {
        "missing_summary": missing_summary(df),
        "missing_rows": rows_with_missing(df),
        "island_counts": island_counts(df),
        "species_counts": species_counts(df),
        "species_stats": species_stats(df),
        "sex_species_stats": sex_species_stats(df),
    }
