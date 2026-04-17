import duckdb
import polars as pl
from pathlib import Path
from typing import Union


def load_penguins(path: Union[str, Path] = "dataset/penguins.csv") -> pl.DataFrame:
    """Load the penguins dataset using DuckDB and return a Polars DataFrame.

    The function also standardizes column names and casts numeric columns.
    """
    path = str(path)
    safe_path = path.replace("'", "''")
    con = duckdb.connect()
    rel = con.execute(
        f"""
        SELECT *
        FROM read_csv_auto(
            '{safe_path}',
            nullstr=['NA', 'na', '']
        )
        """
    )
    # Convert Arrow directly to Polars (no Pandas intermediary).
    df = pl.from_arrow(rel.arrow())
    df = _standardize_columns(df)

    # At this point we already normalized common NA markers via pandas->polars,
    # so no additional per-column replacement is required.

    # Ensure numeric columns are cast to floats when present
    casts = []
    if "bill_length_mm" in df.columns:
        casts.append(pl.col("bill_length_mm").cast(pl.Float64))
    if "bill_depth_mm" in df.columns:
        casts.append(pl.col("bill_depth_mm").cast(pl.Float64))
    if "flipper_length_mm" in df.columns:
        casts.append(pl.col("flipper_length_mm").cast(pl.Float64))
    if "body_mass_g" in df.columns:
        casts.append(pl.col("body_mass_g").cast(pl.Float64))
    if casts:
        df = df.with_columns(casts)

    # Normalize selected string columns without overriding nulls.
    for c in ["species", "island", "sex"]:
        if c in df.columns:
            df = df.with_columns([pl.col(c).str.to_lowercase()])

    return df


def _standardize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename messy Portuguese headers to stable English identifiers.

    Handles the duplicated header 'profundidade do bico' by inspecting
    median values to decide which column is actually flipper length.
    """
    cols = df.columns
    lower = [c.strip().lower() for c in cols]
    rename_map = {}

    # DuckDB may suffix duplicated columns as "_1", so match by prefix.
    depth_prefix = "profundidade do bico"
    idxs = [i for i, c in enumerate(lower) if c.startswith(depth_prefix)]
    if len(idxs) >= 2:
        for idx in idxs:
            series = df[cols[idx]].drop_nulls()
            try:
                median = float(series.median()) if len(series) > 0 else 0
            except Exception:
                median = 0
            # flipper lengths are ~170-230, so median > 100 indicates flipper
            if median and median > 100:
                rename_map[cols[idx]] = "flipper_length_mm"
            else:
                rename_map[cols[idx]] = "bill_depth_mm"

    for c in cols:
        lc = c.strip().lower()
        if lc == "espece":
            rename_map[c] = "species"
        elif lc == "ilha":
            rename_map[c] = "island"
        elif lc == "largura do bico":
            rename_map[c] = "bill_length_mm"
        elif lc == "massa corporal":
            rename_map[c] = "body_mass_g"
        elif lc == "sexo":
            rename_map[c] = "sex"

    df = df.rename(rename_map)

    return df
