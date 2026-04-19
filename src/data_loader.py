import polars as pl
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Palmer Penguins Dataset URL
# Source: Palmer Station Antarctica LTER & Dr. Kristen Gorman
# License: CC-0 (Creative Commons Zero)
# Reference: https://allisonhorst.github.io/palmerpenguins/
PENGUINS_URL = "https://raw.githubusercontent.com/mcnakhaee/palmerpenguins/master/palmerpenguins/data/penguins.csv"


def load_penguins(url: Optional[str] = None) -> pl.DataFrame:
    """Load the Palmer Penguins dataset from remote source using Polars only.

    The Palmer Penguins dataset contains biometric measurements of 344 penguins
    from 3 species collected at Palmer Station, Antarctica (2007-2009).

    Args:
        url: Optional custom URL. Defaults to official Palmer Penguins repository.

    Returns:
        Polars DataFrame with standardized numeric and categorical columns.

    Raises:
        Exception: If dataset cannot be loaded from URL.

    References:
        - Gorman KB, Williams TD, Fraser WR (2014). Ecological sexual dimorphism 
          and environmental variability within a community of Antarctic penguins 
          (genus Pygoscelis). PLoS ONE 9(3):e90081.
          https://doi.org/10.1371/journal.pone.0090081
        
        - Dataset License: CC-0 (https://creativecommons.org/share-your-work/public-domain/cc0/)
    """
    url = url or PENGUINS_URL
    
    try:
        # Load directly using Polars (no DuckDB, no Pandas)
        # Treat "NA" strings as null values for numeric columns
        df = pl.read_csv(url, null_values=["NA", "na", ""])
    except Exception as e:
        raise Exception(f"Failed to load dataset from {url}: {e}")

    # Standardize: ensure numeric columns are Float64, strings are lowercase
    df = _standardize_types(df)

    return df


def _standardize_types(df: pl.DataFrame) -> pl.DataFrame:
    """Standardize data types: numerics → Float64, strings → lowercase.

    The original dataset has correct English column names already:
    - species, island, sex (categorical, lowercase)
    - bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, year (numeric)
    """
    # Cast numeric columns to Float64 (handles nulls gracefully)
    numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    for col in numeric_cols:
        if col in df.columns:
            try:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
            except Exception:
                # If casting fails, leave column as-is
                pass

    # Lowercase string columns (handles NA values naturally)
    for col in ["species", "island", "sex"]:
        if col in df.columns:
            try:
                df = df.with_columns(pl.col(col).str.to_lowercase())
            except Exception:
                # If operation fails, leave column as-is
                pass

    return df
