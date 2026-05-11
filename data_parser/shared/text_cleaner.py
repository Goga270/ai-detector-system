import pandas as pd

from cyberleninka.src.cleaner import clean_dataframe


def apply_clean(df: pd.DataFrame, text_col: str = "text", **kwargs) -> pd.DataFrame:
    """Добавляет text_clean, text_length_clean."""
    return clean_dataframe(
        df, text_col=text_col, annotation_col=None, **kwargs
    )
