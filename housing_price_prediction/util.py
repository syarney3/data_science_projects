import pandas as pd
from sklearn.preprocessing import LabelEncoder

def cap_outliers_percentile(df: pd.DataFrame,
                            columns: list,
                            lower_pct: float = 0.05,
                            upper_pct: float = 0.95
                            ) -> pd.DataFrame:
    """
    Replace outliers in specified columns using percentile capping.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of column names to cap
    lower_pct : float, default=0.01
        Lower percentile (e.g., 1%)
    upper_pct : float, default=0.99
        Upper percentile (e.g., 99%)

    Returns
    -------
    pd.DataFrame
        DataFrame with capped outliers
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

        lower = df[col].quantile(lower_pct)
        upper = df[col].quantile(upper_pct)

        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def impute_median(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Replace missing values in specified columns with the column median.

    Parameters:
        df (pd.DataFrame): Input dataframe
        columns (list): List of column names to impute

    Returns:
        pd.DataFrame: Dataframe with missing values replaced
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            median_value = df[col].median()
            df[col] = df[col].fillna(df[col].median())

    return df

def label_encode_columns(df, cols):
    df_encoded = df.copy()
    le = LabelEncoder()
    
    for col in cols:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    return df_encoded
