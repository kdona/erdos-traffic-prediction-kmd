"""
Generic helper utilities used across scripts.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

# Preprocessing and model utilities
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Optional: XGBoost
try:
    from xgboost import XGBRegressor  # type: ignore
    XGB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    XGB_AVAILABLE = False


# ==================
# Logging & filesystem
# ==================

def log(s: str) -> None:
    """Print a timestamped log message. Useful for tracking progress in long-running scripts."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {s}")


def ensure_output_dir(path: Path) -> None:
    """Create directory if it doesn't exist. Like 'mkdir -p' in bash."""
    path.mkdir(parents=True, exist_ok=True)


# ==================
# Data handling
# ==================

def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the prepared training dataset from parquet file.
    
    Args:
        data_path: Path to X_full_1h.parquet (or similar preprocessed data)
    
    Returns:
        DataFrame with MultiIndex (tmc_code, time_bin) and feature columns
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    log(f"Loading dataset: {data_path}")
    df = pd.read_parquet(data_path)
    log(f"Loaded shape: {getattr(df, 'shape', None)}")
    return df


def attach_time_index(df: pd.DataFrame, time_col: str = "time_bin") -> pd.DataFrame:
    """Ensure we can split chronologically by time.
    Accepts either a MultiIndex with level 'time_bin' or a column named 'time_bin'.
    """
    if isinstance(df.index, pd.MultiIndex) and time_col in df.index.names:
        return df
    # If it's a column, set as second-level index with any existing index as first.
    if time_col in df.columns:
        if df.index.name is None:
            df = df.set_index([df.index.rename("row_id"), time_col]).sort_index()
        else:
            df = df.set_index([df.index, time_col]).sort_index()
        return df
    raise KeyError(
        f"Expected '{time_col}' either as an index level or a column for chronological split.")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log transformations to travel time features.
    
    Why log transform? Travel times are right-skewed (occasional big delays).
    Log transform makes the distribution more normal, which helps many models.
    
    Creates log_tt_per_mile, log_lag1_tt_per_mile, etc. if the base features exist.
    """
    for base in ["tt_per_mile", "lag1_tt_per_mile", "lag2_tt_per_mile", "lag3_tt_per_mile"]:
        if base in df.columns:
            out = f"log_{base}"
            if out not in df.columns:
                # Add small epsilon (1e-6) to avoid log(0)
                df[out] = np.log(df[base].astype(float) + 1e-6)
    return df


def time_split(df: pd.DataFrame, test_ratio: float = 0.2, time_level: str = "time_bin") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train/test sets.
    
    IMPORTANT: We use chronological (time-based) splits, not random splits!
    This simulates real forecasting: train on past data, predict future data.
    
    Args:
        df: DataFrame with MultiIndex including time_level
        test_ratio: Fraction of time periods to use for testing (default 20%)
        time_level: Name of the time index level (default "time_bin")
    
    Returns:
        (train_df, test_df) where train comes before test chronologically
    """
    # Get unique time bins and sort them chronologically
    time_bins = df.index.get_level_values(time_level).unique().sort_values()
    split_idx = int(len(time_bins) * (1 - test_ratio))
    train_times = time_bins[:split_idx]
    test_times = time_bins[split_idx:]
    
    # Use pandas IndexSlice to select all TMCs for specific time periods
    df_train = df.loc[pd.IndexSlice[:, train_times], :]
    df_test = df.loc[pd.IndexSlice[:, test_times], :]
    return df_train, df_test


def balance_training(df_train: pd.DataFrame, event_col: str = "evt_total", negative_frac: float = 0.01, seed: int = 42) -> pd.DataFrame:
    """Balance training data by downsampling non-event periods.
    
    Why? Events are rare (~1% of hours have roadwork). Training on unbalanced data
    can make models just predict "no delay" all the time. This function keeps all
    event periods but samples only a fraction of non-event periods.
    
    Args:
        df_train: Training dataframe
        event_col: Column indicating event presence (>0 means event exists)
        negative_frac: Fraction of non-event rows to keep (0.01 = keep 1%)
        seed: Random seed for reproducibility
    
    Returns:
        Balanced and shuffled dataframe
    """
    if event_col not in df_train.columns:
        log(f"Warning: event_col '{event_col}' not in data. Skipping balancing.")
        return df_train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # Separate event and non-event periods
    any_event = df_train[event_col] > 0
    df_events = df_train[any_event]
    df_no_events = df_train[~any_event]
    
    # Downsample non-event periods
    frac = float(negative_frac)
    if frac <= 0 or frac >= 1:
        neg_sample = df_no_events
    else:
        neg_sample = df_no_events.sample(frac=frac, random_state=seed) if len(df_no_events) else df_no_events
    
    # Combine and shuffle
    df_balanced = pd.concat([df_events, neg_sample], axis=0)
    log(f"Balanced train: events={len(df_events):,}, non-events(sampled)={len(neg_sample):,}, total={len(df_balanced):,}")
    return df_balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ==================
# Feature name helpers & importances
# ==================

def get_feature_names(model) -> Optional[List[str]]:
    """Extract original feature names from the fitted model's preprocessor.
    Returns a list without the ColumnTransformer prefix (e.g., 'num__').
    """
    try:
        pipe = model.regressor_
        pre = pipe.named_steps['pre']
        raw = list(pre.get_feature_names_out())
        # Strip transformer prefix if present, e.g., 'num__feature'
        names = [n.split('__', 1)[1] if '__' in n else n for n in raw]
        return names
    except Exception:
        return None


def extract_linear_feature_importance(model, feature_names: List[str]) -> Optional[pd.DataFrame]:
    try:
        pipe = model.regressor_
        pre = pipe.named_steps['pre']
        reg = pipe.named_steps['reg']
        coefs = getattr(reg, 'coef_', None)
        if coefs is None:
            return None
        # For multi-output or 2D shapes
        coef_arr = np.ravel(coefs)
        return pd.DataFrame({
            'feature': feature_names,
            'coef': coef_arr
        }).sort_values('coef', key=np.abs, ascending=False)
    except Exception:
        return None


def extract_tree_feature_importance(model, feature_names: List[str]) -> Optional[pd.DataFrame]:
    try:
        pipe = model.regressor_
        pre = pipe.named_steps['pre']
        reg = pipe.named_steps['reg']
        importances = getattr(reg, 'feature_importances_', None)
        if importances is None:
            return None
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
    except Exception:
        return None


# ==================
# Model factories & preprocessors
# ==================

def make_preprocessor(features: List[str], scale: bool = True) -> ColumnTransformer:
    if scale:
        return ColumnTransformer([
            ('num', StandardScaler(with_mean=False), features)
        ], remainder='drop')
    return ColumnTransformer([
        ('num', 'passthrough', features)
    ], remainder='drop')


def make_model(preprocessor: ColumnTransformer, regressor, transform_target: bool = True) -> TransformedTargetRegressor | Pipeline:
    pipe = Pipeline([
        ('pre', preprocessor),
        ('reg', regressor)
    ])
    if transform_target:
        log_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=False)
        return TransformedTargetRegressor(regressor=pipe, transformer=log_transformer)
    return pipe


def make_rf(**kwargs) -> RandomForestRegressor:
    # Hyperparameters tuned for tabular travel-time prediction
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=kwargs.pop('random_state', 42),
        **kwargs
    )


def make_gbrt(**kwargs) -> GradientBoostingRegressor:
    # hyperparameters tuned for tabular travel-time prediction
    return GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=kwargs.pop('random_state', 42),
        **kwargs
    )


def make_xgb(**kwargs):
    """hyperparameters tuned for tabular travel-time prediction"""
    if not XGB_AVAILABLE:
        raise RuntimeError("xgboost is not available")
    params = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.9,
        reg_lambda=0.3,
        reg_alpha=0.5,
        min_child_weight=20,
        n_jobs=-1,
        tree_method='hist',  # faster for large datasets
        objective='reg:squarederror',
        random_state=kwargs.pop('random_state', 42),
    )
    params.update(kwargs)
    return XGBRegressor(**params)
