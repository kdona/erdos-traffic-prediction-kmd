"""Centralized feature set definitions for tabular and sequence models.

The repo previously duplicated time / event / lag / road geometry feature lists
across multiple scripts (LSTM, ST-GNN, comparison, event impact). This module
provides a single authoritative source to keep things tidy and consistent.

Design:
- Tabular models NEED explicit lag features (they see one row at a time).
- Sequence models (LSTM, ST-GNN) learn temporal patterns from sequences; lag
  features are redundant and excluded by default.

Functions:
    feature_sets(): returns a dict {'tabular': {...}, 'sequence': {...}}
    get_sequence_full(include_lags: bool=False): convenience accessor for LSTM/ST-GNN.
"""
from __future__ import annotations

from typing import Dict, List

def feature_sets() -> Dict[str, Dict[str, List[str]]]:
    time_features = [
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'hour_of_week_sin', 'hour_of_week_cos', 'is_weekend'
    ]
    evt_features = ['evt_cat_unplanned', 'evt_cat_planned']
    lag_features_raw = ['lag1_tt_per_mile', 'lag2_tt_per_mile', 'lag3_tt_per_mile']
    # Some tabular pipelines create log_ versions; we keep raw here for clarity.
    tmc_features = ['miles', 'reference_speed', 'curve', 'onramp', 'offramp']

    tabular = {
        'base': tmc_features,
        'base_lags': tmc_features + lag_features_raw,
        'evt': evt_features + tmc_features,
        'evt_lags': evt_features + tmc_features + lag_features_raw,
        'cyc': time_features + tmc_features,
        'cyc_lags': time_features + tmc_features + lag_features_raw,
        'full': time_features + evt_features + lag_features_raw + tmc_features,
    }

    sequence = {
        'base': tmc_features,
        'evt': evt_features + tmc_features,
        'cyc': time_features + tmc_features,
        'full': time_features + evt_features + tmc_features,  # no raw lags
        'full_with_lags': time_features + evt_features + lag_features_raw + tmc_features,  # optional experiment
    }

    return {'tabular': tabular, 'sequence': sequence}


def get_sequence_full(include_lags: bool = False) -> List[str]:
    fs = feature_sets()['sequence']
    return fs['full_with_lags'] if include_lags else fs['full']
