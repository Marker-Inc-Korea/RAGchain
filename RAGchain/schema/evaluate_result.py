from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class EvaluateResult:
    results: dict
    """Dictionary of metrics and their results"""
    use_metrics: List[str]
    """List of metrics to use"""
    each_results: pd.DataFrame
    """DataFrame of each results question by question"""
