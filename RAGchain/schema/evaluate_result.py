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

    def __add__(self, other):
        if not isinstance(other, EvaluateResult):
            raise TypeError("unsupported operand type(s) for +: 'EvaluateResult' and '{}'".format(type(other)))
        if self.use_metrics != other.use_metrics:
            raise ValueError("use_metrics must be same for using + operator at EvaluateResult")

        new_each_results = pd.concat([self.each_results, other.each_results], axis=0).reset_index(drop=True)
        new_results = new_each_results[self.use_metrics].mean(axis=0).to_dict()
        return EvaluateResult(
            results=new_results,
            use_metrics=self.use_metrics,
            each_results=new_each_results
        )
