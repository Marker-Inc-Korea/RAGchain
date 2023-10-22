import pandas as pd
import pytest

from RAGchain.schema import EvaluateResult


def test_evaluate_result_add():
    test1 = EvaluateResult(
        results={
            'Recall': 0.4,
            'F1_score': 0.2
        },
        use_metrics=['Recall', 'F1_score'],
        each_results=pd.DataFrame({
            'question': ['q1', 'q2'],
            'Recall': [0.0, 0.8],
            'F1_score': [0.3, 0.1]
        })
    )
    test2 = EvaluateResult(
        results={
            'Recall': 0.6,
            'F1_score': 0.3
        },
        use_metrics=['Recall', 'F1_score'],
        each_results=pd.DataFrame({
            'question': ['q3', 'q4'],
            'Recall': [0.3, 0.9],
            'F1_score': [0.4, 0.2]
        })
    )
    test3 = EvaluateResult(
        results={
            'Recall': 0.15,
            'Accuracy': 0.75
        },
        use_metrics=['Recall', 'Accuracy'],
        each_results=pd.DataFrame({
            'question': ['q5', 'q6'],
            'Recall': [0.05, 0.25],
            'Accuracy': [0.7, 0.8]
        })
    )

    add_result = test1 + test2
    assert add_result.results == {
        'Recall': 0.5,
        'F1_score': 0.25
    }
    assert add_result.use_metrics == ['Recall', 'F1_score']
    assert add_result.each_results.equals(pd.DataFrame({
        'question': ['q1', 'q2', 'q3', 'q4'],
        'Recall': [0.0, 0.8, 0.3, 0.9],
        'F1_score': [0.3, 0.1, 0.4, 0.2]
    }))
    with pytest.raises(ValueError):
        test1 + test3
    with pytest.raises(TypeError):
        test1 + 1
