import os
import pathlib
import pickle
from typing import List, Union
from uuid import UUID
import json

import pytest

from RAGchain.benchmark.retrieval.evaluator import basic_retrieval_evaluation, stretagyqa_retrieval_evaluation

basic_retrieval_eval_values = [
    ({"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, {"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, [1, 2, 3], None, None),
    ({"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, {"1": ["1", "2", "3", "4"], "2": ["3", "5"]}, [1], None, None),
    ({"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, {"1": ["4", "5", "6"], "2": ["6", "7", "8"]}, [1, 2, 3], None, None),
    ({"1": ["1", "2", "3"], "2": ["3", "5", "2"]},
     {"1": ["1", "2", "3"], "2": ["3", "5", "2"], "3": ["3", "5", "2"], "4": ["3", "5", "2"]}, [1, 2, 3], None, None),
    ({"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, {"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, [1, 2, 3],
     {"1": [1, 2, 3], "2": [3, 5, 2]}, {"1": [1, 2, 3], "2": [3, 5, 2]}),
    ({"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, {"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, [1, 2, 3],
     {"1": [1.0, 2.0, 3.0], "2": [3.0, 5.0, 2.0]}, {"1": [1.0, 2.0, 3.0], "2": [3.0, 5.0, 2.0]}),
    ({"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, {"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, [1, 2, 3],
     {"1": [1, 2, 3], "2": [3, 5, 2]}, {"1": [1.0, 2.0, 3.0], "2": [3.0, 5.0, 2.0], "3": [1.0, 2.0, 3.0]}),
    pytest.param({"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, {"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, [1, 4],
                 None, None, marks=pytest.mark.xfail),  # too big k
    pytest.param({"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, {"1.5": ["1", "2", "3"], "2.3": ["3", "5", "2"]}, [1, 4],
                 None, None, marks=pytest.mark.xfail),  # qrels id not in preds
    pytest.param({"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, {"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, [1, 2, 3],
                 {"1.5": [1, 2, 3], "2": [3, 5, 2]}, {"1.5": [1, 2, 3], "2": [3, 5, 2]}, marks=pytest.mark.xfail),
    pytest.param({"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, {"1": ["1", "2", "3"], "2": ["3", "5", "2"]}, [1, 2, 3],
                 {"1": [1, 2, 3], "1": [2, 4, 5], "3": [3, 5, 2]}, {"1": [1, 2, 3], "2": [3, 5, 2]},
                 marks=pytest.mark.xfail)
]
root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
basic_retrieval_eval_value_ids = [f'{order}' for order in range(len(basic_retrieval_eval_values))]

test_strategyqa_gold_small_json_path = os.path.join(root_dir, "resources", "retrieval_files", "ko-strategy-qa_dev.json")
with open(test_strategyqa_gold_small_json_path, 'r') as f:
    strategyqa_gold_small_json = json.load(f)

test_strategyqa_predictions_small_json_path = os.path.join(root_dir, "resources", "retrieval_files", "dev_pred.json")
with open(test_strategyqa_predictions_small_json_path, 'r') as f:
    strategyqa_predictions_small = json.load(f)

stretagy_qa_values = [
    (strategyqa_gold_small_json, strategyqa_predictions_small, [1, 5, 10])
]

stretagy_qa_value_ids = [f'{order}' for order in range(len(stretagy_qa_values))]


@pytest.mark.parametrize('qrels,preds,k_values,qrels_relevance,preds_relevance', basic_retrieval_eval_values,
                         ids=basic_retrieval_eval_value_ids)
def test_basic_retrieval_evaluation(qrels, preds, k_values, qrels_relevance, preds_relevance):
    assert basic_retrieval_evaluation(qrels, preds, k_values, qrels_relevance, preds_relevance)


@pytest.mark.parametrize('qrels,preds,k_values', stretagy_qa_values, ids=stretagy_qa_value_ids)
def test_stretagyqa_retrieval_evaluation(qrels, preds, k_values):
    assert stretagyqa_retrieval_evaluation(qrels, preds, k_values)
