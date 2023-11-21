from RAGchain.benchmark.answer.metrics import *

gt_solutions = ['The dog had bit the man.', 'The man had bitten the dog.']
pred_answer = 'The dog bit the man.'
metric_answers = [
    (BLEU(), 51.1507),
]


def test_answer_metrics():
    for metric, answer in metric_answers:
        assert math.isclose(metric.eval(gt_solutions, pred_answer), answer, abs_tol=0.01)