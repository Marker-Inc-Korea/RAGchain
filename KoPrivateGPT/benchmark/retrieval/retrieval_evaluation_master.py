from base_retrieval_evaluation_factory import RecallFactory, RRFactory, PrecisionFactory, NDCGFactory, DCGFactory, HoleFactory, TopKAccuracyFactory, IDCGFactory, IndDCGFactory, IndIDCGFactory, APFactory, CGFactory

from typing import List, Dict

import json
import click

def basic_retrieval_evaluation(qrels: Dict[str, Dict[str, int]],
                                    results: Dict[str, Dict[str, float]],
                                    k_values: List[int]) -> List[dict[str, float]]:
    """
    :param qrels: The qrels file as a dictionary. Dict[query_id, Dict[doc_id, relevance]]
    :param results: The results file as a dictionary. Dict[query_id, Dict[doc_id, score]]
    :k_values: The k values for which the evaluation should be done. List[int]
    results doc_id can be different from the doc_id in the qrels file.
    """

    metrics_factories = [RecallFactory, RRFactory, PrecisionFactory, NDCGFactory, DCGFactory, HoleFactory, TopKAccuracyFactory, IDCGFactory, IndDCGFactory, IndIDCGFactory, APFactory, CGFactory]

    score_dict = dict()
    for k in k_values:
        for metric_factory in metrics_factories:
            score_dict[f'{metric_factory().metric_name}@{str(k)}'] = metric_factory().eval(qrels, results, k=k)
        Key_dict = {f'{RRFactory().metric_name}@{str(k)}': f'MRR@{str(k)}',
                    f'{APFactory().metric_name}@{str(k)}': f'MAP@{str(k)}'
                    }
        for convert_key in Key_dict.keys():
            if convert_key in score_dict.keys():
                score_dict[Key_dict[convert_key]] = score_dict.pop(convert_key)
    mean_scores = {key: round(sum(value) / len(value), 5) for key, value in score_dict.items()}
    return mean_scores

def stretagyqa_retrieval_evaluation(qrels: Dict[str, Dict[str, int]],
                                    results: Dict[str, Dict[str, float]],
                                    k_values: List[int]) -> List[dict[str, float]]:
    """
    :param qrels: The qrels file as a dictionary. Dict[query_id, Dict[doc_id, relevance]]
    :param results: The results file as a dictionary. Dict[query_id, Dict[doc_id, score]]
    :k_values: The k values for which the evaluation should be done. List[int]
    results doc_id can be different from the doc_id in the qrels file.
    """

    metrics_factories = [RecallFactory, RRFactory, PrecisionFactory, NDCGFactory, DCGFactory, HoleFactory, TopKAccuracyFactory, IDCGFactory, IndDCGFactory, IndIDCGFactory, APFactory, CGFactory]

    score_dict = dict()
    for k in k_values:
        score_dict = score_dict | strategyQA(qrels, results, metrics_factories, k)

    mean_scores = {key: round(sum(value) / len(value), 5) for key, value in score_dict.items()}
    return mean_scores

def strategyQA(solution: dict, pred: dict, metrics_factories: list, k: int) -> dict:
    '''
    k should be smaller than the number of paragraphs in the prediction
    '''
    final_score = {f'{metric_factory().metric_name}@{str(k)}': list() for metric_factory in metrics_factories}
    for key in solution.keys():
        paragraphs = pred[key]['paragraphs']
        if len(paragraphs) < k:
            break
        evidence_per_annotator = list()
        for annotator in solution[key]['evidence']:
            evidence_per_annotator.append(
                set(
                    evidence_id
                    for step in annotator
                    for x in step
                    if isinstance(x, list)
                    for evidence_id in x
                )
            )

        for metric_factory in metrics_factories:
            score_per_annotator = list()
            for evidence in evidence_per_annotator:
                metric = metric_factory()
                score = metric.eval(solution={key: 1 for key in evidence},
                                    pred={key: 1.0 for key in paragraphs[0:k]}, k=k) if len(
                    evidence) > 0 else 0
                score_per_annotator.append(score)
            annotator_max = max(score_per_annotator)
            final_score[f'{metric.metric_name}@{str(k)}'].append(annotator_max)

    Key_dict = {f'{RRFactory().metric_name}@{str(k)}': f'MRR@{str(k)}',
            f'{APFactory().metric_name}@{str(k)}': f'MAP@{str(k)}'
            }
    for convert_key in Key_dict.keys():
        if convert_key in final_score.keys():
            final_score[Key_dict[convert_key]] = final_score.pop(convert_key)

    return final_score

@click.command()
@click.option('--pred', type=click.Path(exists=True), help='prediction file')
@click.option('--sol', type=click.Path(exists=True), help='solution file')
def main(pred, sol):
    if not pred.endswith('.json'):
        raise ValueError('prediction file must be json file.')
    if not sol.endswith('.json'):
        raise ValueError('solution file must be json file.')

    with open(pred, 'r') as f:
        prediction = json.load(f)
    with open(sol, 'r') as f:
        solution = json.load(f)

    # print(f'Metric : {basic_retrieval_evaluation(solution, prediction, [1, 5, 10])}')
    print(f'Metric : {stretagyqa_retrieval_evaluation(solution, prediction, [1, 5, 10])}')


if __name__ == '__main__':
    main()

