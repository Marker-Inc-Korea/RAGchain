from base_retrieval_evaluation_factory import RecallFactory, RRFactory, PrecisionFactory, NDCGFactory, DCGFactory, HoleFactory, TopKAccuracyFactory, IDCGFactory, IndDCGFactory, IndIDCGFactory, APFactory, CGFactory

# from benchmark.retrival.f1 import F1Factory
# from benchmark.retrival.average_precision import AveragePrecisionFactory

# from benchmark.retrival.pfound import PFoundFactory

from typing import List, Dict, Union, Tuple

import json
import click


def retrieval_evaluation_master(qrels: Dict[str, Dict[str, int]],
                                results: Dict[str, Dict[str, float]],
                                k_values: List[int]) -> List[dict[str, float]]:
    """
    This function is the master function for the retrival evaluation.
    It calls the other functions and returns a dictionary with the metrics as keys and the values as values.
    :param qrels: The qrels file as a dictionary. Dict[query_id, Dict[doc_id, relevance]]
    :param results: The results file as a dictionary. Dict[query_id, Dict[doc_id, score]]
    :k_values: The k values for which the evaluation should be done. List[int]
    results doc_id can be different from the doc_id in the qrels file.
    """

    metrics_factories = [RecallFactory, RRFactory, PrecisionFactory, NDCGFactory, DCGFactory, HoleFactory, TopKAccuracyFactory, IDCGFactory, IndDCGFactory, IndIDCGFactory, APFactory, CGFactory]

    score_dict = dict()
    for k in k_values:
        score_dict = score_dict | strategyQA(qrels, results, metrics_factories, k)

    mean_scores = {key: sum(value) / len(value) for key, value in score_dict.items()}
    return mean_scores

def strategyQA(solution: dict, pred: dict, metrics_factories: list, k_value: int):
    '''
    k shuld be smaller than the number of paragraphs in the prediction
    '''
    final_score = {f'{metric_factory.__name__[:-7]}@{str(k_value)}' : list() for metric_factory in metrics_factories}
    for key in solution.keys():
        paragraphs = pred[key]['paragraphs']
        if len(paragraphs) < k_value:
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
        score_per_annotator = list()

        for metric_factory in metrics_factories:
            for evidence in evidence_per_annotator:
                metric = metric_factory()
                score = metric.eval(solution = {key: 1 for key in evidence}, pred = {key: 1.0 for key in paragraphs[0:k_value]}, k = k_value) if len(evidence) > 0 else 0
                score_per_annotator.append(score)
            annotator_max = max(score_per_annotator)
            final_score[f'{metric_factory.__name__[:-7]}@{str(k_value)}'].append(annotator_max)
        

    return final_score

@click.command()
@click.option('--pred', type=click.Path(exists=True), help='prediction file')
@click.option('--sol', type=click.Path(exists=True), help='solution file')
def main(pred, sol):
    if not pred.endswith('.json'):
        raise ValueError('prediction file must be json file.')

    with open(pred, 'r') as f:
        prediction = json.load(f)
    with open(sol, 'r') as f:
        solution = json.load(f)

    print(f'Metric : {retrieval_evaluation_master(solution, prediction, [1,5,10])}')


if __name__ == '__main__':
    main()

