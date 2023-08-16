from base_retrieval_evaluation_factory import RecallFactory, RRFactory, PrecisionFactory, NDCGFactory, DCGFactory, HoleFactory, TopKAccuracyFactory, IDCGFactory, IndDCGFactory, IndIDCGFactory, APFactory, CGFactory
from strategyQA import strategyQA


from typing import List, Dict

import json
import click


def retrieval_evaluation_master(qrels: Dict[str, Dict[str, int]],
                                results: Dict[str, Dict[str, float]],
                                k_values: List[int]) -> List[dict[str, float]]:
    """
    This function is the master function for the retrieval evaluation.
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

    print(f'Metric : {retrieval_evaluation_master(solution, prediction, [1,5,10])}')


if __name__ == '__main__':
    main()

