from base_retrieval_evaluation_factory import RecallFactory, PrecisionFactory, NDCGFactory, MAPFactory, MRRFactory, \
    HoleFactory, TopKAccuracyFactory #RecallCapFactory

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
    evaluation_results = []
    metrics_factory = [RecallFactory, PrecisionFactory, NDCGFactory, MAPFactory, MRRFactory, HoleFactory,
                       TopKAccuracyFactory]#RecallCapFactory

    max_length = max(len(retrieved_docs) for retrieved_docs in results.values())
    fit_k_values = filter(lambda k : (max_length >= k >= 1), k_values)
    
    if k_values != fit_k_values:
        print(f'{list(set(k_values)-set(fit_k_values))} is not used in the evaluation. please enter 1< k_value <num of length of the retrieved documents.')
    print(f'evaluation will be done for the following k values: {fit_k_values}')

    for k in fit_k_values:
        for metric_factory in metrics_factory:
            metric = metric_factory()
            print(str(metric))
            evaluation_results.append(metric.retrieval_metric_function(qrels, results, k))

    return evaluation_results

def stretagyQA(solution: dict, pred: dict):
    qrels = dict()
    results = dict()
    for key in solution.keys():
        paragraphs = pred[key]['paragraphs']
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
        for _, evidence in enumerate(evidence_per_annotator):
            qrel_dict = {}
            paragraph_dict = {}
            for e in evidence:
                qrel_dict[e] = 1
            for p in paragraphs:
                paragraph_dict[p] = 1/len(paragraphs)
            qrels[str(key)+f"_{_}"] = qrel_dict
            results[str(key)+f"_{_}"] = paragraph_dict

    return qrels, results

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
    # qrels, results = stretagyQA(solution, prediction)

    print(f'Metric : {retrieval_evaluation_master(qrels, results, [10])}')


if __name__ == '__main__':
    main()

