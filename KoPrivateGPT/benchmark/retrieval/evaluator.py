from KoPrivateGPT.benchmark.retrieval.metrics import Recall, RR, Precision, NDCG, DCG, Hole, TopKAccuracy, IDCG, IndDCG, \
    IndIDCG, AP, CG, ExactlyMatch, F1

from typing import List, Dict

import json
import click


def basic_retrieval_evaluation(qrels: Dict[str, List[str]], preds: Dict[str, List[str]], k_values: List[int],
                               qrels_relevance: Dict[str, List[int]] = None,
                               preds_relevance: Dict[str, List[float]] = None
                               ) -> List[dict[str, float]]:
    """
    :param qrels: The qrels file as a dictionary. Dict[query_id, Dict[doc_id, relevance]]
    :param preds: The results file as a dictionary. Dict[query_id, Dict[doc_id, score]]
    :k_values: The k values for which the evaluation should be done. List[int]
    results doc_id can be different from the doc_id in the qrels file.
    """

    all_metrics = [Recall(), RR(), Precision(), NDCG(), DCG(), Hole(), TopKAccuracy(), IDCG(), IndDCG(), IndIDCG(),
                   AP(), CG(), ExactlyMatch(), F1()]
    binary_metrics = [TopKAccuracy(), ExactlyMatch(), F1(), Hole(), Recall(), Precision()]
    score_dict = dict()

    is_rank_aware = check_retrieval_eval(qrels, preds, qrels_relevance, preds_relevance, k_values)
    #
    metrics = binary_metrics if is_rank_aware else all_metrics
    # attaching relevance
    if is_rank_aware:
        for query_id in qrels.keys():
            qrels[query_id] = dict(zip(qrels[query_id], qrels_relevance[query_id]))
            preds[query_id] = dict(zip(preds[query_id], preds_relevance[query_id]))
    else:
        for query_id in qrels.keys():
            qrels[query_id] = dict(zip(qrels[query_id], [1] * len(qrels[query_id])))
            preds[query_id] = dict(zip(preds[query_id], [1] * len(preds[query_id])))

    for k in k_values:
        for metric in metrics:
            score_dict[f'{metric.metric_name}@{str(k)}'] = list()
            for query_id in qrels.keys():
                score_dict[f'{metric.metric_name}@{str(k)}'].append(metric.eval(qrels[query_id], preds[query_id], k=k))
        Key_dict = {f'{RR().metric_name}@{str(k)}': f'MRR@{str(k)}',
                    f'{AP().metric_name}@{str(k)}': f'MAP@{str(k)}'
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
    :param qrels: The qrels file as a dictionary.  note. "https://github.com/eladsegal/strategyqa/blob/main/data/strategyqa/dev.json"
    :param results: The results file as a dictionary. Dict[query_id, Dict[doc_id, score]]
    :k_values: The k values for which the evaluation should be done. List[int]
    results doc_id can be different from the doc_id in the qrels file.
    """

    #all_metrics = [Recall(), RR(), Precision(), NDCG(), DCG(), Hole(), TopKAccuracy(), IDCG(), IndDCG(), IndIDCG(),
    #               AP(), CG(), ExactlyMatch(), F1()]
    binary_metrics = [TopKAccuracy(), ExactlyMatch(), F1(), Hole(), Recall(), Precision()]

    score_dict = dict()
    for k in k_values:
        score_dict = score_dict | strategyQA(qrels, results, binary_metrics, k)

    mean_scores = {key: round(sum(value) / len(value), 5) for key, value in score_dict.items()}
    return mean_scores


def strategyQA(solution: dict, pred: dict, metrics: list, k: int) -> dict:
    '''
    @article{geva2021strategyqa,
      title = {{Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies}},
      author = {Geva, Mor and Khashabi, Daniel and Segal, Elad and Khot, Tushar and Roth, Dan and Berant, Jonathan},
      journal = {Transactions of the Association for Computational Linguistics (TACL)},
      year = {2021},
    }
    '''
    final_score = {f'{metric.metric_name}@{str(k)}': list() for metric in metrics}
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

        for metric in metrics:
            score_per_annotator = list()
            for evidence in evidence_per_annotator:

                score = metric.eval(solution={key: 1 for key in evidence},
                                    pred={key: 1.0 for key in paragraphs[0:k]}, k=k) if len(
                    evidence) > 0 else 0
                score_per_annotator.append(score)
            annotator_max = max(score_per_annotator)
            final_score[f'{metric.metric_name}@{str(k)}'].append(annotator_max)

    Key_dict = {f'{RR().metric_name}@{str(k)}': f'MRR@{str(k)}',
                f'{AP().metric_name}@{str(k)}': f'MAP@{str(k)}'
                }
    for convert_key in Key_dict.keys():
        if convert_key in final_score.keys():
            final_score[Key_dict[convert_key]] = final_score.pop(convert_key)

    return final_score


def check_retrieval_eval(qrels: Dict[str, List[str]], preds: Dict[str, List[str]], k_values: List[int],
                         qrels_relevance: Dict[str, List[int]] = None, preds_relevance: Dict[str, List[float]] = None
                         ):
    if set(qrels.keys()) - set(preds.keys()):
        print(f"{set(qrels.keys()) - set(preds.keys())}")
        raise ValueError('prediction Dictionary need to contain a whole query_ids in qrels.')

    if set(preds.keys()) - set(qrels.keys()):
        print("Warning: prediction Dictionary contain more query_ids than qrels. the mismatched ids will be ignored.")

    over_preds = [query_id for query_id, retrieved_docs in preds_relevance.items() if
                  len(retrieved_docs) < max(k_values)]
    min_length_preds = min([len(retrieved_docs) for retrieved_docs in preds_relevance.values()])

    if over_preds:
        raise ValueError(
            f'current min length of preds : {min_length_preds}, each number of retrieved docs need to be larger than k_value')

    if qrels_relevance or preds_relevance:
        if qrels_relevance and preds_relevance:
            if set(qrels.keys()) - set(qrels_relevance.keys()):
                raise ValueError('qrels_relevance need to contain a whole query_ids in qrels.')

            if set(preds.keys()) - set(preds_relevance.keys()):
                raise ValueError('preds_relevance need to contain a whole query_ids in preds.')

            if set(qrels_relevance.keys()) - set(qrels.keys()):
                print("Warning: qrels_relevance contain more query_ids than qrels. the mismatched ids will be ignored.")

            if set(preds_relevance.keys()) - set(preds.keys()):
                print("Warning: preds_relevance contain more query_ids than preds. the mismatched ids will be ignored.")
            return True
        else:
            if qrels_relevance:
                raise ValueError('please check preds_relevance')
            else:
                raise ValueError('please check qrels_relevance')
    else:
        return False


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
