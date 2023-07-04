import click
import json


def _recall(relevant_paragraphs, retrieved_paragraphs):
    result = len(set(relevant_paragraphs).intersection(retrieved_paragraphs)) / len(
        relevant_paragraphs
    )
    return result


def recall(solution: dict, pred: dict):
    result_score = list()
    for key in solution.keys():
        paragraphs = pred[key]['paragraphs']
        evidence_per_annotator = list()
        score_per_annotator = list()
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
        for evidence in evidence_per_annotator:
            score = _recall(evidence, paragraphs)
            score_per_annotator.append(score)
        annotator_max = max(score_per_annotator)
        result_score.append(annotator_max)
    return float(sum(result_score)) / len(result_score)


def accuracy(solution: dict, pred: dict):
    total_cnt = 0
    right_cnt = 0
    for key in solution.keys():
        try:
            prediction = pred[key]['answer']
        except:
            print(f'{key} is not existed in prediction file.')
            continue
        total_cnt += 1
        prediction = bool(prediction)
        if bool(solution[key]['answer']) == prediction:
            right_cnt += 1

    return float(right_cnt) / total_cnt


@click.command()
@click.option('--pred', type=click.Path(exists=True), help='prediction file')
def main(pred):
    if not pred.endswith('.json'):
        raise ValueError('prediction file must be json file.')

    with open(pred, 'r') as f:
        prediction = json.load(f)
    with open('./ko-strategy-qa_dev.json', 'r') as f:
        solution = json.load(f)

    print(f'Accuracy : {accuracy(solution, prediction)}')
    print(f'Recall : {recall(solution, prediction)}')


if __name__ == '__main__':
    main()
