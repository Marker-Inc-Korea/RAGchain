def strategyQA(solution: dict, pred: dict, metrics_factories: list, k_value: int):
    '''
    k shuld be smaller than the number of paragraphs in the prediction
    '''
    final_score = {f'{metric_factory.__name__[:-7]}@{str(k_value)}': list() for metric_factory in metrics_factories}
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
                score = metric.eval(solution={key: 1 for key in evidence},
                                    pred={key: 1.0 for key in paragraphs[0:k_value]}, k=k_value) if len(
                    evidence) > 0 else 0
                score_per_annotator.append(score)
            annotator_max = max(score_per_annotator)
            final_score[f'{metric_factory.__name__[:-7]}@{str(k_value)}'].append(annotator_max)

    return final_score