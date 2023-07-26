from benchmark.retrival.recall import RecallFactory
from benchmark.retrival.precision import PrecisionFactory
#from benchmark.retrival.f1 import F1Factory
#from benchmark.retrival.average_precision import AveragePrecisionFactory
from benchmark.retrival.ndcg import NDCGFactory
from benchmark.retrival.map import MAPFactory
from benchmark.retrival.mrr import MRRFactory
#from benchmark.retrival.pfound import PFoundFactory
from benchmark.retrival.hole import HoleFactory
from benchmark.retrival.recall_cap import RecallCapFactory
from benchmark.retrival.top_k_accuracy import TopKAccuracyFactory
from typing import List, Dict, Union, Tuple

def retrival_evaluation_master(qrels: Dict[str, Dict[str, int]],
                               results: Dict[str, Dict[str, float]],
                               metrics: List[str], k_values: List[int]) -> List[dict[str, float]]:
    """
    This function is the master function for the retrival evaluation.
    It calls the other functions and returns a dictionary with the metrics as keys and the values as values.
    """
    evaluation_results = []
    for k in k_values:
        for metric in metrics:
            if metric in ["recall", "Recall", "RECALL"]:
                metric_factory = RecallFactory(k)
            elif metric in ["precision", "Precision", "PRECISION"]:
                metric_factory = PrecisionFactory(k)
            elif metric in ["NDCG", "ndcg", "Ndcg"]:
                metric_factory = NDCGFactory(k)
            elif metric in ["MAP", "map", "Map"]:
                metric_factory = MAPFactory(k)
            elif metric in ["MRR", "mrr", "Mrr"]:
                metric_factory = MRRFactory(k)
            elif metric in ["HOLE", "hole", "Hole"]:
                metric_factory = HoleFactory(k)
            elif metric in ["RECALLCAP", "recallcap", "Recallcap", "RecallCap"]:
                metric_factory = RecallCapFactory(k)
            elif metric in ["TOPKACCURACY", "topkaccuracy", "Topkaccuracy", "TopKAccuracy", "topKAccuracy", "Accuracy", "accuracy", "ACCURACY", "Acc", "acc", "ACC"]:
                metric_factory = TopKAccuracyFactory(k)
            else:
                print(f"Metric {metric} not found.")
            evaluation_results.append(metric_factory.eval(qrels, results, k))

    return evaluation_results