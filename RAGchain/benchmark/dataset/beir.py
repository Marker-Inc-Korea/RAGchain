from typing import List, Optional

from RAGchain.benchmark.dataset.base import BaseBeirEvaluator
from RAGchain.pipeline.base import BaseRunPipeline


class BeirFEVEREvaluator(BaseBeirEvaluator):
    """
    BeirFEVEREvaluator is a class for evaluating pipeline performance on beir FEVER datasets.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None
                 ):
        file_path = "BeIR/fever"
        # Create support metrics
        super().__init__(run_pipeline=run_pipeline, file_path=file_path, evaluate_size=evaluate_size, metrics=metrics)


class BeirFIQAEvaluator(BaseBeirEvaluator):
    """
    BeirFIQAEvaluator is a class for evaluating pipeline performance on beir fiqa datasets.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None
                 ):
        file_path = "BeIR/fiqa"
        # Create support metrics
        super().__init__(run_pipeline=run_pipeline, file_path=file_path, evaluate_size=evaluate_size, metrics=metrics)


class BeirHOTPOTQAEvaluator(BaseBeirEvaluator):
    """
    BeirHOTPOTQAEvaluator is a class for evaluating pipeline performance on beir hotpotqa datasets.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None
                 ):
        file_path = "BeIR/hotpotqa"
        # Create support metrics
        super().__init__(run_pipeline=run_pipeline, file_path=file_path, evaluate_size=evaluate_size, metrics=metrics)


class BeirQUORAEvaluator(BaseBeirEvaluator):
    """
    BeirQUORAEvaluator is a class for evaluating pipeline performance on beir quora datasets.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None
                 ):
        file_path = "BeIR/quora"
        # Create support metrics
        super().__init__(run_pipeline=run_pipeline, file_path=file_path, evaluate_size=evaluate_size, metrics=metrics)


class BeirSCIDOCSEvaluator(BaseBeirEvaluator):
    """
    BeirSCIDOCSEvaluator is a class for evaluating pipeline performance on beir scidocs datasets.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None
                 ):
        file_path = "BeIR/scidocs"
        # Create support metrics
        super().__init__(run_pipeline=run_pipeline, file_path=file_path, evaluate_size=evaluate_size, metrics=metrics)


class BeirSCIFACTEvaluator(BaseBeirEvaluator):
    """
    BeirSCIFACTEvaluator is a class for evaluating pipeline performance on beir scifact datasets.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None
                 ):
        file_path = "BeIR/scifact"
        # Create support metrics
        super().__init__(run_pipeline=run_pipeline, file_path=file_path, evaluate_size=evaluate_size, metrics=metrics)
